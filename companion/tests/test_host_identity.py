from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from companion.hosting.identity import IdentityStore, InvalidCredential


@pytest.fixture
def identities(tmp_path):
    now = [1000.0]
    store = IdentityStore(tmp_path / "private" / "host.sqlite", clock=lambda: now[0])
    organization = store.create_organization("Test organization")
    user = store.create_user(organization, "Alice")
    return store, organization, user, now


def test_pairing_replay_hash_storage_and_restart(identities):
    store, organization, user, _ = identities
    invite = store.invite(user)
    paired = store.redeem(invite, "Desktop")
    assert paired.principal.organization_id == organization
    assert paired.principal.user_id == user
    assert store.authenticate(paired.token) == paired.principal
    assert paired.token not in repr(paired)
    with pytest.raises(InvalidCredential):
        store.redeem(invite, "Replay")
    assert invite.encode() not in store.path.read_bytes()
    assert paired.token.encode() not in store.path.read_bytes()
    reopened = IdentityStore(store.path, clock=lambda: 1001)
    assert reopened.authenticate(paired.token) == paired.principal
    assert reopened.devices(user) == [{
        "id": paired.principal.device_id, "name": "Desktop", "expires_at": paired.principal.expires_at,
    }]


def test_concurrent_redemption_has_one_winner(identities):
    store, _, user, _ = identities
    invite = store.invite(user)

    def redeem(_):
        try:
            return store.redeem(invite, "Concurrent device")
        except InvalidCredential:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(redeem, range(12)))
    assert sum(result is not None for result in results) == 1
    assert len(store.devices(user)) == 1


def test_expiration_and_revocation_fail_closed(identities):
    store, _, user, now = identities
    expired = store.invite(user, ttl=1)
    now[0] += 1
    with pytest.raises(InvalidCredential):
        store.redeem(expired, "Expired")
    first = store.redeem(store.invite(user), "First")
    second = store.redeem(store.invite(user), "Second")
    assert first.principal.owner == second.principal.owner
    assert store.revoke_device(first.principal.device_id)
    assert not store.revoke_device(first.principal.device_id)
    with pytest.raises(InvalidCredential):
        store.authenticate(first.token)
    assert store.authenticate(second.token) == second.principal
    now[0] = second.principal.expires_at
    with pytest.raises(InvalidCredential):
        store.authenticate(second.token)


def test_disable_user_revokes_devices_and_pending_invites(identities):
    store, _, user, _ = identities
    paired = store.redeem(store.invite(user), "Device")
    pending = store.invite(user)
    assert store.disable_user(user)
    for attempt in (lambda: store.authenticate(paired.token), lambda: store.redeem(pending, "Late")):
        with pytest.raises(InvalidCredential):
            attempt()
    with pytest.raises(ValueError):
        store.invite(user)
    assert store.devices(user) == []


def test_users_and_organizations_do_not_share_quota_identity(identities):
    store, org, user, _ = identities
    other_org = store.create_organization("Other")
    users = [user, store.create_user(org, "Alice"), store.create_user(other_org, "Alice")]
    owners = {store.redeem(store.invite(u), "Desktop").principal.owner for u in users}
    assert len(owners) == 3


def test_pending_invites_and_devices_are_bounded(identities):
    store, _, user, now = identities
    for _ in range(4):
        store.invite(user)
    with pytest.raises(ValueError, match="four"):
        store.invite(user)
    now[0] += 601
    for _ in range(8):
        store.redeem(store.invite(user), "Device")
    invite = store.invite(user)
    with pytest.raises(ValueError, match="device limit"):
        store.redeem(invite, "Ninth")
    store.revoke_device(store.devices(user)[0]["id"])
    assert store.redeem(invite, "Replacement")


@pytest.mark.parametrize("token", ["", "swp_d_bad", "x" * 100000, None, "swp_d_" + "a" * 43],
                         ids=["empty", "malformed", "oversized", "none", "unknown"])
def test_invalid_credentials(identities, token):
    with pytest.raises(InvalidCredential):
        identities[0].authenticate(token)


@pytest.mark.parametrize("ttl", [0, -1, 601, True, float("nan"), float("inf")])
def test_invalid_invitation_lifetimes(identities, ttl):
    with pytest.raises(ValueError):
        identities[0].invite(identities[2], ttl=ttl)


@pytest.mark.parametrize("name", ["", " ", "bad\nlabel", "x" * 81])
def test_invalid_labels(identities, name):
    with pytest.raises(ValueError):
        identities[0].create_organization(name)


def test_unknown_organization_and_user_are_rejected(identities):
    store = identities[0]
    with pytest.raises(ValueError):
        store.create_user("unknown", "User")
    with pytest.raises(ValueError):
        store.invite("unknown")


def test_missing_identity_tables_do_not_accept_existing_token(identities):
    store, _, user, _ = identities
    paired = store.redeem(store.invite(user), "Device")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM devices")
    with pytest.raises(InvalidCredential):
        store.authenticate(paired.token)
