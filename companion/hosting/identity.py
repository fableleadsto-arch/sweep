"""Local host administration and persistent, revocable device credentials.

Only random credential digests are stored. Invitations authorize one device for
an existing host-created user; clients cannot choose a tenant or quota identity.
The database is host-private configuration, not a shared client workspace.
"""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


class InvalidCredential(ValueError):
    """Unknown, expired, revoked or already consumed credential."""


@dataclass(frozen=True)
class Principal:
    organization_id: str
    user_id: str
    device_id: str
    expires_at: float

    @property
    def owner(self) -> str:
        # Multiple devices share the user's quota, not a quota per credential.
        return f"{self.organization_id}_{self.user_id}"


@dataclass(frozen=True)
class Pairing:
    principal: Principal
    token: str = field(repr=False)


def _digest(value: str, prefix: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(re.escape(prefix) + r"[A-Za-z0-9_-]{43}", value):
        raise InvalidCredential("invalid credential")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _label(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 80 or any(ord(c) < 32 for c in value):
        raise ValueError("name must contain 1 to 80 printable characters")
    return value.strip()


class IdentityStore:
    """SQLite transactions make invitation redemption single-use across processes."""

    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("identity database must not be a symbolic link")
        if not self.path.exists():
            # Windows inherits the host account directory ACL; POSIX uses 0600.
            import os
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self.path.is_symlink():
                    raise ValueError("identity database must not be a symbolic link") from None
            else:
                os.close(descriptor)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS organizations (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, organization_id TEXT NOT NULL REFERENCES organizations(id),
                    name TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS invitations (
                    digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS devices (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    name TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS devices_user ON devices(user_id);
                CREATE INDEX IF NOT EXISTS invitations_user ON invitations(user_id);
            """)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_organization(self, name: str) -> str:
        name = _label(name)
        identifier = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT count(*) FROM organizations").fetchone()[0] >= 64:
                raise ValueError("organization limit reached")
            connection.execute("INSERT INTO organizations VALUES (?, ?)", (identifier, name))
        return identifier

    def create_user(self, organization_id: str, name: str) -> str:
        name = _label(name)
        identifier = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT count(*) FROM users").fetchone()[0] >= 512:
                raise ValueError("user limit reached")
            if not connection.execute("SELECT id FROM organizations WHERE id=?", (organization_id,)).fetchone():
                raise ValueError("unknown organization")
            connection.execute("INSERT INTO users(id, organization_id, name) VALUES (?, ?, ?)",
                               (identifier, organization_id, name))
        return identifier

    def invite(self, user_id: str, *, ttl: float = 600) -> str:
        if isinstance(ttl, bool) or not math.isfinite(ttl) or not 1 <= ttl <= 600:
            raise ValueError("invitation lifetime must be 1 to 600 seconds")
        token = "swp_i_" + secrets.token_urlsafe(32)
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM invitations WHERE expires_at<=?", (now,))
            if not connection.execute("SELECT id FROM users WHERE id=? AND enabled=1", (user_id,)).fetchone():
                raise ValueError("unknown or disabled user")
            count = connection.execute("SELECT count(*) FROM invitations WHERE user_id=?", (user_id,)).fetchone()[0]
            if count >= 4:
                raise ValueError("user already has four pending invitations")
            connection.execute("INSERT INTO invitations VALUES (?, ?, ?)",
                               (_digest(token, "swp_i_"), user_id, now + ttl))
        return token

    def redeem(self, invitation: str, device_name: str) -> Pairing:
        digest = _digest(invitation, "swp_i_")
        device_name = _label(device_name)
        now = self._clock()
        token = "swp_d_" + secrets.token_urlsafe(32)
        device_id = uuid.uuid4().hex
        expires = now + 30 * 24 * 60 * 60
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT u.id, u.organization_id FROM invitations i JOIN users u ON u.id=i.user_id
                WHERE i.digest=? AND i.expires_at>? AND u.enabled=1
            """, (digest, now)).fetchone()
            if row is None:
                raise InvalidCredential("invalid credential")
            connection.execute("DELETE FROM devices WHERE expires_at<=?", (now,))
            if connection.execute("SELECT count(*) FROM devices WHERE user_id=?", (row["id"],)).fetchone()[0] >= 8:
                raise ValueError("user device limit reached; revoke an existing device")
            connection.execute("DELETE FROM invitations WHERE digest=?", (digest,))
            connection.execute("INSERT INTO devices VALUES (?, ?, ?, ?, ?)",
                               (device_id, row["id"], device_name, _digest(token, "swp_d_"), expires))
            principal = Principal(row["organization_id"], row["id"], device_id, expires)
        return Pairing(principal, token)

    def authenticate(self, token: str) -> Principal:
        digest = _digest(token, "swp_d_")
        with self._connection() as connection:
            row = connection.execute("""
                SELECT d.id, d.user_id, d.expires_at, u.organization_id
                FROM devices d JOIN users u ON u.id=d.user_id
                WHERE d.digest=? AND d.expires_at>? AND u.enabled=1
            """, (digest, self._clock())).fetchone()
        if row is None:
            raise InvalidCredential("invalid credential")
        return Principal(row["organization_id"], row["user_id"], row["id"], row["expires_at"])

    def revoke_device(self, device_id: str) -> bool:
        with self._connection() as connection:
            return connection.execute("DELETE FROM devices WHERE id=?", (device_id,)).rowcount == 1

    def disable_user(self, user_id: str) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute("UPDATE users SET enabled=0 WHERE id=?", (user_id,)).rowcount == 1
            connection.execute("DELETE FROM invitations WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM devices WHERE user_id=?", (user_id,))
            return changed

    def devices(self, user_id: str) -> list[dict]:
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id, name, expires_at FROM devices WHERE user_id=? AND expires_at>? ORDER BY name",
                (user_id, self._clock()),
            )]
