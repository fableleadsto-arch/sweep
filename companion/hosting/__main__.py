"""Local operator commands. Nothing listens until the explicit `serve` command."""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path

from .identity import IdentityStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep optional text host administration")
    parser.add_argument("--state", type=Path, required=True, help="private host identity database path")
    commands = parser.add_subparsers(dest="command", required=True)
    organization = commands.add_parser("organization", help="create an organization")
    organization.add_argument("name")
    user = commands.add_parser("user", help="create a user within an organization")
    user.add_argument("organization_id")
    user.add_argument("name")
    invite = commands.add_parser("invite", help="issue a single-use invitation, valid for ten minutes")
    invite.add_argument("user_id")
    devices = commands.add_parser("devices", help="list a user's device IDs, never credential secrets")
    devices.add_argument("user_id")
    revoke = commands.add_parser("revoke", help="revoke one device credential")
    revoke.add_argument("device_id")
    disable = commands.add_parser("disable", help="disable a user and all credentials/invitations")
    disable.add_argument("user_id")
    serve = commands.add_parser("serve", help="explicitly start the text-only API")
    serve.add_argument("--config-root", type=Path, required=True, help="host's local inference settings directory")
    serve.add_argument("--bind", default="127.0.0.1", help="literal bind IP; loopback by default")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--certfile", type=Path)
    serve.add_argument("--keyfile", type=Path)
    serve.add_argument("--allow-host", action="append", default=[], help="exact trusted host name for TLS connections")
    args = parser.parse_args(argv)
    if args.command == "serve":
        try:
            address = ipaddress.ip_address(args.bind)
        except ValueError:
            parser.error("--bind must be a literal IP address")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        tls = args.certfile is not None and args.keyfile is not None
        if bool(args.certfile) != bool(args.keyfile):
            parser.error("provide both --certfile and --keyfile")
        if not address.is_loopback and (not tls or not args.allow_host):
            parser.error("network hosting requires TLS and at least one explicit --allow-host")
        if tls and (not args.certfile.is_file() or not args.keyfile.is_file()):
            parser.error("TLS certificate and key must be existing files")
        if any(not host or any(c in host for c in "/*@?#\\ \t\r\n") for host in args.allow_host):
            parser.error("--allow-host must be an exact host name, without a URL or wildcard")
        if not args.config_root.is_dir():
            parser.error("--config-root must be an existing settings directory")
    store = IdentityStore(args.state)
    if args.command == "organization":
        print(store.create_organization(args.name))
    elif args.command == "user":
        print(store.create_user(args.organization_id, args.name))
    elif args.command == "invite":
        print(store.invite(args.user_id))
    elif args.command == "devices":
        print(json.dumps(store.devices(args.user_id)))
    elif args.command == "revoke":
        print(json.dumps({"revoked": store.revoke_device(args.device_id)}))
    elif args.command == "disable":
        print(json.dumps({"disabled": store.disable_user(args.user_id)}))
    elif args.command == "serve":
        import uvicorn

        from companion.compute.pool import FairComputePool

        from .api import create_app
        from .worker import HostChatWorker

        worker = HostChatWorker(args.config_root)
        pool = FairComputePool(worker)
        app = create_app(store, pool, allowed_hosts=tuple({"127.0.0.1", "localhost", "::1", *args.allow_host}))
        try:
            uvicorn.run(app, host=str(address), port=args.port, workers=1, proxy_headers=False,
                        access_log=False, ssl_certfile=str(args.certfile) if tls else None,
                        ssl_keyfile=str(args.keyfile) if tls else None, limit_concurrency=64)
        finally:
            pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
