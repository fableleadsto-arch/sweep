"""Bearer authentication shared by Sweep's local HTTP services."""

import secrets

from fastapi import HTTPException


def authorize(authorization: str | None, token: str) -> None:
    if not token:
        raise HTTPException(503, "API token is not configured. Start via sweep-launcher serve.")
    expected = f"Bearer {token}".encode("utf-8")
    supplied = (authorization or "").encode("utf-8")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
