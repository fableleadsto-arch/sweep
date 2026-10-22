"""An opt-in native-client API, separate from the legacy brain service."""

from __future__ import annotations

import asyncio
import ipaddress
import sqlite3
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from companion.compute.pool import FairComputePool, QueueFullError
from companion.compute.workers import WorkerJob

from .contracts import PairRequest, TextRequest
from .identity import IdentityStore, InvalidCredential, Principal


class _Attempts:
    """Bounded global admission for credential checks; no attacker-keyed map."""

    def __init__(self, limit: int):
        self._limit = limit
        self._recent: deque[float] = deque()
        self._lock = threading.Lock()

    def take(self) -> None:
        now = time.monotonic()
        with self._lock:
            while self._recent and self._recent[0] <= now - 60:
                self._recent.popleft()
            if len(self._recent) >= self._limit:
                raise HTTPException(429, "Too many requests; retry later", headers={"Retry-After": "60"})
            self._recent.append(now)


class NativeTransport:
    """Reject browser origins, unsafe transport and oversized bodies before parsing.

    The runner disables proxy-header trust. A reverse proxy deployment must not
    turn forwarded headers from untrusted clients into an authenticated transport.
    """

    def __init__(self, app, *, allowed_hosts: tuple[str, ...]):
        self.app = app
        self.allowed_hosts = {host.lower() for host in allowed_hosts}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = scope.get("headers", [])
        hosts = [value.decode("latin1") for key, value in headers if key.lower() == b"host"]
        names = [key.lower() for key, _ in headers]
        error = None
        try:
            if len(hosts) != 1 or any(ord(c) < 33 or ord(c) == 127 for c in hosts[0]):
                raise ValueError("invalid host")
            url = urlsplit("//" + hosts[0])
            if (url is None or url.hostname not in self.allowed_hosts or url.username or url.password
                    or url.path or url.query or url.fragment or url.port == 0):
                raise ValueError("invalid host")
            if scope.get("scheme") != "https":
                peer = scope.get("client") or ("", 0)
                if not ipaddress.ip_address(peer[0]).is_loopback:
                    raise ValueError("transport requires TLS")
                if url.hostname != "localhost" and not ipaddress.ip_address(url.hostname).is_loopback:
                    raise ValueError("transport requires TLS")
        except (ValueError, TypeError, IndexError):
            error = (400, "Untrusted host or transport")
        if b"origin" in names or scope.get("query_string"):
            error = (400, "Browser origins and query parameters are not accepted")
        if names.count(b"authorization") > 1 or b"content-encoding" in names:
            error = (400, "Unsupported request headers")
        if error is not None:
            return await JSONResponse({"detail": error[1]}, status_code=error[0],
                                      headers={"Cache-Control": "no-store"})(scope, receive, send)
        body = bytearray()
        try:
            async with asyncio.timeout(5):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > 32 * 1024:
                        return await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
                    if not message.get("more_body"):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "Request body timed out"}, status_code=408)(scope, receive, send)

        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def private_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend([
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                ])
            await send(message)

        await self.app(scope, bounded_receive, private_send)


def create_app(store: IdentityStore, pool: FairComputePool, *,
               allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            await asyncio.to_thread(pool.shutdown)

    app = FastAPI(title="Sweep text host", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(NativeTransport, allowed_hosts=allowed_hosts)
    pairing_attempts = _Attempts(30)
    credential_attempts = _Attempts(1200)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_, __):
        # Default validation errors echo request input, potentially including invites.
        return JSONResponse({"detail": "Invalid request fields or limits"}, status_code=422)

    @app.exception_handler(sqlite3.Error)
    async def unavailable_store(_, __):
        return JSONResponse({"detail": "Host identity store is unavailable"}, status_code=503)

    def identity(authorization: str | None = Header(default=None)) -> Principal:
        credential_attempts.take()
        if authorization is None or not authorization.startswith("Bearer ") or len(authorization) != 56:
            raise HTTPException(401, "Invalid device credential", headers={"WWW-Authenticate": "Bearer"})
        try:
            return store.authenticate(authorization[7:])
        except InvalidCredential:
            raise HTTPException(401, "Invalid device credential", headers={"WWW-Authenticate": "Bearer"}) from None

    authenticated = Depends(identity)

    @app.get("/v1/health")
    def health():
        return {"service": "Sweep text host", "version": 1, "scope": "text_only"}

    @app.post("/v1/pair")
    def pair(request: PairRequest):
        pairing_attempts.take()
        try:
            paired = store.redeem(request.invitation, request.device_name)
        except InvalidCredential:
            raise HTTPException(401, "Invalid or expired invitation") from None
        except ValueError:
            raise HTTPException(400, "Pairing could not complete; contact the host operator") from None
        return {"token": paired.token, **asdict(paired.principal)}

    @app.get("/v1/me")
    def me(principal: Principal = authenticated):
        return asdict(principal)

    @app.delete("/v1/device", status_code=204)
    def disconnect(principal: Principal = authenticated):
        store.revoke_device(principal.device_id)
        return Response(status_code=204)

    @app.post("/v1/chat", status_code=202)
    def chat(request: TextRequest, principal: Principal = authenticated):
        try:
            identifier = pool.submit(principal.owner, WorkerJob(
                kind="inference", backend="local-text", spec=request.model_dump(),
            ))
        except QueueFullError:
            raise HTTPException(429, "Compute queue is full; retry later", headers={"Retry-After": "5"}) from None
        except RuntimeError:
            raise HTTPException(503, "Host is shutting down") from None
        return {"id": identifier}

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str, principal: Principal = authenticated):
        result = pool.status(principal.owner, job_id)
        if result.state == "unknown":
            raise HTTPException(404, "Job not found")
        return result.to_dict()

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, principal: Principal = authenticated):
        if pool.status(principal.owner, job_id).state == "unknown":
            raise HTTPException(404, "Job not found")
        return {"accepted": pool.cancel(principal.owner, job_id)}

    return app
