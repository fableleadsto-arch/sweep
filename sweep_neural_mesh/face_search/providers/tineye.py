"""TinEye provider — direct Search API with HMAC-SHA256 request signing.

Mirrors selfwatch's implementation of TinEye's documented spec. Requires both
TINEYE_API_KEY (public) and TINEYE_PRIVATE_KEY. Supports URL queries and
multipart image-upload queries; each variant is signed differently.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Any, Optional
from urllib.parse import quote, urlencode

from .base import Provider

TINEYE_URL = "https://api.tineye.com/rest/search/"


def _keys() -> tuple[str, str]:
    return (
        os.environ.get("TINEYE_API_KEY", "").strip(),
        os.environ.get("TINEYE_PRIVATE_KEY", "").strip(),
    )


def _hmac_sha256(key: str, message: str) -> str:
    return hmac.new(key.encode(), message.encode(), hashlib.sha256).hexdigest()


def _sign_post_multipart(
    *,
    private_key: str,
    api_key: str,
    image_filename: str,
    content_type: str,
    date: int,
    nonce: str,
) -> str:
    qs = urlencode(sorted({"api_key": api_key, "date": str(date), "nonce": nonce}.items()))
    url = f"{TINEYE_URL}?{qs}"
    string_to_sign = (
        private_key
        + "POST"
        + content_type.lower()
        + quote(image_filename, safe="")
        + str(date)
        + nonce
        + url
    )
    return _hmac_sha256(private_key, string_to_sign)


def _sign_get(
    *,
    private_key: str,
    api_key: str,
    date: int,
    nonce: str,
    extra: dict[str, str] | None = None,
) -> str:
    params: dict[str, str] = {"api_key": api_key, "date": str(date), "nonce": nonce}
    if extra:
        params.update(extra)
    qs = urlencode(sorted(params.items()))
    url = f"{TINEYE_URL}?{qs}"
    string_to_sign = private_key + "GET" + str(date) + nonce + url
    return _hmac_sha256(private_key, string_to_sign)


def _build_multipart(image_bytes: bytes, filename: str) -> tuple[bytes, str, str]:
    boundary = "----sweep" + secrets.token_hex(12)
    safe_filename = filename.replace('"', "").replace("\r", "").replace("\n", "")
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image_upload"; filename="{safe_filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + image_bytes + tail
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type, safe_filename


class TinEyeProvider(Provider):
    name = "tineye"
    accepts_url = True
    accepts_upload = True

    def is_enabled(self) -> bool:
        api_key, private_key = _keys()
        return bool(api_key and private_key)

    def note(self) -> Optional[str]:
        if not self.is_enabled():
            return "Set TINEYE_API_KEY and TINEYE_PRIVATE_KEY to enable."
        return (
            "Experimental — HMAC signing follows TinEye's documented spec but "
            "has not been validated against live credentials in this host."
        )

    def search(
        self,
        client: Any,
        *,
        image_url: Optional[str],
        image_bytes: Optional[bytes],
        image_filename: Optional[str],
        max_results: int,
    ) -> Any:
        from ..models import ProviderResult

        api_key, private_key = _keys()
        if not (api_key and private_key):
            return ProviderResult(provider=self.name, error="missing TinEye credentials")
        date = int(time.time())
        nonce = secrets.token_hex(16)
        try:
            if image_bytes is not None and image_filename:
                body, content_type, safe_filename = _build_multipart(image_bytes, image_filename)
                api_sig = _sign_post_multipart(
                    private_key=private_key,
                    api_key=api_key,
                    image_filename=safe_filename,
                    content_type=content_type,
                    date=date,
                    nonce=nonce,
                )
                params = {"api_key": api_key, "date": str(date), "nonce": nonce, "api_sig": api_sig}
                resp = client.post(
                    TINEYE_URL,
                    params=params,
                    content=body,
                    headers={"Content-Type": content_type},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            elif image_url:
                api_sig = _sign_get(
                    private_key=private_key,
                    api_key=api_key,
                    date=date,
                    nonce=nonce,
                    extra={"image_url": image_url},
                )
                params = {
                    "api_key": api_key,
                    "date": str(date),
                    "nonce": nonce,
                    "image_url": image_url,
                    "api_sig": api_sig,
                }
                resp = client.get(TINEYE_URL, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            else:
                return ProviderResult(provider=self.name, error="tineye requires image bytes or URL")
        except Exception as exc:  # noqa: BLE001 — network/HTTP errors are data
            return ProviderResult(provider=self.name, error=str(exc)[:300])

        code = data.get("code")
        if code is not None and code != 200:
            return ProviderResult(
                provider=self.name,
                error=f"TinEye error {code}: {data.get('messages')}",
            )
        matches = []
        for m in (data.get("results") or {}).get("matches") or []:
            thumb = m.get("image_url")
            score = m.get("score")
            domain = m.get("domain")
            for backlink in m.get("backlinks") or []:
                url = backlink.get("backlink") or backlink.get("url")
                if not url:
                    continue
                matches.append({"link": url, "title": domain, "thumbnail": thumb, "score": score})
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break
        return self._result(self.name, matches, max_results)
