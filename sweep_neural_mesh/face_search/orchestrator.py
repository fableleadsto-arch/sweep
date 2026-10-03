"""Orchestrator — runs the full search pipeline.

Pipeline:
  1. Load the query image, detect + embed faces (face mode only).
  2. Build the provider fleet from config: SerpAPI engines (Lens, Yandex,
     Bing, Google Images) when SERPAPI_KEY is set, TinEye when its keys are
     set, keyless sources (DDG name search, Commons) always.
  3. Public URL: when SWEEP_PUBLIC_BASE_URL is set, the crop is copied to
     ``<repo>/uploads/face_search/<token>.jpg`` and served at
     ``<base>/uploads/face_search/<token>.jpg`` — the caller must expose that
     static path (selfwatch-style). Without it, URL-based providers are
     skipped in face mode (they would fail server-side anyway) and only
     keyless + upload-capable providers run.
  4. Fan out in parallel threads (providers are I/O bound).
  5. Merge + dedupe, classify platforms, compute confidence.
  6. Face verification: for the strongest face, pull each candidate's
     thumbnail and compare ArcFace embeddings (cosine >= threshold).
  7. Rank, render a text report, DELETE all temp files.

Privacy contract: query images and crops live in temp files only and are
removed in ``finally``; no face database is built, nothing persists.
"""
from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from . import face
from .dedupe import merge
from .models import Face, ProfileMatch, ProviderResult, SearchReport
from .providers import keyless, serpapi_sources, tineye as tineye_mod
from .providers.base import Provider

logger = logging.getLogger("sweep.face_search")

# Static uploads dir served by the caller's web server (selfwatch pattern):
# set SWEEP_PUBLIC_BASE_URL=https://your-host to enable URL-based providers.
UPLOADS_DIR = Path(os.environ.get("SWEEP_UPLOADS_DIR", "")) if os.environ.get("SWEEP_UPLOADS_DIR") else Path(tempfile.gettempdir()) / "sweep_face_search_uploads"

VERIFY_COSINE_THRESHOLD = 0.45  # conservative same-person threshold for crops


def build_providers(name_hint: str = "") -> list[Provider]:
    """Assemble the provider fleet.

    Default (zero-configuration) fleet: direct upload scrapers for Bing,
    Yandex, Google Lens and TinEye — no API keys, no public hosting — plus
    keyless name/Commons search when a name hint is given.

    Optional accelerators (auto-enabled only when their env keys are set):
    SerpAPI engines and the TinEye REST API.
    """
    from .providers.keyless_engines import KEYLESS_ENGINE_PROVIDERS

    fleet: list[Provider] = [cls() for cls in KEYLESS_ENGINE_PROVIDERS]
    fleet.append(keyless.NameHintSearchProvider(name_hint=name_hint))
    fleet.append(keyless.CommonsSearchProvider(name_hint=name_hint))
    if serpapi_sources.serpapi_key():  # optional accelerator, never required
        for cls in serpapi_sources.SERPAPI_PROVIDERS:
            fleet.append(cls())
    if tineye_mod.TinEyeProvider().is_enabled():  # optional accelerator
        fleet.append(tineye_mod.TinEyeProvider())
    return fleet


class FaceSearchOrchestrator:
    """Runs one search end-to-end and cleans up after itself."""

    def __init__(self, *, max_workers: int = 6, timeout_s: float = 120.0):
        self.max_workers = max_workers
        self.timeout_s = timeout_s

    # ── public entry point ────────────────────────────────────────────

    def run_search(
        self,
        image: str,
        *,
        mode: str = "face",
        name_hint: str = "",
        verify: bool = True,
        max_results: int = 60,
        fleet: Optional[list[Provider]] = None,
    ) -> SearchReport:
        t0 = time.perf_counter()
        report = SearchReport(query_image=image, mode=mode)
        temp_paths: list[Path] = []

        try:
            # 1. Resolve the image to bytes + a local temp file.
            image_bytes, image_path = self._resolve_image(image, temp_paths)
            if image_bytes is None or image_path is None:
                report.notes.append("could not read/download the query image")
                return report

            # 2. Face pipeline (face mode): detect + embed.
            faces: list[Face] = []
            if mode == "face":
                try:
                    faces = face.detect(str(image_path))
                except Exception as exc:  # noqa: BLE001
                    report.notes.append(f"face detection failed: {exc}")
                if not faces:
                    report.notes.append(
                        "no face found — falling back to generic reverse-image mode"
                    )
                    mode = report.mode = "image"
                else:
                    report.faces_found = len(faces)
                    for f in faces:
                        try:
                            face.embed_face(str(image_path), f)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("embed failed: %s", exc)

            # 3. Host the crop publicly when configured (URL providers).
            public_url = self._maybe_host(image, image_path, faces, temp_paths)
            if public_url is None:
                report.notes.append(
                    "no public base URL (SWEEP_PUBLIC_BASE_URL) — URL-based "
                    "providers skipped; set it or use keyless/upload sources"
                )

            # 4. Fan out.
            fleet = fleet or build_providers(name_hint=name_hint)
            results = self._fan_out(
                fleet,
                image_url=public_url,
                image_bytes=image_bytes,
                image_filename=image_path.name,
                max_results=max_results,
            )
            for r in results:
                if r.ok and r.matches:
                    report.providers_used.append(r.provider)
                elif r.error:
                    report.providers_failed[r.provider] = r.error

            # 5. Merge + classify + confidence.
            report.results = merge(results)

            # 6. Face verification of candidates (face mode only).
            if mode == "face" and verify:
                self._verify_candidates(report.results, faces)

            # 7. Final ranking: verified first, then confidence.
            report.results.sort(
                key=lambda m: (
                    m.verified is not True,  # verified True first
                    -(m.confidence + (0.5 if m.verified else 0.0)),
                    m.domain,
                )
            )
            report.results = report.results[: max_results]
        finally:
            # 8. Privacy: remove every temp file we created.
            for p in temp_paths:
                shutil.rmtree(p.parent if p.parent.name.startswith("sweep_fs_") else p, ignore_errors=True)
            _cleanup_uploads()

        elapsed = time.perf_counter() - t0
        report.stats = {
            "elapsed_s": round(elapsed, 2),
            "providers_ran": len(report.providers_used) + len(report.providers_failed),
            "results": len(report.results),
            "profiles": sum(1 for r in report.results if r.is_profile),
            "verified": sum(1 for r in report.results if r.verified),
        }
        return report

    # ── steps ─────────────────────────────────────────────────────────

    def _resolve_image(self, image: str, temp_paths: list[Path]) -> tuple[Optional[bytes], Optional[Path]]:
        """Load the query image from disk or URL. Registers temp files."""
        if image.startswith(("http://", "https://")):
            try:
                # Use the impersonation session: many hosts (e.g. Wikimedia)
                # 403 default httpx/curl user-agents.
                from .providers.http import HttpLike

                session = HttpLike(timeout=40.0)
                try:
                    resp = session.get(image)
                    resp.raise_for_status()
                    data = resp.content
                finally:
                    session.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("image download failed: %s", exc)
                return None, None
            suffix = Path(urlparse(image).path).suffix or ".jpg"
            if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
                suffix = ".jpg"
        else:
            p = Path(image)
            if not p.is_file():
                return None, None
            data = p.read_bytes()
            suffix = p.suffix or ".jpg"
        tmp = Path(tempfile.mkdtemp(prefix="sweep_fs_")) / f"query{suffix}"
        tmp.write_bytes(data)
        temp_paths.append(tmp)
        return data, tmp

    def _maybe_host(
        self, original: str, image_path: Path, faces: list[Face], temp_paths: list[Path]
    ) -> Optional[str]:
        """Serve a face crop (or the image itself) at a public URL when the
        host has configured SWEEP_PUBLIC_BASE_URL. Returns None otherwise."""
        base = os.environ.get("SWEEP_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if not base:
            return None
        try:
            crop_path: Path
            if faces:
                best = max(faces, key=lambda f: f.area)
                crop = self._crop_bytes(image_path, best)
                if crop is None:
                    return None
                crop_path = UPLOADS_DIR / f"{secrets.token_hex(8)}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop_path.write_bytes(crop)
                temp_paths.append(crop_path)
            else:
                crop_path = image_path  # already a temp copy
            return f"{base}/uploads/face_search/{crop_path.name}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("public hosting failed: %s", exc)
            return None

    def _crop_bytes(self, image_path: Path, face_obj: Face) -> Optional[bytes]:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face_obj.box
        mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
        x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = img[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return bytes(buf) if ok else None

    def _fan_out(
        self,
        fleet: list[Provider],
        *,
        image_url: Optional[str],
        image_bytes: Optional[bytes],
        image_filename: Optional[str],
        max_results: int,
    ) -> list[ProviderResult]:
        """Run every enabled provider in parallel; failures are data."""
        results: list[ProviderResult] = []
        runnable = [p for p in fleet if p.is_enabled()]
        # NOTE: the client context must stay open until every future completes,
        # so the as_completed() drain runs INSIDE the `with httpx.Client` block.
        with httpx.Client(follow_redirects=True) as client:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(runnable) or 1))) as pool:
                futures = {}
                for p in runnable:
                    can_run = (
                        (p.accepts_url and image_url)
                        or (p.accepts_upload and image_bytes)
                        or (not p.accepts_url and not p.accepts_upload)
                    )
                    if not can_run:
                        results.append(
                            ProviderResult(
                                provider=p.name,
                                error="skipped: needs a public image URL it can fetch",
                                note=p.note(),
                            )
                        )
                        continue
                    futures[pool.submit(self._run_one, p, client, image_url, image_bytes, image_filename, max_results)] = p
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        res = fut.result(timeout=self.timeout_s)
                    except Exception as exc:  # noqa: BLE001
                        res = ProviderResult(provider=p.name, error=str(exc)[:300])
                    if res.note is None:
                        res.note = p.note()
                    results.append(res)
        return results

    @staticmethod
    def _run_one(
        p: Provider,
        client: httpx.Client,
        image_url: Optional[str],
        image_bytes: Optional[bytes],
        image_filename: Optional[str],
        max_results: int,
    ) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            res = p.search(
                client,
                image_url=image_url if p.accepts_url else None,
                image_bytes=image_bytes if p.accepts_upload else None,
                image_filename=image_filename if p.accepts_upload else None,
                max_results=max_results,
            )
        except Exception as exc:  # noqa: BLE001 — provider crash is data
            res = ProviderResult(provider=p.name, error=str(exc)[:300])
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res

    def _verify_candidates(self, candidates: list[ProfileMatch], faces: list[Face]) -> None:
        """Compare each candidate thumbnail's face to the query face."""
        query_face = max(faces, key=lambda f: f.area) if faces else None
        if query_face is None or not query_face.embedding:
            return
        with httpx.Client(follow_redirects=True, timeout=20) as client:
            for m in candidates:
                if not m.thumbnail_url:
                    continue
                data = keyless.fetch_image_bytes(client, m.thumbnail_url)
                if not data:
                    continue
                tmp = Path(tempfile.gettempdir()) / f"sweep_verify_{secrets.token_hex(6)}.jpg"
                try:
                    tmp.write_bytes(data)
                    try:
                        found = face.detect(str(tmp))
                    except Exception:  # noqa: BLE001
                        found = []
                    if not found:
                        m.verified = False
                        continue
                    best_cos = None
                    for f in found:
                        try:
                            face.embed_face(str(tmp), f)
                        except Exception:  # noqa: BLE001
                            continue
                        if f.embedding:
                            c = face.cosine(query_face.embedding, f.embedding)
                            if c is not None and (best_cos is None or c > best_cos):
                                best_cos = c
                    if best_cos is not None:
                        m.cosine = best_cos
                        m.verified = best_cos >= VERIFY_COSINE_THRESHOLD
                finally:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    # ── reporting ─────────────────────────────────────────────────────

    @staticmethod
    def render_text(report: SearchReport) -> str:
        lines: list[str] = []
        mode_label = "person photo" if report.mode == "face" else "image"
        lines.append(f"Face/reverse-image search — {mode_label}: {report.query_image}")
        if report.faces_found:
            lines.append(f"  faces detected: {report.faces_found}")
        used = ", ".join(report.providers_used) or "(none)"
        lines.append(f"  providers: {used}")
        for name, err in report.providers_failed.items():
            lines.append(f"  [failed] {name}: {err}")
        for n in report.notes:
            lines.append(f"  [note] {n}")
        if not report.results:
            lines.append("  no matches found")
        profiles = [r for r in report.results if r.is_profile]
        web = [r for r in report.results if not r.is_profile]
        if profiles:
            lines.append(f"\n  profiles ({len(profiles)}):")
            for m in profiles:
                lines.append(f"    {m.render_line()}")
        if web:
            shown = web[:15]
            lines.append(f"\n  other web sources ({len(web)}{' — first 15 shown' if len(web) > 15 else ''}):")
            for m in shown:
                lines.append(f"    {m.render_line()}")
        st = report.stats
        lines.append(
            f"\n  {st.get('results', 0)} results | {st.get('profiles', 0)} profiles | "
            f"{st.get('verified', 0)} face-verified | {st.get('elapsed_s', 0)}s"
        )
        return "\n".join(lines)


def _cleanup_uploads() -> None:
    """Remove hosted crops older than the TTL (privacy + disk hygiene)."""
    try:
        if not UPLOADS_DIR.is_dir():
            return
        now = time.time()
        for f in UPLOADS_DIR.iterdir():
            try:
                if now - f.stat().st_mtime > 3600:
                    f.unlink()
            except OSError:
                continue
    except OSError:
        pass
