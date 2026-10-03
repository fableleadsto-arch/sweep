"""Face/reverse-image search capability for the companion brain.

Wraps ``sweep_neural_mesh.face_search`` so the brain's capability catalog
(``GET /api/compute``) can auto-route tasks like "find this person's social
profiles from a photo" or "reverse image search this picture" to a working
implementation.

Contract (companion/tools/common.py): a tool function takes the payload dict
and returns ``{result, summary, libraries_used}``. Dependencies are optional —
when the face stack is missing, the engine degrades to keyless metadata
search with a note; nothing crashes.
"""
from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path
from typing import Any, Optional

FACE_LIBS = ("cv2", "numpy")  # optional face-stack libs; engine degrades without them


def run_face_search(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a face/reverse-image search.

    Image resolution order:
      1. ``params.image`` — local path or http(s) URL
      2. ``image_base64`` / ``data.image_base64`` — raw or data-URL base64

    Mode is taken from ``params.mode`` or inferred from the task text
    ("person photo"/"find this person" → face; otherwise image).
    """
    params = payload.get("params") or {}
    task = str(payload.get("task") or "")
    image, temp_path = _resolve_image(payload)
    if image is None:
        raise ValueError(
            "No image provided. Send params.image (path or URL) or image_base64."
        )
    mode = _infer_mode(params, task)

    try:
        from sweep_neural_mesh.face_search import FaceSearchOrchestrator

        engine = FaceSearchOrchestrator()
        report = engine.run_search(
            image,
            mode=mode,
            name_hint=str(params.get("name_hint") or params.get("name") or ""),
            verify=bool(params.get("verify", True)),
            max_results=int(params.get("max_results") or 60),
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as ComputeResult.error
        raise RuntimeError(f"face search failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

    d = report.to_dict()
    profiles = [r for r in d["results"] if r["is_profile"]]
    verified = sum(1 for r in d["results"] if r["verified"])
    summary = (
        f"{d['mode']}-mode scan via {len(d['providers_used'])} provider(s): "
        f"{len(d['results'])} results, {len(profiles)} profile-like, "
        f"{verified} face-verified ({report.stats.get('elapsed_s', 0)}s)."
    )
    return {
        "result": d,
        "summary": summary,
        "libraries_used": ["sweep_neural_mesh.face_search"],
    }


# ── helpers ───────────────────────────────────────────────────────────


def _resolve_image(payload: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Returns (image_path_or_url, temp_path_to_cleanup)."""
    params = payload.get("params") or {}
    image = params.get("image")
    if isinstance(image, str) and image.strip():
        return image.strip(), None
    raw = payload.get("image_base64")
    if not raw and isinstance(payload.get("data"), dict):
        raw = payload["data"].get("image_base64")
    if not raw:
        return None, None
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_base64 is not valid base64") from None
    tmp = Path(tempfile.mkdtemp(prefix="sweep_fs_api_")) / "query.jpg"
    tmp.write_bytes(blob)
    return str(tmp), str(tmp.parent)


def _infer_mode(params: dict, task: str) -> str:
    explicit = str(params.get("mode") or "").lower()
    if explicit in ("face", "image"):
        return explicit
    t = task.lower()
    if any(w in t for w in ("person", "people", "profile", "face", "who is", "identify")):
        face = "face"
    else:
        face = "image"
    return face
