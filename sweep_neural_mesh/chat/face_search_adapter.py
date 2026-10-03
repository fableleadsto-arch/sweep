"""Bridge between the chat CLI and the face/reverse-image search engine.

Keeps the CLI thin: argument parsing + text rendering. All engine work
happens in ``sweep_neural_mesh.face_search``.
"""
from __future__ import annotations

import shlex
from typing import Optional


def parse_face_search_args(argline: str) -> Optional[tuple[str, dict]]:
    """Parse a '/face-search' argument line → (image, orchestrator kwargs).

    Supports quoted name hints and flags:
        --mode face|image, --no-verify, --max-results N
    """
    try:
        parts = shlex.split(argline)
    except ValueError:
        parts = argline.split()
    if not parts:
        return None
    image = parts[0]
    opts: dict = {
        "mode": "face",
        "no_verify": False,
        "max_results": 60,
        "name_hint": "",
    }
    i = 1
    while i < len(parts):
        tok = parts[i]
        if tok == "--mode" and i + 1 < len(parts):
            v = parts[i + 1].lower()
            opts["mode"] = v if v in ("face", "image") else "face"
            i += 2
        elif tok == "--no-verify":
            opts["no_verify"] = True
            i += 1
        elif tok == "--max-results" and i + 1 < len(parts):
            try:
                opts["max_results"] = max(1, min(int(parts[i + 1]), 200))
            except ValueError:
                pass
            i += 2
        elif not opts["name_hint"]:
            opts["name_hint"] = tok.strip('"')
            i += 1
        else:
            i += 1
    return image, opts


def run_face_search_text(
    image: str,
    *,
    mode: str = "face",
    no_verify: bool = False,
    max_results: int = 60,
    name_hint: str = "",
) -> str:
    """Run a search and return the human-readable report."""
    from sweep_neural_mesh.face_search import FaceSearchOrchestrator

    engine = FaceSearchOrchestrator()
    report = engine.run_search(
        image,
        mode=mode,
        verify=not no_verify,
        max_results=max_results,
        name_hint=name_hint,
    )
    return FaceSearchOrchestrator.render_text(report)
