"""Command-line entry point for the face/reverse-image search engine.

Usage:
    python -m sweep_neural_mesh.face_search.cli photo.jpg
    python -m sweep_neural_mesh.face_search.cli https://example.com/pic.jpg --mode image
    python -m sweep_neural_mesh.face_search.cli photo.jpg --name "jane dough" --json
    python -m sweep_neural_mesh.face_search.cli --providers
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sweep-face-search",
        description="Search the public internet for a person photo (or any image).",
    )
    ap.add_argument("image", nargs="?", help="image path or URL")
    ap.add_argument("--mode", choices=("face", "image"), default="face",
                    help="face = person photo (default); image = generic reverse image")
    ap.add_argument("--name", default="", help="optional name hint for keyless name search")
    ap.add_argument("--no-verify", action="store_true", help="skip face verification of candidates")
    ap.add_argument("--max-results", type=int, default=60)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--providers", action="store_true", help="list provider status and exit")
    args = ap.parse_args(argv)

    if args.providers:
        from .orchestrator import build_providers

        for p in build_providers(name_hint=args.name):
            status = "enabled" if p.is_enabled() else "disabled"
            print(f"  {p.name:20s} {status:9s} url={p.accepts_url} upload={p.accepts_upload}  {p.note() or ''}")
        return 0

    if not args.image:
        ap.error("an image path or URL is required (or use --providers)")

    from .orchestrator import FaceSearchOrchestrator

    engine = FaceSearchOrchestrator()
    report = engine.run_search(
        args.image,
        mode=args.mode,
        name_hint=args.name,
        verify=not args.no_verify,
        max_results=args.max_results,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(FaceSearchOrchestrator.render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
