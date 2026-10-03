"""Frozen desktop entry point. Keep this separate from the legacy CLI."""
from sweep.desktop.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
