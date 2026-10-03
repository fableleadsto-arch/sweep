"""Allow `python -m sweep_neural_mesh.chat` to launch the CLI."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
