"""Install Sweep in a local virtual environment and open its native desktop window.

Run: python setup_sweep.py
Inspect without installing: python setup_sweep.py --check
Install only: python setup_sweep.py --install-only
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the current Python environment")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Install the CLI without the desktop window")
    parser.add_argument("--extras", choices=["science", "ai", "browser", "datasets", "memory", "dev", "desktop", "build"],
                        action="append", default=[])
    args = parser.parse_args(argv)
    if sys.version_info < (3, 12):
        print("Sweep requires Python 3.12 or newer.", file=sys.stderr)
        return 1
    if args.check:
        from sweep.launcher import doctor
        return doctor()
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if not python.exists():
            print(f"Creating isolated Python environment: {environment}", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        extras = sorted(set(args.extras + ([] if args.headless else ["desktop"])))
        target = str(ROOT) + (f"[{','.join(extras)}]" if extras else "")
        subprocess.run([str(python), "-m", "pip", "install", "-e", target], cwd=ROOT, check=True)
        print("Sweep is installed. Start it again with launch_sweep.cmd (Windows) or .venv/bin/sweep-desktop.")
        if not args.install_only:
            module = "sweep.launcher" if args.headless else "sweep.desktop"
            return subprocess.call([str(python), "-m", module], cwd=ROOT)
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}. Fix the error above and rerun setup_sweep.py.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup cancelled; rerun to continue.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
