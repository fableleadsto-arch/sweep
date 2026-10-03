"""Build a Python-bundled Windows desktop app and its per-user setup executable."""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-only", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("This build script currently targets Windows x64.")
    build = ROOT / "build" / "desktop"
    build.mkdir(parents=True, exist_ok=True)
    from PySide6.QtWidgets import QApplication
    from sweep.desktop.owl import owl_icon
    _application = QApplication.instance() or QApplication([])  # retain Qt ownership
    owl_icon().pixmap(256, 256).save(str(build / "sweep.ico"), "ICO")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
               "--name", "Sweep", "--icon", str(build / "sweep.ico"),
               "--distpath", str(ROOT / "dist"), "--workpath", str(build / "pyinstaller"),
               "--specpath", str(build), "--paths", str(ROOT),
               "--collect-data", "trafilatura", "--collect-data", "courlan", "--collect-data", "tld",
               "--collect-data", "justext"]
    for module in ("torch", "tensorflow", "transformers", "sentence_transformers", "companion.vendor",
                   "sweep_neural_mesh", "sweep_cognitive", "cognition", "pandas", "scipy", "sklearn",
                   "numpy", "matplotlib", "IPython", "pytest", "PySide6.QtQml", "PySide6.QtQuick"):
        command.extend(["--exclude-module", module])
    command.append(str(ROOT / "scripts" / "desktop_entry.py"))
    subprocess.run(command, cwd=ROOT, check=True)
    destination = ROOT / "dist" / "Sweep"
    shutil.copy2(ROOT / "docs" / "DESKTOP.md", destination / "README.txt")
    licenses = destination / "licenses"
    licenses.mkdir(exist_ok=True)
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        for entry in distribution.files or []:
            if any(part.lower().startswith(("license", "copying", "notice")) for part in entry.parts):
                source = Path(distribution.locate_file(entry))
                if source.is_file() and source.stat().st_size < 2_000_000:
                    target = licenses / name / str(entry).replace("/", "_").replace("\\", "_")
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(source, target)
    if not args.app_only:
        payload = build / "payload.zip"
        with ZipFile(payload, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for file in destination.rglob("*"):
                if file.is_file():
                    archive.write(file, file.relative_to(destination))
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
                        "--name", "Sweep-Setup", "--icon", str(build / "sweep.ico"),
                        "--distpath", str(ROOT / "dist"), "--workpath", str(build / "installer"),
                        "--specpath", str(build), "--add-data", f"{payload};.",
                        str(ROOT / "scripts" / "desktop_installer.py")], cwd=ROOT, check=True)
    print(f"Desktop app: {destination / 'Sweep.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
