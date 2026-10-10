"""Build a Python-bundled Windows desktop app and its per-user setup executable."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
RECORDED_DISTRIBUTIONS = (
    "PySide6-Essentials", "shiboken6", "pyinstaller", "httpx", "httpcore", "h11",
    "certifi", "anyio", "pydantic", "pydantic-settings", "python-dotenv",
    "beautifulsoup4", "trafilatura", "lxml", "courlan", "tld", "justext", "pillow", "pypdf",
)


def source_state(root: Path) -> dict:
    """Report provenance without retaining paths, remote URLs, or Git error text."""
    state = {"revision": None, "dirty": None}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=root, capture_output=True,
            text=True, check=False, timeout=10,
        )
        if revision.returncode or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision.stdout.strip()):
            return state
        state["revision"] = revision.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root,
            capture_output=True, text=True, check=False, timeout=10,
        )
        if status.returncode == 0:
            state["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return state


def write_build_info(root: Path, destination: Path) -> Path:
    """Write an explicit, non-sensitive inventory of the build environment."""
    with (root / "pyproject.toml").open("rb") as project_file:
        version = tomllib.load(project_file)["project"]["version"]
    dependencies = {}
    for name in RECORDED_DISTRIBUTIONS:
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    info = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": source_state(root),
        "sweep_version": version,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "dependencies": dependencies,
    }
    path = destination / "build-info.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def write_checksums(destination: Path, artifacts: list[Path]) -> Path:
    """Hash only artifacts from this successful build, using portable relative names."""
    if not artifacts:
        raise ValueError("At least one build artifact is required")
    root = destination.resolve()
    entries = []
    for artifact in artifacts:
        resolved = artifact.resolve()
        relative = resolved.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise ValueError("Artifact names cannot contain line breaks")
        with resolved.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append((relative, digest))
    path = destination / "SHA256SUMS.txt"
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(entries)), encoding="utf-8")
    temporary.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-only", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("This build script currently targets Windows x64.")
    # A failed rebuild must not leave a checksum file that appears to certify it.
    (ROOT / "dist" / "SHA256SUMS.txt").unlink(missing_ok=True)
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
                   "sweep_neural_mesh", "sweep_cognitive", "pandas", "scipy", "sklearn",
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
    write_build_info(ROOT, destination)
    artifacts = [destination / "Sweep.exe", destination / "build-info.json"]
    if not args.app_only:
        payload = build / "payload.zip"
        with ZipFile(payload, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(destination.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(destination))
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
                        "--name", "Sweep-Setup", "--icon", str(build / "sweep.ico"),
                        "--distpath", str(ROOT / "dist"), "--workpath", str(build / "installer"),
                        "--specpath", str(build), "--add-data", f"{payload};.",
                        str(ROOT / "scripts" / "desktop_installer.py")], cwd=ROOT, check=True)
        artifacts.append(ROOT / "dist" / "Sweep-Setup.exe")
    checksums = write_checksums(ROOT / "dist", artifacts)
    print(f"Desktop app: {destination / 'Sweep.exe'}")
    print(f"Build information: {destination / 'build-info.json'}")
    print(f"SHA-256 checksums: {checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
