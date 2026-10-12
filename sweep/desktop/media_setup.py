"""Explicit, bounded setup of the desktop's local speech asset.

Importing this module and checking readiness never downloads or loads a model.
Only ``install_speech_model`` contacts the fixed publisher URL.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import httpx

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
# SHA-256 of the publisher's HTTPS archive checked during release setup.
# This is a pinned transfer checksum, not an independently signed publisher claim.
MODEL_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
MAX_DOWNLOAD_BYTES = 60_000_000
MAX_EXTRACTED_BYTES = 160_000_000
MAX_ARCHIVE_ENTRIES = 300
DOWNLOAD_SECONDS = 120
MANIFEST_NAME = "sweep-model.json"
REQUIRED_FILES = frozenset({
    "am/final.mdl", "conf/mfcc.conf", "conf/model.conf",
    "graph/Gr.fst", "graph/HCLr.fst", "graph/disambig_tid.int",
    "graph/phones/word_boundary.int",
    "ivector/final.dubm", "ivector/final.ie", "ivector/final.mat",
    "ivector/global_cmvn.stats", "ivector/online_cmvn.conf", "ivector/splice.conf",
})
_ALLOWED_FILES = REQUIRED_FILES | {"README"}
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", re.I)


def _redirected(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _directory(root: str | Path, *, create: bool = False) -> Path:
    raw = str(root)
    if raw.startswith(("\\\\", "//")):
        raise ValueError("Local speech assets must be stored on this computer.")
    selected = Path(root).expanduser().absolute()
    if _redirected(selected) or selected.resolve() != selected:
        raise ValueError("Local speech storage must not redirect to another location.")
    for candidate in (selected, selected / "models", selected / "models" / "speech"):
        if _redirected(candidate) or candidate.resolve() != candidate:
            raise ValueError("Local speech storage must not redirect to another location.")
        if candidate.exists() and not candidate.is_dir():
            raise ValueError("Local speech storage must be a directory.")
        if create:
            candidate.mkdir(parents=candidate == selected, exist_ok=True)
            if _redirected(candidate) or candidate.resolve() != candidate:
                raise ValueError("Local speech storage changed during setup.")
    return selected / "models" / "speech"


def speech_model_path(root: str | Path) -> Path:
    """Return the fixed owned asset path without creating directories."""
    return _directory(root) / MODEL_NAME


def _inventory(target: Path, *, include_manifest: bool) -> dict[str, int]:
    if _redirected(target) or not target.is_dir() or target.resolve() != target:
        raise ValueError("The local speech asset is missing or redirects elsewhere.")
    found: dict[str, int] = {}
    total = 0
    entries = 0
    pending = [target]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > MAX_ARCHIVE_ENTRIES:
                    raise ValueError("The local speech asset contains too many files.")
                path = Path(child.path)
                if _redirected(path) or path.resolve() != path:
                    raise ValueError("Local speech assets must not contain linked files.")
                info = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Local speech assets must contain regular files.")
                name = path.relative_to(target).as_posix()
                if name == MANIFEST_NAME and include_manifest:
                    if info.st_size > 16_000:
                        raise ValueError("The local speech setup record is invalid.")
                    continue
                if name not in _ALLOWED_FILES or info.st_size <= 0:
                    raise ValueError("The local speech asset is incomplete or unexpected.")
                found[name] = info.st_size
                total += info.st_size
                if total > MAX_EXTRACTED_BYTES:
                    raise ValueError("The local speech asset exceeds its storage limit.")
    if not REQUIRED_FILES.issubset(found):
        raise ValueError("The local speech asset is incomplete.")
    return found


def model_ready(root: str | Path) -> bool:
    """Check the complete owned layout; do not import inference or use the network."""
    try:
        target = speech_model_path(root)
        found = _inventory(target, include_manifest=True)
        manifest = target / MANIFEST_NAME
        if _redirected(manifest) or not manifest.is_file():
            return False
        with manifest.open("rb") as handle:
            raw = handle.read(16_001)
        if len(raw) > 16_000:
            return False
        record = json.loads(raw)
        return (
            isinstance(record, dict) and record.get("schema_version") == 1
            and record.get("asset") == MODEL_NAME
            and bool(re.fullmatch(r"[0-9a-f]{64}", MODEL_SHA256))
            and record.get("archive_sha256") == MODEL_SHA256
            and record.get("files") == found
        )
    except (OSError, ValueError, TypeError, RecursionError):
        return False


def _download(destination: Path, progress: Callable[[str], None]) -> str:
    digest = hashlib.sha256()
    total = 0
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    reported = -1
    with httpx.Client(
        trust_env=False, follow_redirects=False,
        timeout=httpx.Timeout(30, connect=15),
    ) as client:
        with client.stream("GET", MODEL_URL, headers={"Accept-Encoding": "identity"}) as response:
            if response.status_code != 200:
                raise ValueError("The local speech download could not be retrieved. Try setup again later.")
            declared = response.headers.get("content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError as exc:
                    raise ValueError("The local speech download has an invalid size.") from exc
                if not 0 < length <= MAX_DOWNLOAD_BYTES:
                    raise ValueError("The local speech download exceeds its size limit.")
            with destination.open("xb") as output:
                for chunk in response.iter_bytes(64 * 1024):
                    if time.monotonic() > deadline:
                        raise TimeoutError("The local speech download timed out. Try setup again.")
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("The local speech download exceeds its size limit.")
                    output.write(chunk)
                    digest.update(chunk)
                    megabytes = total // 1_000_000
                    if megabytes // 5 != reported:
                        reported = megabytes // 5
                        progress(f"Downloading local speech support: {megabytes} MB received")
    if total == 0:
        raise ValueError("The local speech download was empty.")
    return digest.hexdigest()


def _extract(archive: Path, stage: Path) -> Path:
    seen: set[str] = set()
    total = 0
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if not 0 < len(members) <= MAX_ARCHIVE_ENTRIES:
            raise ValueError("The local speech archive has too many files.")
        for member in members:
            # ZipInfo normalizes separators and truncates NULs on Windows;
            # validate the original archive name before using the normalized one.
            name = member.orig_filename
            parts = PurePosixPath(name).parts
            if (
                not parts or parts[0] != MODEL_NAME or "\\" in name
                or name != member.filename or any(ord(char) < 32 for char in name)
                or any(part in {".", ".."} or ":" in part or part.rstrip(" .") != part
                       or _RESERVED.fullmatch(part) for part in parts)
                or name.startswith("/") or "//" in name or "/./" in name
            ):
                raise ValueError("The local speech archive contains an unsafe path.")
            folded = "/".join(parts).casefold()
            if folded in seen:
                raise ValueError("The local speech archive contains duplicate paths.")
            seen.add(folded)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or member.external_attr & 0x400:
                raise ValueError("The local speech archive contains linked or special files.")
            if member.flag_bits & 1:
                raise ValueError("The local speech archive must not be encrypted.")
            destination = stage.joinpath(*parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            relative = "/".join(parts[1:])
            if relative not in _ALLOWED_FILES or member.file_size <= 0:
                raise ValueError("The local speech archive contains unexpected files.")
            total += member.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise ValueError("The local speech archive exceeds its extraction limit.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with source.open(member) as incoming, destination.open("xb") as outgoing:
                while chunk := incoming.read(64 * 1024):
                    written += len(chunk)
                    if written > member.file_size or written > MAX_EXTRACTED_BYTES:
                        raise ValueError("The local speech archive exceeds its extraction limit.")
                    outgoing.write(chunk)
            if written != member.file_size:
                raise ValueError("The local speech archive is incomplete.")
    target = stage / MODEL_NAME
    _inventory(target, include_manifest=False)
    return target


def _cleanup(stage: Path, parent: Path) -> None:
    # The stage was created by this invocation; never resolve and delete a link target.
    if (
        stage.parent != parent or not stage.name.startswith(".sweep-speech-")
        or _redirected(parent) or parent.resolve() != parent
        or _redirected(stage) or stage.resolve() != stage
    ):
        return
    if stage.is_dir():
        shutil.rmtree(stage)


def install_speech_model(
    root: str | Path, progress: Callable[[str], None] | None = None,
) -> dict:
    """Download the pinned asset only after the desktop approves explicit setup."""
    notify = progress or (lambda message: None)
    target = speech_model_path(root)
    if model_ready(root):
        return {"status": "ready", "already_installed": True,
                "message": "Local speech processing is ready. Audio stays on this device."}
    if _redirected(target) or target.exists():
        raise ValueError("An incomplete local speech folder already exists. It was preserved; choose a clean setup location or repair it before trying again.")
    if not re.fullmatch(r"[0-9a-f]{64}", MODEL_SHA256):
        raise RuntimeError("Local speech setup is unavailable because its verified asset checksum is missing.")
    parent = _directory(root, create=True)
    stage = Path(tempfile.mkdtemp(prefix=".sweep-speech-", dir=parent))
    try:
        notify("Downloading local speech support (about 40 MB)")
        try:
            digest = _download(stage / "model.zip", notify)
        except httpx.HTTPError as exc:
            raise ValueError("The local speech download was interrupted. Try setup again.") from exc
        if digest != MODEL_SHA256:
            raise ValueError("The local speech download failed its integrity check. Nothing was installed.")
        notify("Checking and installing local speech support")
        try:
            prepared = _extract(stage / "model.zip", stage)
        except zipfile.BadZipFile as exc:
            raise ValueError("The local speech archive is damaged. Nothing was installed.") from exc
        files = _inventory(prepared, include_manifest=False)
        with (prepared / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "asset": MODEL_NAME,
                       "archive_sha256": digest, "files": files}, handle, sort_keys=True)
        if _directory(root) != parent or _redirected(target) or target.exists():
            raise ValueError("Local speech storage changed during setup; existing files were preserved.")
        prepared.rename(target)
        return {"status": "ready", "already_installed": False,
                "message": "Local speech processing is ready. Audio stays on this device."}
    finally:
        _cleanup(stage, parent)
