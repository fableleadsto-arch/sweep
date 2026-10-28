"""Measure the source-only document foundation on fixed synthetic local fixtures.

Run with the project's Python: python scripts/benchmark_documents.py --runs 3
Prints measurements, never customer contents. No models, downloads or cloud calls.
Cold-process wall time includes interpreter startup; reader times do not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def peak_process_bytes() -> int | None:
    """OS process high-water RSS/working set, not Python-only allocation."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def measure(path: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    cpu_started, started = time.process_time(), time.perf_counter()
    from sweep.documents import inspect_source

    snapshot = inspect_source(path, [str(path)])
    first = time.perf_counter() - started
    started = time.perf_counter()
    repeated = inspect_source(path, [str(path)])
    warm = time.perf_counter() - started
    assert repeated == snapshot
    span = snapshot.reference(0, 0, len("SYNTHETIC TEST DOCUMENT"))
    assert snapshot.resolve(span).text.startswith("SYNTHETIC TEST DOCUMENT")
    return {"format": snapshot.format, "file_sha256": snapshot.file_sha256,
            "source_bytes": snapshot.source_bytes, "parser": snapshot.parser,
            "parser_version": snapshot.parser_version, "first_reader_seconds": first,
            "repeated_reader_seconds": warm, "cpu_seconds": time.process_time() - cpu_started,
            "peak_process_bytes": peak_process_bytes(),
            "extraction_sha256": snapshot.extraction_sha256,
            "characters": sum(len(unit.text) for unit in snapshot.units),
            "source_preserved": hashlib.sha256(path.read_bytes()).hexdigest() == snapshot.file_sha256}


def fixtures(directory: Path) -> list[Path]:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    content = "SYNTHETIC TEST DOCUMENT - Invoice TEST-001 - SKU-A quantity 12 pieces"
    plain, docx, pdf = [directory / f"synthetic.{suffix}" for suffix in ("txt", "docx", "pdf")]
    plain.write_text(content, encoding="utf-8")
    with zipfile.ZipFile(docx, "w") as archive:
        # Fixed timestamp makes bytes/hash reproducible across runs.
        entry = zipfile.ZipInfo("word/document.xml", date_time=(2026, 1, 1, 0, 0, 0))
        archive.writestr(entry, '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + content + '</w:t></w:r></w:p></w:body></w:document>')
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 30 700 Td ({content}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(pdf)
    return [plain, docx, pdf]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        print(json.dumps(measure(args.worker)))
        return
    if not 1 <= args.runs <= 30:
        parser.error("--runs must be between 1 and 30")
    samples = []
    with tempfile.TemporaryDirectory(prefix="sweep-document-benchmark-") as directory:
        for path in fixtures(Path(directory)):
            original = path.read_bytes()
            for run in range(args.runs):
                started = time.perf_counter()
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                                        capture_output=True, text=True, check=True, timeout=30)
                wall = time.perf_counter() - started
                measured = json.loads(result.stdout)
                assert measured["source_preserved"] and path.read_bytes() == original
                samples.append({"run": run + 1, "cold_process_seconds": wall, **measured})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    print(json.dumps({"fixture_type": "synthetic; fixed content; no random seed needed",
                      "scope": "source-only ingestion, not TradeOps accuracy or whole desktop RAM",
                      "python": platform.python_version(), "platform": platform.system(),
                      "revision": revision, "dirty": bool(dirty),
                      "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in ("sweep/documents.py", "sweep/desktop/data_tools.py",
                                                     "scripts/benchmark_documents.py")},
                      "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
