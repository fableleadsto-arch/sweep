"""Tests for the vendored-framework layer (companion/vendor + vendor_loader).

Verifies the in-repo source inventory — importable pure-Python packages and
stored source archives of the compiled giants — plus the sys.path fallback
and its integration with availability detection and the capability catalog.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys

import pytest

from companion.tools.common import module_available
from companion.tools.vendor_loader import (
    VENDOR_ROOT,
    add_vendored_paths,
    archive_present,
    bundled_wheels,
    describe_all,
    framework_source,
    inventory_summary,
    is_vendored,
    source_archives,
    vendored_path,
    vendored_packages,
    wheel_path,
    wheel_present,
)

IMPORTABLE_EXPECTED = {"transformers", "langchain", "llama_index", "autogen", "crewai"}
ARCHIVE_EXPECTED = {"torch", "tensorflow", "vllm", "jax", "keras"}
WHEEL_EXPECTED = {
    "torch-cu126-win",
    "torchvision-cu126-win",
    "torchaudio-cu126-win",
    "tensorflow-cp313-win",
    "onnxruntime-gpu-win",
}
NVIDIA_LINUX_EXPECTED = {
    "nvidia-cuda-nvrtc-cu12-linux",
    "nvidia-cuda-runtime-cu12-linux",
    "nvidia-cuda-cupti-cu12-linux",
    "nvidia-cudnn-cu12-linux",
    "nvidia-cublas-cu12-linux",
    "nvidia-cufft-cu12-linux",
    "nvidia-curand-cu12-linux",
    "nvidia-cusolver-cu12-linux",
    "nvidia-cusparse-cu12-linux",
    "nvidia-cusparselt-cu12-linux",
    "nvidia-nccl-cu12-linux",
    "nvidia-nvshmem-cu12-linux",
    "nvidia-nvtx-cu12-linux",
    "nvidia-nvjitlink-cu12-linux",
    "nvidia-cufile-cu12-linux",
}
HEAVY_ROOTS = {"torch", "tensorflow", "jax", "vllm", "transformers", "llama_index", "litellm"}


# ── importable vendored packages ────────────────────────────────────────

def test_vendor_root_exists() -> None:
    assert VENDOR_ROOT.is_dir(), "companion/vendor/ must exist"


def test_importable_packages_present() -> None:
    names = set(vendored_packages())
    missing = IMPORTABLE_EXPECTED - names
    if missing:
        pytest.skip(f"Optional complete vendor bundle not installed: {sorted(missing)}")
    assert IMPORTABLE_EXPECTED <= names, f"missing vendored packages: {IMPORTABLE_EXPECTED - names}"


@pytest.mark.parametrize("name", sorted(IMPORTABLE_EXPECTED))
def test_importable_packages_have_license(name) -> None:
    path = vendored_path(name)
    if path is None:
        pytest.skip(f"Optional vendored source not installed: {name}")
    # Every source package that is present must carry its license.
    license_files = [p for p in path.iterdir() if p.name.upper().startswith("LICENSE")]
    assert license_files, f"{name} vendored source is missing a LICENSE file"


def test_vendored_path_unknown_name() -> None:
    assert vendored_path("definitely-not-vendored") is None
    assert not is_vendored("definitely-not-vendored")


def test_vendored_paths_on_sys_path_append_only() -> None:
    root = str(VENDOR_ROOT)
    add_vendored_paths()
    add_vendored_paths()  # idempotent
    occurrences = [p for p in sys.path if p == root]
    assert occurrences, "vendored root must be on sys.path"
    assert len(occurrences) == 1, "add_vendored_paths must not duplicate entries"


def test_find_spec_resolves_vendored_package() -> None:
    # With the vendored root on sys.path, a framework that is NOT pip-installed
    # must still resolve to its vendored source tree.
    for name in IMPORTABLE_EXPECTED:
        if importlib.util.find_spec(name) is not None:
            continue  # installed — that's fine, installed wins
        path = vendored_path(name)
        assert path is not None and path.is_dir(), f"{name} neither installed nor vendored"


# ── source archives ─────────────────────────────────────────────────────

def test_source_archives_declared() -> None:
    archives = source_archives()
    assert {a.name for a in archives} == ARCHIVE_EXPECTED
    for archive in archives:
        assert archive.filename.endswith(".tar.gz")
        assert archive.license
        assert archive.source


def test_source_archives_stored_on_disk() -> None:
    for name in ARCHIVE_EXPECTED:
        assert archive_present(name), f"source archive for {name} is missing on disk"
        path = framework_source(name)
        assert path is not None and path.is_file()
        assert path.stat().st_size > 100_000, f"{name} archive looks truncated"


# ── bundled wheels (local-only, not committed) ──────────────────────────

def test_bundled_wheels_declared() -> None:
    wheels = bundled_wheels()
    all_names = {w.name for w in wheels}
    assert WHEEL_EXPECTED <= all_names
    assert NVIDIA_LINUX_EXPECTED <= all_names
    for wheel in wheels:
        assert wheel.filename.endswith(".whl")
        assert wheel.license
        assert wheel.source.startswith("https://")
    # The PyTorch trio must come from the official PyTorch CUDA index.
    torch_trio = {w.name for w in wheels if "torch" in w.name and "cu126" in w.filename}
    assert torch_trio == {"torch-cu126-win", "torchvision-cu126-win", "torchaudio-cu126-win"}
    assert all(w.source.startswith("https://download.pytorch.org") for w in wheels if "torch" in w.name and "cu126" in w.filename)


def test_nvidia_linux_wheels_platform() -> None:
    wheels = {w.name: w for w in bundled_wheels()}
    for name in NVIDIA_LINUX_EXPECTED:
        assert wheels[name].platform == "linux", f"{name} must be flagged linux"
    for name in WHEEL_EXPECTED:
        assert wheels[name].platform == "win", f"{name} must be flagged win"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_bundled_wheels_are_gitignored() -> None:
    # The multi-GB wheels must never be staged: GitHub hard-rejects files
    # over 100MB. Guard against accidental `git add companion/vendor/*`.
    import subprocess

    repo_root = VENDOR_ROOT.parent.parent  # companion/vendor -> repo root
    wheel_rel = (repo_root / "companion/vendor/archives/torch-2.9.1+cu126-cp313-cp313-win_amd64.whl").as_posix()
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", wheel_rel],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "bundled wheels must be covered by .gitignore"


# The wheels are gitignored by design (GitHub rejects >100MB files), so they
# only exist on machines where they were downloaded. Presence-dependent tests
# skip elsewhere rather than hard-fail.
@pytest.mark.skipif(
    not wheel_present("torch-cu126-win"),
    reason="PyTorch CUDA wheel not on disk (download to companion/vendor/archives/)",
)
def test_wheel_path_and_present() -> None:
    assert wheel_present("torch-cu126-win"), "the PyTorch CUDA wheel must be on disk"
    path = wheel_path("torch-cu126-win")
    assert path is not None and path.is_file()
    # Sanity: a 2.4GB+ wheel should be at least 2GB on disk (not truncated).
    assert path.stat().st_size > 2_000_000_000, "torch CUDA wheel looks truncated"
    assert wheel_present("torchvision-cu126-win")
    assert wheel_present("torchaudio-cu126-win")
    assert wheel_present("tensorflow-cp313-win")
    assert wheel_present("onnxruntime-gpu-win")


def test_wheel_path_unknown_name() -> None:
    assert wheel_path("definitely-not-a-wheel") is None
    assert not wheel_present("definitely-not-a-wheel")


def test_describe_all_includes_wheels() -> None:
    # Presence-agnostic: describe_all always declares the wheel keys, and
    # ``present`` must be truthful (True when on disk, False otherwise).
    inventory = describe_all()
    wheel_names = {w["name"] for w in inventory["wheels"]}
    assert WHEEL_EXPECTED <= wheel_names
    assert NVIDIA_LINUX_EXPECTED <= wheel_names
    for wheel in inventory["wheels"]:
        assert wheel["present"] is wheel_present(wheel["name"])
        assert "platform" in wheel


# ── boot inventory summary ──────────────────────────────────────────────

def test_inventory_summary_shape() -> None:
    inv = inventory_summary()
    assert inv["root"]
    assert set(inv["importable_packages"]) == set(vendored_packages())
    assert inv["source_archives_total"] == 5
    assert inv["source_archives_present"] == sum(archive_present(a.name) for a in source_archives())
    # Wheels: 20 registered today (5 win + 15 nvidia linux); on-disk presence
    # is machine-dependent, but counts must always be consistent and at least
    # the current registry size (>= so future additions don't break the test).
    assert inv["wheels_total"] >= 20
    assert inv["wheels_present"] + len(inv["wheels_missing"]) == inv["wheels_total"]
    for missing in inv["wheels_missing"]:
        assert missing["name"] and missing["platform"] and missing["filename"]


def test_inventory_summary_missing_is_truthful() -> None:
    inv = inventory_summary()
    present_names = {w.name for w in bundled_wheels() if wheel_present(w.name)}
    missing_names = {m["name"] for m in inv["wheels_missing"]}
    all_names = {w.name for w in bundled_wheels()}
    assert present_names | missing_names == all_names
    assert present_names & missing_names == set()


# ── availability integration ────────────────────────────────────────────

def test_module_available_considers_vendored() -> None:
    # Any framework that is pip-installed OR vendored-importable must report
    # available. This is the property the capability engine relies on.
    for name in IMPORTABLE_EXPECTED | ARCHIVE_EXPECTED:
        installed = importlib.util.find_spec(name) is not None
        if installed or is_vendored(name):
            assert module_available(name), f"{name} should be available (installed or vendored)"
        else:
            # Compiled giants are stored as source archives only — they are
            # NOT importable until built, so availability stays honest.
            assert not module_available(name), f"{name} must not report importable"

    assert not module_available("definitely-not-a-real-package")


# ── dependency-honest load ──────────────────────────────────────────────

def test_load_reports_missing_dependency_as_unavailable(monkeypatch) -> None:
    # When a vendored (or installed) framework exists but one of its deps is
    # missing, load() must surface CapabilityUnavailable naming the missing
    # dep — never a raw traceback.
    from companion.tools import common

    def _boom(name):
        raise ModuleNotFoundError("No module named 'somedep'", name="somedep")

    monkeypatch.setattr(common, "module_available", lambda n: True)
    monkeypatch.setattr("importlib.import_module", _boom)

    from companion.tools.common import CapabilityUnavailable

    with pytest.raises(CapabilityUnavailable) as excinfo:
        common.load("transformers")
    assert "somedep" in str(excinfo.value)
    assert "pip install somedep" in str(excinfo.value)


def test_load_missing_framework_raises_unavailable(monkeypatch) -> None:
    from companion.tools import common

    monkeypatch.setattr(common, "module_available", lambda n: False)

    from companion.tools.common import CapabilityUnavailable

    with pytest.raises(CapabilityUnavailable):
        common.load("definitely-not-a-real-package")


# ── catalog integration ─────────────────────────────────────────────────

def test_vendor_source_capability_registered() -> None:
    from companion.capabilities import CAPABILITIES, list_capabilities

    ids = {cap.id for cap in CAPABILITIES}
    assert "vendor-source" in ids

    catalog = {c.id: c for c in list_capabilities()}
    cap = catalog["vendor-source"]
    assert cap.available  # framework-free capability is always available
    assert cap.libraries == []

    # The internal registry entry carries the search keywords.
    from companion.capabilities import _BY_ID

    assert "which frameworks" in _BY_ID["vendor-source"].keywords


def test_vendor_source_capability_runs() -> None:
    from companion.capabilities import CapabilityEngine

    from companion.schemas import ComputeRequest

    import asyncio

    engine = CapabilityEngine()
    result = asyncio.run(
        engine.run(
            ComputeRequest(task="which frameworks do we have source for", capability="vendor-source")
        )
    )
    assert result.ok
    assert result.result["counts"]["importable"] == len(vendored_packages())
    assert result.result["counts"]["archives"] >= 5
    # Wheels are local-only (gitignored): the capability must always report the
    # key and a truthful count matching the bundled_wheels list, whatever the
    # disk state.
    assert "wheels" in result.result["counts"]
    assert result.result["counts"]["wheels"] == len(result.result["bundled_wheels"])
    assert len(result.result["importable_packages"]) == len(vendored_packages())


def test_describe_all_shape() -> None:
    inventory = describe_all()
    assert inventory["root"]
    assert isinstance(inventory["importable"], list)
    assert isinstance(inventory["archives"], list)
    assert isinstance(inventory["wheels"], list)
    importable_names = {p["name"] for p in inventory["importable"]}
    assert importable_names == set(vendored_packages())


# ── honesty guardrails ──────────────────────────────────────────────────

def test_catalog_build_does_not_import_heavy_frameworks() -> None:
    # Building the catalog (which imports vendor_loader + common) must never
    # pull heavy frameworks into sys.modules. Earlier tests in a full-suite
    # run may legitimately import them, so assert against a baseline taken
    # *before* the catalog module is imported here — not against an empty
    # sys.modules.
    import sys as _sys

    baseline = {name for name in _sys.modules if name.split(".")[0] in HEAVY_ROOTS}
    from companion import capabilities  # noqa: F401 - ensure module imported

    after = {name for name in _sys.modules if name.split(".")[0] in HEAVY_ROOTS}
    assert after <= baseline, f"catalog build imported heavy frameworks: {after - baseline}"
