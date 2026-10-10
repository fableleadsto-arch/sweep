import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sweep.desktop import __main__ as desktop


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native Qt release lifecycle")
def test_native_smoke_launch_exits_and_releases_its_instance_lock():
    pytest.importorskip("PySide6.QtWidgets")
    # The smoke check must exercise the real dock close handler and event loop,
    # not a mocked QApplication or a hidden/offscreen platform plugin.
    with tempfile.TemporaryDirectory(prefix="sweep-native-smoke-") as temporary:
        directory = Path(temporary)
        profile = directory / "profile"
        environment = os.environ.copy()
        environment.update(SWEEP_DESKTOP_DIR=str(profile),
                           SWEEP_CONTROLLER_DIR=str(directory / "controller"),
                           PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                           QT_QPA_PLATFORM="windows")
        process = subprocess.run([sys.executable, "-m", "sweep.desktop", "--smoke-test"],
            cwd=directory, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW)
        assert process.returncode == 0, process.stdout + process.stderr
        assert (profile / "desktop.ini").exists(), "The native dock did not initialize"
        assert not (profile / "desktop.lock").exists(), "The smoke launch left its instance lock"
        assert not (profile / "startup-error.log").exists()


def test_instance_name_is_stable_normalized_and_bounded(tmp_path):
    root = tmp_path / "profile"
    name = desktop.instance_server_name(root)
    assert name == desktop.instance_server_name(root / ".." / "profile")
    assert len(name) == 46
    assert len(desktop.instance_server_name(root / ("long-directory-" * 15))) == 46
    if os.name == "nt":
        assert name == desktop.instance_server_name(type(root)(str(root).upper()))


def test_instance_names_do_not_collapse_distinct_paths(tmp_path):
    assert desktop.instance_server_name(tmp_path / "a_b" / "c") != (
        desktop.instance_server_name(tmp_path / "a" / "b_c"))


def test_failed_connection_is_reported_and_socket_closed(monkeypatch):
    network = pytest.importorskip("PySide6.QtNetwork")
    calls = []
    socket = SimpleNamespace(connectToServer=lambda name: calls.append(name),
                             waitForConnected=lambda timeout: False,
                             abort=lambda: calls.append("closed"))
    monkeypatch.setattr(network, "QLocalSocket", lambda: socket)
    with pytest.raises(RuntimeError, match="could not be reached"):
        desktop.notify_existing("Sweep-test")
    assert calls == ["Sweep-test", "closed"]


def test_unacknowledged_launch_is_not_success(monkeypatch):
    network = pytest.importorskip("PySide6.QtNetwork")
    socket = SimpleNamespace(connectToServer=lambda name: None,
                             waitForConnected=lambda timeout: True,
                             write=lambda data: len(data), bytesToWrite=lambda: 0,
                             canReadLine=lambda: False,
                             waitForReadyRead=lambda timeout: False, abort=lambda: None)
    monkeypatch.setattr(network, "QLocalSocket", lambda: socket)
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        desktop.notify_existing("Sweep-test")


@pytest.mark.parametrize("error, message", [("PermissionError", "permission"),
                                           ("UnknownError", "could not lock")])
def test_lock_errors_do_not_try_to_contact_another_instance(tmp_path, monkeypatch, error, message):
    core = pytest.importorskip("PySide6.QtCore")
    errors = core.QLockFile.LockError
    fake_lock = SimpleNamespace(tryLock=lambda timeout: False,
                                error=lambda: getattr(errors, error))
    factory = lambda path: fake_lock
    factory.LockError = errors
    monkeypatch.setattr(core, "QLockFile", factory)
    monkeypatch.setattr(desktop, "notify_existing", lambda *args, **kwargs: pytest.fail("wrong branch"))
    with pytest.raises(RuntimeError, match=message):
        desktop.claim_instance(tmp_path)


def test_existing_instance_receives_background_flag(tmp_path, monkeypatch):
    core = pytest.importorskip("PySide6.QtCore")
    errors = core.QLockFile.LockError
    fake_lock = SimpleNamespace(tryLock=lambda timeout: False,
                                error=lambda: errors.LockFailedError)
    factory = lambda path: fake_lock
    factory.LockError = errors
    monkeypatch.setattr(core, "QLockFile", factory)
    calls = []
    monkeypatch.setattr(desktop, "notify_existing", lambda name, **kwargs: calls.append((name, kwargs)))
    assert desktop.claim_instance(tmp_path, background=True) is None
    assert calls == [(desktop.instance_server_name(tmp_path), {"background": True})]


def test_failed_listen_releases_owned_lock(tmp_path, monkeypatch):
    core = pytest.importorskip("PySide6.QtCore")
    network = pytest.importorskip("PySide6.QtNetwork")
    calls = []
    fake_lock = SimpleNamespace(tryLock=lambda timeout: True,
                                unlock=lambda: calls.append("unlocked"))
    monkeypatch.setattr(core, "QLockFile", lambda path: fake_lock)
    fake_server = SimpleNamespace(setSocketOptions=lambda option: None,
                                  listen=lambda name: False,
                                  errorString=lambda: "address unavailable",
                                  close=lambda: calls.append("closed"))
    factory = lambda: fake_server
    factory.SocketOption = network.QLocalServer.SocketOption
    factory.removeServer = lambda name: True
    monkeypatch.setattr(network, "QLocalServer", factory)
    with pytest.raises(RuntimeError, match="address unavailable"):
        desktop.claim_instance(tmp_path)
    assert calls == ["closed", "unlocked"]


@pytest.mark.parametrize("quiet", [False, True])
def test_startup_error_returns_failure_and_respects_background(tmp_path, monkeypatch, quiet):
    widgets = pytest.importorskip("PySide6.QtWidgets")
    dialogs = []
    monkeypatch.setattr(widgets.QMessageBox, "critical", lambda *args: dialogs.append(args))
    assert desktop.report_startup_error(tmp_path, "Cannot connect", quiet=quiet) == 1
    assert (tmp_path / "startup-error.log").read_text() == "Cannot connect\n"
    assert bool(dialogs) is not quiet


def test_main_returns_nonzero_for_instance_failure_without_opening_window(tmp_path, monkeypatch):
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from sweep.desktop import platform
    import dotenv
    fake_app = SimpleNamespace(setApplicationName=lambda name: None,
                               setOrganizationName=lambda name: None,
                               setQuitOnLastWindowClosed=lambda value: None)
    monkeypatch.setattr(widgets, "QApplication", lambda argv: fake_app)
    monkeypatch.setattr(platform, "data_directory", lambda: tmp_path)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(widgets.QMessageBox, "critical", lambda *args: pytest.fail("background dialog"))

    def fail(*args, **kwargs):
        raise RuntimeError("Cannot connect to Sweep")

    monkeypatch.setattr(desktop, "claim_instance", fail)
    assert desktop.main(["--background"]) == 1
    assert (tmp_path / "startup-error.log").exists()


@pytest.mark.parametrize("background", [False, True])
def test_local_launch_is_acknowledged_without_showing_background_window(tmp_path, background):
    widgets = pytest.importorskip("PySide6.QtWidgets")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    lock, server = desktop.claim_instance(tmp_path)
    shown = []
    server.newConnection.connect(lambda: desktop.accept_instance_connections(server, lambda: shown.append(True)))
    command = [sys.executable, "-c",
               "from PySide6.QtCore import QCoreApplication; app = QCoreApplication([]); "
               "from sweep.desktop.__main__ import notify_existing; import sys; "
               "notify_existing(sys.argv[1], background=sys.argv[2] == 'True')",
               desktop.instance_server_name(tmp_path), str(background)]
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        deadline = time.monotonic() + 8
        while process.poll() is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert process.poll() is not None, "Local instance request did not finish"
        output, errors = process.communicate(timeout=1)
        assert process.returncode == 0, (output + errors).decode(errors="replace")
        assert shown == ([] if background else [True])
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
        server.close()
        lock.unlock()
        app.processEvents()
