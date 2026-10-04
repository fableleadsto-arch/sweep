"""Launch Sweep in its own native desktop window."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time


def instance_server_name(root: Path) -> str:
    """Bound the OS pipe/socket name and avoid collisions from path separators."""
    normalized = os.path.normcase(str(root.resolve()))
    digest = hashlib.sha256(normalized.encode("utf-8", errors="surrogatepass")).hexdigest()
    return "Sweep-" + digest[:40]


def notify_existing(server_name: str, *, background: bool = False) -> None:
    """Require acknowledgement before reporting a successful second launch."""
    from PySide6.QtNetwork import QLocalSocket
    socket = QLocalSocket()
    try:
        socket.connectToServer(server_name)
        if not socket.waitForConnected(2000):
            raise RuntimeError("Sweep is already running, but could not be reached. "
                               "Wait a moment and try opening Sweep again.")
        request = b"background\n" if background else b"show\n"
        if socket.write(request) != len(request):
            raise RuntimeError("Could not send the request to the running Sweep window.")
        if socket.bytesToWrite() and not socket.waitForBytesWritten(2000):
            raise RuntimeError("Could not send the request to the running Sweep window.")
        deadline = time.monotonic() + 2
        while not socket.canReadLine():
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0 or not socket.waitForReadyRead(remaining):
                raise RuntimeError("The running Sweep window did not acknowledge this launch. "
                                   "Wait a moment and try again.")
        if bytes(socket.readLine(32)) != b"ok\n":
            raise RuntimeError("The running Sweep window returned an invalid response.")
    finally:
        socket.abort()


def claim_instance(root: Path, *, background: bool = False):
    """Return the owned lock/server, or None after notifying the existing app."""
    from PySide6.QtCore import QLockFile
    from PySide6.QtNetwork import QLocalServer
    lock = QLockFile(str(root / "desktop.lock"))
    server_name = instance_server_name(root)
    if not lock.tryLock(100):
        if lock.error() == QLockFile.LockError.PermissionError:
            raise RuntimeError(f"Sweep cannot write its desktop lock in {root}. "
                               "Check that your account has permission to use this folder.")
        if lock.error() != QLockFile.LockError.LockFailedError:
            raise RuntimeError(f"Sweep could not lock its desktop data folder: {root}.")
        notify_existing(server_name, background=background)
        return None
    QLocalServer.removeServer(server_name)
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(server_name):
        error = server.errorString()
        server.close()
        lock.unlock()
        raise RuntimeError(f"Sweep could not start its desktop connection: {error}")
    return lock, server


def accept_instance_connections(server, reveal) -> None:
    """Handle only bounded, explicit requests from another local launch."""
    from PySide6.QtCore import QTimer
    while server.hasPendingConnections():
        socket = server.nextPendingConnection()
        socket.setReadBufferSize(32)
        timer = QTimer(socket)
        timer.setSingleShot(True)
        timer.timeout.connect(socket.abort)
        timer.start(3000)
        socket.disconnected.connect(socket.deleteLater)

        def receive(current=socket, timeout=timer):
            if not current.canReadLine():
                if current.bytesAvailable() >= 32:
                    current.abort()
                return
            request = bytes(current.readLine(32))
            if request not in {b"show\n", b"background\n"}:
                current.abort()
                return
            timeout.stop()
            if request == b"show\n":
                reveal()
            current.write(b"ok\n")
            current.disconnectFromServer()

        socket.readyRead.connect(receive)
        if socket.bytesAvailable():
            receive()


def report_startup_error(root: Path, message: str, *, quiet: bool = False) -> int:
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    try:
        (root / "startup-error.log").write_text(message + "\n", encoding="utf-8")
    except OSError:
        pass
    if not quiet:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Sweep could not start", message)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--events")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    from .platform import data_directory
    root = data_directory()
    root.mkdir(parents=True, exist_ok=True)
    from dotenv import load_dotenv
    if not getattr(sys, "frozen", False):
        load_dotenv(override=False)
    load_dotenv(root / ".env", override=False)
    if args.task:
        if not args.events:
            parser.error("--events is required with --task")
        from .runtime import run_worker
        return run_worker(args.task, args.events)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sweep.Desktop")
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
    except ImportError:
        raise SystemExit("Desktop dependencies missing. Run python setup_sweep.py --install-only.")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Sweep")
    app.setOrganizationName("Sweep")
    app.setQuitOnLastWindowClosed(False)
    try:
        instance = claim_instance(root, background=args.background)
    except (OSError, RuntimeError) as exc:
        return report_startup_error(root, str(exc), quiet=args.background or args.smoke_test)
    if instance is None:
        return 0
    lock, server = instance
    window = None
    try:
        from .dock import DockWindow
        window = DockWindow(root)
        server.newConnection.connect(lambda: accept_instance_connections(server, window.reveal))
        window.show()
        if args.smoke_test:
            QTimer.singleShot(1000, app.quit)
        return app.exec()
    finally:
        try:
            if window is not None:
                window.shutdown()
        finally:
            server.close()
            lock.unlock()


if __name__ == "__main__":
    raise SystemExit(main())
