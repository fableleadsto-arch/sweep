"""Launch Sweep in its own native desktop window."""
from __future__ import annotations

import argparse
import sys


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
        from PySide6.QtCore import QLockFile, QTimer
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
    except ImportError:
        raise SystemExit("Desktop dependencies missing. Run python setup_sweep.py --install-only.")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Sweep")
    app.setOrganizationName("Sweep")
    app.setQuitOnLastWindowClosed(False)
    lock = QLockFile(str(root / "desktop.lock"))
    server_name = "Sweep-" + str(root.resolve()).replace(":", "").replace("\\", "_").replace("/", "_")
    if not lock.tryLock(100):
        socket = QLocalSocket()
        socket.connectToServer(server_name)
        socket.waitForConnected(1000)
        socket.write(b"show")
        socket.waitForBytesWritten(1000)
        return 0
    QLocalServer.removeServer(server_name)
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    server.listen(server_name)
    from .window import SweepWindow
    window = SweepWindow(root)
    def show_existing():
        socket = server.nextPendingConnection()
        if socket:
            socket.disconnectFromServer()
            socket.deleteLater()
        window.reveal()
    server.newConnection.connect(show_existing)
    if not args.background or not window.tray.isVisible():
        window.show()
    if args.smoke_test:
        QTimer.singleShot(1000, app.quit)
    result = app.exec()
    window.shutdown()
    server.close()
    lock.unlock()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
