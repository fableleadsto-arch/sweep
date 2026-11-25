"""Exercise native dock + real workers/models with isolated synthetic data.

Run manually on an unlocked Windows desktop. A full-screen synthetic fixture
temporarily covers the display for the approved screen-reading check. Personal
Sweep history/settings are never edited. --exe selects a packaged worker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sweep.desktop.platform import data_directory
    from sweep.desktop.providers import load_config
    config = load_config(data_directory())
    workspace = Path(tempfile.mkdtemp(prefix="sweep-companion-check-"))
    root = workspace / "desktop"
    root.mkdir()
    (root / "providers.json").write_text(json.dumps({"schema_version": 1, **{
        key: config[key] for key in ("provider", "model", "vision_provider", "vision_model", "ollama_base_url")}}), encoding="utf-8")
    os.environ["SWEEP_DESKTOP_DIR"] = str(root)
    os.environ["SWEEP_CONTROLLER_DIR"] = str(workspace / "controller")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QVBoxLayout, QWidget

    from sweep.desktop import window as engine
    from sweep.desktop.dock import DockWindow
    if args.exe:
        executable = args.exe.resolve(strict=True)
        engine.launch_command = lambda *items: [str(executable), *items]
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    window = DockWindow(root)
    window.settings.setValue("notifications", False)
    window.settings.setValue("reduce_motion", True)
    window.pinned = True
    approvals = []
    def approve(dialog):
        approvals.append(dialog.windowTitle())
        return QMessageBox.StandardButton.Yes
    QMessageBox.exec = approve
    observations = []
    fixture = QWidget()
    fixture.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
    fixture.setStyleSheet("background: white; color: black;")
    layout = QVBoxLayout(fixture)
    label = QLabel("SWEEP SCREEN CHECK\nProject Lumen\nInvoice quantity 12\nPacking quantity 10")
    label.setStyleSheet("font: 32pt 'Arial'; padding: 60px;")
    layout.addWidget(label)

    def run(text, expected, attachment=None):
        if attachment:
            window.set_attachment(attachment)
        window.reveal()
        window.prompt.setPlainText(text)
        start = time.monotonic()
        QTest.keyClick(window.prompt, Qt.Key.Key_Return)
        identifier = window.last_enqueued_id
        first = None
        while (window.process is not None or window.pending) and time.monotonic() - start < 250:
            app.processEvents()
            if window._response_previews.get(identifier) and first is None:
                first = round(time.monotonic() - start, 2)
            time.sleep(.02)
        task = window.task_store.get(identifier)
        assert task["state"] == "completed", task
        result = window.task_store.result(identifier)
        assert expected(result), f"Unexpected result for {text!r} in task {identifier}"
        observations.append({"request": text, "capability": task["capability"],
                             "elapsed_s": round(time.monotonic() - start, 2), "first_preview_s": first,
                             "message": result.get("message")})
        print(json.dumps(observations[-1]), flush=True)
        return result

    try:
        run("Hi Sweep", lambda r: "I'm here" in r["message"])
        run("My project is called Lumen. Suggest one short next step for organizing its notes.",
            lambda r: bool(r.get("message")) and not r.get("proposal"))
        run("What is my project called?", lambda r: "lumen" in r["message"].casefold())
        run("calculate 17 * 23", lambda r: "391" in r["message"])
        run("open character map", lambda r: r.get("status") == "ok")
        data = workspace / "orders.csv"
        original = "item,quantity\nbolt,2\nbolt,2\nnut,3\n"
        data.write_text(original, encoding="utf-8")
        run("Remove duplicate rows", lambda r: r.get("output_rows") == 2, data)
        run("Now convert it to JSON", lambda r: len(json.loads(Path(r["artifacts"][0]["path"]).read_text())) == 2)
        assert data.read_text() == original
        fixture.setGeometry(app.primaryScreen().geometry())
        fixture.showFullScreen()
        fixture.raise_()
        fixture.activateWindow()
        app.processEvents()
        time.sleep(1)
        run("Read my screen", lambda r: all(word in r.get("text", "") for word in ("Lumen", "12", "10")))
        assert window.isVisible()
        run("What were the two quantities in that screen text?", lambda r: "12" in r["message"] and "10" in r["message"])
        fixture.hide()
        window.reveal()
        app.processEvents()
        window.grab().save(str(workspace / "verified-dock.png"))
        window.resize(380, 650)
        app.processEvents()
        window.grab().save(str(workspace / "narrow-dock.png"))
        summary = {"workspace": str(workspace), "worker_executable": str(args.exe or "source"),
                   "approvals": approvals, "observations": observations}
        (workspace / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"passed": True, "workspace": str(workspace)}), flush=True)
    finally:
        fixture.close()
        window.shutdown()
        window.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    main()
