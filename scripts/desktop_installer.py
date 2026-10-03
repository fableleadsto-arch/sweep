"""Per-user Sweep setup wizard, frozen with a bundled local application payload.

Uses only the standard library. It never installs services, requests elevation,
enables startup, or copies development credentials/models into the application.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from zipfile import ZipFile

APP_ID = "Sweep.Desktop.2"


def unpack(payload: Path, destination: Path, progress=lambda message: None):
    """Reject traversal and symlink members before writing any payload content."""
    root = destination.resolve()
    with ZipFile(payload) as archive:
        members = archive.infolist()
        if sum(item.file_size for item in members) > 2_000_000_000:
            raise ValueError("Installer payload is unexpectedly large.")
        for item in members:
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe path in installer payload.")
        for index, item in enumerate(members):
            archive.extract(item, root)
            if index % 100 == 0:
                progress(f"Installing application files · {index + 1} / {len(members)}")


def create_shortcut(target: Path, shortcut: Path):
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    script = "param([string]$TargetPath,[string]$LinkPath)\n$ws = New-Object -ComObject WScript.Shell\n$link = $ws.CreateShortcut($LinkPath)\n$link.TargetPath = $TargetPath\n$link.WorkingDirectory = Split-Path -Parent $TargetPath\n$link.IconLocation = $TargetPath + ',0'\n$link.Save()\n"
    with tempfile.TemporaryDirectory(prefix="sweep-shortcut-") as temp:
        path = Path(temp) / "shortcut.ps1"
        path.write_text(script, encoding="utf-8-sig")
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-File", str(path), "-TargetPath", str(target), "-LinkPath", str(shortcut)],
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def uninstall_script(target: Path, shortcuts: list[Path]) -> str:
    def literal(value):
        return "'" + str(value).replace("'", "''") + "'"
    return "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "Add-Type -AssemblyName PresentationFramework",
        "$answer = [System.Windows.MessageBox]::Show('Uninstall Sweep? Your task history and settings will be kept. Please close Sweep first.', 'Uninstall Sweep', 'YesNo')",
        "if ($answer -ne 'Yes') { exit }",
        f"$target = {literal(target)}",
        "$expected = Join-Path $env:LOCALAPPDATA 'Programs\\Sweep'",
        "if ([IO.Path]::GetFullPath($target) -ne [IO.Path]::GetFullPath($expected)) { throw 'Unexpected uninstall path' }",
        "if ((Get-Item -LiteralPath $target).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing to uninstall a redirected folder' }",
        "$marker = Get-Content -LiteralPath (Join-Path $target 'sweep-install.json') -Raw | ConvertFrom-Json",
        f"if ($marker.app_id -ne '{APP_ID}') {{ throw 'Installation marker missing' }}",
        "try {",
        "  $active = Get-Process Sweep -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $target 'Sweep.exe') }",
        "  if ($active) { throw 'Close Sweep using Quit Sweep in its tray menu, then retry.' }",
        "  Remove-Item -LiteralPath $target -Recurse -Force",
        *[f"  if (Test-Path -LiteralPath {literal(path)}) {{ Remove-Item -LiteralPath {literal(path)} -Force }}" for path in shortcuts],
        "  Remove-Item -LiteralPath 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Sweep' -Force -ErrorAction SilentlyContinue",
        "  Remove-ItemProperty -LiteralPath 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' -Name 'Sweep' -ErrorAction SilentlyContinue",
        "  [System.Windows.MessageBox]::Show('Sweep has been uninstalled. Your personal data was preserved.', 'Sweep') | Out-Null",
        "} catch { [System.Windows.MessageBox]::Show($_.Exception.Message, 'Uninstall needs attention') | Out-Null }",
    ])


def install(payload: Path, desktop: bool, progress):
    import winreg
    programs = Path(os.environ["LOCALAPPDATA"]) / "Programs"
    target = programs / "Sweep"
    programs.mkdir(parents=True, exist_ok=True)
    if target.exists():
        marker = target / "sweep-install.json"
        if not marker.exists() or json.loads(marker.read_text())["app_id"] != APP_ID:
            raise RuntimeError(f"The folder {target} already exists and is not a recognized Sweep installation.")
    staging = programs / ("Sweep-install-" + uuid.uuid4().hex)
    backup = programs / ("Sweep-previous-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        unpack(payload, staging, progress)
        if not (staging / "Sweep.exe").exists():
            raise RuntimeError("The application executable is missing from the payload.")
        (staging / "sweep-install.json").write_text(json.dumps({"app_id": APP_ID, "version": "2.0.0"}), encoding="utf-8")
        if target.exists():
            target.rename(backup)
        try:
            staging.rename(target)
        except OSError:
            if backup.exists():
                backup.rename(target)
            raise
        progress("Creating Start menu entry and uninstall registration")
        shortcut = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Sweep.lnk"
        shortcuts = [shortcut]
        create_shortcut(target / "Sweep.exe", shortcut)
        if desktop:
            # Ask Windows for the actual Desktop path, including OneDrive relocation.
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
                desktop_path = Path(os.path.expandvars(winreg.QueryValueEx(key, "Desktop")[0]))
            shortcut = desktop_path / "Sweep.lnk"
            create_shortcut(target / "Sweep.exe", shortcut)
            shortcuts.append(shortcut)
        script = target / "uninstall.ps1"
        script.write_text(uninstall_script(target, shortcuts), encoding="utf-8-sig")
        uninstall = subprocess.list2cmdline(["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", "-File", str(script)])
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Sweep") as key:
            for name, value in {"DisplayName": "Sweep", "DisplayVersion": "2.0.0", "Publisher": "Sweep",
                                "InstallLocation": str(target), "DisplayIcon": str(target / "Sweep.exe"),
                                "UninstallString": uninstall}.items():
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
            winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
        if backup.exists() and backup.resolve().parent == programs.resolve() and backup.name.startswith("Sweep-previous-"):
            shutil.rmtree(backup)
        return target
    finally:
        if staging.exists() and staging.resolve().parent == programs.resolve() and staging.name.startswith("Sweep-install-"):
            shutil.rmtree(staging)


def main():
    import tkinter as tk
    from tkinter import ttk, messagebox
    root = tk.Tk()
    root.title("Install Sweep")
    root.geometry("590x390")
    root.resizable(False, False)
    root.configure(bg="#f5f4ef")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#f5f4ef")
    style.configure("TLabel", background="#f5f4ef", foreground="#263e38", font=("Segoe UI", 11))
    style.configure("TCheckbutton", background="#f5f4ef", font=("Segoe UI", 10))
    frame = ttk.Frame(root, padding=30)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Meet Sweep.", font=("Segoe UI", 25, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Your desktop companion for research, files and everyday tasks.", wraplength=510).pack(anchor="w", pady=12)
    ttk.Label(frame, text="Installs for your Windows account. Python is included.\nNo browser, administrator account or model download is required.", wraplength=510).pack(anchor="w", pady=8)
    desktop = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Create a desktop shortcut", variable=desktop).pack(anchor="w", pady=12)
    status = tk.StringVar(value="Ready to install in LocalAppData\\Programs\\Sweep")
    ttk.Label(frame, textvariable=status, wraplength=510).pack(anchor="w", pady=12)
    progressbar = ttk.Progressbar(frame, mode="indeterminate")
    progressbar.pack(fill="x", pady=8)
    events = queue.Queue()
    installed = None
    busy = False
    payload = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "payload.zip"

    def start():
        nonlocal busy
        if installed:
            subprocess.Popen([str(installed / "Sweep.exe")], cwd=installed,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            root.destroy()
            return
        busy = True
        button.configure(state="disabled")
        progressbar.start(15)
        desktop_choice = desktop.get()
        def work():
            try:
                events.put(("done", install(payload, desktop_choice, lambda message: events.put(("progress", message)))))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def poll():
        nonlocal installed, busy
        while not events.empty():
            kind, value = events.get()
            if kind == "progress":
                status.set(value)
            else:
                busy = False
                progressbar.stop()
                button.configure(state="normal")
                if kind == "done":
                    installed = value
                    status.set("Sweep is installed. Find it in your Start menu.")
                    button.configure(text="Open Sweep")
                else:
                    status.set("Installation needs attention. You can retry.")
                    messagebox.showerror("Sweep setup", value)
        root.after(100, poll)
    button = ttk.Button(frame, text="Install Sweep", command=start)
    button.pack(anchor="e", pady=8)
    root.protocol("WM_DELETE_WINDOW", lambda: None if busy else root.destroy())
    root.after(100, poll)
    if "--smoke-test" in sys.argv:
        root.after(1000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
