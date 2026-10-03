# Sweep desktop

Sweep runs in its own native Python/Qt window. It does not start a web server,
open a browser to host the interface, or require a localhost URL. Browsers open
only when you explicitly open a source link or request a browser action.

## Install and launch on Windows

Run `dist/Sweep-Setup.exe`. The installer includes Python and the desktop runtime,
installs for the current user under `%LOCALAPPDATA%/Programs/Sweep`, adds a Start
menu entry and registers an uninstaller in Windows Installed apps. A desktop
shortcut is optional. Launch-at-sign-in is off until enabled in Settings.
The local build is unsigned; it is not a signed public release.

You can also run `dist/Sweep/Sweep.exe` directly with its `_internal` folder beside
it. For development, run `python setup_sweep.py`, then `launch_sweep.cmd` or
`.venv/Scripts/python.exe -m sweep.desktop`. Use `--headless` with setup to install
only the terminal workflow.

The installer never bundles `.env`, model weights, personal files or task history.
Uninstall preserves your personal data. Quit Sweep from its tray menu before
upgrading or uninstalling.

## Working in the desktop app

- Enter a request and press Ctrl+Enter. Automatic routing selects an existing
  desktop skill, search, scraping, research, or configured conversation provider.
- Try `what is 15% of 200`, `open calculator`, `search Python documentation`,
  `research Python packaging`, or a public HTTPS URL.
- Choose Attach file, press Ctrl+O, or drop a file for local inspection. CSV
  previews show up to 100 rows/50 columns; text previews are bounded. Images have
  a native preview. Reading a file does not upload it.
- Activity lists actual operations and research steps. Results include source
  links, evidence, data tables or media. Export result opens a native save dialog.
- Stop terminates the task worker. Actions already completed cannot be undone;
  externally launched apps may continue running.
- Recent tasks and result artifacts remain available after restarting. Interrupted
  tasks are labeled; they are not automatically resumed.

Ctrl+Alt+Space brings Sweep forward on Windows, including from other apps.
Ctrl+K focuses the command bar inside Sweep. If another application has already
registered the global shortcut, use the tray icon. Window minimize, maximize,
move, resize and taskbar behavior are provided by the operating system.
Closing the window can keep Sweep in the tray; Quit Sweep exits it completely.
Notifications can be disabled in Settings.

## Permissions and providers

Computer-changing commands require an explicit task approval. Web tasks ask
before sending queries externally; Settings can remember permission for public
web tasks. Conversation separately asks before using configured providers.
File selection grants access to that file for its inspection task. These are
application-level permissions, not an OS sandbox.

Settings, task artifacts and activity live under `%LOCALAPPDATA%/Sweep` on Windows.
They are plaintext local files. Delete selected task directories there if needed.
The existing controller continues using its existing `~/.sweep/controller` store.

For conversation/search provider configuration, place your own `.env` in the
desktop settings directory. Use the existing `.env.example` names, including
`OLLAMA_BASE_URL`/`OLLAMA_MODEL` or configured cloud-provider keys. Restart Sweep
after changes. Development launches also load the repository `.env`; packaged
apps do not carry it. Configured providers may charge for requests.

## Reproduce the Windows build

```powershell
python setup_sweep.py --install-only --extras build --extras dev
.venv\Scripts\python.exe scripts\build_desktop.py
```

Outputs are `dist/Sweep/Sweep.exe` and `dist/Sweep-Setup.exe`. Build on Windows;
PyInstaller bundles the interpreter and native libraries for the build platform.
The setup wizard uses Python's native Tk interface and includes the application
payload. It validates archive paths, creates per-user shortcuts, and registers a
marker-checked uninstaller. No third-party installer compiler is required.

Qt libraries remain dynamically linked. Distribution-provided licenses/notices
are copied to the app's `licenses` directory. Qt/PySide sources and versioned
releases are available from https://code.qt.io/pyside/pyside-setup.git/ and
https://code.qt.io/qt/ . Preserve notices and review release obligations before
public distribution. No signing certificate is configured in this repository.

## Scope

The native shell currently connects desktop skills, public-web search, scraping,
bounded research, selected-file previews and the existing conversation provider
chain. It does not claim completed video tracking, maps/satellite, biometric
identification, voice control, or a general autonomous computer agent. Those
research modules remain separate until integrated and validated through the same
capability/task/permission/event contracts. macOS/Linux can run the Qt source
shell; installers, global shortcuts and startup integration are Windows-only in
this build.
