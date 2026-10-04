# Sweep desktop

Sweep runs in its own native Python/Qt dock at the top of the screen. It does not start a web server,
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

## Working in the dock

- Hover over the top bar or click it to expand the chat. **Pin** keeps it expanded;
  **Esc** or the minus button collapses it. Drag the bar to move it and use the
  bottom-right grip to resize the expanded chat.
- Type a message and press **Enter** to send; **Shift+Enter** inserts a new line.
  Chat, desktop actions, search, scraping and research share this one input.
  Ordinary conversation uses the configured model. A model can propose a
  supported action through **Review and run proposed task**.
- Try `what is 15% of 200`, `open YouTube in Brave`, `open calculator`,
  `search Python documentation`, `research Python packaging`, or
  `scrape https://example.com`. Named sites resolve to actual URLs; if a requested
  browser is unavailable, Sweep reports that instead of silently using another.
- Choose **Attach**, press **Ctrl+O**, or drop a local file. Send an image with a
  question about its text, objects or location clues. Use `inspect locally` or
  `metadata only` for metadata and local OCR without a visual-model request.
  Text and CSV files receive bounded previews in the conversation.
- **History** reopens saved conversations and earlier task results. The plus
  button starts a new chat. Each chat keeps its own recent model context, including
  completed tool results. Context is captured when a message is submitted.
- Continue sending while a task runs to add requests to the queue. Up to 50 tasks
  can wait; one worker runs at a time. Waiting actions receive permission checks
  when dispatched. Use the inline **Cancel waiting task** link or **Stop** for the
  active worker. Completed actions cannot be undone, and launched apps may remain open.
- Results include source links and inline **Export result** and **Retry** actions.
  Retrying creates a new task and preserves the original record. Previous running
  or waiting tasks become **interrupted** after a restart and never resume automatically.

**Ctrl+Alt+Space** opens and focuses the dock from other apps on Windows;
**Ctrl+K** focuses its input. If another app owns the global shortcut, use the tray
icon. Closing the dock collapses it; **Quit Sweep** in the tray exits the app.
Notifications and optional launch-at-sign-in are controlled in Settings.
Workers have a 120-second runtime budget for tools, 240 seconds for conversation
and 300 seconds for image analysis, plus a five-second watchdog grace period.
Local model answers can take longer on a CPU, especially when loading a model.

## Permissions and providers

Computer-changing commands require an explicit task approval. Web tasks ask
before sending queries externally; Settings can remember permission for public
web tasks. Conversation asks before sending a message and recent context, with
an option to remember approval for that conversation. Image-model analysis asks
for each request, including when the model runs locally. File selection grants
access to that file for inspection. These are application-level permissions,
not an OS sandbox.

Settings, task artifacts and activity live under `%LOCALAPPDATA%/Sweep` on Windows.
Chats and task artifacts are plaintext local files; API keys saved through Settings
are protected with Windows account encryption (DPAPI). Do not store passwords in chats.
The existing controller continues using its existing `~/.sweep/controller` store.

Open **Settings** to select the chat provider/model, image provider/model and
Ollama address. Optional OpenAI, Gemini and Anthropic chat keys can be entered
there; image analysis supports Ollama, OpenAI and Gemini. Blank key fields retain
saved keys. An explicitly selected provider is used without falling back to a
different provider. Cloud providers may charge for requests.

Search keys and advanced options can still be configured in a `.env` under the
desktop settings directory using `.env.example` names. Restart after changing
that file. Development launches also load the repository `.env`; packaged apps
do not carry it.

## Local chat and image understanding

Install Ollama separately if it is not already installed. Download the selected
vision model explicitly in a terminal:

```powershell
ollama pull qwen3-vl:2b
```

In Sweep Settings, select **Ollama** for **Image provider**, enter `qwen3-vl:2b`
for **Image-capable model**, and keep **Ollama address** at `http://127.0.0.1:11434`
for inference on this computer. Select **Ollama** for **Chat provider** and choose
an installed chat model, or leave its model field blank for automatic selection.
Use **Check installed local models** or **Start installed local AI (Ollama)**;
these buttons save the provider form before running their task. Starting the server requires an installed
Ollama executable; neither button installs Ollama or downloads models. The address
is a backend connection setting, not a website users need to open.

Attach an image and ask, for example, `Describe the objects and read the sign`.
With this loopback Ollama configuration, model analysis stays on this computer.
Sweep also reads metadata and extracts text locally through installed Tesseract
or Windows OCR with an available language pack. A local-only inspection needs no
vision model. Images are limited to 10 MB and 25 million pixels; supported formats
are PNG, JPEG, WebP, BMP, GIF and TIFF, using the first frame for animated files.

Before model analysis, Sweep resizes the image and removes embedded metadata.
Visual answers may be wrong. Landmark suggestions are unverified hypotheses;
embedded GPS is editable metadata rather than verified scene location. Public
profile searches need a supplied name or handle. Identifying a person or matching
their accounts from a photo is not supported.

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
`dist/Sweep/build-info.json` records the source revision, working-tree status,
runtime versions and build time without credentials or local paths.
`dist/SHA256SUMS.txt` is regenerated only after a successful build. It covers the
app executable, build metadata and setup executable; `--app-only` excludes setup.

Qt libraries remain dynamically linked. Distribution-provided licenses/notices
are copied to the app's `licenses` directory. Qt/PySide sources and versioned
releases are available from https://code.qt.io/pyside/pyside-setup.git/ and
https://code.qt.io/qt/ . Preserve notices and review release obligations before
public distribution. No signing certificate is configured in this repository.

## Scope

The native dock connects desktop skills, public-web search, scraping, bounded
research, selected-file inspection, local OCR, conversation and approved image
analysis. It does not claim completed video tracking, maps/satellite, biometric
identification, voice control, or a general autonomous computer agent. Those
research modules remain separate until integrated and validated through the same
capability/task/permission/event contracts. macOS/Linux can run the Qt source
shell; installers, global shortcuts and startup integration are Windows-only in
this build.
