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
  The composer grows for multiline messages. The conversation title and waiting
  count sit above the chat; **Stop** appears while a task is active. Secondary
  result actions are available under **More**.
  Chat, desktop actions, search, scraping and research share this one input.
  Ordinary conversation runs locally and streams its answer into the dock. Sweep can propose a
  supported action through **Review and run proposed task**.
- Try `what is 15% of 200`, `open YouTube in Brave`, `open calculator`,
  `search Python documentation`, `research Python packaging`, or
  `scrape https://example.com`. Named sites resolve to actual URLs; if a requested
  browser is unavailable, Sweep reports that instead of silently using another.
- Choose **Attach**, press **Ctrl+O**, or drop a local file. Send an image with a
  question about its text, objects or location clues. Use `inspect locally` or
  `metadata only` for metadata and local OCR without a visual-model request.
  Documents and datasets receive readable previews, profiles and output-file
  cards in the conversation. Recordings can produce local transcripts; videos can
  produce sampled-frame previews and descriptions. See the file examples below.
- **History** searches and reopens saved conversations and earlier task results. The plus
  button starts a new chat. Each chat keeps its own recent model context, including
  completed tool results. Context is captured when a message is submitted.
- Follow up on an earlier file or result: `What does this document say about cancellation?`
  or `Compare these sources`. Answers use selected saved excerpts and include a
  **Sources used** action. Original extraction limits still apply. Attach up to four
  PDF, DOCX or text documents together to compare them; other attachment types
  currently accept one file per request.
- After a data transformation, `Now convert it to JSON` or another supported data
  operation uses the earlier output, including when queued before that task
  finishes. A failed source stops its follow-up; it never silently reuses the
  original file or another conversation's result.
- If a follow-up fails or is interrupted, choose **Review recovery**. Keep valid
  original sources or explicitly select a completed retry and review its excerpt.
  For a failed source, choose **Retry source**, wait for it to finish, then return
  to the follow-up's recovery panel. **Retry with these sources** creates a new
  follow-up with your selected results; original tasks remain unchanged. Normal
  task permissions still apply, including approval for device actions.
- Continue sending while a task runs to add requests to the queue. Up to 50 tasks
  can wait; one worker runs at a time. Waiting actions receive permission checks
  when dispatched. Use the inline **Remove from queue** link or **Stop** for the
  active worker. Completed actions cannot be undone, and launched apps may remain open.
- Results include source links and inline **Export result** and **Try again** actions.
  Research also saves a readable Markdown report and a JSON evidence graph, with
  actual excerpts, source URLs, recorded timestamps and unresolved work. Open their
  file cards to preview or save a copy. These records do not verify every source claim.
  Retrying creates a new task and preserves the original record. Previous running
  or waiting tasks become **interrupted** after a restart and never resume automatically.
- Missing or damaged saved results show **Result unavailable** with **Try again**.
  They cannot be exported or reused as conversation evidence. A task is marked
  complete only after its result passes validation and is saved. Result records
  are limited to 2 MB; larger outputs belong in separate generated files.

**Ctrl+Alt+Space** opens and focuses the dock from other apps on Windows;
**Ctrl+K** focuses its input. If another app owns the global shortcut, use the tray
icon. Closing the dock collapses it; **Quit Sweep** in the tray exits the app.
Notifications and optional launch-at-sign-in are controlled in Settings.
Workers have a 120-second runtime budget for ordinary tools, 240 seconds for
conversation, documents and speech setup, and 300 seconds for image analysis,
sampled-video descriptions and transcription, plus a five-second watchdog grace period.
Local model answers can take longer on a CPU, especially when loading a model.
Streaming keeps completed cards and text selections intact while the current
answer changes. Local chat uses a bounded 4,096-token context and allows the
model to unload two minutes after its last request. An engine started by Sweep
loads at most one model and one inference context at a time; settings of an
already-running external engine are left unchanged.

## Local processing and permissions

Computer-changing commands require an explicit task approval. Web tasks ask
before sending queries externally; Settings can remember permission for public
web tasks. Local conversation asks to use the message and recent thread context,
with an option to remember approval for that conversation. Image analysis asks
for each request. File selection grants access to that file for inspection;
data transformations create separate task files after approval. These are
application-level permissions, not an OS sandbox.

Settings, task artifacts and activity live under `%LOCALAPPDATA%/Sweep` on Windows.
Chats and task artifacts are plaintext local files. Do not store passwords in chats.
The existing controller continues using its existing `~/.sweep/controller` store.

The desktop only accepts a local inference engine at an HTTP loopback address.
Remote endpoints, cloud-backed models and remote model aliases are rejected.
Model metadata is checked before prompts or images are sent to the engine.
Environment API keys and saved settings from older cloud-enabled builds cannot
enable cloud inference. Legacy encrypted credential blobs are preserved but not
read or used by the desktop. The separate developer APIs retain their own
configuration; the desktop never falls back to them.

Open **Settings** for startup, notifications, public-web permission and reduced
motion. Model identifiers and the local engine address are under **Advanced local
settings**. **Check readiness** verifies available local processing; **Start local
processing** starts an already installed engine. Both actions save the form first.

Optional public-search keys can still be configured in a `.env` under the
desktop settings directory using `.env.example` names. Restart after changing
that file. Development launches also load the repository `.env`; packaged apps
do not carry it.

## Local chat and image understanding

The current local runtime adapter uses Ollama. If it is not already installed,
open the [official Windows download page](https://ollama.com/download/windows),
choose the manual download and run its installer. Then open a new PowerShell
window and explicitly download a chat model and image-capable model:

```powershell
ollama pull qwen2.5-coder:3b
ollama pull qwen3-vl:2b-instruct
```

Under **Settings → Advanced local settings**, enter `qwen2.5-coder:3b` for
**Conversation model**, `qwen3-vl:2b-instruct` for **Image model**, and
`http://127.0.0.1:11434` for **Local engine address**. The conversation model may
be left blank for automatic selection from installed models. These are example
installed model identifiers, not services Sweep contacts remotely.

Use the explicit instruction variant for image answers. Some reasoning-only model
variants cannot disable internal reasoning with a request flag and may consume the
answer budget before producing usable text. Sweep labels output-limit responses
as incomplete rather than presenting a truncated answer as finished.

Sweep can start the installed engine when conversation needs it, with cloud
support disabled for that process. It does not install the engine or download
models during inference. The engine address is an internal connection setting,
not a website users need to open. A missing or incompatible local model produces
a setup error instead of contacting a cloud provider.

Attach an image and ask, for example, `Describe the objects and read the sign`.
Model analysis stays on this computer.
Sweep also reads metadata and extracts text locally through installed Tesseract
or Windows OCR with an available language pack. A local-only inspection needs no
vision model. Images are limited to 10 MB and 25 million pixels; supported formats
are PNG, JPEG, WebP, BMP, GIF and TIFF, using the first frame for animated files.

Before model analysis, Sweep resizes the image and removes embedded metadata.
Visual answers may be wrong. Landmark suggestions are unverified hypotheses;
embedded GPS is editable metadata rather than verified scene location. Public
profile searches need a supplied name or handle. Identifying a person or matching
their accounts from a photo is not supported.

## Documents and data in the same chat

Attach one file, then state the task. For example:

| Attached file | Request | Result |
|---|---|---|
| PDF, DOCX, TXT or Markdown | `Read this document` or `Summarize this document` | Local text extraction; local answers when requested |
| CSV, TSV, JSON or JSONL | `Summarize this data` | Schema, row count, missing/duplicate counts, numeric statistics and a table preview |
| CSV or JSON records | `Clean this dataset` | Trim text cells, remove empty rows and exact duplicates, save a new file |
| Supported dataset | `Remove duplicates by name, city` | Keep the first row for each combination of named columns |
| Supported dataset | `Filter where age >= 18 and export to JSON` | Apply one comparison and save the matching rows |
| Supported dataset | `Convert to TSV` | Create CSV, TSV, JSON or JSONL output |
| Standalone SQLite backup | `Inspect table 'people'` | Inspect an explicitly selected ordinary table; transform/export through the same requests |

Output cards open a local preview with **Save a copy…**. Sweep never changes the
input file. Artifacts record their size and SHA-256 hash; the result records the
source file's hash. JSON/JSONL exports preserve nested objects and arrays; CSV/TSV
represent those cells as JSON text. CSV formula-like text is retained as data and
is not automatically opened in a spreadsheet application.

Limits are 20 MB per input/output, 20,000 data rows, 128 columns, 40,000 characters
per cell, and 80,000 characters of document text. Result previews contain at most
30 rows; the inline dock table displays up to 15. Preview cells and profile details
are also limited by display size, with truncation disclosed. These display limits
do not shorten exported data. PDF extraction reads at most 100 pages and reports
truncation. Scanned PDFs without
selectable text need page OCR, which is not integrated yet. DOCX reading covers
body paragraphs and table text; headers, tracked changes and layout are not a
faithful document rendering.

Local document questions use the first 5,000 extracted characters, with that
scope shown in the result. They do not perform full-document retrieval. If local
generation is unavailable, the extracted text remains usable and Sweep reports
the missing answer instead of discarding the document result.

SQLite reads an in-memory copy, never arbitrary SQL, views, virtual tables or
generated columns. Live WAL databases require an exported standalone backup.
Binary cells can be previewed but cannot be silently converted to placeholder
text. UTF-8 and BOM-marked UTF-16 text are supported. Unsupported requests such
as joins, sorting, imputation, Excel/Parquet conversion or compound filters return
an explanation instead of claiming a partial transformation completed everything.

## Recordings and video in the same chat

Use **Settings → Set up speech** before the first transcription. Sweep asks before
downloading its fixed English speech model, about 40 MB, from `alphacephei.com`.
The setup task shows progress and checks a pinned archive checksum. The model is
stored separately under `%LOCALAPPDATA%/Sweep/models/speech`; recordings are never
uploaded. Transcription and readiness checks do not download models. If setup is
missing, the result offers **Set up speech**; after setup, use **Try again** on the
original recording task. This build has one English model, not a language or model
picker.

Attach a local audio or video file, then send a request in the usual chat:

| Attached file | Request | Result |
|---|---|---|
| Video | `Inspect this video` or `Show keyframes` | Media details, up to eight sampled frames, a contact sheet and a JSON frame manifest |
| Video | `Describe this video` | The same samples plus a local visual description of their visible content |
| Audio or video | `Transcribe this recording` or `Create subtitles` | English transcript, timestamped segments and TXT, SRT and JSON artifacts |
| Audio recording | `Summarize this recording` or `What was said about the deadline?` | Local transcription followed by a local answer using the first 5,000 recognized characters |
| Audio | `Inspect metadata` | Audio format and stream details without transcription |

The dock shows the video contact sheet or the first 15 transcript segments. Open
artifact cards to preview the saved frames, manifest, transcript or subtitles and
**Save a copy**. Full recognized text and word timings are retained in the JSON
artifact within the transcription limits; the chat may show only an excerpt.
Input files remain unchanged and source/output hashes are recorded.

Supported audio extensions are WAV, MP3, FLAC, OGG/OGA, M4A, AAC, Opus and WMA;
video extensions are MP4/M4V, MOV, MKV, WebM, AVI, WMV and MPEG/MPG. Actual decoding
depends on the file's codec and validity. Inputs are limited to 200 MB. Media
inspection accepts a reported duration up to 30 minutes, bounds frame decoding
and samples at most eight frames within a 45-second decoding budget. Decoded
frames must not exceed 8,294,400 pixels (the area of a 3840 × 2160 frame). Frames use their actual decoded presentation
timestamps; missing times remain unknown. Video descriptions analyze those
sampled stills only. They do not inspect continuous motion, track objects or hear
the audio. Request transcription separately for spoken content in a video.

Transcription processes at most the first five minutes of the first audio track,
with a four-minute processing budget, further output limits and explicit partial-result labels. Recognition
can mishear words, names and numbers; review it against the recording. Speakers
are not identified or separated. If source timing is missing or discontinuous,
timestamps are relative to concatenated decoded audio and may not align with
video playback; the result and JSON record that limitation. SRT includes only
segments with valid word timings. If a subsequent local answer fails, the
transcript remains available.

The installer bundles the Python media decoder and speech runtime. No system
Python or separate FFmpeg executable is required. The speech model is a separate,
explicit setup download; local chat and image models retain their own setup above.
Microphone capture, live voice conversation, spoken replies, in-dock media
playback, object tracking and cross-recording comparison are not implemented.

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

The media build uses PyAV 19.0.1 and Vosk 0.3.45. PyAV's source license is BSD-3-Clause;
Vosk and the fixed small English model declare Apache-2.0. The Windows media wheel
also bundles native codec libraries, including x264 and x265. Its FFmpeg build has
modified license classification, so its reported LGPL string is not a reliable
license summary for the bundled libraries. Curated notices and source references
are included under `licenses/media`. This private local build does not establish
public redistribution compliance; corresponding-source and linked-library
obligations require review before distributing it publicly.

## Scope

The native dock connects desktop skills, public-web search, scraping, bounded
research, selected-file inspection, document extraction, bounded data transformations,
local OCR, conversation, approved local image analysis, sampled-video inspection
and English recording transcription. It does not claim completed video tracking,
maps/satellite, biometric identification, live voice control, or a general autonomous computer agent. Those
research modules remain separate until integrated and validated through the same
capability/task/permission/event contracts. macOS/Linux can run the Qt source
shell; installers, global shortcuts and startup integration are Windows-only in
this build.

The [full product requirements audit](PRODUCT_REQUIREMENTS.md) maps the original
specification to implemented, partial and missing workflows and records the next
integration steps.
