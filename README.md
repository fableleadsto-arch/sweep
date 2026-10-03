# Sweep

Native desktop companion built on Python for computer tasks, web research,
local files, conversation and reusable tool integrations.

## Set up and run

On Windows, run **[Sweep-Setup.exe](dist/Sweep-Setup.exe)** to install the desktop
application and its included Python runtime. Open Sweep from the Start menu.
Its own native window hosts the experience; no browser or localhost page is needed.
See [desktop installation and controls](docs/DESKTOP.md).

For development, Python 3.12 or newer is required. Double-click **setup_sweep.cmd**, or run:

```powershell
python setup_sweep.py
```

Source setup creates `.venv`, installs the Python/Qt package and opens the desktop window. It does
not change your global Python installation, overwrite `.env`, compile C++, or
start training/download model weights. Installation requires internet access.

After setup, double-click **launch_sweep.cmd**. From a terminal:

```powershell
.\.venv\Scripts\python.exe -m sweep.launcher
.\.venv\Scripts\python.exe -m sweep.desktop
.\.venv\Scripts\python.exe -m sweep.launcher doctor
.\.venv\Scripts\python.exe -m sweep --no-llm "what is 15% of 200"
.\.venv\Scripts\python.exe -m sweep --no-llm "open calculator"
.\.venv\Scripts\python.exe -m sweep.launcher search "Python documentation"
.\.venv\Scripts\python.exe -m sweep.launcher scrape "https://example.com" --output page.json
.\.venv\Scripts\python.exe -m sweep.launcher research "Python async HTTP clients" --output research.json
```

On Linux/macOS, use `.venv/bin/python` or `.venv/bin/sweep-launcher`. Desktop app
names are primarily Windows-oriented. JSON exports refuse to overwrite existing files.

`python setup_sweep.py --check` checks the current environment without installing.
Use `--install-only` to skip opening the window, or `--headless` for a CLI-only install.
Optional extras can be installed with
`--extras science`, `--extras ai`, `--extras browser`, `--extras datasets`,
`--extras memory` (Qdrant), or `--extras dev` (repeat the option to combine them).

## Available tasks

| Task | Entry point | Requirements |
| --- | --- | --- |
| Native desktop workspace, tray, shortcuts, file previews and task history | `python -m sweep.desktop` | Desktop setup (default); Windows installer available |
| Open sites/apps/folders; list/create/find files; notes, aliases, math, system information | `python -m sweep` | Controller; app availability depends on the OS |
| Web search | `python -m sweep.launcher search "query"` | Internet; optional provider keys |
| Extract public-page text, metadata and links to JSON | `python -m sweep.launcher scrape URL` | Base dependencies and internet |
| Bounded research with source/evidence collection | `python -m sweep.launcher research "topic"` | Search access; heuristic evidence collection |
| Web API for your tools | `python -m sweep.launcher serve web` | Base dependencies |
| Companion planning, memory, provider/tool APIs | `python -m sweep.launcher serve companion` | Some endpoints need configured providers or optional libraries |
| Model-backed chat | `python -m sweep_neural_mesh.chat --no-shell` | Optional AI dependencies and compatible model assets; may download a model |
| Cognitive/logic/evidence components | Python libraries under `cognition/` and `sweep_cognitive/` | Component-specific inputs; experimental orchestration |

Sweep does not yet perform every computer task or provide general intelligence.
Voice, vision, document models, reverse-image/face search and training remain
optional components, not promises made by the base installer. The face-search
implementation and provider requirements are documented in
[sweep_neural_mesh/face_search/README.md](sweep_neural_mesh/face_search/README.md).

## Local APIs and configuration

The optional API launcher binds to `127.0.0.1` (web port 8787, companion port 8088).
These developer APIs are separate from the desktop application. Open `/docs`
for the schema. API calls require `Authorization: Bearer <token>`. If no token is
configured, the launcher prints the path of a generated token file under
`.sweep-runtime/`; that file is removed when the server stops normally. On Windows
it inherits your directory's access permissions. Keep that directory private.

For persistent tokens, configure `SWEEP_API_TOKEN` and `BRAIN_SERVICE_TOKEN` in the
environment or `.env`. Use your own randomly generated values, not example tokens.
Search/provider options are listed in `.env.example`; existing configuration is
never overwritten by setup. Use `SWEEP_DEBUG`, not the generic `DEBUG` variable.

`CORS_ORIGINS` accepts explicit comma-separated origins. Generated Python
execution, background ingestion and remote browsers default off. See
[architecture and security boundaries](docs/ARCHITECTURE.md) before enabling them.
The built-in notes/history stores are plaintext; do not use them for passwords.

## Use from Python

```python
import asyncio
from sweep import Controller

async def main():
    result = await Controller().execute("calculate 2 + 2", allow_llm=False)
    print(result.message)

asyncio.run(main())
```

For your own tools, `Controller.execute()` returns a structured `ActionResult`.
Web endpoints expose search, extraction, browse and research; companion endpoints
expose the provider/tool system. These remain separate systems today.

## Development

```powershell
python setup_sweep.py --install-only --extras dev
.\.venv\Scripts\python.exe -m pytest tests sweep/tests sweep_core/tests cognition/tests
python scripts/audit_codebase.py
```

The root pytest configuration also discovers the broader companion, cognitive,
neural-mesh and intelligence suites. Some require optional frameworks/model
assets and can be slow. Use an isolated pytest `--basetemp` when necessary.

`pyproject.toml` is the authoritative package configuration. `pip install -e .`
installs the base Python runtime. The legacy `requirements.txt` is an optional
large research stack, not the recommended base setup. C++ acceleration is opt-in:
install pybind11 and a C++20 compiler, then `python setup.py build_ext --inplace`.

See [the codebase map](docs/ARCHITECTURE.md) for subsystem roles, current boundaries,
remaining engineering work and cleanup decisions.

Private - all rights reserved.
