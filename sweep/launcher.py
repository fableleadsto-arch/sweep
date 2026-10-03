"""Task launcher for existing Sweep capabilities; imports optional stacks lazily."""
from __future__ import annotations

import argparse
import asyncio
from importlib import metadata
import json
import os
from pathlib import Path
import secrets
import sys


def doctor() -> int:
    """Report installed distributions without loading models or making requests."""
    required = ("fastapi", "uvicorn", "pydantic", "pydantic-settings", "httpx",
                "beautifulsoup4", "trafilatura", "lxml", "numpy", "pandas", "scipy", "scikit-learn",
                "python-dotenv")
    missing = []
    print(f"Python {sys.version.split()[0]} | {sys.executable}")
    for name in required:
        try:
            print(f"  {name}: {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
            print(f"  {name}: MISSING")
    for name in ("torch", "transformers", "playwright", "datasets"):
        try:
            print(f"  optional {name}: {metadata.version(name)} (model readiness not checked)")
        except metadata.PackageNotFoundError:
            print(f"  optional {name}: not installed")
    print("Desktop controller: available. Web tasks need the base dependencies and internet.")
    print("Model chat, vision, audio and training need additional dependencies/assets.")
    return 1 if missing or sys.version_info < (3, 12) else 0


def _emit(value, output: str | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        # Exclusive creation prevents an export from silently replacing a user's file.
        with Path(output).open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"Saved {output}")
    else:
        print(text)


async def _web_task(args) -> int:
    if args.task == "search":
        from app.search.engine import relai_search
        value = await relai_search(args.query, limit=args.limit)
        _emit(value, args.output)
        return 0 if value.get("hits") else 1
    if args.task == "scrape":
        from app.core.http import relai_fetch
        from app.extraction.page_data import extract_page_data
        fetched = await relai_fetch(args.url)
        if not fetched.ok:
            print(f"Fetch failed: {fetched.error or fetched.status}", file=sys.stderr)
            return 1
        page = extract_page_data(html=fetched.text, url=fetched.url,
                                 status=fetched.status, content_type=fetched.content_type)
        _emit(page.model_dump(mode="json"), args.output)
        return 0
    from app.research.engine import start_research, get_research
    session = await start_research(args.objective, max_searches=5, max_pages=10,
                                   max_runtime_ms=60000)
    async with asyncio.timeout(75):
        while session.status.value == "running":
            await asyncio.sleep(0.2)
            session = get_research(session.id)
    _emit(session.model_dump(mode="json"), args.output)
    return 0 if session.status.value == "complete" and session.evidence else 1


def _serve(service: str, port: int | None) -> int:
    import uvicorn
    # Token is ephemeral unless configured; never overwrite .env or print secrets.
    variable = "SWEEP_API_TOKEN" if service == "web" else "BRAIN_SERVICE_TOKEN"
    if service == "web":
        from app.config import get_settings
        configured = get_settings().sweep_api_token
    else:
        from companion.config import get_settings
        configured = get_settings().brain_service_token
    if not configured:
        token = secrets.token_urlsafe(32)
        os.environ[variable] = token
        get_settings.cache_clear()
        directory = Path(os.environ.get("SWEEP_RUNTIME_DIR", ".sweep-runtime"))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{service}-{secrets.token_hex(4)}.token"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
        print(f"API token file: {path.resolve()} (send Authorization: Bearer <token>)")
    else:
        path = None
        print(f"Using configured {variable}.")
    module = "app.main:app" if service == "web" else "companion.main:app"
    try:
        uvicorn.run(module, host="127.0.0.1", port=port or (8787 if service == "web" else 8088))
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
            os.environ.pop(variable, None)
            get_settings.cache_clear()
    return 0


def _menu() -> int:
    while True:
        print("\nSweep\n  1 Desktop commands\n  2 Search web\n  3 Scrape a public page"
              "\n  4 Research a topic\n  5 Web API\n  6 Companion API\n  7 Check installation\n  0 Exit")
        try:
            choice = input("Select: ").strip()
            if choice == "0":
                return 0
            if choice == "1":
                main(["controller"])
            elif choice in {"2", "3", "4"}:
                task = {"2": "search", "3": "scrape", "4": "research"}[choice]
                value = input("Public URL: " if task == "scrape" else "Topic: ").strip()
                if value:
                    main([task, value])
            elif choice in {"5", "6"}:
                main(["serve", "web" if choice == "5" else "companion"])
            elif choice == "7":
                doctor()
        except (EOFError, KeyboardInterrupt):
            return 0


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task")
    commands.add_parser("doctor", help="Check installation without network access")
    controller = commands.add_parser("controller", help="Desktop controller (LLM routing disabled)")
    controller.add_argument("command", nargs="*")
    for name, argument in (("search", "query"), ("scrape", "url"), ("research", "objective")):
        task = commands.add_parser(name)
        task.add_argument(argument)
        task.add_argument("--output", help="Write JSON to a new file")
        if name == "search":
            task.add_argument("--limit", type=int, choices=range(1, 51), default=5)
    server = commands.add_parser("serve")
    server.add_argument("service", choices=["web", "companion"], default="web", nargs="?")
    server.add_argument("--port", type=_port)
    args = parser.parse_args(argv)
    try:
        if args.task is None:
            return _menu()
        if args.task == "doctor":
            return doctor()
        if args.task == "controller":
            from .__main__ import main as controller_main
            return controller_main(["--no-llm", *args.command])
        if args.task == "serve":
            return _serve(args.service, args.port)
        return asyncio.run(_web_task(args))
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run python setup_sweep.py.", file=sys.stderr)
        return 1
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"Task failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
