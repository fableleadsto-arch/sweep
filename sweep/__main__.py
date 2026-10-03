"""Terminal entrypoint for the Sweep controller.

Usage:
    python -m sweep "open youtube"      # run one command
    python -m sweep                     # interactive REPL
    python -m sweep --yes "run pip list"  # skip confirmations

Run interactively, then just type plain English: "open youtube",
"please open gmail", "search for python docs", "what time is it", "exit".
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from . import Controller
from .store import ControllerStore

BANNER = """
Sweep Controller — I turn plain English into day-to-day actions.
  • "open youtube"        • "search for python docs"
  • "what is 15% of 200"  • "remember that my wifi password is X"
  • "list files"          • "run pip version" (asks first)
  • "help" shows everything · "exit" to quit


""".strip()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sweep", description="Multipurpose terminal controller.")
    parser.add_argument("command", nargs="*", help="One-shot English command. Omit for the REPL.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts.")
    parser.add_argument("--dir", help="Override the state directory (SWEEP_CONTROLLER_DIR).")
    parser.add_argument("--no-llm", action="store_true", help="Disable the LLM intent fallback.")
    return parser.parse_args(argv)


def _build_controller(args: argparse.Namespace) -> Controller:
    import os

    if args.dir:
        os.environ["SWEEP_CONTROLLER_DIR"] = args.dir
    return Controller()


async def _run_once(controller: Controller, text: str, yes: bool, allow_llm: bool) -> str:
    from .skills import SkillContext

    result = await controller.execute(
        text,
        ctx=SkillContext(store=controller.store, yes=yes),
        allow_llm=allow_llm,
    )
    if result.status == "exit":
        return result.message
    return result.message


def _main_one_shot(controller: Controller, text: str, yes: bool, allow_llm: bool) -> int:
    from .skills import SkillContext

    result = asyncio.run(controller.execute(
        text, ctx=SkillContext(store=controller.store, yes=yes), allow_llm=allow_llm,
    ))
    print(result.message)
    return 1 if result.status == "error" else 0


def _main_repl(controller: Controller, yes: bool, allow_llm: bool) -> int:
    print(BANNER)
    from .skills import SkillContext

    while True:
        try:
            line = input("sweep> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not line:
            continue
        controller.store.append_history(line)
        lower = line.lower()
        if lower in {"exit", "quit", "bye"}:
            print("Goodbye.")
            return 0
        try:
            result = asyncio.run(
                controller.execute(
                    line,
                    ctx=SkillContext(store=controller.store, yes=yes),
                    allow_llm=allow_llm,
                )
            )
        except KeyboardInterrupt:
            print()
            continue
        print(result.message)
        if result.status == "exit":
            return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    controller = _build_controller(args)
    command = " ".join(args.command).strip()
    if command:
        return _main_one_shot(controller, command, args.yes, not args.no_llm)
    return _main_repl(controller, args.yes, not args.no_llm)


if __name__ == "__main__":
    sys.exit(main())
