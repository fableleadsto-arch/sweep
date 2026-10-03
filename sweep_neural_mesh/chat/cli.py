"""Interactive CLI: chat with Sweep in English from the terminal.

Usage:
    python -m sweep_neural_mesh.chat
    python -m sweep_neural_mesh.chat --no-shell   (chat + mesh only)

Commands:
    /status    engine status
    /why       show the detailed thought process behind the last mesh answer
    /reset     clear conversation history
    /raw       toggle raw command-output display
    /voice     one voice turn: speak, then Sweep answers by voice
    /voice-auto  continuous voice conversation (Ctrl+C to stop)
    /vision [seconds]  live object tracking from camera (Ctrl+C or timeout)
    /face-search <image> [name]  find a person's profiles from a photo
                                 (or any reverse-image search with --mode image)
    /mics      list microphone devices
    /quit      exit
"""
from __future__ import annotations

import argparse
import sys
import time

BANNER = r"""
   _____ _____                _      _         _
  / ____/ ____|              | |    (_)       | |
 | (___| (___   __ _ _ __ ___| |     _ _ __ __| |
  \___ \\___ \ / _` | '__/ _ \ |    | | '__/ _` |
  ____) |___) | (_| | | |  __/ |____| | | | (_| |
 |_____/|____/ \__,_|_|  \___|______|_|_|  \__,_|

 Neural-mesh assistant — chat, reason, run terminal tasks.
 Type /quit to exit, /status for engine state.
"""


def main() -> int:
    ap = argparse.ArgumentParser(prog="sweep-chat")
    ap.add_argument("--no-shell", action="store_true", help="disable terminal execution")
    ap.add_argument("--shell", action="store_true", help="enable terminal commands with approval for each command")
    ap.add_argument("--no-mesh", action="store_true", help="disable neural mesh routing")
    args = ap.parse_args()

    from .engine import SweepChat
    from .commands import CommandResult

    def confirm_command(command: str) -> bool:
        try:
            return input(f"Run {command!r}? [y/N] ").strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            return False

    chat = SweepChat(enable_shell=args.shell and not args.no_shell,
                     enable_mesh=not args.no_mesh, confirm_command=confirm_command)
    print(BANNER)
    print("Loading chat base in the background… (first answer may take a minute)\n")

    history: list[dict] = []
    show_raw = False

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit", "exit", "quit"):
            print("bye!")
            return 0
        if line == "/status":
            st = chat.status()
            print(f"  chat base: loaded={st['chat_base']['loaded']} "
                  f"loading={st['chat_base']['loading']} "
                  f"lora={st['chat_base']['lora_adapter']}")
            print(f"  neural mesh: {st['mesh']} | shell: {st['shell']}")
            continue
        if line == "/reset":
            history.clear()
            print("  (conversation cleared)")
            continue
        if line == "/raw":
            show_raw = not show_raw
            print(f"  raw output display: {show_raw}")
            continue
        if line in ("/why", "/thoughts", "/chain"):
            res = chat.last_reasoning_result
            tc = getattr(res.trace, "thought_chain", None) if res is not None else None
            if tc is None:
                print("  no mesh reasoning recorded yet — ask a claim/logic/math "
                      "question first, then run /why")
            else:
                print(tc.to_text())
            continue
        if line == "/mics":
            try:
                from .voice import list_microphones
                for m in list_microphones():
                    print(" ", m)
            except Exception as e:
                print(f"  mic error: {e}")
            continue
        if line == "/voice":
            _voice_turn(chat, history)
            continue
        if line == "/voice-auto":
            _voice_loop(chat, history)
            continue
        if line == "/face-search":
            _face_search_help()
            continue
        if line.startswith("/face-search "):
            _face_search_session(line[len("/face-search "):].strip())
            continue
        if line.startswith("/vision"):
            arg = line.split()[1] if len(line.split()) > 1 else ""
            _vision_session(chat, history,
                            max_seconds=float(arg) if arg.isdigit() else None)
            continue

        t0 = time.perf_counter()
        resp = chat.chat(line, history)
        # maintain conversation history for multi-turn coherence
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": resp.text[:1000]})

        tag = {"shell": "🛠️ ", "mesh": "🧠 ", "chat": "💬 "}.get(resp.engine, "")
        if show_raw and resp.raw_command_result is not None:
            r: CommandResult = resp.raw_command_result
            print(f"  [cmd] {r.command}")
            if r.output:
                print(r.output[:2000])
        print(f"\n{tag}sweep> {resp.text}\n")
        if resp.engine == "shell" and resp.command:
            print(f"  (ran: {resp.command} — {resp.latency_ms:.0f} ms)")
        elif resp.latency_ms:
            print(f"  ({resp.engine}, {resp.latency_ms:.0f} ms)")
        print()


# ── face search ──────────────────────────────────────────────────────

def _face_search_help() -> None:
    print(
        "  usage: /face-search <image-path-or-url> [\"name hint\"] [--mode image] [--no-verify]"
    )
    print("  example: /face-search photo.jpg \"jane dough\"")
    print("  example: /face-search https://example.com/pic.jpg --mode image")
    print("  API providers (SERPAPI_KEY / TINEYE keys) light up automatically;")
    print("  keyless name search runs when a name hint is given.")


def _face_search_session(argline: str) -> None:
    try:
        from .face_search_adapter import parse_face_search_args, run_face_search_text
    except Exception as e:
        print(f"  face search unavailable: {e}")
        return
    parsed = parse_face_search_args(argline)
    if parsed is None:
        _face_search_help()
        return
    image, opts = parsed
    print(f"  scanning… (mode={opts['mode']}, verify={not opts['no_verify']})")
    try:
        print(run_face_search_text(image, **opts))
    except Exception as e:
        print(f"  face search failed: {e.__class__.__name__}: {e}")


# ── voice/vision helpers ─────────────────────────────────────────────

def _voice_turn(chat, history) -> None:
    """One push-to-talk voice turn."""
    try:
        from .voice import record_utterance, transcribe, Speaker
    except Exception as e:
        print(f"  voice unavailable: {e}")
        return
    speaker = Speaker()
    try:
        print("  🎙️  listening… (speak, then pause)")
        pcm = record_utterance()
        if not pcm:
            print("  (nothing captured)")
            return
        result = transcribe(pcm)
        text = result.text.strip()
        print(f"  🗣️  you> {text}")
        if text:
            resp = chat.chat(text, history)
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": resp.text[:1000]})
            print(f"\nsweep> {resp.text}\n")
            speaker.speak(resp.text)
    except RuntimeError as e:
        print(f"  {e}")
    except Exception as e:
        print(f"  voice error: {e.__class__.__name__}: {e}")


def _voice_loop(chat, history) -> None:
    """Continuous voice conversation until Ctrl+C."""
    print("  🎙️  continuous voice mode — Ctrl+C to stop")
    while True:
        try:
            _voice_turn(chat, history)
        except KeyboardInterrupt:
            print("\n  (voice mode ended)")
            return


def _vision_session(chat, history, max_seconds: float | None) -> None:
    """Live object tracking; notable events are described and routed."""
    try:
        from .vision import CameraSession, describe_scene
    except Exception as e:
        print(f"  vision unavailable: {e}")
        return
    print("  👁️  camera session — press q in the preview window to stop")
    try:
        with CameraSession() as cam:
            t0 = time.time()
            for events, snapshot, _frame in cam.stream(max_seconds=max_seconds):
                notable = [e for e in events.values()
                           if e["event"] in ("entered", "left")]
                if notable:
                    desc = describe_scene(snapshot)
                    names = ", ".join(f"{e['event']}: {e['label']}" for e in notable)
                    print(f"  [{time.time()-t0:6.1f}s] {names} | scene: {desc}")
                if max_seconds and time.time() - t0 > max_seconds:
                    break
            print(f"  final scene: {describe_scene(cam.tracker.snapshot())}")
    except RuntimeError as e:
        print(f"  {e}")
    except KeyboardInterrupt:
        print("\n  (vision session ended)")


if __name__ == "__main__":
    sys.exit(main())
