"""SweepChat — the conversational engine tying together:

  1. Shell-command routing (RUN: lines from the LoRA'd chat base)
  2. The neural mesh (ReasoningCortex: evidence/logic/knowledge tasks)
  3. The LoRA'd Qwen2-0.5B chat base (general conversation & generation)

Routing order (fast -> slow):
  - explicit terminal intent  -> shell execute
  - claim/evidence/verdict questions, pure logic/math -> neural mesh
  - everything else (chat, explanations, open-ended)  -> Qwen chat base

The chat base is lazily loaded from the LoRA adapter if present, else the
stock Qwen2-0.5B-Instruct; in both cases the Qwen chat template is used.
"""
from __future__ import annotations

from services.model_loading import serialized_model_load

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .commands import extract_run_line, run_command, CommandResult

logger = logging.getLogger(__name__)

LORA_DIR = Path(__file__).parent.parent / "training" / "neural_models" / "chat_lora"
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are Sweep, a helpful AI assistant with a neural reasoning mesh. "
    "You answer questions, reason step by step, and can perform terminal "
    "tasks when asked. Be concise and accurate."
)

# Patterns that indicate the neural mesh is the right engine
_RE_CLAIM = re.compile(
    r"\b(claim|verdict|evidence|contradict\w*|supports?|refutes?)\b", re.I
)
_RE_LOGIC = re.compile(
    r"(what comes next|next number|is \d+ prime|simplify|evaluate:|"
    r"\btrue\s+(and|or|nand|xor)\s+(true|false)\b|not\s*\(|syllogism|"
    r"all .+ are .+\. .+ is a .+)", re.I
)
_RE_MATH = re.compile(
    r"\d+\s*[\+\-\*/x×÷]\s*\d+|\bcalculate\b|\bwhat is \d+|\bsolve\b", re.I
)
_RE_SHELL = re.compile(
    r"\b(list|show|run|execute|open|create|make|delete|copy|move|check)\b.*"
    r"\b(files?|folder|directory|tests?|git|terminal|command|process(es)?|"
    r"memory|disk|version)\b", re.I
)


@dataclass
class ChatResponse:
    text: str
    engine: str          # "shell" | "mesh" | "chat"
    confidence: float = 0.0
    latency_ms: float = 0.0
    command: str = ""    # shell path only
    raw_command_result: CommandResult | None = None
    meta: dict = field(default_factory=dict)


class _ChatBase:
    """Lazy-loading wrapper around the LoRA'd (or stock) Qwen chat model."""

    def __init__(self):
        self._model = None
        self._tok = None
        self._loaded = False
        self._loading = False
        self._lock = threading.Lock()
        self._adapter = None  # None -> try LoRA dir, else stock

    def _start_load(self) -> None:
        if self._loaded or self._loading:
            return
        self._loading = True
        threading.Thread(target=self._load_worker, daemon=True).start()

    @serialized_model_load
    def _load_worker(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            src = str(LORA_DIR) if (LORA_DIR / "adapter_config.json").exists() else BASE_MODEL
            self._adapter = src != BASE_MODEL
            tok = AutoTokenizer.from_pretrained(src)
            model = AutoModelForCausalLM.from_pretrained(
                src, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            model.eval()
            with self._lock:
                self._tok, self._model, self._loaded = tok, model, True
            logger.info("Chat base loaded from %s", src)
        except Exception as e:
            logger.warning("Chat base load failed: %s", e)
        finally:
            self._loading = False

    def ready(self, timeout: float = 120.0) -> bool:
        self._start_load()
        t0 = time.time()
        while not self._loaded and time.time() - t0 < timeout and self._loading:
            time.sleep(0.2)
        return self._loaded

    def generate(self, history: list[dict], max_new_tokens: int = 220) -> str:
        if not self.ready():
            return ""
        import torch
        tok, model = self._tok, self._model
        prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                repetition_penalty=1.1,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


class SweepChat:
    """Main entry: chat(user_text, history) -> ChatResponse."""

    def __init__(self, enable_shell: bool = False, enable_mesh: bool = True, confirm_command=None):
        self.enable_shell = enable_shell
        self.confirm_command = confirm_command
        self.enable_mesh = enable_mesh
        self._base = _ChatBase()
        self._mesh = None
        self._mesh_checked = False
        self.last_reasoning_result = None  # full ReasoningResult from last mesh pass

    # ── neural mesh (lazy) ───────────────────────────────────────────
    def _get_mesh(self):
        if self._mesh_checked:
            return self._mesh
        self._mesh_checked = True
        if not self.enable_mesh:
            return None
        try:
            from sweep_neural_mesh.neurons.cortex import ReasoningCortex
            self._mesh = ReasoningCortex()
        except Exception as e:
            logger.warning("Neural mesh unavailable: %s", e)
            self._mesh = None
        return self._mesh

    def _mesh_answer(self, text: str) -> str | None:
        mesh = self._get_mesh()
        if mesh is None:
            return None
        try:
            res = mesh.reason(text, evidence=[])
            self.last_reasoning_result = res  # kept for /why (thought chain)
            dec = (res.decision or "").strip().lower()
            if dec in ("unknown", "insufficient", ""):
                return None
            conf = float(getattr(res, "confidence", 0.0) or 0.0)
            if conf < 0.55:
                return None
            answer = getattr(res, "reasoning", "") or dec
            # keep it short; the full reasoning chain stays in the trace
            if len(answer) > 600:
                answer = answer[:600].rsplit(".", 1)[0] + "."
            return f"[mesh:{dec}@{conf:.2f}] {answer}" if answer else None
        except Exception as e:
            logger.debug("mesh path failed: %s", e)
            return None

    # ── routing ──────────────────────────────────────────────────────
    def _looks_like_shell(self, text: str) -> bool:
        if not self.enable_shell:
            return False
        return bool(_RE_SHELL.search(text)) and len(text) < 200

    def _looks_like_mesh(self, text: str) -> bool:
        if not self.enable_mesh:
            return False
        if _RE_CLAIM.search(text) or _RE_LOGIC.search(text) or _RE_MATH.search(text):
            return True
        # short factual questions also go through the mesh first — but plain
        # greetings/chat (“hello”, “how are you”) must go to the chat base,
        # not the mesh, otherwise every “hi” pays a 2.5-minute retrieval cost.
        if re.match(r"^(hello|hi|hey|thanks|thank you|ok|yo|good\s(morning|evening|night)|how are you|what'?s up)\b", text.strip(), re.I):
            return False
        return bool(re.match(r"^(what|who|when|where|how many|is|are|does|did)\b", text.strip(), re.I)) and len(text) < 120

    def _shell_flow(self, text: str) -> ChatResponse | None:
        """Ask the chat base for a RUN: line, then execute + summarize."""
        if not self._base.ready(timeout=90):
            return None
        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"{text}\n\nIf this requires running a terminal command, "
                f"reply with a single line 'RUN: <command>'. Otherwise answer normally."},
        ]
        raw = self._base.generate(history, max_new_tokens=60)
        cmd = extract_run_line(raw or "")
        if not cmd:
            return None
        if self.confirm_command is None or not self.confirm_command(cmd):
            return ChatResponse(text=f"Command requires approval: {cmd}", engine="shell", command=cmd)
        result = run_command(cmd)
        if not result.ok and result.refused_reason:
            return ChatResponse(
                text=f"I refused to run that command: {result.refused_reason}",
                engine="shell", command=cmd, raw_command_result=result,
            )
        # summarize output with the chat base
        summary = raw  # default: the model's own text
        if result.ok:
            follow = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"RUN: {cmd}"},
                {"role": "user", "content": f"[command output]\n{result.output or '(no output)'}\n\nSummarize this for the user in one or two sentences."},
            ]
            summary = self._base.generate(follow, max_new_tokens=120) or f"Ran `{cmd}`.\n```\n{result.output[:500]}\n```"
        return ChatResponse(
            text=f"```bash\n$ {cmd}\n```\n{summary}" if result.ok else f"Command failed: {result.error}",
            engine="shell", command=cmd, raw_command_result=result,
        )

    def chat(self, text: str, history: list[dict] | None = None) -> ChatResponse:
        t0 = time.perf_counter()
        text = (text or "").strip()
        if not text:
            return ChatResponse(text="Say something and I'll help.", engine="chat")

        history = history or []

        # 1. shell intent
        if self._looks_like_shell(text):
            resp = self._shell_flow(text)
            if resp is not None:
                resp.latency_ms = (time.perf_counter() - t0) * 1000
                return resp
            # fall through to mesh/chat if the base isn't ready

        # 2. neural mesh (fast, deterministic, trained heads)
        if self._looks_like_mesh(text):
            mesh_ans = self._mesh_answer(text)
            if mesh_ans:
                return ChatResponse(
                    text=mesh_ans, engine="mesh",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # 3. chat base (general conversation, generation, everything else)
        chat_history = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:]
        chat_history.append({"role": "user", "content": text})
        reply = self._base.generate(chat_history)
        if not reply:
            # last resort: mesh without threshold, else apology
            fallback = self._mesh_answer(text)
            reply = fallback or "I couldn't load my language model — try again in a moment."
            engine = "mesh" if fallback else "chat"
            return ChatResponse(text=reply, engine=engine,
                                latency_ms=(time.perf_counter() - t0) * 1000)
        return ChatResponse(text=reply, engine="chat",
                            latency_ms=(time.perf_counter() - t0) * 1000)

    def status(self) -> dict:
        return {
            "chat_base": {
                "loaded": self._base._loaded,
                "loading": self._base._loading,
                "lora_adapter": (LORA_DIR / "adapter_config.json").exists(),
            },
            "mesh": self._mesh is not None,
            "shell": self.enable_shell,
        }
