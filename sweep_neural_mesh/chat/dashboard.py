"""Live training dashboard — real-time visualization of the chat-base LoRA run.

Serves a single-page dashboard on http://localhost:8765 showing:
  - animated neural-mesh diagram (routing cortex -> task heads -> LLM base)
  - live loss curve (parsed from the training log)
  - progress bar, ETA, current epoch/step
  - training-throughput stats

The mesh view shows the Sweep architecture (not Qwen's internals): incoming
signals flow through the router into the three engines (shell / mesh / chat)
with animated pulses whose speed tracks the actual step rate.

Usage:
    python -m sweep_neural_mesh.chat.dashboard [--log benchmark/results/chat_lora.log] [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG_DEFAULT = Path(__file__).parent.parent.parent / "benchmark" / "results" / "chat_lora.log"

_STEP_RE = re.compile(
    r"ep(?P<epoch>\d+) step (?P<step>\d+)/(?P<total>\d+) "
    r"loss=(?P<loss>[\d.]+) elapsed=(?P<elapsed>\d+)s eta=(?P<eta>\d+)m"
)
_EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+): train_loss=(?P<tl>[\d.]+) val_loss=(?P<vl>[\d.]+)"
)
_EX_RE = re.compile(r"Dataset: (?P<n>\d+) usable examples")

_state_lock = threading.Lock()
_state = {
    "steps": [],          # [(step_index_global, loss)]
    "epochs": [],         # [{"epoch":1,"train_loss":..,"val_loss":..}]
    "current": {"epoch": 1, "step": 0, "total": 0, "loss": None, "eta_min": None, "elapsed_s": 0},
    "usable_examples": None,
    "started": time.time(),
    "alive": False,
}


def parse_log(path: Path) -> None:
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    steps = []
    cur = dict(_state["current"])
    epochs = []
    usable = None
    for line in text.splitlines():
        m = _STEP_RE.search(line)
        if m:
            d = m.groupdict()
            cur = {
                "epoch": int(d["epoch"]), "step": int(d["step"]),
                "total": int(d["total"]), "loss": float(d["loss"]),
                "eta_min": int(d["eta"]), "elapsed_s": int(d["elapsed"]),
            }
            steps.append(float(d["loss"]))
            continue
        m = _EPOCH_RE.search(line)
        if m:
            d = m.groupdict()
            epochs.append({"epoch": int(d["epoch"]), "train_loss": float(d["tl"]),
                           "val_loss": float(d["vl"])})
            continue
        m = _EX_RE.search(line)
        if m:
            usable = int(m.group("n"))
    # keep the LARGEST usable count (train split logs before val split)
    with _state_lock:
        if usable is not None and _state["usable_examples"]:
            usable = max(usable, _state["usable_examples"])
        _state["steps"] = steps[-600:]
        _state["epochs"] = epochs
        _state["current"] = cur
        _state["usable_examples"] = usable
        _state["alive"] = _log_is_fresh(path)


def _log_is_fresh(path: Path) -> bool:
    # The trainer writes only every 50 steps (~11 min), so allow a generous
    # window; also treat a running trainer process as alive.
    try:
        fresh = time.time() - path.stat().st_mtime < 1500
        if fresh:
            return True
        return _trainer_process_alive()
    except Exception:
        return False


def _trainer_process_alive() -> bool:
    try:
        import psutil
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if "train_chat_lora" in cmd:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def watch_loop(path: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        parse_log(path)
        stop.wait(2.0)


# ────────────────────────────────────────────────────────────────────
# Dashboard HTML (vanilla JS, no CDN dependency — works offline)
# ────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sweep — Live Training</title>
<style>
  :root { --bg:#0b0e14; --panel:#131826; --edge:#232b41; --ink:#dbe4ff;
          --dim:#7a86a8; --acc:#5eead4; --hot:#f472b6; --warm:#fbbf24; }
  body { background:var(--bg); color:var(--ink); font:14px/1.45 'Segoe UI',sans-serif;
         margin:0; padding:18px; }
  h1 { font-size:18px; margin:0 0 4px; } h1 .live { color:var(--acc); }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .grid { display:grid; grid-template-columns: 460px 1fr; gap:14px; }
  .panel { background:var(--panel); border:1px solid var(--edge); border-radius:12px; padding:14px; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.12em;
              color:var(--dim); margin:0 0 10px; }
  svg { display:block; width:100%; height:auto; }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:12px; }
  .stat { background:var(--bg); border:1px solid var(--edge); border-radius:8px; padding:8px 10px; }
  .stat .v { font-size:20px; font-weight:600; color:var(--acc); }
  .stat .k { font-size:10px; color:var(--dim); text-transform:uppercase; letter-spacing:.1em; }
  .bar { height:14px; background:var(--bg); border:1px solid var(--edge);
         border-radius:7px; overflow:hidden; margin:6px 0 2px; }
  .bar > div { height:100%; width:0%;
    background:linear-gradient(90deg,var(--acc),#38bdf8); transition:width .8s ease; }
  .row { display:flex; justify-content:space-between; font-size:11px; color:var(--dim); }
  .eplog { font-size:11px; color:var(--dim); margin-top:10px; max-height:120px; overflow:auto; }
  .pulse { animation: pp 1.6s ease-in-out infinite; }
  @keyframes pp { 0%,100%{opacity:.25} 50%{opacity:1} }
</style></head><body>
<h1>Sweep <span class="live" id="live">● LIVE</span> — chat-base LoRA training</h1>
<div class="sub">neural-mesh view · updates every 2 s from the live training log</div>
<div class="grid">
  <div class="panel">
    <h2>Neural mesh</h2>
    <svg id="mesh" viewBox="0 0 440 430">
      <defs>
        <radialGradient id="gCore"><stop offset="0%" stop-color="#5eead4"/><stop offset="100%" stop-color="#134e4a"/></radialGradient>
        <radialGradient id="gUser"><stop offset="0%" stop-color="#fbbf24"/><stop offset="100%" stop-color="#78350f"/></radialGradient>
        <radialGradient id="gShell"><stop offset="0%" stop-color="#f472b6"/><stop offset="100%" stop-color="#831843"/></radialGradient>
        <radialGradient id="gMeshN"><stop offset="0%" stop-color="#38bdf8"/><stop offset="100%" stop-color="#0c4a6e"/></radialGradient>
        <radialGradient id="gChat"><stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#4c1d95"/></radialGradient>
      </defs>
      <!-- edges -->
      <g stroke="#232b41" stroke-width="2">
        <line id="e1" x1="220" y1="60"  x2="110" y2="180"/>
        <line id="e2" x1="220" y1="60"  x2="220" y2="180"/>
        <line id="e3" x1="220" y1="60"  x2="330" y2="180"/>
        <line id="e4" x1="110" y1="230" x2="110" y2="320"/>
        <line id="e5" x1="220" y1="230" x2="220" y2="320"/>
        <line id="e6" x1="330" y1="230" x2="330" y2="320"/>
        <line x1="110" y1="360" x2="220" y2="395"/>
        <line x1="220" y1="360" x2="220" y2="395"/>
        <line x1="330" y1="360" x2="220" y2="395"/>
      </g>
      <!-- animated pulses travel along e1..e6 via JS -->
      <circle id="p1" r="5" fill="#fbbf24" opacity=".9"/>
      <circle id="p2" r="5" fill="#fbbf24" opacity=".9"/>
      <circle id="p3" r="5" fill="#fbbf24" opacity=".9"/>
      <!-- nodes -->
      <circle cx="220" cy="60"  r="34" fill="url(#gUser)"/><text x="220" y="64"  text-anchor="middle" font-size="11" fill="#0b0e14" font-weight="700">INPUT</text>
      <circle cx="110" cy="205" r="40" fill="url(#gShell)"/><text x="110" y="202" text-anchor="middle" font-size="10" fill="#0b0e14" font-weight="700">SHELL</text><text x="110" y="215" text-anchor="middle" font-size="9"  fill="#0b0e14">commands</text>
      <circle cx="220" cy="205" r="40" fill="url(#gMeshN)"/><text x="220" y="202" text-anchor="middle" font-size="10" fill="#0b0e14" font-weight="700">MESH</text><text x="220" y="215" text-anchor="middle" font-size="9"  fill="#0b0e14">evidence·logic</text>
      <circle cx="330" cy="205" r="40" fill="url(#gChat)"/><text x="330" y="202" text-anchor="middle" font-size="10" fill="#0b0e14" font-weight="700">CHAT</text><text x="330" y="215" text-anchor="middle" font-size="9"  fill="#0b0e14">Qwen+LoRA</text>
      <!-- mesh internals -->
      <g font-size="9" fill="#7a86a8" text-anchor="middle">
        <circle cx="110" cy="342" r="22" fill="#131826" stroke="#232b41"/><text x="110" y="346">claims</text>
        <circle cx="220" cy="342" r="22" fill="#131826" stroke="#232b41"/><text x="220" y="346">logic</text>
        <circle cx="330" cy="342" r="22" fill="#131826" stroke="#232b41"/><text x="330" y="346">knowledge</text>
        <circle cx="220" cy="398" r="26" fill="url(#gCore)"/><text x="220" y="402" font-size="10" fill="#0b0e14" font-weight="700">CORTEX</text>
      </g>
      <text id="routing" x="220" y="120" text-anchor="middle" font-size="10" fill="#7a86a8">routing by intent…</text>
    </svg>
  </div>
  <div class="panel">
    <h2>Training</h2>
    <div class="stats">
      <div class="stat"><div class="v" id="s-loss">—</div><div class="k">loss</div></div>
      <div class="stat"><div class="v" id="s-step">—</div><div class="k">step</div></div>
      <div class="stat"><div class="v" id="s-eta">—</div><div class="k">eta</div></div>
      <div class="stat"><div class="v" id="s-ex">—</div><div class="k">examples</div></div>
    </div>
    <div class="bar"><div id="bar"></div></div>
    <div class="row"><span id="pct">0%</span><span id="epinfo">epoch —</span></div>
    <h2 style="margin-top:14px">Loss curve</h2>
    <svg id="curve" viewBox="0 0 640 220">
      <rect x="0" y="0" width="640" height="220" fill="#0b0e14" rx="8"/>
      <g id="grid" stroke="#1a2136" stroke-width="1">
        <line x1="40" y1="30"  x2="620" y2="30"/><line x1="40" y1="80"  x2="620" y2="80"/>
        <line x1="40" y1="130" x2="620" y2="130"/><line x1="40" y1="180" x2="620" y2="180"/>
      </g>
      <path id="lossline" fill="none" stroke="#5eead4" stroke-width="2"/>
      <text id="ymax" x="8" y="34" font-size="9" fill="#7a86a8">2.5</text>
      <text id="ymin" x="8" y="184" font-size="9" fill="#7a86a8">0.0</text>
    </svg>
    <div class="eplog" id="eplog"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let pulseT = 0, stepRate = 0;

async function poll() {
  try {
    const r = await fetch('/api/state'); const s = await r.json();
    const c = s.current;
    $('live').style.color = s.alive ? '#5eead4' : '#f472b6';
    $('live').textContent = s.alive ? '● LIVE' : '○ STOPPED';
    $('s-loss').textContent = c.loss != null ? c.loss.toFixed(3) : '—';
    $('s-step').textContent = c.total ? `${c.step}/${c.total}` : '—';
    $('s-eta').textContent = c.eta_min != null ? `${c.eta_min}m` : '—';
    $('s-ex').textContent = s.usable_examples ?? '—';
    const pct = c.total ? Math.min(100, (c.step / c.total) * 100) : 0;
    $('bar').style.width = pct + '%';
    $('pct').textContent = pct.toFixed(1) + '%';
    $('epinfo').textContent = `epoch ${c.epoch} · elapsed ${Math.round(c.elapsed_s/60)}m`;
    // loss curve
    const pts = s.steps;
    if (pts.length > 1) {
      const mx = Math.max(...pts), mn = Math.min(...pts);
      const top = Math.min(2.6, mx + .05), bot = Math.max(0, mn - .05);
      const W = 580, H = 160, x0 = 48, y0 = 26;
      const X = i => x0 + i * W / (pts.length - 1);
      const Y = v => y0 + H * (1 - (v - bot) / (top - bot || 1));
      $('lossline').setAttribute('d',
        'M' + pts.map((v,i)=>`${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' L'));
      $('ymax').textContent = top.toFixed(1);
      $('ymin').textContent = bot.toFixed(1);
      stepRate = pts.length / Math.max(1, c.elapsed_s);
    }
    // epoch log
    $('eplog').innerHTML = s.epochs.map(e =>
      `<div>epoch ${e.epoch}: train ${e.train_loss.toFixed(4)} · val ${e.val_loss.toFixed(4)}</div>`).join('');
  } catch (e) {}
}

// animate pulses along router edges; rate reflects step rate
const EDGES = [
  ['p1', [220,60],  [110,180]],
  ['p2', [220,60],  [220,180]],
  ['p3', [220,60],  [330,180]],
];
function animate() {
  pulseT += 0.004 + Math.min(0.03, stepRate * 0.02);
  const tt = pulseT % 1;
  EDGES.forEach(([id, a, b], i) => {
    const local = (pulseT * (0.9 + i*0.25)) % 1;
    const el = $(id);
    el.setAttribute('cx', a[0] + (b[0]-a[0])*local);
    el.setAttribute('cy', a[1] + (b[1]-a[1])*local);
    el.setAttribute('opacity', 0.25 + 0.65*Math.sin(local*Math.PI));
  });
  requestAnimationFrame(animate);
}
animate();
setInterval(poll, 2000);
poll();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence request logging
        pass

    def do_GET(self):
        if self.path == "/api/state":
            with _state_lock:
                body = json.dumps(_state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(LOG_DEFAULT))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    stop = threading.Event()
    threading.Thread(target=watch_loop, args=(Path(args.log), stop), daemon=True).start()
    parse_log(Path(args.log))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Sweep training dashboard: http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
