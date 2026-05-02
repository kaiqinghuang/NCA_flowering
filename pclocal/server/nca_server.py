"""FastAPI WebSocket server: NCA simulation loop + RAW RGBA frame stream.

PC-local build: this server lives in the same process and on the same
host as the browser, so we ditch image compression entirely. Each frame
is sent as **raw RGBA bytes** (length = ``W * H * 4``) over a loopback
WebSocket, and the browser blits it directly to a canvas via
``putImageData``. No JPEG/WebP encoder, no compression artifacts, no
adaptive-quality controller — every pixel on screen is the exact NCA
output. At 960×540 / 30 fps the loopback bandwidth is ~60 MB/s, well
within what the kernel handles for free.

Run:
    uvicorn nca_server:app --host 127.0.0.1 --port 8000

Protocol (single bidirectional WS at /ws):
  Client → Server  (JSON text frames)
    {"op": "list_models"}
        → {"type":"models","models":[base...],"brush_models":[brush...]}
    {"op": "load_base",  "slot": 0..3, "path": "..."}  # path is relative to MODELS_DIR
    {"op": "load_brush", "path": "..."}                # path is relative to BRUSH_MODELS_DIR
    {"op": "remove_brush", "id": int}
    {"op": "select_brush", "id": int}
    {"op": "stamp", "id": int, "x": int, "y": int, "r": float, "erase": bool}
    {"op": "clear_brush", "id": int}
    {"op": "clear_state"}
    {"op": "set_param", "name": "...", "value": <number|bool>}
    {"op": "reseed"}

  Server → Client
    Binary frame: raw RGBA bytes, ``len == W * H * 4`` (latest render).
    JSON text:    {"type": "status", ...} ack/diagnostics

A separate asyncio task runs the NCA loop at TARGET_STEPS_PER_SEC and
hands a fresh RGBA buffer to clients at TARGET_FPS.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import socket as _socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nca_model import BASE_MODELS, NCASimulator, Params
from npy_loader import load_model


# ---------- Config (override via env) ----------
H = int(os.environ.get("NCA_H", "540"))
W = int(os.environ.get("NCA_W", "960"))
# Local build: 30 fps / 10 sps default. FPS and SPS are *intentionally
# decoupled* — they control different things.
#
#   * NCA_FPS = render/broadcast rate. This is the visual refresh rate
#     the user perceives — Kinect cursor smoothness, paint stroke
#     liveness, overall "the screen feels alive" feel. Each broadcast
#     sends a fresh snapshot of `sim.state`, even if the state hasn't
#     advanced since the last broadcast (cheap — render cost is a
#     ~10ms GPU clamp+transfer, no NCA work). Keep this comfortably
#     above ~24 fps so motion looks continuous.
#
#   * NCA_SPS = how often the NCA simulator actually steps forward.
#     This is the *aesthetic pace* of evolution — how fast patterns
#     grow, how fast drips flow. It's also the single biggest driver
#     of GPU load: each step costs ~20–25ms on Apple Silicon at
#     960×540 (one big depthwise conv + per-model mixing). Targeting
#     anything close to the per-step budget guarantees stutter, so we
#     keep this well below the ceiling.
#
# Default 30/10:
#   * 30 fps × ~10ms render = 300ms/s render work (~30% of the render
#     thread). Gives a smooth-feeling canvas matched to the ~30Hz
#     Kinect hand stream.
#   * 10 sps × ~25ms step = 250ms/s step work (~25%). Plenty of GPU
#     headroom for paint-event spikes and drip evolution. The sim
#     evolves at a calm, deliberate pace — patterns drift visibly but
#     never feel "frantic", which fits the slow-painting aesthetic.
#
# Side-effect on drips: `spawn_drips`'s `base_speed` was tuned around
# the original 60-sps loop, so at sps=10 drips advance ~6× slower in
# wall-clock time. If you want them livelier, push the UI **Drip spd**
# slider higher; you can also raise the **Speed** slider
# (`steps_per_frame`) to run multiple NCA steps per sim tick without
# bumping NCA_SPS — e.g. Speed=3 + sps=10 evolves NCA at the equivalent
# of ~30 sps while keeping the loop pacing & GPU spike profile of 10.
#
# Override with NCA_FPS / NCA_SPS env vars per machine.
TARGET_FPS = float(os.environ.get("NCA_FPS", "30"))
TARGET_STEPS_PER_SEC = float(os.environ.get("NCA_SPS", "10"))
PAINT_QUEUE_MAX = int(os.environ.get("NCA_PAINT_QUEUE_MAX", "4096"))
# Cap how many paint events one step can drain. Lower value = smoother
# step times during fast painting (excess events queue up and get processed
# in the next steps), at the cost of a tiny amount of input latency.
PAINT_BATCH_LIMIT = int(os.environ.get("NCA_PAINT_BATCH_LIMIT", "96"))
# Throttle drip spawning + evolution to every Nth sim step. evolve_drips is
# the dominant per-step cost once many drips are alive (each drip costs ~7
# GPU dispatches), so halving its frequency directly halves that long-tail
# cost. base_speed in spawn_drips is divided by the same factor to keep the
# wall-clock drip flow speed unchanged.
DRIP_EVOLVE_EVERY = int(os.environ.get("NCA_DRIP_EVOLVE_EVERY", "2"))
SCRIPT_DIR = Path(__file__).parent.resolve()
# Two separate weight folders, picked independently:
#
#   * MODELS_DIR        — used for the four base slots (A/B/C/D). This is
#                         the original `texture_model` library.
#   * BRUSH_MODELS_DIR  — used only by the paint brush picker (load_brush).
#                         Curated subset that the operator drops .npy files
#                         into; intentionally *not* the same folder as the
#                         base library so the two pools don't pollute each
#                         other.
#
# Each folder is resolved with the same priority chain via `_pick_dir`:
#   1. Env var (absolute, or relative to server dir)
#   2. Hardcoded absolute paths for known dev machines (so the demo PC
#      doesn't need any env var to start)
#   3. <repo-root>/<folder_name>  (i.e. pclocal/../../<folder_name>)
#   4. pclocal/<folder_name>
# The first candidate that exists AND contains at least one .npy file
# wins. If nothing matches, we still fall back to candidate #3 so the
# log clearly shows the expected layout (an empty/missing folder is
# legal — the picker will just be empty until you put files in).
_KNOWN_BASE_MODEL_DIRS: list[Path] = [
    Path(r"C:\Users\xiaoh\Documents\NCA_flowering\texture_model"),
]
_FALLBACK_BASE_MODEL_DIRS: list[Path] = [
    (SCRIPT_DIR / ".." / ".." / "texture_model").resolve(),
    (SCRIPT_DIR / ".." / "texture_model").resolve(),
]

_KNOWN_BRUSH_MODEL_DIRS: list[Path] = [
    Path(r"C:\Users\xiaoh\Documents\NCA_flowering\brush_model"),
]
_FALLBACK_BRUSH_MODEL_DIRS: list[Path] = [
    (SCRIPT_DIR / ".." / ".." / "brush_model").resolve(),
    (SCRIPT_DIR / ".." / "brush_model").resolve(),
]


def _pick_dir(env_var: str, known: list[Path], fallback: list[Path]) -> Path:
    """Pick a weights folder using the standard priority chain.

    See the comment block above for the priority order. Returns an
    absolute Path; the folder is *not* required to exist — callers must
    handle missing/empty folders gracefully (e.g. by returning [] from a
    listing helper).
    """
    env = os.environ.get(env_var)
    if env:
        p = Path(env)
        return p.resolve() if p.is_absolute() else (SCRIPT_DIR / p).resolve()
    for cand in known + fallback:
        try:
            if cand.is_dir() and any(cand.glob("*.npy")):
                return cand
        except OSError:
            continue
    return fallback[0]


MODELS_DIR = _pick_dir("NCA_MODELS_DIR", _KNOWN_BASE_MODEL_DIRS, _FALLBACK_BASE_MODEL_DIRS)
BRUSH_MODELS_DIR = _pick_dir(
    "NCA_BRUSH_MODELS_DIR", _KNOWN_BRUSH_MODEL_DIRS, _FALLBACK_BRUSH_MODEL_DIRS
)
CLIENT_DIR = SCRIPT_DIR.parent / "client"

# Pick best available device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------- Global state ----------
app = FastAPI()
sim = NCASimulator(Params(H=H, W=W), DEVICE)
STEP_EXECUTOR: Optional[ThreadPoolExecutor] = None

# Frame protocol constants advertised in the hello message. Clients
# read these to size their ImageData buffer.
FRAME_FORMAT = "rgba8"            # raw 8-bit RGBA, row-major, top-left origin
FRAME_BYTES = W * H * 4           # bytes per frame on the wire

# Latest fully-rendered RGBA frame, produced by the step thread right after
# each NCA step (Fix B). The broadcast loop just hands this bytes object to
# clients at TARGET_FPS pacing — no GPU work happens on the asyncio side.
# `bytes` assignment is atomic under the GIL, so a single global reference
# is enough for cross-thread publication; readers always see either the old
# or the new pointer, never a torn value.
_latest_rgba: Optional[bytes] = None


class ClientConnection:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        # Keep only the freshest frame to minimize latency and jitter
        # under brief stalls (e.g. browser tab in background).
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self.send_task = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        try:
            while True:
                payload = await self.queue.get()
                await self.ws.send_bytes(payload)
        except Exception:
            pass  # Handled by the main endpoint disconnect

    def cancel(self):
        self.send_task.cancel()


clients: set[ClientConnection] = set()
clients_lock = asyncio.Lock()
paint_queue = deque(maxlen=PAINT_QUEUE_MAX)
paint_queue_lock = threading.Lock()
_drip_tick = 0


# ---------- Windows multimedia timer resolution (Fix A) ----------
# Windows' default scheduling tick is ~15.6ms, so `asyncio.sleep(0.023)`
# (the 33ms broadcast pacer minus a 10ms render) actually sleeps 31ms,
# and `asyncio.sleep(0.069)` (the 100ms sim pacer minus a 30ms step)
# actually sleeps 78–94ms. That alone caps SPS at ~9 and FPS at ~22 with
# ±15ms jitter, which is what we were seeing in perf logs even when the
# GPU had plenty of headroom. timeBeginPeriod(1) drops the kernel
# scheduler tick to 1ms system-wide; sleep precision on the asyncio loop
# follows. NB: this is a process-global Windows setting; we restore it
# on shutdown to be a polite citizen.
WIN_TIMER_PERIOD_MS = 1
_win_timer_active = False


def _try_set_windows_timer_resolution(period_ms: int) -> bool:
    """Bump Windows multimedia timer resolution. No-op on non-Windows."""
    if os.name != "nt":
        return False
    try:
        winmm = ctypes.WinDLL("winmm")
        rc = winmm.timeBeginPeriod(period_ms)
        return rc == 0  # TIMERR_NOERROR
    except Exception:
        return False


def _try_clear_windows_timer_resolution(period_ms: int) -> None:
    """Undo a prior timeBeginPeriod call. No-op on non-Windows."""
    if os.name != "nt":
        return
    try:
        winmm = ctypes.WinDLL("winmm")
        winmm.timeEndPeriod(period_ms)
    except Exception:
        pass


def _try_set_tcp_nodelay(ws: WebSocket) -> bool:
    """Best-effort: disable Nagle on the underlying TCP socket.

    For loopback the impact is small, but a partial segment held by the
    kernel for ~40ms still spikes our frame latency. Starlette/uvicorn
    doesn't expose the socket officially, so we probe common attribute
    paths and fall back silently if none match.
    """
    try:
        candidates = []
        send = getattr(ws, "_send", None)
        if send is not None:
            holder = getattr(send, "__self__", None)
            if holder is not None:
                candidates.append(holder)
        for attr in ("_transport", "transport", "_protocol"):
            v = getattr(ws, attr, None)
            if v is not None:
                candidates.append(v)
        for c in candidates:
            obj = c
            for _ in range(4):
                sock = None
                if hasattr(obj, "get_extra_info"):
                    try:
                        sock = obj.get_extra_info("socket")
                    except Exception:
                        sock = None
                if sock is not None:
                    try:
                        sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                        return True
                    except Exception:
                        return False
                obj = getattr(obj, "transport", None)
                if obj is None:
                    break
    except Exception:
        pass
    return False


def _list_npy_in(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.glob("*.npy")])


def list_available_base_models() -> list[str]:
    return _list_npy_in(MODELS_DIR)


def list_available_brush_models() -> list[str]:
    return _list_npy_in(BRUSH_MODELS_DIR)


# ---------- WS handler ----------
async def handle_message(ws: WebSocket, msg: dict):
    op = msg.get("op")
    if op == "list_models":
        await ws.send_text(json.dumps({
            "type": "models",
            "models": list_available_base_models(),
            "brush_models": list_available_brush_models(),
        }))
    elif op == "load_base":
        slot = int(msg["slot"])
        path = MODELS_DIR / msg["path"]
        w = load_model(path, DEVICE)
        sim.set_base_model(slot, w)
        await ws.send_text(json.dumps({"type": "loaded_base", "slot": slot, "name": w.name}))
    elif op == "load_brush":
        # Brush models live in their own folder so the picker can be
        # curated independently of the base library (see BRUSH_MODELS_DIR).
        path = BRUSH_MODELS_DIR / msg["path"]
        print(f"[ws] load_brush path={path} exists={path.exists()}")
        w = load_model(path, DEVICE)
        bm_id = sim.add_brush_model(w)
        bm = sim.get_brush(bm_id)
        payload = {
            "type": "loaded_brush", "id": bm_id, "name": w.name,
            "color": [float(c) for c in bm.color],
        }
        print(f"[ws] loaded_brush -> {payload}")
        await ws.send_text(json.dumps(payload))
    elif op == "remove_brush":
        sim.remove_brush_model(int(msg["id"]))
    elif op == "stamp":
        _enqueue_paint_event({
            "kind": "stamp",
            "id": int(msg["id"]),
            "x": int(msg["x"]),
            "y": int(msg["y"]),
            "r": float(msg.get("r", 2.0)),
            "erase": bool(msg.get("erase", False)),
        })
    elif op == "stroke":
        _enqueue_paint_event({
            "kind": "stroke",
            "id": int(msg["id"]),
            "x0": int(msg["x0"]),
            "y0": int(msg["y0"]),
            "x1": int(msg["x1"]),
            "y1": int(msg["y1"]),
            "r": float(msg.get("r", 2.0)),
            "erase": bool(msg.get("erase", False)),
        })
    elif op == "clear_brush":
        sim.clear_brush_mask(int(msg["id"]))
    elif op == "clear_state":
        sim.clear_state()
    elif op == "reseed":
        seed = int(np.random.randint(0, 2 ** 31 - 1))
        sim.reseed_noise(seed)
    elif op == "set_param":
        _apply_param(msg["name"], msg["value"])
    else:
        await ws.send_text(json.dumps({"type": "error", "msg": f"unknown op: {op}"}))


def _apply_param(name: str, value):
    p = sim.p
    if name in ("alignment", "rotation_deg"):
        setattr(p, name, type(getattr(p, name))(value))
        sim.update_direction()
    elif name in ("noise_scale", "octaves", "half_width", "noise_z_scale", "layer_freq_spread"):
        setattr(p, name, float(value))
        sim.mark_altitude_dirty()
    elif name in ("noise_z_speed",):
        p.noise_z_speed = float(value)
    elif name in ("mask_threshold", "mask_edge_sharpness"):
        setattr(p, name, float(value))
        sim.mark_altitude_dirty()
    elif name in ("steps_per_frame",):
        p.steps_per_frame = int(value)
    elif name in ("spray_splatter_amount", "drip_gravity"):
        setattr(p, name, int(value))
    elif name in (
        "spray_splatter_radius", "spray_drip_threshold",
        "spray_drip_speed", "spray_drip_wobble",
        "spray_drip_min_width", "spray_drip_chance",
    ):
        setattr(p, name, float(value))
    elif name in ("disturbance", "show_mask_tint", "active"):
        setattr(p, name, bool(value))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    nodelay_ok = _try_set_tcp_nodelay(ws)
    print(f"[ws] client connected tcp_nodelay={'on' if nodelay_ok else 'unset'}")
    client_conn = ClientConnection(ws)
    async with clients_lock:
        clients.add(client_conn)
    try:
        # Send initial inventory
        await ws.send_text(json.dumps({
            "type": "hello",
            "device": str(DEVICE),
            "H": H, "W": W,
            "fps": TARGET_FPS,
            "frame_format": FRAME_FORMAT,
            "frame_bytes": FRAME_BYTES,
            "models": list_available_base_models(),
            "brush_models": list_available_brush_models(),
        }))
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
                await handle_message(ws, msg)
            except Exception as e:
                import traceback
                traceback.print_exc()
                await ws.send_text(json.dumps({"type": "error", "msg": str(e)}))
    except WebSocketDisconnect:
        pass
    finally:
        client_conn.cancel()
        async with clients_lock:
            clients.discard(client_conn)


# ---------- Background simulation + broadcast ----------
# Per-second perf counters. After Fix B the step thread does both the
# NCA step and the render, so we track them as separate phases:
#   * `calls`   = number of `_step_blocking` invocations (≈ TARGET_SPS)
#   * `steps`   = NCA steps executed (= calls × steps_per_frame, used for sps)
#   * `renders` = renders done in the step thread (= calls when there's
#                 at least one client; 0 otherwise — we skip the render
#                 phase entirely when nobody is connected to save GPU)
#   * `frames`  = unique RGBA payloads pushed to clients by broadcast_loop
#                 (skip-duplicate: equals min(SPS, TARGET_FPS) when active)
_perf = {
    "calls": 0,
    "steps": 0,
    "renders": 0,
    "frames": 0,
    "step_ms": 0.0,
    "render_ms": 0.0,
    "paint_applied": 0,
    "paint_dropped": 0,
    "last": time.perf_counter(),
}


def _enqueue_paint_event(event: dict):
    with paint_queue_lock:
        was_full = len(paint_queue) == PAINT_QUEUE_MAX
        paint_queue.append(event)
        if was_full:
            _perf["paint_dropped"] += 1


def _drain_paint_events(limit: int) -> list[dict]:
    out: list[dict] = []
    with paint_queue_lock:
        n = min(limit, len(paint_queue))
        for _ in range(n):
            out.append(paint_queue.popleft())
    return out


def _render_to_bytes() -> bytes:
    """Snapshot NCA state and return a packed RGBA byte buffer ready for WS.

    Returns ``W * H * 4`` bytes (row-major, R,G,B,255 per pixel) ready to
    feed straight into the browser's ``putImageData`` — no encoder, no
    compression, no quality knob. Pixel values are exactly what the NCA
    simulator produced this tick.

    Called from the step thread immediately after the NCA step completes
    (Fix B). Both the step writes and this render now run on the same
    CUDA default stream in the same thread, so the implicit `.cpu()` sync
    only waits for *this thread's* ops — no cross-thread GPU contention.
    """
    with sim.lock:
        state_snap = sim.state.detach().clone()
        if sim.p.show_mask_tint and sim.mask is not None:
            tint_snap = sim._compose_tint().detach().clone()
        else:
            tint_snap = None
    rgb = state_snap[0, :3].clamp(-1, 1).mul(0.5).add(0.5)
    if tint_snap is not None:
        rgb = rgb * 0.76 + tint_snap * 0.24
    rgb_u8 = rgb.clamp(0, 1).mul(255).to(torch.uint8)            # (3, H, W)
    alpha = torch.full((1, H, W), 255, dtype=torch.uint8, device=rgb_u8.device)
    rgba = torch.cat([rgb_u8, alpha], dim=0).permute(1, 2, 0).contiguous()
    return rgba.cpu().numpy().tobytes()


def _step_blocking():
    """Run one NCA step batch + render in a single worker thread.

    Per Fix B, the render runs in this same thread on the CUDA default
    stream right after the step. This eliminates the GPU-stream contention
    that caused render's `.cpu()` to wait for step's conv2d (and vice
    versa) when they were on different threads, and removes the
    "phase-locking" failure mode where the two loops aligned and the
    effective FPS halved.

    Side effect: writes the rendered RGBA payload to the module-global
    `_latest_rgba`. The broadcast loop (asyncio side) only paces sends
    and never touches the GPU.
    """
    global _latest_rgba
    t_call_start = time.perf_counter()

    # Apply queued brush ops inside the step thread to avoid lock contention with WS.
    events = _drain_paint_events(PAINT_BATCH_LIMIT)
    for ev in events:
        if ev["kind"] == "stamp":
            sim.stamp_disk(ev["id"], ev["x"], ev["y"], ev["r"], ev["erase"])
        else:
            sim.paint_segment(ev["id"], ev["x0"], ev["y0"], ev["x1"], ev["y1"], ev["r"], ev["erase"])
    _perf["paint_applied"] += len(events)

    # Spawn new drips from the wet pool laid down by the just-applied paint
    # events, then evolve all live drips one tick. Both run even when sim is
    # paused so paint visibly drips after you stop painting. Throttled to
    # every Nth step (see DRIP_EVOLVE_EVERY) to cap drip-related GPU load.
    global _drip_tick
    _drip_tick += 1
    if _drip_tick >= DRIP_EVOLVE_EVERY:
        _drip_tick = 0
        with sim.lock:
            sim.spawn_drips()
            sim.evolve_drips()

    n = 0
    if sim.p.active:
        if sim.p.disturbance:
            sim.apply_disturbance(time.time())
        n = max(1, sim.p.steps_per_frame)
        for _ in range(n):
            sim.step()
    # Sync here so `step_ms` reflects real GPU completion (without this,
    # CUDA work would still be queued and the actual cost would land
    # inside the render's `.cpu()` instead).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_step_end = time.perf_counter()
    _perf["step_ms"] += (t_step_end - t_call_start) * 1000.0
    _perf["steps"] += n
    _perf["calls"] += 1

    # Render right after step on the same default stream. Skip entirely
    # when nobody's connected so we don't burn GPU on bytes nobody reads.
    # (`len(set)` is a single int read under the GIL — no lock needed.)
    if len(clients) > 0:
        payload = _render_to_bytes()  # `.cpu()` inside is the implicit sync
        _latest_rgba = payload
        _perf["render_ms"] += (time.perf_counter() - t_step_end) * 1000.0
        _perf["renders"] += 1


async def perf_loop():
    """Print per-second timing so the bottleneck is visible."""
    while True:
        await asyncio.sleep(2.0)
        now = time.perf_counter()
        dt = now - _perf["last"]
        if dt < 0.1:
            continue
        sps = _perf["steps"] / dt
        fps = _perf["frames"] / dt
        # Per-call averages: step_ms is accumulated per `_step_blocking`
        # invocation, divided by call count; render_ms is only added when
        # there's at least one client (so we divide by `renders`, not
        # `calls`, to avoid showing 0.00 when nobody is connected).
        avg_step = _perf["step_ms"] / max(1, _perf["calls"])
        avg_render = _perf["render_ms"] / max(1, _perf["renders"])
        with paint_queue_lock:
            queued = len(paint_queue)
        # frame_mb = bytes/sec actually pushed to clients (skip-duplicate
        # means this drops to ~0 when SPS=0 even if FPS target is high).
        frame_mb_s = fps * FRAME_BYTES / 1_048_576.0
        print(
            f"[perf] sps={sps:5.1f}  fps={fps:4.1f}  step_avg={avg_step:5.2f}ms  "
            f"render_avg={avg_render:5.2f}ms  raw_mbps={frame_mb_s:5.1f}MB/s  "
            f"paint={_perf['paint_applied']} drop={_perf['paint_dropped']} "
            f"q={queued} clients={len(clients)}"
        )
        _perf["calls"] = 0
        _perf["steps"] = 0
        _perf["renders"] = 0
        _perf["frames"] = 0
        _perf["step_ms"] = 0.0
        _perf["render_ms"] = 0.0
        _perf["paint_applied"] = 0
        _perf["paint_dropped"] = 0
        _perf["last"] = now


async def sim_loop():
    """Schedule NCA step batches in a thread pool — keeps asyncio loop free."""
    loop = asyncio.get_running_loop()
    interval = 1.0 / TARGET_STEPS_PER_SEC
    next_t = time.perf_counter()
    while True:
        if sim.count_loaded_models() > 0:
            await loop.run_in_executor(STEP_EXECUTOR, _step_blocking)
        next_t += interval
        delay = next_t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_t = time.perf_counter()


async def broadcast_loop():
    """Pace network sends at TARGET_FPS — no GPU work happens here.

    Per Fix B, the step thread renders right after each NCA step and
    publishes the resulting RGBA bytes to `_latest_rgba`. This loop just
    copies that reference into per-client send queues at the configured
    FPS rate.

    Skip-duplicate: when SPS < TARGET_FPS (e.g. SPS=10, FPS=30), the same
    payload reference would otherwise be sent multiple times. We compare
    by identity (`is`) — any new render publishes a fresh `bytes` object,
    so an unchanged reference means the NCA hasn't advanced and there's
    nothing new to deliver. Skipping those duplicate sends:
        * frees the asyncio loop / WS layer from pointless 2 MB writes
          (which was a real source of jitter in the old setup), and
        * spares the browser putImageData calls on identical pixels.

    Net effect: `fps` perf counter reports min(SPS, TARGET_FPS) — the
    rate at which the user actually sees new content.
    """
    interval = 1.0 / TARGET_FPS
    next_t = time.perf_counter()
    last_sent: Optional[bytes] = None
    while True:
        payload = _latest_rgba
        if payload is not None and payload is not last_sent and clients:
            async with clients_lock:
                for client in clients:
                    try:
                        client.queue.put_nowait(payload)
                    except asyncio.QueueFull:
                        try:
                            _ = client.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            continue
                        try:
                            client.queue.put_nowait(payload)
                        except asyncio.QueueFull:
                            pass
            last_sent = payload
            _perf["frames"] += 1
        next_t += interval
        delay = next_t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_t = time.perf_counter()


@app.on_event("startup")
async def on_startup():
    global STEP_EXECUTOR, _win_timer_active
    raw_mb_per_s = TARGET_FPS * FRAME_BYTES / 1_048_576.0
    print(
        f"[nca_server] device={DEVICE} grid={W}×{H} fps={TARGET_FPS} "
        f"sps={TARGET_STEPS_PER_SEC} format={FRAME_FORMAT} "
        f"frame_bytes={FRAME_BYTES} (~{raw_mb_per_s:.0f} MB/s on loopback)"
    )
    print(
        f"[nca_server] base  models dir: {MODELS_DIR} "
        f"({len(list_available_base_models())} .npy found)"
    )
    print(
        f"[nca_server] brush models dir: {BRUSH_MODELS_DIR} "
        f"({len(list_available_brush_models())} .npy found)"
    )
    _win_timer_active = _try_set_windows_timer_resolution(WIN_TIMER_PERIOD_MS)
    print(
        f"[nca_server] win_timer_{WIN_TIMER_PERIOD_MS}ms="
        f"{'on' if _win_timer_active else 'off (non-Windows or failed)'}"
    )
    # Single dedicated thread for the combined step+render work. We no longer
    # use a separate render executor — render now runs in this thread right
    # after each step on the same CUDA default stream, eliminating the GPU
    # contention / phase-locking that produced the uneven FPS pattern.
    STEP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nca-step")
    asyncio.create_task(sim_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(perf_loop())


@app.on_event("shutdown")
async def on_shutdown():
    if STEP_EXECUTOR is not None:
        STEP_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    if _win_timer_active:
        _try_clear_windows_timer_resolution(WIN_TIMER_PERIOD_MS)


# ---------- Static client ----------
if CLIENT_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(CLIENT_DIR)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(
            str(CLIENT_DIR / "index.html"),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
