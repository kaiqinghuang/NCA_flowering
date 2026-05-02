"""FastAPI WebSocket server: NCA simulation loop + RAW RGBA frame stream.

PC-local build: this server lives in the same process and on the same
host as the browser, so we ditch image compression entirely. Each frame
is sent as **raw RGBA bytes** (length = ``W * H * 4``) over a loopback
WebSocket, and the browser blits it directly to a canvas via
``putImageData``. No JPEG/WebP encoder, no compression artifacts, no
adaptive-quality controller — every pixel on screen is the exact NCA
output. At 960×540 / 15 fps the loopback bandwidth is ~30 MB/s, well
within what the kernel handles for free.

Run:
    uvicorn nca_server:app --host 127.0.0.1 --port 8000

Protocol (single bidirectional WS at /ws):
  Client → Server  (JSON text frames)
    {"op": "list_models"}
    {"op": "load_base", "slot": 0..3, "path": "..."}
    {"op": "load_brush", "path": "..."}
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
# Local build: 15 fps / 15 sps default. This is *intentionally slow*.
#
# Rationale: the previous 60/60 target was aspirational — on Apple
# Silicon (MPS) the actual NCA step costs ~20–25ms at 960×540, plus
# ~10ms render and asyncio dispatch overhead. The loop would fire steps
# faster than the GPU could complete them, so wall-clock throughput
# settled around 12 sps with highly *uneven* per-frame latency. Visually
# that reads as "stuttering / dropped frames" even though the GPU is the
# real bottleneck.
#
# Picking a target the GPU can hit comfortably (15 sps × ~25ms = 375ms
# of step work per second, ≈40% utilization) makes every frame land on
# time, so the simulation looks *evenly slow* — a calm, flowing pace
# rather than a frantic-but-stuttery one. Drips and NCA evolution end
# up running at ~1/4 the previous wall-clock speed (since `spawn_drips`
# was tuned around 60 sps), which fits the slow-painting aesthetic.
#
# To temporarily speed things back up without touching the loop rate,
# turn up the UI `Speed` slider (`steps_per_frame`); e.g. Speed=4 with
# sps=15 ≈ the old 60 sps evolution rate.
#
# Override with NCA_FPS / NCA_SPS env vars if you want a faster machine
# to push more.
TARGET_FPS = float(os.environ.get("NCA_FPS", "15"))
TARGET_STEPS_PER_SEC = float(os.environ.get("NCA_SPS", "15"))
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
# Where to look for .npy weights, in priority order:
#   1. NCA_MODELS_DIR env var (absolute or relative-to-server-dir)
#   2. Hardcoded absolute paths for known dev machines (so the demo PC
#      doesn't need any env var to start). Add your own machine's path
#      to the list below.
#   3. <repo-root>/texture_model  (i.e. pclocal/../../texture_model)
#   4. pclocal/texture_model
# The first candidate that exists AND contains at least one .npy file
# wins. If nothing matches, we still fall back to candidate #3 so the
# log clearly shows the expected layout.
_KNOWN_MODEL_DIRS: list[Path] = [
    Path(r"C:\Users\xiaoh\Documents\NCA_flowering\texture_model"),
]
_FALLBACK_MODEL_DIRS: list[Path] = [
    (SCRIPT_DIR / ".." / ".." / "texture_model").resolve(),
    (SCRIPT_DIR / ".." / "texture_model").resolve(),
]


def _pick_models_dir() -> Path:
    env = os.environ.get("NCA_MODELS_DIR")
    if env:
        p = Path(env)
        return p.resolve() if p.is_absolute() else (SCRIPT_DIR / p).resolve()
    for cand in _KNOWN_MODEL_DIRS + _FALLBACK_MODEL_DIRS:
        try:
            if cand.is_dir() and any(cand.glob("*.npy")):
                return cand
        except OSError:
            continue
    return _FALLBACK_MODEL_DIRS[0]


MODELS_DIR = _pick_models_dir()
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
RENDER_EXECUTOR: Optional[ThreadPoolExecutor] = None

# Frame protocol constants advertised in the hello message. Clients
# read these to size their ImageData buffer.
FRAME_FORMAT = "rgba8"            # raw 8-bit RGBA, row-major, top-left origin
FRAME_BYTES = W * H * 4           # bytes per frame on the wire


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


def list_available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.glob("*.npy")])


# ---------- WS handler ----------
async def handle_message(ws: WebSocket, msg: dict):
    op = msg.get("op")
    if op == "list_models":
        await ws.send_text(json.dumps({"type": "models", "models": list_available_models()}))
    elif op == "load_base":
        slot = int(msg["slot"])
        path = MODELS_DIR / msg["path"]
        w = load_model(path, DEVICE)
        sim.set_base_model(slot, w)
        await ws.send_text(json.dumps({"type": "loaded_base", "slot": slot, "name": w.name}))
    elif op == "load_brush":
        path = MODELS_DIR / msg["path"]
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
            "models": list_available_models(),
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
# Per-second perf counters
_perf = {
    "steps": 0,
    "frames": 0,
    "render_ms": 0.0,
    "step_ms": 0.0,
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


def _step_blocking():
    """Run one batch of NCA steps in a worker thread (releases asyncio loop)."""
    t0 = time.perf_counter()
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _perf["step_ms"] += (time.perf_counter() - t0) * 1000.0
    _perf["steps"] += n


def _render_raw_blocking() -> bytes:
    """Snapshot NCA state on GPU under sim lock, then transfer to a flat
    RGBA byte buffer.

    Returns ``W * H * 4`` bytes (row-major, R,G,B,255 per pixel) ready to
    feed straight into the browser's ``putImageData`` — no encoder, no
    compression, no quality knob. Pixel values are exactly what the NCA
    simulator produced this tick.

    Concurrency:
        - Lock window is short: just a tensor clone (kernel-queue submit
          on the device, returns immediately on CUDA / MPS).
        - GPU→CPU transfer + alpha pad happens outside the lock so the
          step thread isn't blocked waiting on GPU sync.
    """
    t0 = time.perf_counter()
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
    # Pad an opaque alpha channel on-device, then transfer one (H, W, 4)
    # block. One contiguous CPU array → no extra copy when we hand it to
    # FastAPI's send_bytes (it'll bytes() the buffer once).
    alpha = torch.full((1, H, W), 255, dtype=torch.uint8, device=rgb_u8.device)
    rgba = torch.cat([rgb_u8, alpha], dim=0).permute(1, 2, 0).contiguous()
    rgba_np = rgba.cpu().numpy()                                  # (H, W, 4) uint8
    payload = rgba_np.tobytes()
    _perf["render_ms"] += (time.perf_counter() - t0) * 1000.0
    _perf["frames"] += 1
    return payload


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
        avg_step = _perf["step_ms"] / max(1, _perf["steps"])
        avg_render = _perf["render_ms"] / max(1, _perf["frames"])
        with paint_queue_lock:
            queued = len(paint_queue)
        # frame_mb = bytes/sec on the wire (uncompressed RGBA)
        frame_mb_s = fps * FRAME_BYTES / 1_048_576.0
        print(
            f"[perf] sps={sps:5.1f}  fps={fps:4.1f}  step_avg={avg_step:5.2f}ms  "
            f"render_avg={avg_render:5.2f}ms  raw_mbps={frame_mb_s:5.1f}MB/s  "
            f"paint={_perf['paint_applied']} drop={_perf['paint_dropped']} "
            f"q={queued} clients={len(clients)}"
        )
        _perf["steps"] = 0
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
    """Render + send the current frame at TARGET_FPS — render runs in a
    worker thread. Each client gets at most one un-sent frame queued; if
    it's still un-sent when the next frame is ready we replace the stale
    one with the fresh one (browser tab in background, etc.)."""
    loop = asyncio.get_running_loop()
    interval = 1.0 / TARGET_FPS
    next_t = time.perf_counter()
    while True:
        if clients and sim.count_loaded_models() > 0:
            try:
                payload = await loop.run_in_executor(RENDER_EXECUTOR, _render_raw_blocking)
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
            except Exception as e:
                print(f"[broadcast] render/send failed: {e}")
        next_t += interval
        delay = next_t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_t = time.perf_counter()


@app.on_event("startup")
async def on_startup():
    global STEP_EXECUTOR, RENDER_EXECUTOR
    raw_mb_per_s = TARGET_FPS * FRAME_BYTES / 1_048_576.0
    print(
        f"[nca_server] device={DEVICE} grid={W}×{H} fps={TARGET_FPS} "
        f"sps={TARGET_STEPS_PER_SEC} format={FRAME_FORMAT} "
        f"frame_bytes={FRAME_BYTES} (~{raw_mb_per_s:.0f} MB/s on loopback)"
    )
    print(f"[nca_server] models dir: {MODELS_DIR} ({len(list_available_models())} found)")
    # Dedicated single-thread executors keep step/render timing stable.
    STEP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nca-step")
    RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nca-render")
    asyncio.create_task(sim_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(perf_loop())


@app.on_event("shutdown")
async def on_shutdown():
    if STEP_EXECUTOR is not None:
        STEP_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    if RENDER_EXECUTOR is not None:
        RENDER_EXECUTOR.shutdown(wait=False, cancel_futures=True)


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
