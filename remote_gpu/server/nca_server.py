"""FastAPI WebSocket server: NCA simulation loop + WebP frame stream.

Run:
    uvicorn nca_server:app --host 0.0.0.0 --port 8000

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
    Binary frame: WebP-encoded RGB image (latest render).
    JSON text:    {"type": "status", ...} ack/diagnostics

A separate asyncio task runs the NCA loop at TARGET_STEPS_PER_SEC and
encodes a frame at TARGET_FPS, broadcasting to all connected clients.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
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
from PIL import Image

from nca_model import BASE_MODELS, NCASimulator, Params
from npy_loader import load_model


# ---------- Config (override via env) ----------
H = int(os.environ.get("NCA_H", "540"))
W = int(os.environ.get("NCA_W", "960"))
TARGET_FPS = float(os.environ.get("NCA_FPS", "30"))
TARGET_STEPS_PER_SEC = float(os.environ.get("NCA_SPS", "60"))
WEBP_QUALITY = int(os.environ.get("NCA_WEBP_Q", "98"))
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
_default_models = (SCRIPT_DIR / ".." / ".." / "texture_model").resolve()
_env_models = os.environ.get("NCA_MODELS_DIR")
if _env_models:
    _md = Path(_env_models)
    MODELS_DIR = _md.resolve() if _md.is_absolute() else (SCRIPT_DIR / _md).resolve()
else:
    MODELS_DIR = _default_models
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
ENCODE_EXECUTOR: Optional[ThreadPoolExecutor] = None

class ClientConnection:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        # Keep only the freshest frame to minimize latency and jitter under bursts.
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
    "encode_ms": 0.0,
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


def _render_and_encode_blocking() -> bytes:
    """Snapshot state on GPU under sim lock, then transfer + encode unlocked."""
    t0 = time.perf_counter()
    # Take quick GPU snapshot under the lock (kernel queue submit, very fast).
    with sim.lock:
        state_snap = sim.state.detach().clone()
        if sim.p.show_mask_tint and sim.mask is not None:
            tint_snap = sim._compose_tint().detach().clone()
        else:
            tint_snap = None
    # Heavy work outside the lock: GPU sync + transfer + encode
    rgb = state_snap[0, :3].clamp(-1, 1).mul(0.5).add(0.5)
    if tint_snap is not None:
        rgb = rgb * 0.76 + tint_snap * 0.24
    rgb_np = rgb.clamp(0, 1).permute(1, 2, 0).mul(255).to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    # JPEG encoding is 5-10x faster than WebP and reduces Python GIL blocking significantly
    Image.fromarray(rgb_np, mode="RGB").save(buf, format="JPEG", quality=WEBP_QUALITY)
    payload = buf.getvalue()
    _perf["encode_ms"] += (time.perf_counter() - t0) * 1000.0
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
        avg_enc = _perf["encode_ms"] / max(1, _perf["frames"])
        with paint_queue_lock:
            queued = len(paint_queue)
        print(
            f"[perf] sps={sps:5.1f}  fps={fps:4.1f}  step_avg={avg_step:5.2f}ms  "
            f"enc_avg={avg_enc:5.2f}ms  paint={_perf['paint_applied']} "
            f"drop={_perf['paint_dropped']} q={queued} clients={len(clients)}"
        )
        _perf["steps"] = 0
        _perf["frames"] = 0
        _perf["step_ms"] = 0.0
        _perf["encode_ms"] = 0.0
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
    """Encode + send the current render at TARGET_FPS — encode runs in a worker thread."""
    loop = asyncio.get_running_loop()
    interval = 1.0 / TARGET_FPS
    next_t = time.perf_counter()
    while True:
        if clients and sim.count_loaded_models() > 0:
            try:
                payload = await loop.run_in_executor(ENCODE_EXECUTOR, _render_and_encode_blocking)
                async with clients_lock:
                    for client in clients:
                        try:
                            client.queue.put_nowait(payload)
                        except asyncio.QueueFull:
                            # Slow client: drop stale frame, keep freshest one.
                            try:
                                _ = client.queue.get_nowait()
                            except asyncio.QueueEmpty:
                                continue
                            try:
                                client.queue.put_nowait(payload)
                            except asyncio.QueueFull:
                                pass
            except Exception as e:
                print(f"[broadcast] encode/send failed: {e}")
        next_t += interval
        delay = next_t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_t = time.perf_counter()


@app.on_event("startup")
async def on_startup():
    global STEP_EXECUTOR, ENCODE_EXECUTOR
    print(f"[nca_server] device={DEVICE} grid={W}×{H} fps={TARGET_FPS} sps={TARGET_STEPS_PER_SEC}")
    print(f"[nca_server] models dir: {MODELS_DIR} ({len(list_available_models())} found)")
    # Dedicated single-thread executors keep step/encode timing stable.
    STEP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nca-step")
    ENCODE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nca-encode")
    asyncio.create_task(sim_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(perf_loop())


@app.on_event("shutdown")
async def on_shutdown():
    if STEP_EXECUTOR is not None:
        STEP_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    if ENCODE_EXECUTOR is not None:
        ENCODE_EXECUTOR.shutdown(wait=False, cancel_futures=True)


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
