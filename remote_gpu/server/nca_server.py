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
import time
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
WEBP_QUALITY = int(os.environ.get("NCA_WEBP_Q", "70"))
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
clients: set[WebSocket] = set()
clients_lock = asyncio.Lock()


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
        sim.stamp_disk(int(msg["id"]), int(msg["x"]), int(msg["y"]),
                       float(msg.get("r", 2.0)), bool(msg.get("erase", False)))
    elif op == "stroke":
        sim.paint_segment(
            int(msg["id"]),
            int(msg["x0"]), int(msg["y0"]),
            int(msg["x1"]), int(msg["y1"]),
            float(msg.get("r", 2.0)),
            bool(msg.get("erase", False)),
        )
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
    elif name in ("noise_scale", "octaves", "half_width", "noise_z_scale"):
        setattr(p, name, float(value))
        sim.mark_altitude_dirty()
    elif name in ("noise_z_speed",):
        p.noise_z_speed = float(value)
    elif name in ("mask_threshold", "mask_edge_sharpness"):
        setattr(p, name, float(value))
        sim.mask_dirty = True
    elif name in ("steps_per_frame",):
        p.steps_per_frame = int(value)
    elif name in ("disturbance", "show_mask_tint", "active"):
        setattr(p, name, bool(value))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        # Send initial inventory
        await ws.send_text(json.dumps({
            "type": "hello",
            "device": str(DEVICE),
            "H": H, "W": W,
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
        async with clients_lock:
            clients.discard(ws)


# ---------- Background simulation + broadcast ----------
# Per-second perf counters
_perf = {"steps": 0, "frames": 0, "encode_ms": 0.0, "step_ms": 0.0, "last": time.perf_counter()}


def _step_blocking():
    """Run one batch of NCA steps in a worker thread (releases asyncio loop)."""
    t0 = time.perf_counter()
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
    # Heavy work outside the lock: GPU sync + transfer + WebP encode
    rgb = state_snap[0, :3].clamp(-1, 1).mul(0.5).add(0.5)
    if tint_snap is not None:
        rgb = rgb * 0.76 + tint_snap * 0.24
    rgb_np = rgb.clamp(0, 1).permute(1, 2, 0).mul(255).to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(rgb_np, mode="RGB").save(buf, format="WEBP", quality=WEBP_QUALITY, method=0)
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
        print(f"[perf] sps={sps:5.1f}  fps={fps:4.1f}  step_avg={avg_step:5.2f}ms  enc_avg={avg_enc:5.2f}ms  clients={len(clients)}")
        _perf["steps"] = 0
        _perf["frames"] = 0
        _perf["step_ms"] = 0.0
        _perf["encode_ms"] = 0.0
        _perf["last"] = now


async def sim_loop():
    """Schedule NCA step batches in a thread pool — keeps asyncio loop free."""
    interval = 1.0 / TARGET_STEPS_PER_SEC
    next_t = time.perf_counter()
    while True:
        if sim.p.active and sim.count_loaded_models() > 0:
            await asyncio.to_thread(_step_blocking)
        next_t += interval
        delay = next_t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_t = time.perf_counter()


async def broadcast_loop():
    """Encode + send the current render at TARGET_FPS — encode runs in a worker thread."""
    interval = 1.0 / TARGET_FPS
    next_t = time.perf_counter()
    while True:
        if clients:
            try:
                payload = await asyncio.to_thread(_render_and_encode_blocking)
                async with clients_lock:
                    dead = []
                    for ws in clients:
                        try:
                            await ws.send_bytes(payload)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        clients.discard(ws)
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
    print(f"[nca_server] device={DEVICE} grid={W}×{H} fps={TARGET_FPS} sps={TARGET_STEPS_PER_SEC}")
    print(f"[nca_server] models dir: {MODELS_DIR} ({len(list_available_models())} found)")
    asyncio.create_task(sim_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(perf_loop())


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
