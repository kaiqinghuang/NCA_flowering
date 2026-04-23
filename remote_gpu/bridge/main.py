"""Local Kinect→browser bridge.

Run on the Windows laptop that has the Kinect:

    python -m uvicorn bridge.main:app --host 0.0.0.0 --port 7000

The browser connects to `ws://localhost:7000/ws` (same-origin works via the
main NCA client page; the bridge address is configurable). Protocol:

Browser → Bridge (JSON text):
    {"op": "start_calibration"}
    {"op": "cancel_calibration"}
    {"op": "reset_calibration"}

Bridge → Browser (JSON text, broadcast to all connected clients):
    {"op": "hello", "canvas": [W, H], "cal_ready": bool}
    {"op": "hand", "t": ..., "tracked": bool, "confident": bool,
     "cal_ready": bool, "cal_mode": "idle|capturing",
     "cal_status": {...},
     "cx": float?, "cy": float?,   // only if tracked AND calibrated
     "pinch": bool}
    {"op": "cal_started", ...}  {"op": "cal_cancelled"}  {"op": "cal_reset"}
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .calibration import Calibration
from .gesture import GestureProcessor
from .kinect_source import KinectSource, RawHandFrame


# ------------------ Config ------------------
HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("BRIDGE_PORT", "7000"))
CANVAS_W = int(os.environ.get("BRIDGE_CANVAS_W", "960"))
CANVAS_H = int(os.environ.get("BRIDGE_CANVAS_H", "540"))
EMA_ALPHA = float(os.environ.get("BRIDGE_EMA_ALPHA", "0.35"))
BROADCAST_HZ = float(os.environ.get("BRIDGE_BROADCAST_HZ", "30"))
PINCH_DIST_M = float(os.environ.get("BRIDGE_PINCH_DIST_M", "0.03"))
DEBOUNCE_FRAMES = int(os.environ.get("BRIDGE_DEBOUNCE", "1"))

SCRIPT_DIR = Path(__file__).parent.resolve()
CAL_PATH = str(SCRIPT_DIR / "calibration.json")


# ------------------ Globals ------------------
app = FastAPI(title="NCA Kinect Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clients: set[WebSocket] = set()
clients_lock = asyncio.Lock()

# Latest raw frame from Kinect thread (protected by thread lock).
_latest_raw: Optional[RawHandFrame] = None
_raw_lock = threading.Lock()
_main_loop: Optional[asyncio.AbstractEventLoop] = None
_new_raw_event: Optional[asyncio.Event] = None

calibration = Calibration(CAL_PATH, (CANVAS_W, CANVAS_H))
gesture = GestureProcessor(
    ema_alpha=EMA_ALPHA,
    pinch_dist_m=PINCH_DIST_M,
    debounce_frames=DEBOUNCE_FRAMES,
)


def _on_kinect_frame(frame: RawHandFrame) -> None:
    """Called from the Kinect thread."""
    global _latest_raw
    with _raw_lock:
        _latest_raw = frame
    if _main_loop is not None and _new_raw_event is not None:
        # Thread-safe: schedule Event.set() on the asyncio loop.
        _main_loop.call_soon_threadsafe(_new_raw_event.set)


kinect = KinectSource(_on_kinect_frame)


# ------------------ Broadcast helper ------------------
async def _broadcast(msg: dict) -> None:
    data = json.dumps(msg)
    async with clients_lock:
        stale: list[WebSocket] = []
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                stale.append(ws)
        for ws in stale:
            clients.discard(ws)


# ------------------ Processor loop ------------------
async def processor_loop() -> None:
    """Translate raw Kinect frames → canvas events + broadcast."""
    assert _new_raw_event is not None
    interval = 1.0 / max(1.0, BROADCAST_HZ)
    last_send = 0.0
    last_tracked = False

    while True:
        # Wait for at least one new raw frame, then drain pacing.
        await _new_raw_event.wait()
        _new_raw_event.clear()

        with _raw_lock:
            raw = _latest_raw
        if raw is None:
            continue

        now = time.perf_counter()
        if now - last_send < interval * 0.9:
            # Too soon — let more raw frames arrive; they'll coalesce via `_latest_raw`.
            continue
        last_send = now

        tracked = bool(raw.tracked)
        pinch_raw = False
        pinch_dist = -1.0
        cal_status: dict = {"state": "idle"}
        cx_canvas: Optional[float] = None
        cy_canvas: Optional[float] = None

        if tracked:
            xz = (raw.hand_pos[0], raw.hand_pos[2])
            dx = raw.hand_pos[0] - raw.thumb_pos[0]
            dy = raw.hand_pos[1] - raw.thumb_pos[1]
            dz = raw.hand_pos[2] - raw.thumb_pos[2]
            pinch_dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            pinch_raw = gesture.raw_pinch(
                raw.hand_pos, raw.thumb_pos, raw.hand_state, raw.confident,
            )

            if calibration.mode == "capturing":
                # Route pinch-held samples to the calibration state machine.
                cal_status = calibration.on_sample(xz, pinch_raw)
                # Don't emit canvas coords while capturing; prevents
                # accidental paint strokes during the guided wizard.
            elif calibration.ready:
                mapped = calibration.apply(xz)
                if mapped is not None:
                    cx_canvas, cy_canvas = mapped

        # Track loss → reset EMA so re-entry doesn't produce a wide snap.
        if not tracked and last_tracked:
            gesture.reset_xy()
            gesture.reset_pinch()
        last_tracked = tracked

        # Clamp + smooth; only reports pinch when tracked and calibrated.
        ema_xy: Optional[tuple[float, float]] = None
        stable_pinch = False
        if cx_canvas is not None and cy_canvas is not None:
            cx_clamped = max(0.0, min(CANVAS_W - 1.0, cx_canvas))
            cy_clamped = max(0.0, min(CANVAS_H - 1.0, cy_canvas))
            (ex, ey), stable_pinch = gesture.update(cx_clamped, cy_clamped, pinch_raw)
            ema_xy = (ex, ey)

        payload = {
            "op": "hand",
            "t": raw.t,
            "tracked": tracked,
            "confident": bool(raw.confident),
            "cal_ready": calibration.ready,
            "cal_mode": calibration.mode,
            "cal_status": cal_status,
            "pinch": bool(stable_pinch),
            "pinch_raw": bool(pinch_raw),
            "pinch_dist_m": float(pinch_dist),
            "debug_joints": {
                "hand_color": [raw.hand_color[0], raw.hand_color[1]],
                "thumb_color": [raw.thumb_color[0], raw.thumb_color[1]],
                "wrist_color": [raw.wrist_color[0], raw.wrist_color[1]],
            },
        }
        if ema_xy is not None:
            payload["cx"] = ema_xy[0]
            payload["cy"] = ema_xy[1]

        await _broadcast(payload)


# ------------------ Lifecycle ------------------
@app.on_event("startup")
async def on_startup() -> None:
    global _main_loop, _new_raw_event
    _main_loop = asyncio.get_running_loop()
    _new_raw_event = asyncio.Event()
    kinect.start()
    asyncio.create_task(processor_loop())
    print(f"[bridge] canvas={CANVAS_W}x{CANVAS_H} ema={EMA_ALPHA} "
          f"pinch<={PINCH_DIST_M*100:.1f}cm  listening on ws://{HOST}:{PORT}/ws")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    kinect.stop()


# ------------------ WebSocket endpoint ------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        await ws.send_text(json.dumps({
            "op": "hello",
            "canvas": [CANVAS_W, CANVAS_H],
            "cal_ready": calibration.ready,
            "cal_mode": calibration.mode,
            "kinect_ok": kinect.started_ok,
        }))
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except Exception:  # noqa: BLE001
                continue
            op = msg.get("op")
            if op == "start_calibration":
                calibration.start()
                await _broadcast({"op": "cal_started", "corner": 0,
                                  "total": 4})
            elif op == "cancel_calibration":
                calibration.cancel()
                await _broadcast({"op": "cal_cancelled"})
            elif op == "reset_calibration":
                calibration.reset()
                await _broadcast({"op": "cal_reset"})
            elif op == "ping":
                await ws.send_text(json.dumps({"op": "pong", "t": time.time()}))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[bridge] ws handler error: {e}")
    finally:
        async with clients_lock:
            clients.discard(ws)


# ------------------ Health ------------------
@app.get("/")
async def root():
    return {
        "service": "nca-kinect-bridge",
        "canvas": [CANVAS_W, CANVAS_H],
        "cal_ready": calibration.ready,
        "kinect_ok": kinect.started_ok,
        "clients": len(clients),
    }


@app.get("/debug/color.jpg")
async def debug_color_jpg():
    jpeg = kinect.get_debug_jpeg()
    if not jpeg:
        return Response(status_code=404)
    return Response(content=jpeg, media_type="image/jpeg")
