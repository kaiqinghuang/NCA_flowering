"""Local Kinect→browser bridge.

Run on the Windows laptop that has the Kinect:

    python -m uvicorn bridge.main:app --host 0.0.0.0 --port 7000

Browser → Bridge (JSON text):
    {"op": "tv_calib_start"}              # enter 4-corner wizard
    {"op": "tv_calib_confirm"}            # commit current corner = latest HandTipRight
    {"op": "tv_calib_redo"}               # discard last captured corner
    {"op": "tv_calib_cancel"}             # exit wizard without saving
    {"op": "tv_calib_reset"}              # delete saved tv_calibration.json

Bridge → Browser (JSON text, broadcast):
    {"op": "hello", "canvas": [W, H], "kinect_ok": bool, "kinect_mode": "...",
     "tv_ready": bool, "tv_status": {...}}
    {"op": "hand", "t": ..., "tracked": bool, "confident": bool,
     "tv_ready": bool, "tv_status": {...},
     "body_tip_xyz": [x,y,z]?, "body_tracked": bool,
     "cx": float?, "cy": float?,           # only when tv_ready AND tracked
     "pinch": bool}                        # true while a fingertip is in the box
    {"op": "tv_calib_started"} {"op": "tv_calib_progress", ...}
    {"op": "tv_calib_done", "ready": bool, "reason": str}
    {"op": "tv_calib_cancelled"} {"op": "tv_calib_reset"}
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

from .kinect_source import KinectSource, RawHandFrame
from .tv_calibration import TVCalibration


# ------------------ Config ------------------
HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("BRIDGE_PORT", "7000"))
CANVAS_W = int(os.environ.get("BRIDGE_CANVAS_W", "960"))
CANVAS_H = int(os.environ.get("BRIDGE_CANVAS_H", "540"))
BROADCAST_HZ = float(os.environ.get("BRIDGE_BROADCAST_HZ", "30"))
KINECT_MODE = os.environ.get("BRIDGE_KINECT_MODE", "depth").strip().lower()
DEPTH_BAND_MIN_M = float(os.environ.get("BRIDGE_DEPTH_BAND_MIN_M", "0.02"))
DEPTH_BAND_MAX_M = float(os.environ.get("BRIDGE_DEPTH_BAND_MAX_M", "0.45"))

SCRIPT_DIR = Path(__file__).parent.resolve()
TV_CAL_PATH = str(SCRIPT_DIR / "tv_calibration.json")


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

_latest_raw: Optional[RawHandFrame] = None
_raw_lock = threading.Lock()
_main_loop: Optional[asyncio.AbstractEventLoop] = None
_new_raw_event: Optional[asyncio.Event] = None

tv_calibration = TVCalibration(TV_CAL_PATH, (CANVAS_W, CANVAS_H))


def _on_kinect_frame(frame: RawHandFrame) -> None:
    """Called from the Kinect thread."""
    global _latest_raw
    with _raw_lock:
        _latest_raw = frame
    if _main_loop is not None and _new_raw_event is not None:
        _main_loop.call_soon_threadsafe(_new_raw_event.set)


if KINECT_MODE == "depth":
    from .kinect_depth_source import KinectDepthSource

    kinect = KinectDepthSource(
        _on_kinect_frame,
        tv_calibration,
        box_near_m=DEPTH_BAND_MIN_M,
        box_far_m=DEPTH_BAND_MAX_M,
    )
    print("[bridge] Kinect mode: depth (body+depth, depth fingertip)")
else:
    kinect = KinectSource(_on_kinect_frame)
    print("[bridge] Kinect mode: body (SDK joints only)")


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


def _safe_xyz(t):
    if t is None:
        return None
    try:
        return [float(t[0]), float(t[1]), float(t[2])]
    except Exception:  # noqa: BLE001
        return None


# ------------------ Processor loop ------------------
async def processor_loop() -> None:
    """Translate raw Kinect frames → broadcast hand events."""
    assert _new_raw_event is not None
    interval = 1.0 / max(1.0, BROADCAST_HZ)
    last_send = 0.0

    while True:
        await _new_raw_event.wait()
        _new_raw_event.clear()

        with _raw_lock:
            raw = _latest_raw
        if raw is None:
            continue

        now = time.perf_counter()
        if now - last_send < interval * 0.9:
            continue
        last_send = now

        tracked = bool(raw.tracked)
        cx_canvas: Optional[float] = None
        cy_canvas: Optional[float] = None
        if tracked and tv_calibration.ready:
            mapped = tv_calibration.project_xyz_to_canvas(raw.hand_pos)
            if mapped is not None:
                cx_canvas, cy_canvas = mapped
                cx_canvas = max(0.0, min(CANVAS_W - 1.0, cx_canvas))
                cy_canvas = max(0.0, min(CANVAS_H - 1.0, cy_canvas))

        body_tip = _safe_xyz(raw.body_tip_xyz)
        body_tracked = bool(raw.body_tracked) if raw.body_tracked is not None else None

        payload = {
            "op": "hand",
            "t": raw.t,
            "tracked": tracked,
            "confident": bool(raw.confident),
            "tv_ready": tv_calibration.ready,
            "tv_status": tv_calibration.status_dict(),
            "body_tip_xyz": body_tip,
            "body_tracked": body_tracked,
            "tip_signed_dist_m": (
                float(raw.pinch_dist_direct_m)
                if raw.pinch_dist_direct_m is not None and raw.pinch_dist_direct_m >= 0
                else None
            ),
        }
        if cx_canvas is not None and cy_canvas is not None:
            payload["cx"] = cx_canvas
            payload["cy"] = cy_canvas
            payload["pinch"] = True   # continuous draw whenever fingertip is in box
        else:
            payload["pinch"] = False

        await _broadcast(payload)

        drain = getattr(kinect, "drain_pending_msgs", None)
        if callable(drain):
            for extra in drain():
                await _broadcast(extra)


# ------------------ Lifecycle ------------------
@app.on_event("startup")
async def on_startup() -> None:
    global _main_loop, _new_raw_event
    _main_loop = asyncio.get_running_loop()
    _new_raw_event = asyncio.Event()
    kinect.start()
    asyncio.create_task(processor_loop())
    print(
        f"[bridge] canvas={CANVAS_W}x{CANVAS_H}  mode={KINECT_MODE}  "
        f"box=[{DEPTH_BAND_MIN_M:.02f},{DEPTH_BAND_MAX_M:.02f}]m  "
        f"tv_ready={'yes' if tv_calibration.ready else 'no'}  "
        f"listening on ws://{HOST}:{PORT}/ws"
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    kinect.stop()


# ------------------ WebSocket endpoint ------------------
async def _broadcast_status(extra: Optional[dict] = None) -> None:
    msg = {
        "op": "tv_calib_progress",
        "status": tv_calibration.status_dict(),
        "tv_ready": tv_calibration.ready,
    }
    if extra:
        msg.update(extra)
    await _broadcast(msg)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        await ws.send_text(json.dumps({
            "op": "hello",
            "canvas": [CANVAS_W, CANVAS_H],
            "kinect_ok": kinect.started_ok,
            "kinect_mode": KINECT_MODE,
            "tv_ready": tv_calibration.ready,
            "tv_status": tv_calibration.status_dict(),
            "box_near_m": DEPTH_BAND_MIN_M,
            "box_far_m": DEPTH_BAND_MAX_M,
        }))
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except Exception:  # noqa: BLE001
                continue
            op = msg.get("op")

            if op == "tv_calib_start":
                tv_calibration.start()
                await _broadcast({
                    "op": "tv_calib_started",
                    "status": tv_calibration.status_dict(),
                })

            elif op == "tv_calib_confirm":
                getter = getattr(kinect, "get_latest_body_tip", None)
                tip = getter() if callable(getter) else None
                res = tv_calibration.confirm(tip)
                await _broadcast_status({
                    "op": "tv_calib_progress",
                    "result": res,
                    "captured": len(tv_calibration.captured),
                })
                if res.get("done"):
                    await _broadcast({
                        "op": "tv_calib_done",
                        "ready": tv_calibration.ready,
                        "reason": res.get("reason", ""),
                        "status": tv_calibration.status_dict(),
                    })

            elif op == "tv_calib_redo":
                res = tv_calibration.redo()
                await _broadcast_status({"op": "tv_calib_progress", "result": res})

            elif op == "tv_calib_cancel":
                tv_calibration.cancel()
                await _broadcast({
                    "op": "tv_calib_cancelled",
                    "status": tv_calibration.status_dict(),
                })

            elif op == "tv_calib_reset":
                tv_calibration.reset()
                await _broadcast({
                    "op": "tv_calib_reset",
                    "status": tv_calibration.status_dict(),
                    "tv_ready": False,
                })

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
        "kinect_ok": kinect.started_ok,
        "kinect_mode": KINECT_MODE,
        "tv_ready": tv_calibration.ready,
        "tv_status": tv_calibration.status_dict(),
        "clients": len(clients),
    }


@app.get("/debug/depth.jpg")
async def debug_depth_jpg():
    """Depth-mode only: TV polygon + interaction box + fingertip overlay (~14 Hz)."""
    getter = getattr(kinect, "get_debug_depth_jpeg", None)
    jpeg = getter() if callable(getter) else None
    if not jpeg:
        return Response(
            status_code=404,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
