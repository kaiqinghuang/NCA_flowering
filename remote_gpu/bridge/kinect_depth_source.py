"""Unified Kinect v2 runtime: Body (for TV calibration) + Depth (for runtime).

Opens **Body | Depth** so we can:

* Stream `HandTipRight` 3D position from the SDK skeleton — used by the manual
  4-corner TV calibration wizard (one click per corner).
* Process the depth frame inside the calibrated TV interaction box to find a
  fingertip (point with smallest signed distance to the TV plane) — used at
  runtime to drive the canvas brush.

The RGB color stream is **not** opened (saves USB bandwidth and CPU; the old
debug-color JPEG was removed in the depth-first redesign).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from .depth_processing import analyze_depth_frame, render_depth_debug_bgr
from .kinect_source import HAND_STATE_NOT_TRACKED, HAND_STATE_OPEN, RawHandFrame


class KinectDepthSource:
    def __init__(
        self,
        on_frame: Callable[[RawHandFrame], None],
        tv_cal,                        # TVCalibration instance — shared with main.py
        box_near_m: float = 0.02,
        box_far_m: float = 0.45,
        surface_eps_m: Optional[float] = None,
    ):
        self.on_frame = on_frame
        self.tv_cal = tv_cal
        self.box_near_m = float(box_near_m)
        self.box_far_m = float(box_far_m)
        self._surface_eps_m = (
            float(surface_eps_m)
            if surface_eps_m is not None
            else float(os.environ.get("BRIDGE_DEBUG_SURFACE_EPS_M", "0.03"))
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_ok = False

        self._debug_lock = threading.Lock()
        self._debug_depth_jpeg: Optional[bytes] = None

        self._body_lock = threading.Lock()
        self._latest_body_tip: Optional[tuple] = None
        self._latest_body_tracked: bool = False
        self._latest_body_t: float = 0.0

        self._pending_lock = threading.Lock()
        self._pending_msgs: list[dict] = []

    # ---------- public API ----------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="kinect-depth")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_debug_depth_jpeg(self) -> Optional[bytes]:
        with self._debug_lock:
            return self._debug_depth_jpeg

    def get_latest_body_tip(self) -> Optional[tuple]:
        """Most recent SDK HandTipRight 3D position (or None if no body tracked)."""
        with self._body_lock:
            if self._latest_body_tip is None:
                return None
            # Stale guard: drop if older than ~0.4s
            if time.time() - self._latest_body_t > 0.4:
                return None
            return tuple(self._latest_body_tip)

    def drain_pending_msgs(self) -> list[dict]:
        with self._pending_lock:
            out = self._pending_msgs[:]
            self._pending_msgs.clear()
            return out

    def _push_msg(self, msg: dict) -> None:
        with self._pending_lock:
            self._pending_msgs.append(msg)

    # ---------- runtime thread ----------
    def _apply_pykinect_compat_shims(self) -> None:
        if not hasattr(time, "clock"):
            time.clock = time.perf_counter  # type: ignore[attr-defined]
        try:
            if "object" not in np.__dict__:
                np.object = object  # type: ignore[attr-defined]
        except Exception:
            pass

    def _run(self) -> None:
        self._apply_pykinect_compat_shims()
        try:
            from pykinect2.PyKinectV2 import (
                FrameSourceTypes_Body,
                FrameSourceTypes_Depth,
                JointType_HandTipRight,
                TrackingState_Tracked,
            )
            from pykinect2 import PyKinectRuntime
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-depth] PyKinect2 import failed: {e}")
            self._emit_untracked_forever()
            return

        try:
            runtime = PyKinectRuntime.PyKinectRuntime(
                FrameSourceTypes_Body | FrameSourceTypes_Depth
            )
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-depth] Failed to open Kinect runtime: {e}")
            self._emit_untracked_forever()
            return

        self.started_ok = True
        print("[kinect-depth] Runtime started (body + depth).")

        last_debug_push = 0.0

        while not self._stop.is_set():
            # ----- Body frame (cheap, run every loop iter when present) -----
            if runtime.has_new_body_frame():
                try:
                    bodies = runtime.get_last_body_frame()
                    if bodies is not None:
                        tip_xyz: Optional[tuple] = None
                        tip_tracked = False
                        for i in range(6):
                            b = bodies.bodies[i]
                            if not b.is_tracked:
                                continue
                            j = b.joints[JointType_HandTipRight]
                            # Kinect SDK CameraSpacePoint uses Y-up; the rest of
                            # the bridge (depth-pixel projection, plane fit,
                            # signed-distance test) uses standard CV Y-down.
                            # Flip Y once here so everything downstream is
                            # consistent.
                            tip_xyz = (
                                float(j.Position.x),
                                -float(j.Position.y),
                                float(j.Position.z),
                            )
                            tip_tracked = j.TrackingState == TrackingState_Tracked
                            break
                        with self._body_lock:
                            self._latest_body_tip = tip_xyz
                            self._latest_body_tracked = tip_tracked
                            self._latest_body_t = time.time()
                except Exception:  # noqa: BLE001
                    pass

            # ----- Depth frame (drives RawHandFrame emission) -----
            if not runtime.has_new_depth_frame():
                time.sleep(0.002)
                continue

            depth = runtime.get_last_depth_frame()
            if depth is None:
                continue

            depth_flat = np.asarray(depth, dtype=np.uint16).ravel()

            analysis = analyze_depth_frame(
                depth_flat,
                self.tv_cal,
                box_near_m=self.box_near_m,
                box_far_m=self.box_far_m,
                surface_eps_m=self._surface_eps_m,
            )

            with self._body_lock:
                body_tip_snapshot = self._latest_body_tip
                body_tracked_snapshot = self._latest_body_tracked

            now = time.time()
            if now - last_debug_push > 0.07:
                last_debug_push = now
                self._update_depth_debug_jpeg(depth_flat, analysis, body_tip_snapshot)

            res = analysis["result"]
            if res.tracked:
                self._emit(RawHandFrame(
                    t=now,
                    tracked=True,
                    hand_pos=res.tip_xyz,
                    thumb_pos=res.tip_xyz,
                    wrist_pos=res.tip_xyz,
                    hand_state=HAND_STATE_OPEN,
                    confident=res.confident,
                    hand_color=(res.debug_uv_tip[0], res.debug_uv_tip[1]),
                    thumb_color=(-1.0, -1.0),
                    wrist_color=(-1.0, -1.0),
                    pinch_raw_direct=True,            # always "drawing" while in interaction box
                    pinch_dist_direct_m=res.tip_signed_dist_m,
                    body_tip_xyz=body_tip_snapshot,
                    body_tracked=body_tracked_snapshot,
                ))
            else:
                self._emit(RawHandFrame(
                    t=now,
                    tracked=False,
                    hand_pos=(0.0, 0.0, 0.0),
                    thumb_pos=(0.0, 0.0, 0.0),
                    wrist_pos=(0.0, 0.0, 0.0),
                    hand_state=HAND_STATE_NOT_TRACKED,
                    confident=False,
                    hand_color=(-1.0, -1.0),
                    thumb_color=(-1.0, -1.0),
                    wrist_color=(-1.0, -1.0),
                    pinch_raw_direct=False,
                    pinch_dist_direct_m=-1.0,
                    body_tip_xyz=body_tip_snapshot,
                    body_tracked=body_tracked_snapshot,
                ))

        try:
            runtime.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------- helpers ----------
    def _emit(self, frame: RawHandFrame) -> None:
        try:
            self.on_frame(frame)
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-depth] on_frame callback raised: {e}")

    def _emit_untracked_forever(self) -> None:
        while not self._stop.is_set():
            self._emit(RawHandFrame(
                t=time.time(), tracked=False,
                hand_pos=(0.0, 0.0, 0.0), thumb_pos=(0.0, 0.0, 0.0),
                wrist_pos=(0.0, 0.0, 0.0),
                hand_state=HAND_STATE_NOT_TRACKED, confident=False,
                hand_color=(-1.0, -1.0), thumb_color=(-1.0, -1.0), wrist_color=(-1.0, -1.0),
            ))
            time.sleep(0.1)

    def _update_depth_debug_jpeg(
        self,
        depth_flat: np.ndarray,
        analysis: dict,
        body_tip_xyz: Optional[tuple],
    ) -> None:
        try:
            import cv2

            bgr = render_depth_debug_bgr(
                depth_flat,
                self.tv_cal,
                analysis=analysis,
                body_tip_xyz=body_tip_xyz,
                box_near_m=self.box_near_m,
                box_far_m=self.box_far_m,
                surface_eps_m=self._surface_eps_m,
            )
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                with self._debug_lock:
                    self._debug_depth_jpeg = enc.tobytes()
        except Exception:  # noqa: BLE001
            return
