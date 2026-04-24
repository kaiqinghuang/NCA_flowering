"""Kinect v2 depth-based hand blob → fingertip (no body skeleton).

Requires a fitted TV plane (`plane.json`) for best segmentation; see
`PlaneCalibration` + WS op `capture_tv_plane`.

Debug preview still uses color when available.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from .depth_processing import process_depth_frame, render_depth_debug_bgr
from .kinect_source import HAND_STATE_NOT_TRACKED, HAND_STATE_OPEN, RawHandFrame
from .plane_calibration import PlaneCalibration, PlaneModel


class KinectDepthSource:
    def __init__(
        self,
        on_frame: Callable[[RawHandFrame], None],
        plane_cal: PlaneCalibration,
        band_min_m: float = 0.02,
        band_max_m: float = 0.40,
        pinch_eigen_ratio: float = 2.8,
    ):
        self.on_frame = on_frame
        self.plane_cal = plane_cal
        self.band_min_m = band_min_m
        self.band_max_m = band_max_m
        self.pinch_eigen_ratio = pinch_eigen_ratio
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_ok = False
        self._debug_jpeg: Optional[bytes] = None
        self._debug_depth_jpeg: Optional[bytes] = None
        self._debug_lock = threading.Lock()
        self._debug_frame_idx = 0
        self._plane_capture = threading.Event()
        self._pending_msgs: list[dict] = []
        self._pending_lock = threading.Lock()
        self._plane_capture_buf: list[np.ndarray] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="kinect-depth")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_debug_jpeg(self) -> Optional[bytes]:
        with self._debug_lock:
            return self._debug_jpeg

    def get_debug_depth_jpeg(self) -> Optional[bytes]:
        with self._debug_lock:
            return self._debug_depth_jpeg

    def request_plane_capture(self) -> None:
        """Signal the depth thread to grab ~1s of frames and fit / save the plane."""
        self._plane_capture_buf.clear()
        self._plane_capture.set()

    def drain_pending_msgs(self) -> list[dict]:
        with self._pending_lock:
            out = self._pending_msgs[:]
            self._pending_msgs.clear()
            return out

    def _push_msg(self, msg: dict) -> None:
        with self._pending_lock:
            self._pending_msgs.append(msg)

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
            from pykinect2.PyKinectV2 import FrameSourceTypes_Color, FrameSourceTypes_Depth
            from pykinect2 import PyKinectRuntime
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-depth] PyKinect2 import failed: {e}")
            self._emit_untracked_forever()
            return

        try:
            runtime = PyKinectRuntime.PyKinectRuntime(
                FrameSourceTypes_Depth | FrameSourceTypes_Color
            )
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-depth] Failed to open Kinect runtime: {e}")
            self._emit_untracked_forever()
            return

        self.started_ok = True
        print("[kinect-depth] Runtime started (depth + color).")
        color_shape = (1080, 1920, 4)
        last_debug_push = 0.0
        last_depth_debug_push = 0.0

        while not self._stop.is_set():
            if self._plane_capture.is_set():
                self._do_plane_capture(runtime)
                self._plane_capture.clear()

            if runtime.has_new_color_frame():
                try:
                    cf = runtime.get_last_color_frame()
                    if cf is not None:
                        self._update_debug_jpeg(cf, color_shape, time.time(), last_debug_push)
                        last_debug_push = time.time()
                except Exception:
                    pass

            if not runtime.has_new_depth_frame():
                time.sleep(0.002)
                continue

            depth = runtime.get_last_depth_frame()
            if depth is None:
                continue

            depth_flat = np.asarray(depth, dtype=np.uint16).ravel()
            plane: Optional[PlaneModel] = self.plane_cal.plane
            res = process_depth_frame(
                depth_flat,
                plane,
                band_min_m=self.band_min_m,
                band_max_m=self.band_max_m,
                pinch_eigen_ratio_thresh=self.pinch_eigen_ratio,
            )

            now = time.time()
            if now - last_depth_debug_push > 0.07:
                last_depth_debug_push = now
                self._update_depth_debug_jpeg(depth_flat, plane, res)

            if not res.tracked:
                self._emit(RawHandFrame(
                    t=now, tracked=False,
                    hand_pos=(0.0, 0.0, 0.0), thumb_pos=(0.0, 0.0, 0.0),
                    wrist_pos=(0.0, 0.0, 0.0),
                    hand_state=HAND_STATE_NOT_TRACKED,
                    confident=False,
                    hand_color=(-1.0, -1.0), thumb_color=(-1.0, -1.0), wrist_color=(-1.0, -1.0),
                ))
                continue

            tip = res.tip_xyz
            thumb = res.thumb_proxy_xyz
            # Map depth (u,v) to fake "color" coords for debug overlay scaling in client.
            du, dv = res.debug_uv_tip
            hc = (du * (1920.0 / 512.0), dv * (1080.0 / 424.0)) if du >= 0 else (-1.0, -1.0)
            tc = hc
            wc = hc

            self._emit(RawHandFrame(
                t=now, tracked=True,
                hand_pos=tip, thumb_pos=thumb, wrist_pos=thumb,
                hand_state=HAND_STATE_OPEN,
                confident=res.confident,
                hand_color=hc, thumb_color=tc, wrist_color=wc,
                pinch_raw_direct=res.pinch_raw,
                pinch_dist_direct_m=res.pinch_dist_m,
            ))

        try:
            runtime.close()
        except Exception:  # noqa: BLE001
            pass

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

    def _do_plane_capture(self, runtime) -> None:
        """Grab depth frames without requiring a new PyKinect API."""
        buf: list[np.ndarray] = []
        t0 = time.time()
        while time.time() - t0 < 1.2 and len(buf) < 35 and not self._stop.is_set():
            if runtime.has_new_depth_frame():
                d = runtime.get_last_depth_frame()
                if d is not None:
                    buf.append(np.asarray(d, dtype=np.uint16).copy())
            time.sleep(0.028)
        if len(buf) < 8:
            self._push_msg({"op": "plane_capture_failed", "reason": "not enough depth frames"})
            return
        from .depth_processing import fit_plane_from_depth_stack

        plane = fit_plane_from_depth_stack(buf)
        if plane is None:
            self._push_msg({"op": "plane_capture_failed", "reason": "ransac failed"})
            return
        self.plane_cal.save(plane)
        self._push_msg({"op": "plane_ready", "a": plane.a, "b": plane.b, "c": plane.c, "d": plane.d})

    def _update_debug_jpeg(
        self, flat_color_frame, color_shape, now: float, last_ref: float,
    ) -> None:
        if now - last_ref < 0.09:
            return
        try:
            import cv2
            arr = np.asarray(flat_color_frame, dtype=np.uint8).reshape(color_shape)
            bgr = arr[:, :, :3]
            small = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
            self._debug_frame_idx += 1
            stamp = f"depth#{self._debug_frame_idx}  plane={'ok' if self.plane_cal.ready else 'no'}"
            cv2.rectangle(small, (8, 8), (420, 36), (0, 0, 0), -1)
            cv2.putText(
                small, stamp, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (40, 255, 120), 1, cv2.LINE_AA,
            )
            ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ok:
                with self._debug_lock:
                    self._debug_jpeg = enc.tobytes()
        except Exception:
            return

    def _update_depth_debug_jpeg(
        self,
        depth_flat: np.ndarray,
        plane: Optional[PlaneModel],
        res,
    ) -> None:
        try:
            import cv2

            bgr = render_depth_debug_bgr(
                depth_flat, plane, self.band_min_m, self.band_max_m, res,
            )
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                with self._debug_lock:
                    self._debug_depth_jpeg = enc.tobytes()
        except Exception:
            return
