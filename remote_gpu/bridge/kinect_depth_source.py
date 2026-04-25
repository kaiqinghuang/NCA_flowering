"""Unified Kinect v2 runtime: Body (for SDK debug) + Depth (everything else).

Opens **Body | Depth** so we can:

* Stream `HandTipRight` 3D position from the SDK skeleton — surfaced as a
  yellow marker in the depth-debug overlay (handy reference during
  calibration, but no longer used for the TV plane fit).
* Process the depth frame inside the calibrated TV interaction box to find
  a fingertip (point with smallest signed distance to the TV plane) — used
  at runtime to drive the canvas brush.
* Run a one-shot **auto TV calibration** on demand: capture ~35 depth
  frames, RANSAC-fit the dominant plane, isolate the largest co-planar
  blob (= the TV screen), derive its 4 rectangle corners, sort them
  TL/TR/BR/BL by (X, Z), and hand them off to the bridge for commit.

The RGB color stream is **not** opened (saves USB bandwidth and CPU; the
debug-color JPEG was removed in the depth-first redesign).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from .depth_processing import (
    analyze_depth_frame,
    auto_calibrate_tv_from_depth,
    render_depth_debug_bgr,
)
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

        # Auto-calibration trigger flag (set from any thread, consumed by _run).
        self._autocal_request = threading.Event()
        self._autocal_in_progress = False

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

    def request_auto_calibration(self) -> bool:
        """Schedule a one-shot TV plane auto-fit on the kinect thread.

        Returns False if a previous request is still being processed (the
        UI should debounce the button anyway).
        """
        if self._autocal_in_progress:
            return False
        self._autocal_request.set()
        return True

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
            # ----- One-shot auto-calibration request -----
            if self._autocal_request.is_set():
                self._autocal_request.clear()
                self._do_auto_calibration(runtime)

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

    def _do_auto_calibration(self, runtime) -> None:
        """Block the kinect thread briefly to capture a depth-frame stack and
        fit the TV plane + 4 corners. Pushes ``tv_autocalib_*`` messages back
        to the bridge via ``drain_pending_msgs``; the actual ``commit_auto``
        on the shared TVCalibration runs on the asyncio thread (see main.py)
        so we never race with the live frame loop.
        """
        self._autocal_in_progress = True
        try:
            # The "tv_autocalib_started" broadcast is sent synchronously from
            # main.py's WS handler so the user gets immediate feedback. We
            # do NOT push it from here because the processor loop only drains
            # pending messages when a new RawHandFrame arrives, and we are
            # about to block this thread for ~1.5s without emitting any.
            buf: list[np.ndarray] = []
            t_start = time.time()
            t_deadline = t_start + 1.6
            target_frames = 35
            while (
                not self._stop.is_set()
                and time.time() < t_deadline
                and len(buf) < target_frames
            ):
                if runtime.has_new_depth_frame():
                    d = runtime.get_last_depth_frame()
                    if d is not None:
                        buf.append(np.asarray(d, dtype=np.uint16).copy())
                else:
                    time.sleep(0.005)
            print(f"[kinect-depth] autocal: captured {len(buf)} depth frames in "
                  f"{time.time() - t_start:.2f}s")
            if len(buf) < 6:
                self._push_msg({
                    "op": "tv_autocalib_failed",
                    "reason": f"only captured {len(buf)} depth frames",
                })
                return

            try:
                # Env-var overrides let the operator tune behaviour without
                # touching code (matches knobs documented in README.md).
                eps_m = float(os.environ.get("BRIDGE_AUTOFIT_EPS_M", "0.015"))
                open_px = int(os.environ.get("BRIDGE_AUTOFIT_OPEN_PX", "3"))
                trim_frac = float(os.environ.get("BRIDGE_AUTOFIT_TRIM_FRAC", "0.78"))
                trim_min_m = float(os.environ.get("BRIDGE_AUTOFIT_TRIM_MIN_M", "0.10"))
                trim_bin_m = float(os.environ.get("BRIDGE_AUTOFIT_TRIM_BIN_M", "0.01"))
                result = auto_calibrate_tv_from_depth(
                    buf,
                    on_plane_eps_m=eps_m,
                    morph_open_px=open_px,
                    density_trim_frac=trim_frac,
                    density_trim_bin_m=trim_bin_m,
                    density_trim_min_run_m=trim_min_m,
                )
            except Exception as e:  # noqa: BLE001
                self._push_msg({
                    "op": "tv_autocalib_failed",
                    "reason": f"exception: {e}",
                })
                return

            if not result.get("ok"):
                self._push_msg({
                    "op": "tv_autocalib_failed",
                    "reason": result.get("reason", "unknown"),
                })
                return

            trim_info = result.get("trim_info") or {}
            trim_log = ""
            if trim_info.get("trimmed"):
                trim_log = (
                    f"  trim={trim_info.get('trim_w_m', 0.0) * 100:.1f}cm×"
                    f"{trim_info.get('trim_h_m', 0.0) * 100:.1f}cm"
                    f"  kept={trim_info.get('kept_w_m', 0.0):.3f}×"
                    f"{trim_info.get('kept_h_m', 0.0):.3f}m"
                )
            print(
                f"[kinect-depth] autocal: blob={result['n_blob_px']}px  "
                f"area={result['area_m2']:.3f}m²  "
                f"edges={result['edge_a_m']:.3f}×{result['edge_b_m']:.3f}m"
                f"{trim_log}"
            )
            self._push_msg({
                "op": "tv_autocalib_corners_ready",
                "corners_3d": result["corners_3d"],
                "plane": list(result["plane"]),
                "n_blob_px": int(result["n_blob_px"]),
                "area_m2": float(result["area_m2"]),
                "edge_a_m": float(result["edge_a_m"]),
                "edge_b_m": float(result["edge_b_m"]),
                "ransac_inlier_pts": int(result.get("ransac_inlier_pts", 0)),
                "trim_info": trim_info,
            })
        finally:
            self._autocal_in_progress = False

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
