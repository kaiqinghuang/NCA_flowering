"""Kinect v2 depth-only runtime.

Opens **Depth + Color** (no SDK skeleton / body source — we used to also
open Body for `HandTipRight`, but the bridge now derives the fingertip
purely from depth, so the body channel was costing CPU + bus bandwidth
for nothing). The two responsibilities are:

* **Live loop** — read a depth frame each iteration, analyse it inside
  the calibrated TV interaction box, find the fingertip (PCA-based
  geometric extremity disambiguated by mean signed distance to the
  plane), apply temporal EMA smoothing, and push a `RawHandFrame` to the
  bridge processor.
* **One-shot auto TV calibration** on demand — capture ~35 depth frames
  + 1 color frame, RANSAC-fit the dominant plane, isolate the largest
  co-planar blob, then use the SDK CoordinateMapper to look up the RGB
  colour of every blob pixel and discard anything that isn't near-black
  (the TV screen is dark, the surrounding wood slats are bright). The
  colour-refined blob's PCA + percentile fit gives the 4 corners (sorted
  A/B/C/D, mapped to canvas TL/TR/BR/BL by `tv_calibration`).

The Color stream costs ~30 MB/s on USB-3 but we don't process it during
the live loop — we only sample one frame during the calibration burst.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .depth_processing import (
    analyze_depth_frame,
    auto_calibrate_tv_from_depth,
    render_depth_debug_bgr,
)


# ----------------------------------------------------------------------
# Frame contract: one tick of fingertip tracking pushed to the bridge.
# Lean by design — depth-only, so no body / thumb / wrist / hand-state
# fields anymore.
# ----------------------------------------------------------------------
@dataclass
class RawHandFrame:
    t: float                              # wall-clock time when the frame was read
    tracked: bool                         # is a hand currently inside the interaction box
    hand_pos: Tuple[float, float, float]  # fingertip (x, y, z) in Kinect camera meters
    confident: bool                       # large enough hand blob (≥ 2 × min_box_px)
    pinch_dist_direct_m: float = -1.0     # fingertip → TV-plane signed distance (m); -1 if untracked


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
        # Morph-open kernel size used to clean the in-box mask before the
        # largest-CC pass — kills 1-px specks and breaks thin filaments
        # that would otherwise glue noise blobs onto the real hand. 0/1
        # disables it.
        self._noise_filter_px = int(
            os.environ.get("BRIDGE_DEBUG_NOISE_FILTER_PX", "3")
        )
        # Temporal EMA on the fingertip 3D position + image-space (u,v),
        # to suppress sub-frame jitter. α = 1.0 disables smoothing (raw),
        # lower α = more smoothing. 0.5 is a good "feels live but stable"
        # default at 30 Hz; drop to 0.3 if it still jitters.
        self._tip_ema_alpha = max(
            0.05,
            min(1.0, float(os.environ.get("BRIDGE_TIP_EMA_ALPHA", "0.5"))),
        )
        self._tip_ema_xyz: Optional[Tuple[float, float, float]] = None
        self._tip_ema_uv: Optional[Tuple[float, float]] = None
        self._tip_ema_t: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_ok = False

        self._debug_lock = threading.Lock()
        self._debug_depth_jpeg: Optional[bytes] = None

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
                FrameSourceTypes_Color,
                FrameSourceTypes_Depth,
            )
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
        print("[kinect-depth] Runtime started (depth + color, no body).")

        last_debug_push = 0.0

        while not self._stop.is_set():
            # ----- One-shot auto-calibration request -----
            if self._autocal_request.is_set():
                self._autocal_request.clear()
                self._do_auto_calibration(runtime)

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
                noise_filter_px=self._noise_filter_px,
            )

            now = time.time()
            res = analysis["result"]

            # Temporal EMA on the fingertip — only when actively tracked.
            # On loss-of-track we reset the EMA state so re-acquisition
            # snaps to the new position instead of dragging from the old
            # one.
            if res.tracked:
                a = self._tip_ema_alpha
                stale = (now - self._tip_ema_t) > 0.3
                if (
                    self._tip_ema_xyz is None
                    or self._tip_ema_uv is None
                    or stale
                    or a >= 1.0
                ):
                    sm_xyz = res.tip_xyz
                    sm_uv = res.debug_uv_tip
                else:
                    sm_xyz = (
                        a * res.tip_xyz[0] + (1.0 - a) * self._tip_ema_xyz[0],
                        a * res.tip_xyz[1] + (1.0 - a) * self._tip_ema_xyz[1],
                        a * res.tip_xyz[2] + (1.0 - a) * self._tip_ema_xyz[2],
                    )
                    sm_uv = (
                        a * res.debug_uv_tip[0] + (1.0 - a) * self._tip_ema_uv[0],
                        a * res.debug_uv_tip[1] + (1.0 - a) * self._tip_ema_uv[1],
                    )
                self._tip_ema_xyz = sm_xyz
                self._tip_ema_uv = sm_uv
                self._tip_ema_t = now

                # Mutate analysis["result"] so the debug overlay's red
                # square is drawn at the smoothed location too.
                from dataclasses import replace as _dc_replace
                analysis["result"] = _dc_replace(
                    res, tip_xyz=sm_xyz, debug_uv_tip=sm_uv,
                )
                emit_xyz = sm_xyz
                emit_dist = res.tip_signed_dist_m
                emit_conf = res.confident
            else:
                self._tip_ema_xyz = None
                self._tip_ema_uv = None
                emit_xyz = (0.0, 0.0, 0.0)
                emit_dist = -1.0
                emit_conf = False

            if now - last_debug_push > 0.07:
                last_debug_push = now
                self._update_depth_debug_jpeg(depth_flat, analysis)

            self._emit(RawHandFrame(
                t=now,
                tracked=res.tracked,
                hand_pos=emit_xyz,
                confident=emit_conf,
                pinch_dist_direct_m=emit_dist,
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
                t=time.time(),
                tracked=False,
                hand_pos=(0.0, 0.0, 0.0),
                confident=False,
                pinch_dist_direct_m=-1.0,
            ))
            time.sleep(0.1)

    def _do_auto_calibration(self, runtime) -> None:
        """Block the kinect thread briefly to capture a depth-frame stack and
        one color frame, then fit the TV plane + 4 corners with a colour
        refinement pass that excludes co-planar non-black structures (wood
        slats, bezels, etc).

        Pushes ``tv_autocalib_*`` messages back to the bridge via
        ``drain_pending_msgs``; the actual ``commit_auto`` on the shared
        TVCalibration runs on the asyncio thread (see main.py) so we never
        race with the live frame loop.
        """
        self._autocal_in_progress = True
        try:
            # The "tv_autocalib_started" broadcast is sent synchronously from
            # main.py's WS handler so the user gets immediate feedback. We
            # do NOT push it from here because the processor loop only drains
            # pending messages when a new RawHandFrame arrives, and we are
            # about to block this thread for ~1.5s without emitting any.
            buf: list[np.ndarray] = []
            last_color_bgr: Optional[np.ndarray] = None
            last_depth_for_mapping: Optional[np.ndarray] = None
            t_start = time.time()
            t_deadline = t_start + 1.6
            target_frames = 35
            while (
                not self._stop.is_set()
                and time.time() < t_deadline
                and len(buf) < target_frames
            ):
                got_anything = False
                if runtime.has_new_depth_frame():
                    d = runtime.get_last_depth_frame()
                    if d is not None:
                        df = np.asarray(d, dtype=np.uint16).copy()
                        buf.append(df)
                        last_depth_for_mapping = df  # remember for mapper
                        got_anything = True
                if runtime.has_new_color_frame():
                    c = runtime.get_last_color_frame()
                    if c is not None:
                        # Kinect v2 color frame: 1920x1080 BGRA uint8 (4ch).
                        try:
                            ca = np.asarray(c, dtype=np.uint8).reshape(1080, 1920, 4)
                            last_color_bgr = ca[:, :, :3].copy()  # drop alpha
                        except Exception:  # noqa: BLE001
                            pass
                        got_anything = True
                if not got_anything:
                    time.sleep(0.005)
            elapsed = time.time() - t_start
            print(
                f"[kinect-depth] autocal: captured {len(buf)} depth frames + "
                f"{'1' if last_color_bgr is not None else '0'} color in "
                f"{elapsed:.2f}s"
            )
            if len(buf) < 6:
                self._push_msg({
                    "op": "tv_autocalib_failed",
                    "reason": f"only captured {len(buf)} depth frames",
                })
                return

            # Compute depth->color mapping for the LAST captured depth frame
            # using the SDK's CoordinateMapper. Returns (DEPTH_H*DEPTH_W, 2)
            # float32 array of color-image (x, y) coordinates per depth pixel.
            depth_to_color_xy: Optional[np.ndarray] = None
            color_enable = os.environ.get("BRIDGE_AUTOFIT_COLOR_ENABLE", "1") not in ("0", "false", "False")
            if color_enable and last_color_bgr is not None and last_depth_for_mapping is not None:
                try:
                    depth_to_color_xy = self._map_depth_to_color(
                        runtime, last_depth_for_mapping
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[kinect-depth] CoordinateMapper failed: {e}")
                    depth_to_color_xy = None

            try:
                # Env-var overrides let the operator tune behaviour without
                # touching code (matches knobs documented in README.md).
                eps_m = float(os.environ.get("BRIDGE_AUTOFIT_EPS_M", "0.015"))
                open_px = int(os.environ.get("BRIDGE_AUTOFIT_OPEN_PX", "3"))
                color_max_v = float(os.environ.get("BRIDGE_AUTOFIT_COLOR_MAX_V", "90"))
                color_close_px = int(os.environ.get("BRIDGE_AUTOFIT_COLOR_CLOSE_PX", "3"))
                # Sides (left/right) = 0, back = 0, front = 1.5%.
                # Only the front edge of the TV picks up bezel / transition
                # pixels that survive the colour pass; the other three sides
                # are clean so we leave them tight to the blob.
                trim_pct = float(os.environ.get("BRIDGE_AUTOFIT_TRIM_PCT", "0"))
                tf_env = os.environ.get("BRIDGE_AUTOFIT_TRIM_FRONT_PCT", "1.5")
                tb_env = os.environ.get("BRIDGE_AUTOFIT_TRIM_BACK_PCT", "0")
                trim_front = float(tf_env) if tf_env != "" else None
                trim_back = float(tb_env) if tb_env != "" else None
                result = auto_calibrate_tv_from_depth(
                    buf,
                    color_bgr=last_color_bgr if color_enable else None,
                    depth_to_color_xy=depth_to_color_xy if color_enable else None,
                    on_plane_eps_m=eps_m,
                    morph_open_px=open_px,
                    color_max_v=color_max_v,
                    color_close_px=color_close_px,
                    trim_pct=trim_pct,
                    trim_pct_front=trim_front,
                    trim_pct_back=trim_back,
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

            color_info = result.get("color_info") or {}
            color_log = ""
            if color_info.get("color_refined"):
                color_log = (
                    f"  color: {color_info.get('n_blob_before', 0)}→"
                    f"{color_info.get('n_after_largest_cc', 0)}px "
                    f"(removed {color_info.get('removed_pct', 0.0):.1f}% as bright)"
                )
            elif color_info:
                color_log = f"  color: skipped ({color_info.get('reason', '?')})"
            tu = result.get("trim_used") or {}
            trim_log = (
                f"trim=sides{tu.get('sides', 0):.1f}/"
                f"front{tu.get('front', 0):.1f}/"
                f"back{tu.get('back', 0):.1f}% "
                f"({tu.get('fb_axis', '?')} axis = front-back)"
            )
            print(
                f"[kinect-depth] autocal: blob={result['n_blob_px']}px  "
                f"area={result['area_m2']:.3f}m²  "
                f"edges={result['edge_a_m']:.3f}×{result['edge_b_m']:.3f}m  "
                f"{trim_log}"
                f"{color_log}"
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
                "color_info": color_info,
            })
        finally:
            self._autocal_in_progress = False

    @staticmethod
    def _map_depth_to_color(runtime, depth_frame: np.ndarray) -> np.ndarray:
        """Use Kinect SDK CoordinateMapper to project every depth pixel to
        color-image coordinates.

        Args:
            runtime:      live ``PyKinectRuntime`` instance
            depth_frame:  uint16 depth frame, length DEPTH_W*DEPTH_H

        Returns:
            (DEPTH_H*DEPTH_W, 2) float32 array of (color_x, color_y).
            Pixels with no valid mapping have value -inf and should be
            filtered by the caller before sampling the color image.
        """
        import ctypes
        from pykinect2.PyKinectV2 import _ColorSpacePoint

        DEPTH_W, DEPTH_H = 512, 424
        L = DEPTH_W * DEPTH_H

        depth_arr = np.ascontiguousarray(depth_frame, dtype=np.uint16).ravel()
        depth_ptr = depth_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        csps = (_ColorSpacePoint * L)()
        runtime._mapper.MapDepthFrameToColorSpace(
            ctypes.c_uint(L), depth_ptr, ctypes.c_uint(L), csps
        )

        # _ColorSpacePoint is { float x; float y; } so the underlying buffer
        # is 2 * L float32. Copy out so it survives the ctypes lifetime.
        return np.frombuffer(csps, dtype=np.float32, count=L * 2).reshape(L, 2).copy()

    def _update_depth_debug_jpeg(
        self,
        depth_flat: np.ndarray,
        analysis: dict,
    ) -> None:
        try:
            import cv2

            bgr = render_depth_debug_bgr(
                depth_flat,
                self.tv_cal,
                analysis=analysis,
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
