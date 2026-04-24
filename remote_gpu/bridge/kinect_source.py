"""Kinect v2 body-tracking source, isolated in a daemon thread.

PyKinect2 is polling-based and Windows-only. We spin it in its own thread and
push each parsed `RawHandFrame` to a user-supplied callback. The asyncio side
of the bridge consumes those frames via a thread-safe hand-off.

On non-Windows (or when the import fails) we emit a warning and do nothing —
the rest of the bridge still runs (useful for dev machines), it just never
reports a hand frame.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


# Kinect HandState enum (PyKinectV2.HandState_*). Duplicated here so that the
# rest of the bridge doesn't need to import PyKinect2 (which would crash on
# macOS/Linux during unit tests).
HAND_STATE_UNKNOWN = 0
HAND_STATE_NOT_TRACKED = 1
HAND_STATE_OPEN = 2
HAND_STATE_CLOSED = 3
HAND_STATE_LASSO = 4


@dataclass
class RawHandFrame:
    """One tick of right-hand tracking straight from Kinect (meters, seconds)."""
    t: float                # wall-clock time when the frame was read
    tracked: bool           # is there a tracked body with a right hand at all
    hand_pos: tuple         # (x, y, z) of JointType_HandTipRight in Kinect frame
    thumb_pos: tuple        # (x, y, z) of JointType_ThumbRight
    wrist_pos: tuple        # (x, y, z) of JointType_WristRight
    hand_state: int         # HAND_STATE_* above (right hand)
    confident: bool         # both joints reported TrackingState_Tracked (strict)
    hand_color: tuple       # (x, y) in Kinect color image space
    thumb_color: tuple      # (x, y) in Kinect color image space
    wrist_color: tuple      # (x, y) in Kinect color image space
    # When set, `main.processor_loop` uses these instead of skeleton pinch logic.
    pinch_raw_direct: Optional[bool] = None
    pinch_dist_direct_m: Optional[float] = None


class KinectSource:
    def __init__(self, on_frame: Callable[[RawHandFrame], None]):
        self.on_frame = on_frame
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_ok = False  # set True once the Kinect runtime is up
        self._debug_jpeg: Optional[bytes] = None
        self._debug_lock = threading.Lock()
        self._debug_frame_idx = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="kinect")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def get_debug_jpeg(self) -> Optional[bytes]:
        with self._debug_lock:
            return self._debug_jpeg

    def get_debug_depth_jpeg(self) -> Optional[bytes]:
        """Body mode has no depth debug stream."""
        return None

    # -- internal --

    def _apply_pykinect_compat_shims(self):
        """Patch legacy PyKinect2 assumptions for modern Python/Numpy.

        Upstream PyKinect2 still references:
        - time.clock()  (removed in Python 3.8+)
        - numpy.object  (removed in NumPy 2.x)
        """
        # PyKinectRuntime uses time.clock() in its startup path.
        if not hasattr(time, "clock"):
            time.clock = time.perf_counter  # type: ignore[attr-defined]
        try:
            import numpy as np
            if "object" not in np.__dict__:
                np.object = object  # type: ignore[attr-defined]
        except Exception:
            # If numpy isn't importable here, PyKinect2 import will still
            # raise below and we'll surface that error as before.
            pass

    def _run(self):
        self._apply_pykinect_compat_shims()
        try:
            from pykinect2 import PyKinectV2  # noqa: F401
            from pykinect2.PyKinectV2 import (
                JointType_HandTipRight,
                JointType_ThumbRight,
                JointType_HandRight,
                JointType_WristRight,
                TrackingState_Tracked,
                FrameSourceTypes_Body,
                FrameSourceTypes_Color,
            )
            from pykinect2 import PyKinectRuntime
        except Exception as e:  # noqa: BLE001
            print(f"[kinect] PyKinect2 import failed: {e}")
            print("[kinect] Bridge will run without Kinect input. "
                  "Install PyKinect2 on a Windows machine to enable gestures.")
            self._emit_untracked_forever()
            return

        try:
            runtime = PyKinectRuntime.PyKinectRuntime(
                FrameSourceTypes_Body | FrameSourceTypes_Color
            )
        except Exception as e:  # noqa: BLE001
            print(f"[kinect] Failed to open Kinect runtime: {e}")
            self._emit_untracked_forever()
            return

        self.started_ok = True
        print("[kinect] Runtime started; waiting for body frames.")
        color_shape = (1080, 1920, 4)  # BGRA
        last_debug_push = 0.0

        while not self._stop.is_set():
            if runtime.has_new_color_frame():
                try:
                    cf = runtime.get_last_color_frame()
                    if cf is not None:
                        self._update_debug_jpeg(cf, color_shape, now=time.time(), last_ref=last_debug_push)
                        last_debug_push = time.time()
                except Exception:
                    pass

            if not runtime.has_new_body_frame():
                time.sleep(0.003)
                continue
            bodies = runtime.get_last_body_frame()
            if bodies is None:
                continue

            # Pick the first tracked body. (Future: prefer closest / specific id.)
            chosen = None
            for i in range(6):
                b = bodies.bodies[i]
                if b.is_tracked:
                    chosen = b
                    break

            now = time.time()
            if chosen is None:
                self._emit(RawHandFrame(
                    t=now, tracked=False,
                    hand_pos=(0.0, 0.0, 0.0), thumb_pos=(0.0, 0.0, 0.0),
                    wrist_pos=(0.0, 0.0, 0.0),
                    hand_state=HAND_STATE_NOT_TRACKED, confident=False,
                    hand_color=(-1.0, -1.0), thumb_color=(-1.0, -1.0), wrist_color=(-1.0, -1.0),
                ))
                continue

            joints = chosen.joints
            hand = joints[JointType_HandTipRight]
            thumb = joints[JointType_ThumbRight]
            wrist = joints[JointType_WristRight]
            hand_joint = joints[JointType_HandRight]
            confident = (
                hand.TrackingState == TrackingState_Tracked
                and thumb.TrackingState == TrackingState_Tracked
            )
            hp = (hand.Position.x, hand.Position.y, hand.Position.z)
            tp = (thumb.Position.x, thumb.Position.y, thumb.Position.z)
            wp = (wrist.Position.x, wrist.Position.y, wrist.Position.z)
            hc = (-1.0, -1.0)
            tc = (-1.0, -1.0)
            wc = (-1.0, -1.0)
            try:
                cpoints = runtime.body_joints_to_color_space(joints)
                hcp = cpoints[JointType_HandTipRight]
                tcp = cpoints[JointType_ThumbRight]
                wcp = cpoints[JointType_WristRight]
                hhp = cpoints[JointType_HandRight]
                hc = (float(hcp.x), float(hcp.y))
                tc = (float(tcp.x), float(tcp.y))
                wc = (float(wcp.x), float(wcp.y))
                # If HandTip is low confidence but HandRight is available, use it.
                if hc[0] < 0 and hhp is not None:
                    hc = (float(hhp.x), float(hhp.y))
            except Exception:
                pass
            try:
                state = int(chosen.hand_right_state)
            except Exception:  # noqa: BLE001
                state = HAND_STATE_UNKNOWN

            self._emit(RawHandFrame(
                t=now, tracked=True, hand_pos=hp, thumb_pos=tp,
                wrist_pos=wp,
                hand_state=state, confident=confident,
                hand_color=hc, thumb_color=tc, wrist_color=wc,
            ))

        try:
            runtime.close()
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, frame: RawHandFrame):
        try:
            self.on_frame(frame)
        except Exception as e:  # noqa: BLE001
            print(f"[kinect] on_frame callback raised: {e}")

    def _emit_untracked_forever(self):
        # Keeps the processor loop alive with "no hand" frames so the WS layer
        # can still report health status to any connected browser.
        while not self._stop.is_set():
            self._emit(RawHandFrame(
                t=time.time(), tracked=False,
                hand_pos=(0.0, 0.0, 0.0), thumb_pos=(0.0, 0.0, 0.0),
                wrist_pos=(0.0, 0.0, 0.0),
                hand_state=HAND_STATE_NOT_TRACKED, confident=False,
                hand_color=(-1.0, -1.0), thumb_color=(-1.0, -1.0), wrist_color=(-1.0, -1.0),
            ))
            time.sleep(0.1)

    def _update_debug_jpeg(self, flat_color_frame, color_shape, now: float, last_ref: float):
        # Keep debug feed lightweight: ~10 FPS JPEG preview is enough.
        if now - last_ref < 0.09:
            return
        try:
            import numpy as np
            import cv2
            arr = np.asarray(flat_color_frame, dtype=np.uint8).reshape(color_shape)
            bgr = arr[:, :, :3]  # BGRA -> BGR
            small = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
            self._debug_frame_idx += 1
            stamp = f"debug#{self._debug_frame_idx}  t={now:.3f}"
            cv2.rectangle(small, (8, 8), (320, 36), (0, 0, 0), -1)
            cv2.putText(
                small,
                stamp,
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 255, 120),
                1,
                cv2.LINE_AA,
            )
            ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                with self._debug_lock:
                    self._debug_jpeg = enc.tobytes()
        except Exception:
            # Debug view should never break hand tracking loop.
            return
