"""Kinect v2 body-tracking source (legacy "body" mode), isolated in a daemon thread.

PyKinect2 is polling-based and Windows-only. We spin it in its own thread and
push each parsed `RawHandFrame` to a user-supplied callback. The asyncio side
of the bridge consumes those frames via a thread-safe hand-off.

On non-Windows (or when the import fails) we emit a warning and do nothing —
the rest of the bridge still runs (useful for dev machines), it just never
reports a hand frame.

NOTE: As of the depth-first redesign, this body-only mode is kept around for
debugging the SDK in isolation; the default runtime is :class:`KinectDepthSource`,
which opens both body and depth and uses depth for fingertip detection. The
old RGB color debug stream has been removed from both sources.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


HAND_STATE_UNKNOWN = 0
HAND_STATE_NOT_TRACKED = 1
HAND_STATE_OPEN = 2
HAND_STATE_CLOSED = 3
HAND_STATE_LASSO = 4


@dataclass
class RawHandFrame:
    """One tick of right-hand tracking emitted to the bridge processor."""
    t: float                # wall-clock time when the frame was read
    tracked: bool           # is the *primary signal* (depth tip OR body tip) usable
    hand_pos: tuple         # (x, y, z) in Kinect camera meters — runtime fingertip
    thumb_pos: tuple        # (x, y, z) — body source: ThumbRight; depth source: centroid proxy
    wrist_pos: tuple        # (x, y, z) — body source: WristRight; depth source: centroid proxy
    hand_state: int         # HAND_STATE_*
    confident: bool         # source-specific confidence (joint TrackingState_Tracked / large blob)
    hand_color: tuple       # (x, y) in Kinect color image space (legacy; depth source: depth-pixel)
    thumb_color: tuple      # (x, y) in Kinect color image space (legacy)
    wrist_color: tuple      # (x, y) in Kinect color image space (legacy)
    # Direct pinch override from depth source. None = use raw skeleton-pinch path in main.py.
    pinch_raw_direct: Optional[bool] = None
    pinch_dist_direct_m: Optional[float] = None
    # SDK HandTipRight (always emitted by KinectDepthSource when body is tracked) —
    # main.py uses this for the manual TV calibration wizard.
    body_tip_xyz: Optional[tuple] = None      # (x, y, z) meters in camera frame
    body_tracked: Optional[bool] = None       # SDK's TrackingState_Tracked for HandTipRight


class KinectSource:
    """Legacy body-only source (no depth). Fires `RawHandFrame` from `HandTipRight`."""

    def __init__(self, on_frame: Callable[[RawHandFrame], None]):
        self.on_frame = on_frame
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_ok = False  # set True once the Kinect runtime is up

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="kinect-body")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    # Kept for API compatibility with main.py — body-only mode has no debug stream.
    def get_debug_depth_jpeg(self) -> Optional[bytes]:
        return None

    def get_latest_body_tip(self) -> Optional[tuple]:
        return None

    def drain_pending_msgs(self) -> list[dict]:
        return []

    def _apply_pykinect_compat_shims(self) -> None:
        """Patch legacy PyKinect2 assumptions for modern Python/Numpy."""
        if not hasattr(time, "clock"):
            time.clock = time.perf_counter  # type: ignore[attr-defined]
        try:
            import numpy as np
            if "object" not in np.__dict__:
                np.object = object  # type: ignore[attr-defined]
        except Exception:
            pass

    def _run(self) -> None:
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
            )
            from pykinect2 import PyKinectRuntime
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-body] PyKinect2 import failed: {e}")
            print("[kinect-body] Bridge will run without Kinect input.")
            self._emit_untracked_forever()
            return

        try:
            runtime = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Body)
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-body] Failed to open Kinect runtime: {e}")
            self._emit_untracked_forever()
            return

        self.started_ok = True
        print("[kinect-body] Runtime started; waiting for body frames.")

        while not self._stop.is_set():
            if not runtime.has_new_body_frame():
                time.sleep(0.003)
                continue
            bodies = runtime.get_last_body_frame()
            if bodies is None:
                continue

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
            confident = (
                hand.TrackingState == TrackingState_Tracked
                and thumb.TrackingState == TrackingState_Tracked
            )
            hp = (hand.Position.x, hand.Position.y, hand.Position.z)
            tp = (thumb.Position.x, thumb.Position.y, thumb.Position.z)
            wp = (wrist.Position.x, wrist.Position.y, wrist.Position.z)
            try:
                state = int(chosen.hand_right_state)
            except Exception:  # noqa: BLE001
                state = HAND_STATE_UNKNOWN

            self._emit(RawHandFrame(
                t=now, tracked=True, hand_pos=hp, thumb_pos=tp,
                wrist_pos=wp,
                hand_state=state, confident=confident,
                hand_color=(-1.0, -1.0), thumb_color=(-1.0, -1.0), wrist_color=(-1.0, -1.0),
                body_tip_xyz=hp,
                body_tracked=hand.TrackingState == TrackingState_Tracked,
            ))

        try:
            runtime.close()
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, frame: RawHandFrame) -> None:
        try:
            self.on_frame(frame)
        except Exception as e:  # noqa: BLE001
            print(f"[kinect-body] on_frame callback raised: {e}")

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
