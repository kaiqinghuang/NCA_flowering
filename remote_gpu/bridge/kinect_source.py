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
    hand_state: int         # HAND_STATE_* above (right hand)
    confident: bool         # both joints reported TrackingState_Tracked (strict)


class KinectSource:
    def __init__(self, on_frame: Callable[[RawHandFrame], None]):
        self.on_frame = on_frame
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_ok = False  # set True once the Kinect runtime is up

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="kinect")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    # -- internal --

    def _run(self):
        try:
            from pykinect2 import PyKinectV2  # noqa: F401
            from pykinect2.PyKinectV2 import (
                JointType_HandTipRight,
                JointType_ThumbRight,
                TrackingState_Tracked,
                FrameSourceTypes_Body,
            )
            from pykinect2 import PyKinectRuntime
        except Exception as e:  # noqa: BLE001
            print(f"[kinect] PyKinect2 import failed: {e}")
            print("[kinect] Bridge will run without Kinect input. "
                  "Install PyKinect2 on a Windows machine to enable gestures.")
            self._emit_untracked_forever()
            return

        try:
            runtime = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Body)
        except Exception as e:  # noqa: BLE001
            print(f"[kinect] Failed to open Kinect runtime: {e}")
            self._emit_untracked_forever()
            return

        self.started_ok = True
        print("[kinect] Runtime started; waiting for body frames.")

        while not self._stop.is_set():
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
                    hand_state=HAND_STATE_NOT_TRACKED, confident=False,
                ))
                continue

            joints = chosen.joints
            hand = joints[JointType_HandTipRight]
            thumb = joints[JointType_ThumbRight]
            confident = (
                hand.TrackingState == TrackingState_Tracked
                and thumb.TrackingState == TrackingState_Tracked
            )
            hp = (hand.Position.x, hand.Position.y, hand.Position.z)
            tp = (thumb.Position.x, thumb.Position.y, thumb.Position.z)
            try:
                state = int(chosen.hand_right_state)
            except Exception:  # noqa: BLE001
                state = HAND_STATE_UNKNOWN

            self._emit(RawHandFrame(
                t=now, tracked=True, hand_pos=hp, thumb_pos=tp,
                hand_state=state, confident=confident,
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
                hand_state=HAND_STATE_NOT_TRACKED, confident=False,
            ))
            time.sleep(0.1)
