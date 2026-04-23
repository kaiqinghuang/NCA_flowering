"""Pinch detection + EMA smoothing for the Kinect hand stream.

Two independent jobs:
  * Decide whether the current frame counts as a pinch ("is user pinching
    right now"). Source signals: Kinect's HandState classifier + the 3D
    euclidean distance between the HandTip and Thumb joints. OR-fused with
    a tiny temporal debounce to kill single-frame flips.
  * Smooth canvas (x, y) with an EMA low-pass so the displayed cursor
    doesn't jitter. Resets whenever tracking drops, so the cursor doesn't
    "snap" across the canvas when the hand re-enters the FOV.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from .kinect_source import HAND_STATE_CLOSED


class GestureProcessor:
    def __init__(
        self,
        ema_alpha: float = 0.35,
        pinch_dist_m: float = 0.03,
        debounce_frames: int = 1,
    ):
        self.ema_alpha = float(ema_alpha)
        self.pinch_dist_m = float(pinch_dist_m)
        self.debounce_frames = int(debounce_frames)

        self._ema_cx: Optional[float] = None
        self._ema_cy: Optional[float] = None
        self._raw_history: list[bool] = []
        self._stable_pinch = False

    def reset_xy(self) -> None:
        """Call when tracking drops. Next update() starts EMA from scratch."""
        self._ema_cx = None
        self._ema_cy = None

    def reset_pinch(self) -> None:
        self._raw_history = []
        self._stable_pinch = False

    def raw_pinch(
        self,
        hand_pos: Tuple[float, float, float],
        thumb_pos: Tuple[float, float, float],
        hand_state: int,
        confident: bool,
    ) -> bool:
        """Single-frame pinch signal. Conservative: requires joint confidence."""
        if not confident:
            return False
        dx = hand_pos[0] - thumb_pos[0]
        dy = hand_pos[1] - thumb_pos[1]
        dz = hand_pos[2] - thumb_pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        closed = hand_state == HAND_STATE_CLOSED
        # Either the SDK says closed, OR the fingers are physically close.
        # Having two signals in OR is robust against the side-view
        # classifier sometimes misreading a real pinch as Open.
        return closed or dist < self.pinch_dist_m

    def update(
        self,
        cx: float,
        cy: float,
        raw_pinch_now: bool,
    ) -> Tuple[Tuple[float, float], bool]:
        """Push one valid frame. Returns (smoothed xy, debounced pinch)."""
        a = self.ema_alpha
        if self._ema_cx is None:
            self._ema_cx = cx
            self._ema_cy = cy
        else:
            self._ema_cx = a * cx + (1.0 - a) * self._ema_cx
            self._ema_cy = a * cy + (1.0 - a) * self._ema_cy

        # Debounce pinch: needs `debounce_frames + 1` consecutive agreeing
        # frames to flip state. Default = 2 frames (~66ms at 30Hz) → kills
        # single-tick noise without adding noticeable lag.
        self._raw_history.append(raw_pinch_now)
        window = self.debounce_frames + 1
        if len(self._raw_history) > window:
            self._raw_history = self._raw_history[-window:]

        if len(self._raw_history) >= window:
            if all(self._raw_history):
                self._stable_pinch = True
            elif not any(self._raw_history):
                self._stable_pinch = False
            # else: mixed frames → keep previous stable state (hysteresis)

        return (self._ema_cx, self._ema_cy), self._stable_pinch
