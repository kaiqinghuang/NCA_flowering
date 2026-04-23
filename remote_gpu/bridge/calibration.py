"""4-point calibration that maps Kinect (X, Z) → canvas (cx, cy).

We ignore the Kinect Y axis (height above TV) entirely per the current setup:
only the horizontal position above the TV matters for canvas coords. A 3x3
homography comfortably handles Kinect mounting tilt and any skew the user's
TV/Kinect pairing throws at us.

Corner capture flow:
    Browser issues `start_calibration` → bridge walks through TL → TR → BR → BL.
    At each step, user holds pinch over the on-canvas target marker for ~0.5s.
    Bridge samples while pinch is held and *commits* the median on release.
    After 4 corners we solve the homography via `cv2.getPerspectiveTransform`
    (falls back to numpy SVD if OpenCV is unavailable).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


CORNER_LABELS = ("TL", "TR", "BR", "BL")
MIN_SAMPLES = 8           # need at least this many frames of hold to commit
SAMPLE_CAP = 120          # stop accumulating after this many samples
MAX_SAMPLE_JITTER_M = 0.05  # >5cm std across samples → reject as too shaky


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 matrix mapping src (Nx2) → dst (Nx2). Requires N >= 4."""
    try:
        import cv2  # type: ignore
        return cv2.getPerspectiveTransform(
            src.astype(np.float32), dst.astype(np.float32)
        )
    except Exception:
        # Numpy fallback: solve the 8-DOF linear system directly.
        n = src.shape[0]
        A = np.zeros((2 * n, 8))
        b = np.zeros(2 * n)
        for i in range(n):
            x, y = src[i]
            u, v = dst[i]
            A[2 * i]     = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            b[2 * i]     = u
            b[2 * i + 1] = v
        h, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.array([[h[0], h[1], h[2]],
                         [h[3], h[4], h[5]],
                         [h[6], h[7], 1.0]])


@dataclass
class CalibrationResult:
    ok: bool
    reason: str = ""


class Calibration:
    def __init__(self, path: str, canvas_size: tuple[int, int]):
        self.path = path
        self.canvas_size = canvas_size   # (W, H)
        self.M: Optional[np.ndarray] = None   # 3x3
        self.ready = False

        # Capture state
        self.mode = "idle"                # "idle" | "capturing"
        self.current_corner = 0           # 0..3 in CORNER_LABELS order
        self.sample_buf: list[tuple[float, float]] = []
        self.captured: list[tuple[float, float]] = []

        self.load()

    # -------- persistence --------
    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            M = np.asarray(data["M"], dtype=np.float64)
            if M.shape != (3, 3):
                return
            self.M = M
            self.ready = True
            print(f"[cal] loaded calibration from {self.path}")
        except Exception as e:
            print(f"[cal] failed to load calibration ({e}); will need fresh capture")

    def save(self) -> None:
        if self.M is None:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "M": self.M.tolist(),
                "canvas": list(self.canvas_size),
                "order": list(CORNER_LABELS),
            }, f, indent=2)

    # -------- control --------
    def start(self) -> None:
        self.mode = "capturing"
        self.current_corner = 0
        self.sample_buf = []
        self.captured = []

    def cancel(self) -> None:
        self.mode = "idle"
        self.sample_buf = []

    def reset(self) -> None:
        self.mode = "idle"
        self.M = None
        self.ready = False
        self.sample_buf = []
        self.captured = []
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:  # noqa: BLE001
            pass

    # -------- frame hook --------
    def on_sample(self, xz: tuple[float, float], pinch_held: bool) -> dict:
        """Feed one frame of hand (X,Z) + pinch state. Returns a status dict
        suitable for broadcasting back to the browser."""
        if self.mode != "capturing":
            return {"state": "idle"}

        label = CORNER_LABELS[self.current_corner]

        if pinch_held:
            if len(self.sample_buf) < SAMPLE_CAP:
                self.sample_buf.append(xz)
            return {
                "state": "sampling",
                "corner": self.current_corner,
                "label": label,
                "samples": len(self.sample_buf),
                "need": MIN_SAMPLES,
            }

        # Pinch released: commit corner if we got enough stable samples.
        if len(self.sample_buf) >= MIN_SAMPLES:
            arr = np.asarray(self.sample_buf, dtype=np.float64)
            jitter = float(np.linalg.norm(arr.std(axis=0)))
            if jitter > MAX_SAMPLE_JITTER_M:
                # Too shaky — reject and ask them to redo this corner.
                self.sample_buf = []
                return {
                    "state": "retry",
                    "corner": self.current_corner,
                    "label": label,
                    "reason": f"hand too shaky ({jitter*100:.1f}cm)",
                }
            center = np.median(arr, axis=0)
            self.captured.append((float(center[0]), float(center[1])))
            self.sample_buf = []
            self.current_corner += 1

            if self.current_corner >= 4:
                result = self._finalize()
                self.mode = "idle"
                if result.ok:
                    return {"state": "done"}
                return {"state": "failed", "reason": result.reason}

            return {
                "state": "corner_done",
                "corner": self.current_corner,
                "label": CORNER_LABELS[self.current_corner],
            }

        # Not yet enough samples — user released too early. Clear and wait.
        self.sample_buf = []
        return {
            "state": "waiting",
            "corner": self.current_corner,
            "label": label,
        }

    # -------- runtime apply --------
    def apply(self, xz: tuple[float, float]) -> Optional[tuple[float, float]]:
        if not self.ready or self.M is None:
            return None
        x, z = xz
        p = self.M @ np.array([x, z, 1.0], dtype=np.float64)
        w = p[2]
        if abs(w) < 1e-9:
            return None
        return float(p[0] / w), float(p[1] / w)

    # -------- internals --------
    def _finalize(self) -> CalibrationResult:
        W, H = self.canvas_size
        src = np.asarray(self.captured, dtype=np.float64)
        # Destination corners MUST match CORNER_LABELS order (TL→TR→BR→BL).
        dst = np.asarray([
            [0.0, 0.0],
            [W,   0.0],
            [W,   H  ],
            [0.0, H  ],
        ], dtype=np.float64)
        try:
            M = _solve_homography(src, dst)
        except Exception as e:  # noqa: BLE001
            return CalibrationResult(ok=False, reason=f"solve failed: {e}")

        # Sanity: mapping each captured corner should land near its target.
        maxerr = 0.0
        for s, d in zip(src, dst):
            p = M @ np.array([s[0], s[1], 1.0])
            p = p[:2] / p[2]
            maxerr = max(maxerr, float(np.linalg.norm(p - d)))
        if maxerr > 2.0:  # more than 2px off → something's wrong
            return CalibrationResult(ok=False, reason=f"max-error {maxerr:.2f}px")

        self.M = M
        self.ready = True
        self.save()
        print(f"[cal] committed (max corner error {maxerr:.2f}px)")
        return CalibrationResult(ok=True)
