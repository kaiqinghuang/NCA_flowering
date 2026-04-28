"""4-corner TV calibration: SDK HandTipRight 3D points → TV plane + canvas homography.

Replaces the old separate `Calibration` (canvas pinch corners) and
`PlaneCalibration` (auto RANSAC plane). One pass of 4 manually-confirmed
corners (A → B → C → D, clockwise from the front-left) gives us:

    1. The TV plane (least-squares fit of the 4 captured 3D points).
    2. A right-handed orthonormal basis (origin = A, u along A→B,
       v along A→D projected onto the plane).
    3. The polygon (4 corners) projected to that 2D basis — used to
       restrict depth-mode segmentation to "above the TV" only.
    4. A 3x3 homography from plane-local (u, v) to canvas (cx, cy).

The 4 corner labels A/B/C/D map to canvas position TL/TR/BR/BL:
    A = front-left  (smaller Z, smaller X) → canvas top-left
    B = front-right (smaller Z, larger  X) → canvas top-right
    C = back-right  (larger  Z, larger  X) → canvas bottom-right
    D = back-left   (larger  Z, smaller X) → canvas bottom-left

Persisted to `tv_calibration.json` next to this module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


CORNER_LABELS: Tuple[str, str, str, str] = ("A", "B", "C", "D")


@dataclass
class TVPlane:
    a: float
    b: float
    c: float
    d: float  # ax + by + cz + d = 0  with (a,b,c) unit normal

    def signed_distance(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1:
            return float(p[0] * self.a + p[1] * self.b + p[2] * self.c + self.d)
        return p[..., 0] * self.a + p[..., 1] * self.b + p[..., 2] * self.c + self.d


def _orient_normal_toward_origin(a: float, b: float, c: float, d: float):
    """Flip plane orientation so that camera origin (0,0,0) has signed_distance > 0.

    With this convention, the interaction box (above the TV, on the camera
    side) corresponds to ``signed_distance > 0`` for any point in front of
    the surface. Makes downstream depth filtering unambiguous.
    """
    if d < 0:
        return -a, -b, -c, -d
    return a, b, c, d


def _fit_plane_svd(points_xyz) -> Optional[TVPlane]:
    p = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 3:
        return None
    cent = p.mean(axis=0)
    q = p - cent
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[-1]
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return None
    n /= nn
    a, b, c = float(n[0]), float(n[1]), float(n[2])
    d = float(-(a * cent[0] + b * cent[1] + c * cent[2]))
    a, b, c, d = _orient_normal_toward_origin(a, b, c, d)
    return TVPlane(a, b, c, d)


def _project_to_plane(p: np.ndarray, plane: TVPlane) -> np.ndarray:
    """Drop the plane-normal component. Works for (3,) or (...,3)."""
    s = plane.signed_distance(p)
    n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
    if np.ndim(s) == 0:
        return np.asarray(p, dtype=np.float64) - s * n
    return p - s[..., None] * n


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    try:
        import cv2
        return cv2.getPerspectiveTransform(
            src.astype(np.float32), dst.astype(np.float32),
        )
    except Exception:
        # Numpy fallback for the 8-DOF linear system.
        n = src.shape[0]
        A = np.zeros((2 * n, 8), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            x, y = src[i]
            u, v = dst[i]
            A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            b[2 * i] = u
            b[2 * i + 1] = v
        h, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.array(
            [[h[0], h[1], h[2]],
             [h[3], h[4], h[5]],
             [h[6], h[7], 1.0]], dtype=np.float64,
        )


@dataclass
class TVCalibrationResult:
    ok: bool
    reason: str = ""


class TVCalibration:
    """State machine + persistence for the manual 4-corner TV calibration."""

    def __init__(self, path: str, canvas_size: Tuple[int, int]):
        self.path = path
        self.canvas_size = canvas_size  # (W, H)

        # Wizard state
        self.mode = "idle"  # "idle" | "capturing"
        self.current_corner = 0  # 0..3 during capture
        self.captured: list[Tuple[float, float, float]] = []

        # Finalized fields (populated after 4 corners committed)
        self.corners_3d: Optional[list[Tuple[float, float, float]]] = None
        self.plane: Optional[TVPlane] = None
        self.basis_origin: Optional[np.ndarray] = None  # (3,)
        self.basis_u: Optional[np.ndarray] = None       # (3,)
        self.basis_v: Optional[np.ndarray] = None       # (3,)
        self.corners_uv: Optional[np.ndarray] = None    # (4, 2)
        self.M: Optional[np.ndarray] = None             # 3x3 homography (u,v)→(cx,cy)
        self.ready: bool = False

        self.load()

    # -------- live status helpers --------
    @property
    def label(self) -> str:
        i = max(0, min(3, self.current_corner))
        return CORNER_LABELS[i]

    def status_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ready": bool(self.ready),
            "current_corner": int(self.current_corner),
            "label": self.label,
            "captured": len(self.captured),
            "total": 4,
            "labels": list(CORNER_LABELS),
        }

    # -------- wizard control --------
    def start(self) -> None:
        self.mode = "capturing"
        self.current_corner = 0
        self.captured = []

    def cancel(self) -> None:
        self.mode = "idle"
        self.captured = []
        self.current_corner = 0

    def confirm(self, tip_xyz: Optional[Tuple[float, float, float]]) -> dict:
        if self.mode != "capturing":
            return {"ok": False, "reason": "not capturing"}
        if tip_xyz is None:
            return {"ok": False, "reason": "no body tip available — keep your right hand visible"}
        self.captured.append((float(tip_xyz[0]), float(tip_xyz[1]), float(tip_xyz[2])))

        if len(self.captured) >= 4:
            res = self._finalize()
            self.mode = "idle"
            self.captured = []
            self.current_corner = 0
            if res.ok:
                return {"ok": True, "done": True}
            return {"ok": False, "done": True, "reason": res.reason}

        self.current_corner = len(self.captured)
        return {
            "ok": True,
            "done": False,
            "next_corner": self.current_corner,
            "next_label": self.label,
        }

    def redo(self) -> dict:
        if self.mode != "capturing":
            return {"ok": False, "reason": "not capturing"}
        if not self.captured:
            return {"ok": False, "reason": "nothing to redo"}
        self.captured.pop()
        self.current_corner = len(self.captured)
        return {"ok": True, "current_corner": self.current_corner, "label": self.label}

    def reset(self) -> None:
        self.cancel()
        self.corners_3d = None
        self.plane = None
        self.basis_origin = None
        self.basis_u = None
        self.basis_v = None
        self.corners_uv = None
        self.M = None
        self.ready = False
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:  # noqa: BLE001
            pass

    def commit_auto(self, corners_abcd) -> TVCalibrationResult:
        """Commit a 4-corner calibration sourced from auto depth-plane fit.

        ``corners_abcd`` must be a sequence of 4 (x, y, z) points
        already ordered as A, B, C, D in camera space (front-left,
        front-right, back-right, back-left). We re-fit the plane with
        SVD on those 4 points (so the plane stays consistent with the
        corners), then run the same finalisation as the manual wizard
        (basis, polygon (u, v), homography → canvas).
        """
        seq = list(corners_abcd)
        if len(seq) != 4:
            return TVCalibrationResult(False, "auto-commit needs exactly 4 corners")
        try:
            self.captured = [
                (float(c[0]), float(c[1]), float(c[2])) for c in seq
            ]
        except Exception as e:  # noqa: BLE001
            return TVCalibrationResult(False, f"auto-commit bad corner format: {e}")
        # Reset wizard state — auto-commit is one-shot.
        self.mode = "idle"
        self.current_corner = 0
        res = self._finalize()
        if not res.ok:
            self.captured = []
        else:
            # _finalize already cleared self.captured? No, only the wizard does.
            self.captured = []
        return res

    # -------- finalization (4 corners → plane + basis + homography) --------
    def _finalize(self) -> TVCalibrationResult:
        if len(self.captured) != 4:
            return TVCalibrationResult(False, "need exactly 4 corners")

        plane = _fit_plane_svd(self.captured)
        if plane is None:
            return TVCalibrationResult(False, "plane SVD failed")

        n = np.array([plane.a, plane.b, plane.c], dtype=np.float64)
        proj = np.stack([_project_to_plane(np.asarray(p, dtype=np.float64), plane)
                         for p in self.captured])  # (4, 3)
        origin = proj[0].copy()
        u_dir_raw = proj[1] - proj[0]  # A → B
        u_dir_raw -= n * np.dot(u_dir_raw, n)
        u_norm = float(np.linalg.norm(u_dir_raw))
        if u_norm < 1e-6:
            return TVCalibrationResult(False, "A→B direction is degenerate")
        u_basis = u_dir_raw / u_norm
        v_basis = np.cross(n, u_basis)
        vn = float(np.linalg.norm(v_basis))
        if vn < 1e-9:
            return TVCalibrationResult(False, "v basis degenerate")
        v_basis /= vn
        # Make sure v_basis points A → D (canvas +y).
        if np.dot(proj[3] - proj[0], v_basis) < 0:
            v_basis = -v_basis

        rel = proj - origin
        u_coords = rel @ u_basis
        v_coords = rel @ v_basis
        corners_uv = np.stack([u_coords, v_coords], axis=1)  # (4, 2)

        W, H = self.canvas_size
        canvas_uv = np.array(
            [[0.0, 0.0], [float(W), 0.0], [float(W), float(H)], [0.0, float(H)]],
            dtype=np.float64,
        )
        try:
            M = _solve_homography(corners_uv, canvas_uv)
        except Exception as e:  # noqa: BLE001
            return TVCalibrationResult(False, f"homography solve failed: {e}")

        # Sanity: each corner should round-trip near its target
        max_err = 0.0
        for s_uv, dst in zip(corners_uv, canvas_uv):
            h = M @ np.array([s_uv[0], s_uv[1], 1.0], dtype=np.float64)
            if abs(h[2]) < 1e-9:
                return TVCalibrationResult(False, "homography degenerate")
            p = h[:2] / h[2]
            max_err = max(max_err, float(np.linalg.norm(p - dst)))
        if max_err > 8.0:  # 4-pt fit should round-trip exactly; >8px = numeric trouble
            return TVCalibrationResult(False, f"corner round-trip error {max_err:.2f}px")

        self.corners_3d = [tuple(map(float, c)) for c in self.captured]
        self.plane = plane
        self.basis_origin = origin
        self.basis_u = u_basis
        self.basis_v = v_basis
        self.corners_uv = corners_uv
        self.M = M
        self.ready = True
        self.save()
        print(f"[tv_cal] committed (round-trip max {max_err:.2f}px)")
        return TVCalibrationResult(True)

    # -------- runtime mapping --------
    def project_xyz_to_plane_uv(self, xyz) -> Optional[Tuple[float, float]]:
        if not self.ready or self.plane is None:
            return None
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        proj = _project_to_plane(p, self.plane)
        rel = proj - self.basis_origin
        return float(np.dot(rel, self.basis_u)), float(np.dot(rel, self.basis_v))

    def project_xyz_to_canvas(self, xyz) -> Optional[Tuple[float, float]]:
        if not self.ready or self.M is None:
            return None
        uv = self.project_xyz_to_plane_uv(xyz)
        if uv is None:
            return None
        h = self.M @ np.array([uv[0], uv[1], 1.0], dtype=np.float64)
        if abs(h[2]) < 1e-9:
            return None
        return float(h[0] / h[2]), float(h[1] / h[2])

    # -------- persistence --------
    def save(self) -> None:
        if self.plane is None or self.basis_origin is None or self.M is None:
            return
        data = {
            "corners_3d": self.corners_3d,
            "plane": {"a": self.plane.a, "b": self.plane.b, "c": self.plane.c, "d": self.plane.d},
            "basis": {
                "origin": self.basis_origin.tolist(),
                "u": self.basis_u.tolist(),
                "v": self.basis_v.tolist(),
            },
            "corners_uv": self.corners_uv.tolist(),
            "M": self.M.tolist(),
            "canvas": list(self.canvas_size),
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[tv_cal] saved to {self.path}")
        except Exception as e:  # noqa: BLE001
            print(f"[tv_cal] save failed ({e})")

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = data["plane"]
            self.plane = TVPlane(float(p["a"]), float(p["b"]), float(p["c"]), float(p["d"]))
            b = data["basis"]
            self.basis_origin = np.asarray(b["origin"], dtype=np.float64)
            self.basis_u = np.asarray(b["u"], dtype=np.float64)
            self.basis_v = np.asarray(b["v"], dtype=np.float64)
            self.corners_uv = np.asarray(data["corners_uv"], dtype=np.float64)
            self.corners_3d = [tuple(map(float, c)) for c in data["corners_3d"]]
            self.M = np.asarray(data["M"], dtype=np.float64)
            self.ready = True
            print(f"[tv_cal] loaded from {self.path}")
        except Exception as e:  # noqa: BLE001
            print(f"[tv_cal] load failed ({e}); starting fresh")
            self.ready = False
