"""Fit and persist the TV / interaction surface as a plane in Kinect camera space.

Plane: a*x + b*y + c*z + d = 0 with (a,b,c) unit normal, distances in meters.
Used by depth-mode segmentation to keep only pixels slightly in front of the surface.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlaneModel:
    a: float
    b: float
    c: float
    d: float  # normalized so sqrt(a^2+b^2+c^2)==1

    def signed_distance(self, xyz: np.ndarray) -> np.ndarray:
        """(N,3) -> (N,) signed distance to plane (meters)."""
        p = np.asarray(xyz, dtype=np.float64)
        return p[:, 0] * self.a + p[:, 1] * self.b + p[:, 2] * self.c + self.d


def fit_plane_svd(xyz: np.ndarray) -> PlaneModel:
    """Least-squares plane through N>=3 points (N,3)."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cent = p.mean(axis=0)
    q = p - cent
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[-1].astype(np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n /= nn
    a, b, c = float(n[0]), float(n[1]), float(n[2])
    d = float(-(a * cent[0] + b * cent[1] + c * cent[2]))
    return PlaneModel(a=a, b=b, c=c, d=d)


def ransac_plane(
    xyz: np.ndarray,
    iterations: int = 120,
    inlier_thresh_m: float = 0.012,
    min_inliers: int = 800,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PlaneModel]:
    """Robust plane for many noisy 3D points (N,3)."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 200:
        return None
    rng = rng or np.random.default_rng()
    best_inliers = 0
    best_plane: Optional[PlaneModel] = None
    n = p.shape[0]
    for _ in range(iterations):
        ii = rng.choice(n, size=3, replace=False)
        tri = p[ii]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        nn = np.cross(v1, v2)
        ln = float(np.linalg.norm(nn))
        if ln < 1e-9:
            continue
        nn /= ln
        a, b, c = float(nn[0]), float(nn[1]), float(nn[2])
        d = float(-(a * tri[0, 0] + b * tri[0, 1] + c * tri[0, 2]))
        dist = np.abs(p @ np.array([a, b, c], dtype=np.float64) + d)
        inl = int(np.count_nonzero(dist < inlier_thresh_m))
        if inl > best_inliers:
            best_inliers = inl
            q = p[dist < inlier_thresh_m * 2.0]
            if q.shape[0] >= 50:
                best_plane = fit_plane_svd(q)
            else:
                best_plane = PlaneModel(a=a, b=b, c=c, d=d)
    if best_plane is None or best_inliers < min_inliers:
        return None
    return best_plane


class PlaneCalibration:
    def __init__(self, path: str):
        self.path = path
        self.plane: Optional[PlaneModel] = None
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.plane = PlaneModel(
                float(data["a"]),
                float(data["b"]),
                float(data["c"]),
                float(data["d"]),
            )
            print(f"[plane] loaded from {self.path}")
        except Exception as e:  # noqa: BLE001
            print(f"[plane] load failed ({e})")

    def save(self, plane: PlaneModel) -> None:
        self.plane = plane
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"a": plane.a, "b": plane.b, "c": plane.c, "d": plane.d}, f, indent=2)
        print(f"[plane] saved to {self.path}")

    @property
    def ready(self) -> bool:
        return self.plane is not None
