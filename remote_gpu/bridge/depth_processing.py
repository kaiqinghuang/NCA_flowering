"""Depth-map hand extraction: band in front of TV plane → cluster → fingertip + pinch heuristic.

Kinect v2 depth 512×424, units in millimeters in the buffer (0 = invalid).
Camera space follows the usual pinhole model (x right, y down, z forward), meters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .plane_calibration import PlaneModel, ransac_plane


# Kinect v2 depth intrinsics @ 512×424 (from Microsoft calibration bundle, meters).
DEPTH_FX = 365.456
DEPTH_FY = 365.456
DEPTH_CX = 254.878
DEPTH_CY = 205.395
DEPTH_W = 512
DEPTH_H = 424


@dataclass
class DepthHandResult:
    tracked: bool
    confident: bool
    pinch_raw: bool
    pinch_dist_m: float  # geometric proxy for UI / tuning
    tip_xyz: Tuple[float, float, float]  # camera space meters
    thumb_proxy_xyz: Tuple[float, float, float]  # for debug lines (centroid)
    debug_uv_tip: Tuple[float, float]  # depth pixel (u,v) for overlay


def depth_to_xyz(u: np.ndarray, v: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    """u,v,z (meters) -> (N,3) camera coordinates."""
    x = (u.astype(np.float64) - DEPTH_CX) * z_m.astype(np.float64) / DEPTH_FX
    y = (v.astype(np.float64) - DEPTH_CY) * z_m.astype(np.float64) / DEPTH_FY
    return np.stack([x, y, z_m.astype(np.float64)], axis=-1)


def depth_frame_to_points(
    depth_mm: np.ndarray,
    stride: int = 2,
    z_min_m: float = 0.5,
    z_max_m: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, z_m) flat arrays for valid pixels (subsampled)."""
    d = np.asarray(depth_mm, dtype=np.uint16)
    if d.size != DEPTH_W * DEPTH_H:
        d = d.reshape(DEPTH_H, DEPTH_W)
    else:
        d = d.reshape(DEPTH_H, DEPTH_W)
    v_coords, u_coords = np.meshgrid(
        np.arange(0, DEPTH_H, stride, dtype=np.int32),
        np.arange(0, DEPTH_W, stride, dtype=np.int32),
        indexing="ij",
    )
    dv = d[v_coords, u_coords]
    valid = (dv > 0) & (dv < 8192)
    z_m = dv[valid].astype(np.float64) * 0.001
    valid_z = (z_m >= z_min_m) & (z_m <= z_max_m)
    u_flat = u_coords[valid][valid_z].astype(np.float64)
    v_flat = v_coords[valid][valid_z].astype(np.float64)
    z_flat = z_m[valid_z]
    return u_flat, v_flat, z_flat


def band_mask_front_of_plane(
    xyz: np.ndarray,
    plane: PlaneModel,
    d_min_m: float,
    d_max_m: float,
) -> np.ndarray:
    """Keep points with signed_distance in [d_min, d_max] (in front of plane along +n)."""
    s = (
        xyz[:, 0] * plane.a
        + xyz[:, 1] * plane.b
        + xyz[:, 2] * plane.c
        + plane.d
    )
    return (s >= d_min_m) & (s <= d_max_m)


def process_depth_frame(
    depth_mm_flat: np.ndarray,
    plane: Optional[PlaneModel],
    band_min_m: float = 0.02,
    band_max_m: float = 0.40,
    min_cluster_px: int = 180,
    pinch_eigen_ratio_thresh: float = 2.8,
    neighborhood_m: float = 0.07,
) -> DepthHandResult:
    """One depth frame → optional fingertip + pinch heuristic."""
    u, v, z_m = depth_frame_to_points(depth_mm_flat, stride=2)
    if u.size < min_cluster_px:
        return _empty()

    xyz = depth_to_xyz(u, v, z_m)
    if plane is not None:
        m = band_mask_front_of_plane(xyz, plane, band_min_m, band_max_m)
        u, v, z_m = u[m], v[m], z_m[m]
        xyz = xyz[m]
    if xyz.shape[0] < min_cluster_px:
        return _empty()

    try:
        from scipy import ndimage
    except ImportError:
        # Fallback: single blob via no scipy — coarse nearest-neighbor grid
        return _process_without_scipy(u, v, xyz, neighborhood_m, pinch_eigen_ratio_thresh)

    # Project to depth grid occupancy for connected components
    occ = np.zeros((DEPTH_H, DEPTH_W), dtype=np.uint8)
    ui = np.clip(np.round(u).astype(np.int32), 0, DEPTH_W - 1)
    vi = np.clip(np.round(v).astype(np.int32), 0, DEPTH_H - 1)
    occ[vi, ui] = 1
    labeled, num = ndimage.label(occ)
    if num < 1:
        return _empty()

    sizes = ndimage.sum(occ, labeled, index=np.arange(1, num + 1))
    best_label = int(np.argmax(sizes)) + 1
    if sizes[best_label - 1] < min_cluster_px:
        return _empty()

    sel = labeled[vi, ui] == best_label
    if int(np.count_nonzero(sel)) < min_cluster_px:
        return _empty()

    P = xyz[sel]
    uu = u[sel]
    vv = v[sel]
    centroid = P.mean(axis=0)
    dcent = np.linalg.norm(P - centroid, axis=1)
    tip_idx = int(np.argmax(dcent))
    tip = P[tip_idx]
    tip_u, tip_v = float(uu[tip_idx]), float(vv[tip_idx])

    # Local neighborhood for pinch vs open hand
    dist_tip = np.linalg.norm(P - tip, axis=1)
    loc = P[dist_tip < neighborhood_m]
    pinch_raw, pinch_proxy = _pinch_from_local_cloud(loc, pinch_eigen_ratio_thresh)

    # confident if cluster is reasonably large and tip not at border
    margin = 6
    confident = (
        P.shape[0] >= min_cluster_px * 1.2
        and margin < tip_u < DEPTH_W - margin
        and margin < tip_v < DEPTH_H - margin
    )

    return DepthHandResult(
        tracked=True,
        confident=bool(confident),
        pinch_raw=bool(pinch_raw),
        pinch_dist_m=float(pinch_proxy),
        tip_xyz=(float(tip[0]), float(tip[1]), float(tip[2])),
        thumb_proxy_xyz=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        debug_uv_tip=(tip_u, tip_v),
    )


def _empty() -> DepthHandResult:
    return DepthHandResult(
        tracked=False,
        confident=False,
        pinch_raw=False,
        pinch_dist_m=-1.0,
        tip_xyz=(0.0, 0.0, 0.0),
        thumb_proxy_xyz=(0.0, 0.0, 0.0),
        debug_uv_tip=(-1.0, -1.0),
    )


def _pinch_from_local_cloud(loc: np.ndarray, eigen_ratio_thresh: float) -> Tuple[bool, float]:
    """Open hand: elongated 2D spread in tangent plane; pinch: more compact."""
    if loc.shape[0] < 12:
        return False, -1.0
    c = loc.mean(axis=0)
    q = loc - c
    _, s, _ = np.linalg.svd(q, full_matrices=False)
    s = np.maximum(s, 1e-9)
    ratio = float(s[0] / s[1]) if s[1] > 1e-9 else 10.0
    # Pinch / fist → lower elongation ratio
    pinch = ratio < eigen_ratio_thresh
    proxy_dist = float(1.0 / ratio)
    return pinch, proxy_dist


def _process_without_scipy(
    u: np.ndarray,
    v: np.ndarray,
    xyz: np.ndarray,
    neighborhood_m: float,
    pinch_eigen_ratio_thresh: float,
) -> DepthHandResult:
    """Single-blob fallback when scipy is missing."""
    if xyz.shape[0] < 200:
        return _empty()
    centroid = xyz.mean(axis=0)
    dcent = np.linalg.norm(xyz - centroid, axis=1)
    tip_idx = int(np.argmax(dcent))
    tip = xyz[tip_idx]
    tip_u, tip_v = float(u[tip_idx]), float(v[tip_idx])
    dist_tip = np.linalg.norm(xyz - tip, axis=1)
    loc = xyz[dist_tip < neighborhood_m]
    pinch_raw, pinch_proxy = _pinch_from_local_cloud(loc, pinch_eigen_ratio_thresh)
    return DepthHandResult(
        tracked=True,
        confident=True,
        pinch_raw=bool(pinch_raw),
        pinch_dist_m=float(pinch_proxy),
        tip_xyz=(float(tip[0]), float(tip[1]), float(tip[2])),
        thumb_proxy_xyz=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        debug_uv_tip=(tip_u, tip_v),
    )


def fit_plane_from_depth_stack(
    depth_frames: list[np.ndarray],
    max_points: int = 12000,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PlaneModel]:
    """Combine several depth frames (flat uint16) and RANSAC-fit the dominant plane."""
    rng = rng or np.random.default_rng(42)
    all_xyz: list[np.ndarray] = []
    for df in depth_frames:
        u, v, z_m = depth_frame_to_points(df, stride=3, z_min_m=0.4, z_max_m=3.5)
        if u.size < 100:
            continue
        xyz = depth_to_xyz(u, v, z_m)
        all_xyz.append(xyz)
    if not all_xyz:
        return None
    pts = np.concatenate(all_xyz, axis=0)
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return ransac_plane(pts, iterations=160, inlier_thresh_m=0.015, min_inliers=600, rng=rng)
