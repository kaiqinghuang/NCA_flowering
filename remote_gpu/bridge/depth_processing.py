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


def render_depth_debug_bgr(
    depth_mm_flat: np.ndarray,
    plane: Optional[PlaneModel],
    band_min_m: float,
    band_max_m: float,
    result: Optional[DepthHandResult] = None,
    surface_eps_m: float = 0.03,
) -> np.ndarray:
    """BGR image for live debugging: depth colormap + TV-plane slab + interaction band + tip.

    **How to verify the plane matches the real TV:** look for the **magenta** overlay.
    Those pixels are within ``surface_eps_m`` of the fitted plane in 3D — they should
    line up with the physical TV rectangle (modulo depth noise). **Green** is the
    interaction shell (``band_min_m``…``band_max_m`` in front of the plane) where
    hand blobs are segmented.
    """
    try:
        import cv2
    except ImportError:
        return np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)

    dmm = np.asarray(depth_mm_flat, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
    valid = (dmm > 0) & (dmm < 8192)
    z_m = dmm.astype(np.float64) * 0.001

    v_idx, u_idx = np.indices((DEPTH_H, DEPTH_W))
    u_f = u_idx.astype(np.float64)
    v_f = v_idx.astype(np.float64)
    x = (u_f - DEPTH_CX) * z_m / DEPTH_FX
    y = (v_f - DEPTH_CY) * z_m / DEPTH_FY

    # Turbo colormap on clipped depth (mm)
    d_clip = np.clip(dmm.astype(np.float32), 500.0, 4500.0)
    norm = ((d_clip - 500.0) / (4500.0 - 500.0) * 255.0).astype(np.uint8)
    norm[~valid] = 0
    bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    bgr[~valid] = (40, 40, 40)

    n_on_plane = 0
    if plane is not None:
        s = x * plane.a + y * plane.b + z_m * plane.c + plane.d
        s[~valid] = np.nan
        # Slab around the infinite plane: should coincide with the TV surface in image space.
        on_plane = valid & np.isfinite(s) & (np.abs(s) <= surface_eps_m)
        n_on_plane = int(np.count_nonzero(on_plane))
        mag = np.array([255.0, 0.0, 255.0], dtype=np.float32)  # BGR magenta
        bf = bgr.astype(np.float32)
        om = on_plane[:, :, None]
        bf = np.where(om, bf * 0.48 + mag * 0.52, bf)
        bgr = bf.astype(np.uint8)

        band = (
            valid
            & np.isfinite(s)
            & (s >= band_min_m)
            & (s <= band_max_m)
        )
        green = np.zeros_like(bgr, dtype=np.float32)
        green[band] = (0.0, 220.0, 0.0)
        bf = bgr.astype(np.float32)
        bm = band[:, :, None]
        blended = bf * 0.55 + green * 0.45
        bgr = np.where(bm, blended, bf).astype(np.uint8)

    tip_u, tip_v = -1.0, -1.0
    pinch = False
    tracked = False
    conf = False
    if result is not None:
        tracked = result.tracked
        conf = result.confident
        pinch = result.pinch_raw
        tip_u, tip_v = result.debug_uv_tip

    if tip_u >= 0 and tip_v >= 0:
        tu = int(np.clip(round(tip_u), 0, DEPTH_W - 1))
        tv = int(np.clip(round(tip_v), 0, DEPTH_H - 1))
        col = (60, 60, 255) if pinch else (255, 255, 255)
        cv2.drawMarker(bgr, (tu, tv), col, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.circle(bgr, (tu, tv), 10, col, 1, cv2.LINE_AA)

    status = (
        f"plane={'ok' if plane else 'no'}  "
        f"tvPx={n_on_plane // 1000}k  "
        f"trk={'Y' if tracked else 'n'}  "
        f"conf={'Y' if conf else 'n'}  "
        f"pinch={'Y' if pinch else 'n'}"
    )
    cv2.rectangle(bgr, (4, 4), (min(DEPTH_W - 4, 500), 46), (0, 0, 0), -1)
    cv2.putText(
        bgr, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 220), 1, cv2.LINE_AA,
    )
    if plane is not None:
        hint = (
            f"magenta = within {int(surface_eps_m * 1000)}mm of fitted plane (compare to TV)  "
            f"green = {int(band_min_m * 1000)}-{int(band_max_m * 1000)}mm shell"
        )
    else:
        hint = "capture TV plane first — then magenta shows fitted surface slab"
    cv2.putText(
        bgr, hint[: min(len(hint), 95)], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 220, 255), 1, cv2.LINE_AA,
    )

    # Readable size in browser (~2x native)
    return cv2.resize(bgr, (DEPTH_W * 2, DEPTH_H * 2), interpolation=cv2.INTER_NEAREST)


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
