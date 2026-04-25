"""Depth-map fingertip extraction restricted to the calibrated TV interaction box.

Pipeline (per Kinect v2 depth frame, 512×424, millimeters):

    1. Convert valid depth pixels to 3D camera coordinates (meters).
    2. Compute signed distance ``s`` to the TV plane (n unit, oriented so the
       camera side has ``s > 0``).
    3. Project each pixel onto the plane and express it in the calibrated
       (u, v) basis. Reject pixels whose (u, v) falls outside the TV's
       4-corner polygon — wood strips, floor, etc. are gone.
    4. Keep only the slab ``s ∈ [box_near_m, box_far_m]`` (the interaction
       volume "above" the screen). All remaining pixels are hand candidates.
    5. **Fingertip = pixel with smallest s** (closest to the screen) inside
       the box. We use the median of the K closest pixels (default 20) so
       the result is robust to lone-pixel depth noise.

There is **no pinch heuristic** here: while a fingertip exists in the box,
the bridge emits ``pinch=True`` so the canvas paints continuously.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Kinect v2 depth intrinsics @ 512×424 (Microsoft factory calibration, meters).
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
    tip_xyz: Tuple[float, float, float]      # camera space meters
    tip_signed_dist_m: float                 # plane-relative depth of the tip
    debug_uv_tip: Tuple[float, float]        # depth pixel coordinate (for overlay)
    in_box_px: int                           # how many depth pixels in interaction box


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def _depth_to_xyz_full(dmm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z_m, valid) per depth pixel, full 512×424."""
    valid = (dmm > 0) & (dmm < 8192)
    z_m = dmm.astype(np.float64) * 0.001
    v_idx, u_idx = np.indices((DEPTH_H, DEPTH_W))
    x = (u_idx.astype(np.float64) - DEPTH_CX) * z_m / DEPTH_FX
    y = (v_idx.astype(np.float64) - DEPTH_CY) * z_m / DEPTH_FY
    return x, y, z_m, valid


def points_in_quad_2d(
    u_coord: np.ndarray, v_coord: np.ndarray, quad_uv: np.ndarray
) -> np.ndarray:
    """Vectorized point-in-quadrilateral test (assumes the 4 vertices form a convex quad)."""
    crosses = []
    for i in range(4):
        a = quad_uv[i]
        b = quad_uv[(i + 1) % 4]
        ex, ey = float(b[0] - a[0]), float(b[1] - a[1])
        rx = u_coord - float(a[0])
        ry = v_coord - float(a[1])
        crosses.append(ex * ry - ey * rx)
    pos = (
        (crosses[0] >= 0) & (crosses[1] >= 0)
        & (crosses[2] >= 0) & (crosses[3] >= 0)
    )
    neg = (
        (crosses[0] <= 0) & (crosses[1] <= 0)
        & (crosses[2] <= 0) & (crosses[3] <= 0)
    )
    return pos | neg


def _empty_result() -> DepthHandResult:
    return DepthHandResult(
        tracked=False,
        confident=False,
        tip_xyz=(0.0, 0.0, 0.0),
        tip_signed_dist_m=-1.0,
        debug_uv_tip=(-1.0, -1.0),
        in_box_px=0,
    )


# ----------------------------------------------------------------------
# Per-frame work — returns both the result and intermediate maps used by debug overlay
# ----------------------------------------------------------------------
def analyze_depth_frame(
    depth_mm_flat: np.ndarray,
    tv_cal,  # TVCalibration; can be None or not-yet-ready
    box_near_m: float = 0.02,
    box_far_m: float = 0.45,
    surface_eps_m: float = 0.03,
    min_box_px: int = 80,
    tip_neighbors: int = 20,
) -> dict:
    """Analyze one depth frame; produce result + masks for the debug overlay.

    Returns a dict with:
        result      : DepthHandResult
        valid       : (H, W) bool
        s           : (H, W) signed distance to plane (NaN where invalid)
        in_polygon  : (H, W) bool — pixel projects inside the TV quad
        on_surface  : (H, W) bool — within ``surface_eps_m`` of plane AND in polygon
        in_box      : (H, W) bool — slab ``[near, far]`` AND in polygon
    """
    dmm = np.asarray(depth_mm_flat, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
    x, y, z_m, valid = _depth_to_xyz_full(dmm)

    out = {
        "valid": valid,
        "s": np.full((DEPTH_H, DEPTH_W), np.nan, dtype=np.float64),
        "in_polygon": np.zeros((DEPTH_H, DEPTH_W), dtype=bool),
        "on_surface": np.zeros((DEPTH_H, DEPTH_W), dtype=bool),
        "in_box": np.zeros((DEPTH_H, DEPTH_W), dtype=bool),
        "result": _empty_result(),
    }

    if tv_cal is None or not getattr(tv_cal, "ready", False):
        return out

    plane = tv_cal.plane
    s = x * plane.a + y * plane.b + z_m * plane.c + plane.d
    s[~valid] = np.nan
    out["s"] = s

    # Project to plane (vectorized): p_proj = p - s * n
    px = x - s * plane.a
    py = y - s * plane.b
    pz = z_m - s * plane.c

    rel_x = px - tv_cal.basis_origin[0]
    rel_y = py - tv_cal.basis_origin[1]
    rel_z = pz - tv_cal.basis_origin[2]
    u_coord = (
        rel_x * tv_cal.basis_u[0]
        + rel_y * tv_cal.basis_u[1]
        + rel_z * tv_cal.basis_u[2]
    )
    v_coord = (
        rel_x * tv_cal.basis_v[0]
        + rel_y * tv_cal.basis_v[1]
        + rel_z * tv_cal.basis_v[2]
    )
    in_polygon = points_in_quad_2d(u_coord, v_coord, tv_cal.corners_uv) & valid
    out["in_polygon"] = in_polygon

    on_surface = in_polygon & np.isfinite(s) & (np.abs(s) <= surface_eps_m)
    out["on_surface"] = on_surface

    in_box = (
        in_polygon
        & np.isfinite(s)
        & (s >= box_near_m)
        & (s <= box_far_m)
    )
    out["in_box"] = in_box

    n_box = int(np.count_nonzero(in_box))
    if n_box < min_box_px:
        return out

    # Fingertip = closest-to-plane pixel (smallest s) inside the box.
    # Robustness: use median of the K smallest-s pixels.
    s_box = np.where(in_box, s, np.inf)
    flat = s_box.ravel()
    k = min(tip_neighbors, n_box)
    if k <= 1:
        idx_min = int(np.argmin(flat))
        tv = idx_min // DEPTH_W
        tu = idx_min % DEPTH_W
        tip_x, tip_y, tip_z = float(x[tv, tu]), float(y[tv, tu]), float(z_m[tv, tu])
        tip_s = float(s[tv, tu])
        debug_uv = (float(tu), float(tv))
    else:
        # k smallest values via partition (O(N))
        idx_part = np.argpartition(flat, k - 1)[:k]
        sel_v = idx_part // DEPTH_W
        sel_u = idx_part % DEPTH_W
        xs = x[sel_v, sel_u]
        ys = y[sel_v, sel_u]
        zs = z_m[sel_v, sel_u]
        ss = s[sel_v, sel_u]
        tip_x = float(np.median(xs))
        tip_y = float(np.median(ys))
        tip_z = float(np.median(zs))
        tip_s = float(np.median(ss))
        debug_uv = (float(np.median(sel_u)), float(np.median(sel_v)))

    confident = n_box >= max(min_box_px * 2, 200)

    out["result"] = DepthHandResult(
        tracked=True,
        confident=bool(confident),
        tip_xyz=(tip_x, tip_y, tip_z),
        tip_signed_dist_m=tip_s,
        debug_uv_tip=debug_uv,
        in_box_px=n_box,
    )
    return out


# ----------------------------------------------------------------------
# Debug overlay
# ----------------------------------------------------------------------
def render_depth_debug_bgr(
    depth_mm_flat: np.ndarray,
    tv_cal,                          # TVCalibration | None
    analysis: Optional[dict] = None,
    body_tip_xyz: Optional[Tuple[float, float, float]] = None,
    box_near_m: float = 0.02,
    box_far_m: float = 0.45,
    surface_eps_m: float = 0.03,
) -> np.ndarray:
    """BGR debug image overlay.

    * **Magenta** = pixels within ``surface_eps_m`` of the fitted plane AND
      inside the calibrated TV polygon → real TV surface confirmation.
    * **Light green** = interaction box (slab ``[box_near, box_far]`` along
      the plane normal, inside the polygon) → where hand will be detected.
    * **Magenta polygon outline** = the calibrated 4-corner TV rectangle.
    * **Yellow circle** = SDK HandTipRight (when visible, useful during
      calibration).
    * **White cross + ring** = the depth-derived fingertip used at runtime.
    """
    try:
        import cv2
    except ImportError:
        return np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)

    dmm = np.asarray(depth_mm_flat, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
    valid = (dmm > 0) & (dmm < 8192)

    # Pseudo-color depth as the base layer
    d_clip = np.clip(dmm.astype(np.float32), 500.0, 4500.0)
    norm = ((d_clip - 500.0) / (4500.0 - 500.0) * 255.0).astype(np.uint8)
    norm[~valid] = 0
    bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    bgr[~valid] = (40, 40, 40)

    n_surface = 0
    n_box = 0
    if analysis is not None and tv_cal is not None and getattr(tv_cal, "ready", False):
        on_surface = analysis.get("on_surface")
        in_box = analysis.get("in_box")
        if on_surface is not None and on_surface.shape == bgr.shape[:2]:
            n_surface = int(np.count_nonzero(on_surface))
            if n_surface > 0:
                mag = np.array([255.0, 0.0, 255.0], dtype=np.float32)  # BGR magenta
                bf = bgr.astype(np.float32)
                m = on_surface[:, :, None]
                bgr = np.where(m, bf * 0.45 + mag * 0.55, bf).astype(np.uint8)
        if in_box is not None and in_box.shape == bgr.shape[:2]:
            n_box = int(np.count_nonzero(in_box))
            if n_box > 0:
                green = np.array([0.0, 255.0, 120.0], dtype=np.float32)  # BGR light green
                bf = bgr.astype(np.float32)
                m = in_box[:, :, None]
                bgr = np.where(m, bf * 0.55 + green * 0.45, bf).astype(np.uint8)

        # Outline: project the 4 captured 3D corners to depth pixels
        try:
            corners_uv_px = []
            for c3 in tv_cal.corners_3d or []:
                cx, cy, cz = c3
                if cz <= 0.05:
                    corners_uv_px = []
                    break
                u_px = DEPTH_CX + (cx * DEPTH_FX) / cz
                v_px = DEPTH_CY + (cy * DEPTH_FY) / cz
                corners_uv_px.append((int(round(u_px)), int(round(v_px))))
            if len(corners_uv_px) == 4:
                pts = np.array(corners_uv_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(bgr, [pts], isClosed=True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                for i, (u_px, v_px) in enumerate(corners_uv_px):
                    cv2.circle(bgr, (u_px, v_px), 5, (255, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(
                        bgr, "TLTRBRBL"[i * 2:i * 2 + 2],
                        (u_px + 6, v_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 255), 1, cv2.LINE_AA,
                    )
        except Exception:  # noqa: BLE001
            pass

    # SDK HandTipRight marker (if available) — useful during calibration
    if body_tip_xyz is not None:
        try:
            bx, by, bz = body_tip_xyz
            if bz > 0.05:
                u_px = DEPTH_CX + (bx * DEPTH_FX) / bz
                v_px = DEPTH_CY + (by * DEPTH_FY) / bz
                ui, vi = int(round(u_px)), int(round(v_px))
                if 0 <= ui < DEPTH_W and 0 <= vi < DEPTH_H:
                    cv2.circle(bgr, (ui, vi), 8, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(bgr, (ui, vi), (0, 220, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=14, thickness=2)
        except Exception:  # noqa: BLE001
            pass

    # Depth-derived fingertip (white)
    tracked = False
    conf = False
    tip_uv = (-1.0, -1.0)
    if analysis is not None:
        res = analysis.get("result")
        if res is not None:
            tracked = bool(res.tracked)
            conf = bool(res.confident)
            tip_uv = res.debug_uv_tip
    if tip_uv[0] >= 0 and tip_uv[1] >= 0:
        tu = int(np.clip(round(tip_uv[0]), 0, DEPTH_W - 1))
        tv = int(np.clip(round(tip_uv[1]), 0, DEPTH_H - 1))
        cv2.drawMarker(bgr, (tu, tv), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.circle(bgr, (tu, tv), 12, (255, 255, 255), 1, cv2.LINE_AA)

    # Status text
    plane_ready = tv_cal is not None and getattr(tv_cal, "ready", False)
    status = (
        f"plane={'OK' if plane_ready else 'NO'}  "
        f"tvPx={n_surface // 100}/100  "
        f"boxPx={n_box}  "
        f"trk={'Y' if tracked else 'n'}  "
        f"conf={'Y' if conf else 'n'}"
    )
    cv2.rectangle(bgr, (4, 4), (min(DEPTH_W - 4, 510), 46), (0, 0, 0), -1)
    cv2.putText(
        bgr, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 220), 1, cv2.LINE_AA,
    )
    if plane_ready:
        hint = (
            f"magenta=TV surface (within {int(surface_eps_m * 1000)}mm)  "
            f"green={int(box_near_m * 1000)}-{int(box_far_m * 1000)}mm interaction box"
        )
    else:
        hint = "TV plane not calibrated — run TV calibration first"
    cv2.putText(
        bgr, hint[: min(len(hint), 100)], (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 220, 255), 1, cv2.LINE_AA,
    )

    return cv2.resize(bgr, (DEPTH_W * 2, DEPTH_H * 2), interpolation=cv2.INTER_NEAREST)
