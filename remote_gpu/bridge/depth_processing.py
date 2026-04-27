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

This module also exposes ``auto_calibrate_tv_from_depth`` — a one-shot
routine that fits the TV plane via RANSAC on a stack of depth frames,
isolates the largest co-planar blob (= the TV screen), and derives the
4 rectangle corners (rotated bounding box) sorted into TL/TR/BR/BL by
their (X, Z) position in camera space.

There is **no pinch heuristic** here: while a fingertip exists in the box,
the bridge emits ``pinch=True`` so the canvas paints continuously.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Optional, Tuple

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


# ----------------------------------------------------------------------
# Auto-calibration: depth point cloud → TV plane + 4 rectangle corners
# ----------------------------------------------------------------------
def _fit_plane_svd_xyz(points: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Least-squares plane through points (N, 3). Returns (a, b, c, d) or None."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
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
    return a, b, c, d


def _ransac_plane(
    pts: np.ndarray,
    iterations: int = 200,
    thresh: float = 0.02,
    min_inliers: int = 600,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """RANSAC + SVD refit for a 3D plane on (N, 3) points."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_pts = int(pts.shape[0])
    if n_pts < max(min_inliers // 2, 30):
        return None
    best_count = 0
    best_inliers: Optional[np.ndarray] = None
    for _ in range(iterations):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n /= nn
        d = float(-np.dot(n, p0))
        s = pts @ n + d
        inliers = np.abs(s) < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
    if best_inliers is None or best_count < min_inliers:
        return None
    refit = _fit_plane_svd_xyz(pts[best_inliers])
    return refit


def _refine_blob_with_color(
    blob_mask_depth: np.ndarray,
    depth_to_color_xy: np.ndarray,
    color_bgr: np.ndarray,
    max_v: float,
    morph_close_px: int,
    cv2_mod,
) -> Tuple[np.ndarray, dict]:
    """Refine a depth-derived blob mask using the corresponding RGB pixels.

    Co-planar non-screen structures (wood slats, bezel ornaments, …)
    survive the depth-only pipeline because they sit *on* the TV plane.
    But visually they are bright/saturated, while the powered-off (or
    dark-content) TV screen is near-black. We exploit that asymmetry:

        for each depth pixel in the blob:
            (cx, cy) = SDK CoordinateMapper(depth pixel)        # done by caller
            sample color_bgr[cy, cx]
            keep the depth pixel iff  max(B, G, R) ≤ max_v

    A small morphological close fills speckle holes from screen
    reflections / interpolation noise, then we keep only the largest
    connected component as the final TV mask.

    Args:
        blob_mask_depth:    (DEPTH_H, DEPTH_W) bool, the depth-only blob.
        depth_to_color_xy:  (DEPTH_H*DEPTH_W, 2) float32 — the SDK's
                            ``MapDepthFrameToColorSpace`` output for the
                            same depth frame the blob was derived from.
                            Invalid mappings are encoded as ±inf.
        color_bgr:          (Hc, Wc, 3) uint8 BGR — the colour frame
                            captured at the same instant.
        max_v:              maximum brightness (0–255) to count as "TV".
                            Wood and bezels are brighter than this.
        morph_close_px:     square kernel size for closing pinholes in
                            the kept mask (px in depth space). 0 = off.
        cv2_mod:            the imported ``cv2`` module (passed in to
                            avoid re-importing it inside this hot path).

    Returns:
        refined_mask:  (DEPTH_H, DEPTH_W) bool, after color filter +
                       morph close + largest-CC. May fall back to
                       ``blob_mask_depth`` if the colour pass cannot
                       confidently produce a usable refinement; the
                       returned ``info`` dict explains which path was
                       taken.
        info:          diagnostics (counts, percentages, why-skipped, …).
    """
    Hc, Wc = color_bgr.shape[:2]
    blob_idx = np.flatnonzero(blob_mask_depth.ravel())
    n_total = int(blob_idx.size)
    if n_total == 0:
        return blob_mask_depth.copy(), {
            "color_refined": False,
            "reason": "empty blob",
        }

    cx = depth_to_color_xy[blob_idx, 0]
    cy = depth_to_color_xy[blob_idx, 1]
    valid = (
        np.isfinite(cx)
        & np.isfinite(cy)
        & (cx >= 0)
        & (cy >= 0)
        & (cx < Wc - 0.5)
        & (cy < Hc - 0.5)
    )
    n_valid = int(valid.sum())
    if n_valid < 200:
        return blob_mask_depth.copy(), {
            "color_refined": False,
            "reason": f"only {n_valid} blob pixels mapped into color frame",
            "n_blob_before": n_total,
        }

    cx_v = cx[valid].astype(np.int32)
    cy_v = cy[valid].astype(np.int32)
    bgr_samples = color_bgr[cy_v, cx_v]            # (n_valid, 3) uint8
    v_chan = bgr_samples.max(axis=1)               # ≈ HSV V (fast proxy)
    keep = v_chan <= max_v                         # bool mask over valid subset
    n_keep = int(keep.sum())
    if n_keep < max(200, n_total // 50):
        # The colour filter wiped almost everything out — likely the TV is
        # actually showing bright content during calibration, or max_v is
        # set too low. Fall back to the unrefined depth blob so the user
        # at least gets a calibration; the printed warning + UI message
        # will point at BRIDGE_AUTOFIT_COLOR_MAX_V / the color frame.
        return blob_mask_depth.copy(), {
            "color_refined": False,
            "reason": (
                f"color filter kept only {n_keep}/{n_valid} pixels "
                f"(max_v={max_v}); falling back to depth-only blob"
            ),
            "n_blob_before": n_total,
            "n_valid_color": n_valid,
            "n_dark_kept": n_keep,
            "max_v_used": float(max_v),
        }

    # Build refined mask in depth-pixel space.
    keep_indices = blob_idx[valid][keep]
    refined = np.zeros(blob_mask_depth.shape, dtype=np.uint8)
    refined.ravel()[keep_indices] = 1

    # Tiny holes from depth/colour misalignment or screen reflections —
    # close them so we don't fragment the TV into many small CCs.
    if morph_close_px and morph_close_px > 0:
        k = max(1, int(morph_close_px))
        ker = cv2_mod.getStructuringElement(cv2_mod.MORPH_RECT, (k, k))
        refined = cv2_mod.morphologyEx(refined, cv2_mod.MORPH_CLOSE, ker)

    # Largest connected component = the TV.
    n_lbl, labels, stats, _ = cv2_mod.connectedComponentsWithStats(refined, connectivity=8)
    if n_lbl < 2:
        return blob_mask_depth.copy(), {
            "color_refined": False,
            "reason": "no connected component after color filter",
            "n_blob_before": n_total,
            "n_valid_color": n_valid,
            "n_dark_kept": n_keep,
        }
    sizes = stats[1:, cv2_mod.CC_STAT_AREA]
    best = 1 + int(np.argmax(sizes))
    final_mask = labels == best
    final_size = int(sizes[best - 1])

    info = {
        "color_refined": True,
        "n_blob_before": n_total,
        "n_valid_color": n_valid,
        "n_dark_kept": n_keep,
        "n_after_largest_cc": final_size,
        "removed_pct": float(1.0 - final_size / max(1, n_total)) * 100.0,
        "max_v_used": float(max_v),
        "color_frame_size": [int(Hc), int(Wc)],
    }
    return final_mask, info


def _assign_corners_TL_TR_BR_BL(corners_3d: np.ndarray) -> list:
    """Permute 4 (x, y, z) corners to canvas-order [TL, TR, BR, BL].

    Convention (matches user's setup, after Y-down convention is enforced):
        - canvas TopLeft  ↔ smaller Z (front, closer to Kinect) AND smaller X (left)
        - canvas TopRight ↔ smaller Z (front)                    AND larger  X (right)
        - canvas BottomRight ↔ larger Z (back)                   AND larger  X
        - canvas BottomLeft  ↔ larger Z (back)                   AND smaller X

    Robust to arbitrary in-plane rotation: we score by angle around the
    centroid in the (X, Z) plane. ``atan2(dx, -dz)`` puts +"front" at angle 0
    growing CCW toward right. For each canvas corner we pick the in-plane
    angle that best matches; brute-forcing all 24 permutations guarantees a
    unique assignment.
    """
    pts = np.asarray(corners_3d, dtype=np.float64).reshape(4, 3)
    cx = float(pts[:, 0].mean())
    cz = float(pts[:, 2].mean())
    angles = np.arctan2(pts[:, 0] - cx, -(pts[:, 2] - cz))  # 0=front, +π/2=right
    target = np.array([
        -np.pi / 4.0,   # TL: front-left  (dx<0, dz<0)
        +np.pi / 4.0,   # TR: front-right (dx>0, dz<0)
        +3 * np.pi / 4.0,  # BR: back-right (dx>0, dz>0)
        -3 * np.pi / 4.0,  # BL: back-left  (dx<0, dz>0)
    ])
    best_perm = None
    best_cost = np.inf
    for perm in permutations(range(4)):
        cost = 0.0
        for tgt_i, c_i in enumerate(perm):
            d = angles[c_i] - target[tgt_i]
            d = float(np.arctan2(np.sin(d), np.cos(d)))  # wrap to [-π, π]
            cost += abs(d)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        return [tuple(map(float, p)) for p in pts]
    return [tuple(map(float, pts[best_perm[i]])) for i in range(4)]


def auto_calibrate_tv_from_depth(
    depth_frames: Iterable[np.ndarray],
    *,
    color_bgr: Optional[np.ndarray] = None,
    depth_to_color_xy: Optional[np.ndarray] = None,
    on_plane_eps_m: float = 0.015,
    ransac_iters: int = 240,
    ransac_inlier_thresh_m: float = 0.02,
    ransac_min_inliers: int = 800,
    min_blob_px: int = 1500,
    max_ransac_pts: int = 12000,
    z_min_m: float = 0.4,
    z_max_m: float = 4.5,
    morph_open_px: int = 3,
    color_max_v: float = 90.0,
    color_close_px: int = 3,
    trim_pct: float = 1.5,
) -> dict:
    """One-shot TV plane + 4-corner derivation, depth + colour fused.

    Steps:
        1. Stack ``depth_frames`` into one (sub-sampled) point cloud.
        2. RANSAC fit the dominant plane (refit via SVD on inliers).
        3. Compute on-plane mask on the LAST frame; morph-open to break
           thin depth-connected bridges to neighbouring co-planar wood
           strips (cheap, doesn't fight us further down).
        4. Largest connected component = depth-only TV blob candidate.
        5. **Colour refinement.** If a colour frame and the SDK's
           depth→colour mapping were supplied, look up each blob pixel's
           RGB value and discard everything brighter than ``color_max_v``
           — wood/bezels are bright, the TV screen is near-black. Take
           the largest connected component of what survives. (Falls back
           gracefully to the depth-only blob if mapping is unavailable
           or the filter wipes everything.)
        6. Project the refined blob to the plane (u, v), then derive 4
           corners via **PCA + percentile trim** instead of
           ``cv2.minAreaRect``. PCA finds the TV's long/short axes; we
           take the ``[trim_pct, 100 − trim_pct]`` percentile of the
           projection along each axis as the rectangle bounds. This is
           robust to a small fraction of co-planar outliers that survive
           the colour pass (e.g. bezel transition pixels at the front
           edge of the TV) — those would otherwise drag a strict min-area
           rect outward.
        7. Reconstruct 3D corners → sort into TL/TR/BR/BL.

    Args:
        trim_pct: percentage of points to discard at *each* end of *each*
                  PCA axis when fitting the rectangle. ``1.5`` keeps the
                  central 97% of points along each axis. Set to ``0`` to
                  fall back to a strict min/max bounding box.

    Returns a dict (always with ``ok``):
        ok, reason
        corners_3d        : list of 4 (x, y, z) tuples (TL, TR, BR, BL)
        plane             : (a, b, c, d) with origin-side positive
        n_blob_px         : int   pixel count of the chosen TV blob
                                  (after colour refinement, if applied)
        area_m2           : float TV rectangle area in m²
        ransac_inlier_pts : int   how many points fed to RANSAC
        color_info        : diagnostic dict from the colour-refine stage
        trim_pct_used     : float trim percentage actually applied
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"ok": False, "reason": "OpenCV (cv2) not installed"}

    frames = list(depth_frames)
    if not frames:
        return {"ok": False, "reason": "no depth frames"}

    z_min_mm = int(z_min_m * 1000.0)
    z_max_mm = int(z_max_m * 1000.0)

    # ---- 1. Build combined point cloud ----
    v_idx, u_idx = np.indices((DEPTH_H, DEPTH_W))
    u_f = u_idx.astype(np.float64)
    v_f = v_idx.astype(np.float64)

    cloud_chunks = []
    for df in frames:
        d = np.asarray(df, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
        m = (d > z_min_mm) & (d < z_max_mm)
        if not m.any():
            continue
        z_m = d.astype(np.float64) * 0.001
        x = (u_f - DEPTH_CX) * z_m / DEPTH_FX
        y = (v_f - DEPTH_CY) * z_m / DEPTH_FY
        cloud_chunks.append(np.stack([x[m], y[m], z_m[m]], axis=1))
    if not cloud_chunks:
        return {"ok": False, "reason": "no valid depth pixels in [%.2f, %.2f]m" % (z_min_m, z_max_m)}
    pts = np.concatenate(cloud_chunks, axis=0)

    rng = np.random.default_rng(42)
    if pts.shape[0] > max_ransac_pts:
        idx = rng.choice(pts.shape[0], max_ransac_pts, replace=False)
        ransac_pts = pts[idx]
    else:
        ransac_pts = pts

    # ---- 2. RANSAC plane ----
    plane = _ransac_plane(
        ransac_pts,
        iterations=ransac_iters,
        thresh=ransac_inlier_thresh_m,
        min_inliers=ransac_min_inliers,
        rng=rng,
    )
    if plane is None:
        return {"ok": False, "reason": f"RANSAC failed (need ≥{ransac_min_inliers} inliers)"}
    a, b, c, d = plane
    # Orient normal so camera origin (0,0,0) lies on +n side.
    if d < 0:
        a, b, c, d = -a, -b, -c, -d

    # ---- 3. On-plane mask on the LAST captured frame ----
    df_last = np.asarray(frames[-1], dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
    valid_last = (df_last > 0) & (df_last < 8192)
    z_last = df_last.astype(np.float64) * 0.001
    x_last = (u_f - DEPTH_CX) * z_last / DEPTH_FX
    y_last = (v_f - DEPTH_CY) * z_last / DEPTH_FY
    s_last = x_last * a + y_last * b + z_last * c + d
    on_plane = valid_last & (np.abs(s_last) <= on_plane_eps_m)

    n_on = int(np.count_nonzero(on_plane))
    if n_on < min_blob_px:
        return {"ok": False, "reason": f"only {n_on} on-plane pixels (need ≥{min_blob_px})"}

    mask_u8 = on_plane.astype(np.uint8) * 255
    if morph_open_px and morph_open_px > 0:
        k = max(1, int(morph_open_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---- 4. Largest connected blob (depth-only candidate) ----
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_lbl <= 1:
        return {"ok": False, "reason": "no connected on-plane blobs"}
    sizes = stats[1:, cv2.CC_STAT_AREA]
    best_local = int(np.argmax(sizes))
    best_lbl = best_local + 1
    best_size = int(sizes[best_local])
    if best_size < min_blob_px:
        return {"ok": False, "reason": f"largest blob too small ({best_size}px < {min_blob_px})"}
    blob_mask = labels == best_lbl

    # ---- 5. Colour refinement: throw away non-black pixels (wood, bezel). ----
    if color_bgr is not None and depth_to_color_xy is not None:
        if depth_to_color_xy.shape[0] != DEPTH_H * DEPTH_W:
            color_info = {
                "color_refined": False,
                "reason": (
                    f"depth_to_color_xy length {depth_to_color_xy.shape[0]} "
                    f"!= expected {DEPTH_H * DEPTH_W}"
                ),
            }
        else:
            blob_mask, color_info = _refine_blob_with_color(
                blob_mask,
                depth_to_color_xy,
                np.ascontiguousarray(color_bgr, dtype=np.uint8),
                max_v=float(color_max_v),
                morph_close_px=int(color_close_px),
                cv2_mod=cv2,
            )
            if color_info.get("color_refined"):
                best_size = int(color_info.get("n_after_largest_cc", best_size))
            if best_size < min_blob_px:
                return {
                    "ok": False,
                    "reason": (
                        f"after color refine only {best_size}px (<{min_blob_px}); "
                        "TV may be showing bright content during calibration "
                        "or BRIDGE_AUTOFIT_COLOR_MAX_V is too low"
                    ),
                }
    else:
        color_info = {
            "color_refined": False,
            "reason": "no color frame / mapping supplied (depth-only fit)",
        }

    # ---- 6. Project refined blob to plane, then PCA + percentile fit. ----
    # Build a provisional orthonormal basis (u_tmp, v_tmp) on the plane —
    # this is just any 2D frame; the PCA below realigns to the TV's actual
    # long/short axes regardless of how (u_tmp, v_tmp) happen to be rotated.
    n_vec = np.array([a, b, c], dtype=np.float64)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_tmp = arbitrary - np.dot(arbitrary, n_vec) * n_vec
    u_tmp_n = float(np.linalg.norm(u_tmp))
    if u_tmp_n < 1e-9:
        return {"ok": False, "reason": "degenerate provisional basis"}
    u_tmp /= u_tmp_n
    v_tmp = np.cross(n_vec, u_tmp)

    bx = x_last[blob_mask]
    by = y_last[blob_mask]
    bz = z_last[blob_mask]
    bs = bx * a + by * b + bz * c + d
    px = bx - bs * a
    py = by - bs * b
    pz = bz - bs * c
    origin_tmp = np.array([px.mean(), py.mean(), pz.mean()], dtype=np.float64)
    rel_x = px - origin_tmp[0]
    rel_y = py - origin_tmp[1]
    rel_z = pz - origin_tmp[2]
    u_coords = rel_x * u_tmp[0] + rel_y * u_tmp[1] + rel_z * u_tmp[2]
    v_coords = rel_x * v_tmp[0] + rel_y * v_tmp[1] + rel_z * v_tmp[2]

    pts2d = np.stack([u_coords, v_coords], axis=1)  # (N, 2) in meters
    if pts2d.shape[0] < 4:
        return {"ok": False, "reason": "too few blob points after projection"}

    # PCA on the in-plane points: principal axis = TV's long edge.
    centroid_uv = pts2d.mean(axis=0)
    centered = pts2d - centroid_uv
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigvals; columns of eigvecs
    order = np.argsort(eigvals)[::-1]        # descending: long axis first
    e1 = eigvecs[:, order[0]]                # long  edge direction in (u_tmp, v_tmp)
    e2 = eigvecs[:, order[1]]                # short edge direction
    # Force a right-handed basis (det > 0) for a stable orientation —
    # doesn't change geometry, just makes the corner ordering deterministic.
    if e1[0] * e2[1] - e1[1] * e2[0] < 0:
        e2 = -e2

    proj_e1 = centered @ e1                  # (N,) projection on long axis
    proj_e2 = centered @ e2                  # (N,) projection on short axis

    # Percentile bounds on each axis: discard the most extreme `trim_pct`%
    # of points at each end. This rejects the small population of bezel /
    # transition pixels that survive the colour pass and would otherwise
    # drag a strict bounding box outward (especially on the front edge).
    trim = max(0.0, min(15.0, float(trim_pct)))
    if trim > 0.0:
        e1_lo, e1_hi = np.percentile(proj_e1, [trim, 100.0 - trim])
        e2_lo, e2_hi = np.percentile(proj_e2, [trim, 100.0 - trim])
    else:
        e1_lo, e1_hi = float(proj_e1.min()), float(proj_e1.max())
        e2_lo, e2_hi = float(proj_e2.min()), float(proj_e2.max())

    edge_a = float(e1_hi - e1_lo)
    edge_b = float(e2_hi - e2_lo)
    if edge_a < 1e-3 or edge_b < 1e-3:
        return {"ok": False, "reason": f"degenerate rect from PCA fit ({edge_a:.4f}m × {edge_b:.4f}m)"}

    # 4 corners in (u_tmp, v_tmp) space. Order: long-min/short-min →
    # long-max/short-min → long-max/short-max → long-min/short-max
    # (CCW in PCA frame). Final TL/TR/BR/BL assignment is done by
    # `_assign_corners_TL_TR_BR_BL` based on (X, Z) in camera space, so
    # any consistent CCW order here is fine.
    box_pca = np.array([
        [e1_lo, e2_lo],
        [e1_hi, e2_lo],
        [e1_hi, e2_hi],
        [e1_lo, e2_hi],
    ], dtype=np.float64)
    box2d = centroid_uv + box_pca[:, 0:1] * e1 + box_pca[:, 1:2] * e2  # (4, 2)

    raw_corners = []
    for u_c, v_c in box2d:
        c3 = origin_tmp + float(u_c) * u_tmp + float(v_c) * v_tmp
        raw_corners.append(c3)
    raw_corners_arr = np.stack(raw_corners, axis=0)  # (4, 3)

    area_m2 = edge_a * edge_b

    sorted_corners = _assign_corners_TL_TR_BR_BL(raw_corners_arr)

    return {
        "ok": True,
        "corners_3d": sorted_corners,
        "plane": (float(a), float(b), float(c), float(d)),
        "n_blob_px": best_size,
        "area_m2": area_m2,
        "ransac_inlier_pts": int(ransac_pts.shape[0]),
        "edge_a_m": edge_a,
        "edge_b_m": edge_b,
        "color_info": color_info,
        "trim_pct_used": float(trim),
    }
