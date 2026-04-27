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
        on_surface  : (H, W) bool — within ``surface_eps_m`` of plane (NOT
                      gated by polygon — this is the *plane-fit*
                      visualization, which should track the entire TV
                      surface even when the calibrated polygon is an
                      inscribed rectangle that's tighter than the TV).
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

    # Plane-fit visualization: every valid pixel within ``surface_eps_m`` of
    # the calibrated plane, NOT clipped by the polygon. The 4-corner polygon
    # is intentionally an INSCRIBED rectangle (slightly smaller than the TV
    # so all 4 corners stay inside the on-plane blob), so gating
    # ``on_surface`` by the polygon would cosmetically shrink the magenta
    # fill and make the plane fit *look* worse than it is. ``in_box``
    # below (used by the runtime fingertip detector) IS still gated by
    # the polygon — that's correct, the interaction box should match the
    # calibrated quadrilateral.
    on_surface = valid & np.isfinite(s) & (np.abs(s) <= surface_eps_m)
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


def _max_rect_in_binary_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Largest axis-aligned all-ones rectangle in a binary mask.

    Standard histogram-stack method, O(H·W):

        For each row r, ``h[c]`` = #consecutive 1-cells in column ``c``
        ending at row ``r``. The largest rectangle of 1s ending at row r
        is the largest rectangle in the histogram h, found in O(W) with a
        monotonic stack. Track the global max over all rows.

    Args:
        mask: (H, W) uint8/bool, non-zero = inside blob.

    Returns:
        ``(top, left, height, width)`` of the largest all-ones rectangle.
        ``mask[top:top+height, left:left+width]`` is fully 1. If the mask
        has no 1s, returns ``(0, 0, 0, 0)``.
    """
    H, W = int(mask.shape[0]), int(mask.shape[1])
    if H == 0 or W == 0:
        return (0, 0, 0, 0)
    h = [0] * W
    best_area = 0
    best = (0, 0, 0, 0)
    for r in range(H):
        row = mask[r]
        for c in range(W):
            h[c] = h[c] + 1 if row[c] else 0
        # Largest rect in histogram h (with sentinel via c == W → cur = 0).
        stack: list[int] = []
        for c in range(W + 1):
            cur = 0 if c == W else h[c]
            while stack and h[stack[-1]] >= cur:
                top_h = h[stack.pop()]
                left = stack[-1] if stack else -1
                width = c - left - 1
                area = top_h * width
                if area > best_area:
                    best_area = area
                    best = (r - top_h + 1, left + 1, top_h, width)
            stack.append(c)
    return best


def _largest_inscribed_aligned_rect_uv(
    pts2d: np.ndarray,
    rect_center: np.ndarray,
    ru: np.ndarray,
    rv: np.ndarray,
    grid_step_m: float,
    cv2_mod,
    tolerance_m: float = 0.025,
) -> Optional[np.ndarray]:
    """Largest rectangle aligned with ``(ru, rv)`` and (mostly) inscribed in the blob.

    Unlike ``cv2.minAreaRect`` (which produces an *outer* bounding rect
    whose corners stick out beyond the blob — visible in the debug
    overlay as a polygon outline that overshoots the magenta on-plane
    fill), this rasterises the blob's plane-projected (u, v) points into
    a binary grid in the rect-aligned frame, then finds the largest
    axis-aligned all-ones rectangle in that grid via the histogram
    method.

    A pure 100%-inscribed LIR tends to come out *much* smaller than the
    actual TV — depth dropouts, specular reflections and slight bezel
    bleed leave small holes / jagged edges in the blob, and any single
    hole inside the rect kills it. To recover a sensible rect we first
    DILATE the rasterised blob by ``tolerance_m`` (rounded to grid
    cells); the LIR is then computed on the dilated mask, so the
    returned rectangle:

        * spans across small inner holes / specks of black noise inside
          the blob (those holes are ≤ tolerance and got filled by
          dilation), and
        * may extend up to ``tolerance_m`` past the blob's actual outer
          boundary (the dilation grew the boundary outward).

    With ``tolerance_m = 0`` this reduces to the strict-inscribed LIR
    where every cell of the returned rectangle is guaranteed to be a
    blob cell.

    Args:
        pts2d:        (N, 2) blob points in the plane's (u, v) coords (m).
        rect_center:  (2,)  origin of the rect-aligned frame in (u, v).
        ru, rv:       (2,)  unit, perpendicular axes of the rect frame.
        grid_step_m:  rasterisation step (meters/cell). 5–10 mm is a good
                      range: small enough that quantisation loses < 1 cm
                      of corner accuracy, large enough that adjacent
                      depth pixels (~4 mm at 1.5 m range) land in the
                      same cell so the grid is dense.
        cv2_mod:      passed-in ``cv2`` module (avoid re-importing).
        tolerance_m:  how much "black noise" the rect is allowed to
                      contain — larger ⇒ rect can grow across bigger
                      holes / jagged edges, but is allowed to extend
                      that much past the blob's true boundary too.
                      Default 25 mm covers typical Kinect depth-dropout
                      gaps and TV-content-induced holes; bump it up if
                      the LIR is consistently coming out smaller than
                      the actual TV.

    Returns:
        ``(4, 2)`` ndarray of (u, v) corners in rect-frame TL/TR/BR/BL
        order, *or* ``None`` if no usable rectangle exists (caller
        should fall back to the bounding rect).
    """
    if pts2d.shape[0] < 4:
        return None
    rel = pts2d - rect_center
    proj_u = rel @ ru
    proj_v = rel @ rv

    u_lo, u_hi = float(proj_u.min()), float(proj_u.max())
    v_lo, v_hi = float(proj_v.min()), float(proj_v.max())
    if (u_hi - u_lo) < 4 * grid_step_m or (v_hi - v_lo) < 4 * grid_step_m:
        return None

    # Pad the grid by tolerance + a couple of cells so the dilated blob
    # has room to grow without clipping at the grid border (which would
    # artificially cap the rect on that side).
    tol_cells = max(0, int(round(float(tolerance_m) / float(grid_step_m))))
    pad = (tol_cells + 2) * grid_step_m
    u_origin = u_lo - pad
    v_origin = v_lo - pad
    W = int(np.ceil((u_hi - u_lo + 2.0 * pad) / grid_step_m)) + 1
    H = int(np.ceil((v_hi - v_lo + 2.0 * pad) / grid_step_m)) + 1
    if H < 4 or W < 4 or H * W > 1_000_000:
        return None

    j = ((proj_u - u_origin) / grid_step_m).astype(np.int32)
    i = ((proj_v - v_origin) / grid_step_m).astype(np.int32)
    np.clip(j, 0, W - 1, out=j)
    np.clip(i, 0, H - 1, out=i)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[i, j] = 1

    # Always close 1-cell pinholes from grid-quantisation aliasing
    # (independent of user-tunable tolerance).
    base_ker = cv2_mod.getStructuringElement(cv2_mod.MORPH_RECT, (3, 3))
    mask = cv2_mod.morphologyEx(mask, cv2_mod.MORPH_CLOSE, base_ker)

    # Tolerance-driven dilation. This is the user-facing knob: every cell
    # of the dilated mask is either a blob cell or within ``tol_cells``
    # of one. The LIR computed on this is therefore allowed to cover
    # non-blob "noise" cells, but only up to that radius.
    if tol_cells > 0:
        k = 2 * tol_cells + 1
        ker = cv2_mod.getStructuringElement(cv2_mod.MORPH_RECT, (k, k))
        mask = cv2_mod.dilate(mask, ker)

    # Largest CC: defends against any stray dot that survived earlier
    # depth/colour morphology and got dilated into a small island.
    n_lbl, labels, stats, _ = cv2_mod.connectedComponentsWithStats(mask, connectivity=8)
    if n_lbl < 2:
        return None
    sizes = stats[1:, cv2_mod.CC_STAT_AREA]
    best_lbl = 1 + int(np.argmax(sizes))
    mask = (labels == best_lbl).astype(np.uint8)

    rect_top, rect_left, rect_h, rect_w = _max_rect_in_binary_mask(mask)
    if rect_w < 2 or rect_h < 2:
        return None

    # Inscribed rect in rect-frame coords. Use cell *inner* corners
    # (rect_left+0.5, rect_top+0.5) → (rect_left+rect_w-0.5, rect_top+rect_h-0.5)
    # so we don't leak ½ cell out of the rasterised mask near the edges.
    u0 = u_origin + (rect_left + 0.5) * grid_step_m
    u1 = u_origin + (rect_left + rect_w - 0.5) * grid_step_m
    v0 = v_origin + (rect_top + 0.5) * grid_step_m
    v1 = v_origin + (rect_top + rect_h - 0.5) * grid_step_m

    rect_corners_local = np.array(
        [[u0, v0], [u1, v0], [u1, v1], [u0, v1]],  # rect-frame TL/TR/BR/BL
        dtype=np.float64,
    )
    uv_corners = rect_center + (
        rect_corners_local[:, 0:1] * ru + rect_corners_local[:, 1:2] * rv
    )
    return uv_corners


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
    inscribe_tolerance_m: float = 0.025,
    inscribe_grid_step_m: float = 0.007,
) -> dict:
    """One-shot TV plane + 4-corner derivation, depth + colour fused.

    Steps:
        1. Stack ``depth_frames`` into one (sub-sampled) point cloud.
        2. RANSAC fit the dominant plane (refit via SVD on inliers).
        3. Compute on-plane mask on the LAST frame; morph-open to break
           thin depth-connected bridges to neighbouring co-planar wood
           strips (cheap, doesn't fight us further down).
        4. Largest connected component = depth-only TV blob candidate.
        5. **(NEW) Colour refinement.** If a colour frame and the SDK's
           depth→colour mapping were supplied, look up each blob pixel's
           RGB value and discard everything brighter than ``color_max_v``
           — wood/bezels are bright, the TV screen is near-black. Take
           the largest connected component of what survives. (Falls back
           gracefully to the depth-only blob if mapping is unavailable
           or the filter wipes everything.)
        6. Project the refined blob to the plane (u, v); use
           ``cv2.minAreaRect`` ONLY for the orientation, then find the
           largest rectangle (mostly) INSCRIBED in the blob along that
           orientation (LIR — Largest Inscribed Rectangle, with
           ``inscribe_tolerance_m`` slack so small black-noise holes
           inside the blob are forgiven). The 4 corners and 4 edges of
           the resulting rectangle therefore sit inside the on-plane
           blob (within tolerance), unlike ``minAreaRect``'s outer
           bounding box whose corners always overshoot.
        7. Reconstruct 3D corners → sort into TL/TR/BR/BL.

    Returns a dict (always with ``ok``):
        ok, reason
        corners_3d        : list of 4 (x, y, z) tuples (TL, TR, BR, BL)
        plane             : (a, b, c, d) with origin-side positive
        n_blob_px         : int   pixel count of the chosen TV blob
                                  (after colour refinement, if applied)
        area_m2           : float TV rectangle area in m²
        ransac_inlier_pts : int   how many points fed to RANSAC
        color_info        : diagnostic dict from the colour-refine stage
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

    # ---- 6. Project refined blob to plane → orientation + INSCRIBED rect. ----
    # Why not the bounding minAreaRect on its own: it must enclose every
    # blob pixel, so its 4 corners stick out into empty space whenever
    # the on-plane blob is anything less than a perfect rectangle —
    # nearly always, because the screen edges always have some depth
    # dropout / noise. In the debug overlay this looks like the magenta
    # polygon outline overshooting the magenta on-plane fill at one or
    # more corners.
    #
    # Fix: use minAreaRect ONLY to obtain a stable orientation, then
    # find the largest rectangle (mostly) INSCRIBED in the blob along
    # that orientation. The inscribed rectangle is computed by
    # rasterising the blob in the rect-aligned frame, dilating by
    # ``inscribe_tolerance_m`` to forgive small black-noise holes, and
    # running the standard histogram-stack LIR algorithm. Every corner
    # and every edge of the returned rectangle therefore sits inside
    # the on-plane blob (within tolerance), so it never overshoots.
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

    pts2d = np.stack([u_coords, v_coords], axis=1)  # (N, 2) float64, full
    if pts2d.shape[0] < 4:
        return {"ok": False, "reason": "refined blob has <4 in-plane points"}

    # Stable orientation from minAreaRect (subsampled for speed; the
    # angle is robust to subsampling, the bounding box itself we
    # discard).
    pts2d_f32 = pts2d.astype(np.float32)
    if pts2d_f32.shape[0] > 8000:
        idx = rng.choice(pts2d_f32.shape[0], 8000, replace=False)
        sample = pts2d_f32[idx]
    else:
        sample = pts2d_f32
    rect = cv2.minAreaRect(sample.reshape(-1, 1, 2))
    rect_center = np.array(rect[0], dtype=np.float64)
    angle_deg = float(rect[2])
    theta = float(np.deg2rad(angle_deg))
    ru = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    rv = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    # Largest rectangle (mostly) inscribed in the blob, aligned with
    # (ru, rv). ``inscribe_tolerance_m`` is the user-tunable knob: how
    # much "black noise" the rect is allowed to contain — bumping it up
    # lets the rect span across bigger holes / jagged edges, but means
    # the rect may extend that much past the blob's actual outer
    # boundary too. Default 25 mm is enough to forgive typical Kinect
    # depth-dropout gaps and TV-content-induced holes without
    # noticeably overshooting the real screen.
    box_uv = _largest_inscribed_aligned_rect_uv(
        pts2d, rect_center, ru, rv,
        grid_step_m=float(inscribe_grid_step_m),
        cv2_mod=cv2,
        tolerance_m=float(inscribe_tolerance_m),
    )
    inscribed = box_uv is not None
    if box_uv is None:
        # Fallback (very unusual — degenerate blob): use the bounding
        # rect so calibration still completes; corners may sit slightly
        # outside the blob in this branch.
        box_uv = np.asarray(cv2.boxPoints(rect), dtype=np.float64)

    raw_corners_arr = np.stack(
        [origin_tmp + float(uc) * u_tmp + float(vc) * v_tmp
         for uc, vc in box_uv],
        axis=0,
    )  # (4, 3)

    # LIR returns a perfect rectangle in the (u, v) plane, so consecutive
    # edge lengths are the rect's two side lengths. (For the bounding-box
    # fallback the same formulas hold — boxPoints is also a rectangle.)
    edge_a = float(np.linalg.norm(raw_corners_arr[1] - raw_corners_arr[0]))
    edge_b = float(np.linalg.norm(raw_corners_arr[2] - raw_corners_arr[1]))
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
        "inscribed": bool(inscribed),
        "inscribe_tolerance_m": float(inscribe_tolerance_m),
    }
