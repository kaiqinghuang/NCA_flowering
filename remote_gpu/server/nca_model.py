"""PyTorch NCA simulator — port of the JS ncaStep loop.

Holds the full simulation state on a GPU device:
  - state tensor [1, CH, H, W] (PyTorch NCHW; JS used NHWC)
  - perception kernel (depthwise conv, 4 filters per channel)
  - direction map (alignment + rotation)
  - mask tensor [1, total_models, H, W] (per-pixel weight per model)
  - per-brush soft mask (deposited by stroke events)

The brush "drip + diffusion + life decay" evolution from the JS reference
is intentionally omitted in this skeleton (see TODO at end). The skeleton
keeps a sigmoid-thresholded deposit mask which is sufficient to drive the
priority-stack mask compositor.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from npy_loader import CH, FILTER_N, PERCEPTION_N, NCAWeights
from perlin import Perlin3D, altitude_weights


BASE_MODELS = 4

BASE_RGB = np.array([
    [74, 106, 212],
    [61, 173, 106],
    [201, 162, 39],
    [196, 74, 156],
], dtype=np.float32)

BRUSH_PALETTE = np.array([
    [210, 95, 55], [96, 188, 222], [223, 168, 66],
    [173, 112, 223], [117, 205, 122], [230, 120, 160],
], dtype=np.float32)


@dataclass
class BrushModel:
    id: int
    name: str
    weights: NCAWeights
    color: np.ndarray  # [3] float32 0..255
    mask: torch.Tensor  # [H, W] float32 on device — soft deposit
    active: bool = False


@dataclass
class Params:
    H: int = 270
    W: int = 480
    alignment: int = 2          # 0=cartesian, 1=polar, 2=bipolar
    rotation_deg: float = 0.0
    steps_per_frame: int = 1
    noise_scale: float = 1.5
    octaves: float = 1.0
    half_width: float = 0.02
    noise_z: float = 0.0
    noise_z_speed: float = 0.0  # >0 → animate
    noise_z_scale: float = 1.0
    mask_threshold: float = 0.5
    mask_edge_sharpness: float = 30.0
    disturbance: bool = False
    show_mask_tint: bool = False
    active: bool = True


def build_perception_kernel(device: torch.device) -> torch.Tensor:
    """Depthwise [CH*FILTER_N, 1, 3, 3] conv weight: identity, sobel_x, sobel_y, lap."""
    identity = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    sobel_x = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], dtype=np.float32) / 8.0
    sobel_y = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1], dtype=np.float32) / 8.0
    lap = np.array([1, 2, 1, 2, -12, 2, 1, 2, 1], dtype=np.float32) / 8.0
    filters = np.stack([identity, sobel_x, sobel_y, lap], axis=0).reshape(FILTER_N, 3, 3)
    # Repeat per channel: depthwise weight shape [CH*FILTER_N, 1, 3, 3]
    out = np.empty((CH * FILTER_N, 1, 3, 3), dtype=np.float32)
    for c in range(CH):
        for f in range(FILTER_N):
            out[c * FILTER_N + f, 0] = filters[f]
    return torch.from_numpy(out).to(device)


def build_direction_map(H: int, W: int, alignment: int, rotation_deg: float, device: torch.device):
    """Returns (cos_map, sin_map) each [1, 1, H, W]."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    if alignment == 0:  # cartesian: down
        dir_x = np.zeros((H, W), dtype=np.float32)
        dir_y = np.ones((H, W), dtype=np.float32)
    elif alignment == 1:  # polar: outward from center
        dx = xs - W * 0.5
        dy = ys - H * 0.5
        ln = np.sqrt(dx * dx + dy * dy)
        ln = np.where(ln > 1e-3, ln, 1.0)
        dir_x = np.where(ln > 1e-3, dx / ln, 0.0).astype(np.float32)
        dir_y = np.where(ln > 1e-3, dy / ln, 1.0).astype(np.float32)
    else:  # bipolar (default in JS)
        v1x = xs - W * 0.25
        v1y = ys - H * 0.25
        v2x = W * 0.75 - xs
        v2y = H * 0.75 - ys
        l1 = np.sqrt(v1x * v1x + v1y * v1y)
        l2 = np.sqrt(v2x * v2x + v2y * v2y)
        l1_3 = np.where(l1 > 0, l1 ** 3, 1.0)
        l2_3 = np.where(l2 > 0, l2 ** 3, 1.0)
        dx = v1x / l1_3 + v2x / l2_3
        dy = v1y / l1_3 + v2y / l2_3
        ln = np.sqrt(dx * dx + dy * dy)
        ok = ln > 1e-3
        dir_x = np.where(ok, dx / np.where(ok, ln, 1.0), 0.0).astype(np.float32)
        dir_y = np.where(ok, dy / np.where(ok, ln, 1.0), 1.0).astype(np.float32)

    rad = math.radians(rotation_deg)
    g_cos, g_sin = math.cos(rad), math.sin(rad)
    rdx = dir_x * g_cos - dir_y * g_sin
    rdy = dir_x * g_sin + dir_y * g_cos
    # JS layout (counterintuitive): cosData = rdy, sinData = rdx. Preserve.
    cos_t = torch.from_numpy(rdy).to(device).reshape(1, 1, H, W)
    sin_t = torch.from_numpy(rdx).to(device).reshape(1, 1, H, W)
    return cos_t, sin_t


def periodic_pad(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Wrap-around pad on H and W axes. x shape: [N, C, H, W]."""
    x = torch.cat([x[:, :, -pad:, :], x, x[:, :, :pad, :]], dim=2)
    x = torch.cat([x[:, :, :, -pad:], x, x[:, :, :, :pad]], dim=3)
    return x


def compute_dx(perceived: torch.Tensor, w: NCAWeights) -> torch.Tensor:
    """perceived: [N, perception_n, H, W] → dx: [N, CH, H, W]."""
    # Layer 1: 1x1 conv. Stored weight [in, out] → conv2d weight [out, in, 1, 1].
    w1 = w.dense1k.t().reshape(w.hidden, PERCEPTION_N, 1, 1)
    h1 = F.conv2d(perceived, w1, bias=w.dense1b)
    h1 = F.relu(h1)
    w2 = w.dense2k.t().reshape(CH, w.hidden, 1, 1)
    return F.conv2d(h1, w2, bias=w.dense2b)


class NCASimulator:
    def __init__(self, params: Params, device: torch.device, seed: int = 1):
        self.p = params
        self.device = device
        # Single big lock — all mutations + step + render snapshot serialize on this.
        # Critical sections are short (kernel queue submits + GPU clones).
        self.lock = threading.RLock()
        self.state = torch.zeros(1, CH, params.H, params.W, device=device)
        self.perception = build_perception_kernel(device)
        self.dir_cos, self.dir_sin = build_direction_map(
            params.H, params.W, params.alignment, params.rotation_deg, device
        )
        self.perlin = Perlin3D(seed)
        self.altitude = self._compute_altitude()  # [4, H, W] on device
        self.altitude_dirty = False
        self.base_models: list[Optional[NCAWeights]] = [None, None, None, None]
        self.brush_models: list[BrushModel] = []
        self.mask: Optional[torch.Tensor] = None
        self.mask_dirty = True
        self._next_brush_id = 1
        self.step_count = 0

    # ---------- Altitude / mask ----------
    def _compute_altitude(self) -> torch.Tensor:
        wts = altitude_weights(
            self.p.H, self.p.W, self.perlin,
            noise_scale=self.p.noise_scale,
            octaves=self.p.octaves,
            half_width=self.p.half_width,
            offset_x=0.0, offset_y=0.0,
            z=self.p.noise_z * self.p.noise_z_scale,
        )
        return torch.from_numpy(wts).permute(2, 0, 1).contiguous().to(self.device)  # [4, H, W]

    def reseed_noise(self, seed: int):
        with self.lock:
            self.perlin = Perlin3D(seed)
            self.p.noise_z = 0.0
            self.altitude_dirty = True
            self.mask_dirty = True

    def mark_altitude_dirty(self):
        with self.lock:
            self.altitude_dirty = True
            self.mask_dirty = True

    def _rebuild_mask(self):
        """Priority-stack composition: brushes (newest = bottom) over altitude bands."""
        if self.altitude_dirty:
            self.altitude = self._compute_altitude()
            self.altitude_dirty = False

        n_brush = len(self.brush_models)
        total = BASE_MODELS + n_brush
        H, W = self.p.H, self.p.W
        mask = torch.zeros(1, total, H, W, device=self.device)

        if n_brush > 0:
            thr = float(np.clip(self.p.mask_threshold, 0.02, 0.98))
            sharp = self.p.mask_edge_sharpness
            # Per-brush sigmoid claim from soft mask
            claims = torch.stack([
                torch.sigmoid((bm.mask - thr) * sharp) * (bm.mask > 1e-6).float()
                for bm in self.brush_models
            ], dim=0)  # [n_brush, H, W]
            # Priority stack: iterate brushes in reverse, accumulate remainder
            rem = torch.ones(H, W, device=self.device)
            for k in range(n_brush - 1, -1, -1):
                wv = rem * claims[k]
                mask[0, BASE_MODELS + k] = wv
                rem = rem * (1.0 - claims[k])
            # Base bands receive the remainder
            for b in range(BASE_MODELS):
                mask[0, b] = self.altitude[b] * rem
        else:
            for b in range(BASE_MODELS):
                mask[0, b] = self.altitude[b]

        self.mask = mask
        self.mask_dirty = False

    # ---------- Direction ----------
    def update_direction(self):
        with self.lock:
            self.dir_cos, self.dir_sin = build_direction_map(
                self.p.H, self.p.W, self.p.alignment, self.p.rotation_deg, self.device
            )

    # ---------- Models ----------
    def set_base_model(self, slot: int, weights: Optional[NCAWeights]):
        assert 0 <= slot < BASE_MODELS
        with self.lock:
            self.base_models[slot] = weights

    def add_brush_model(self, weights: NCAWeights) -> int:
        with self.lock:
            bm_id = self._next_brush_id
            self._next_brush_id += 1
            color = BRUSH_PALETTE[(bm_id - 1) % len(BRUSH_PALETTE)]
            bm = BrushModel(
                id=bm_id, name=weights.name, weights=weights, color=color,
                mask=torch.zeros(self.p.H, self.p.W, device=self.device),
            )
            self.brush_models.append(bm)
            self.mask_dirty = True
            return bm_id

    def remove_brush_model(self, bm_id: int):
        with self.lock:
            self.brush_models = [b for b in self.brush_models if b.id != bm_id]
            self.mask_dirty = True

    def get_brush(self, bm_id: int) -> Optional[BrushModel]:
        # Read-only; list traversal is brief, GIL gives atomicity for ref reads.
        for bm in self.brush_models:
            if bm.id == bm_id:
                return bm
        return None

    # ---------- Brush deposit (skeleton: simple disk, no drip/decay) ----------
    def stamp_disk(self, bm_id: int, cx: int, cy: int, radius: float, erase: bool = False):
        with self.lock:
            bm = self.get_brush(bm_id)
            if bm is None:
                return
            H, W = self.p.H, self.p.W
            r = max(0.5, radius)
            ri = int(math.ceil(r))
            x0, x1 = max(0, cx - ri), min(W, cx + ri + 1)
            y0, y1 = max(0, cy - ri), min(H, cy + ri + 1)
            if x0 >= x1 or y0 >= y1:
                return
            ys = torch.arange(y0, y1, device=self.device).reshape(-1, 1).float()
            xs = torch.arange(x0, x1, device=self.device).reshape(1, -1).float()
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            inside = d2 <= r * r
            if erase:
                bm.mask[y0:y1, x0:x1] = torch.where(inside, torch.zeros_like(bm.mask[y0:y1, x0:x1]),
                                                    bm.mask[y0:y1, x0:x1])
            else:
                per_pixel = 0.08 + 0.12 / max(1.0, r)
                add = torch.where(inside, torch.full_like(bm.mask[y0:y1, x0:x1], per_pixel),
                                  torch.zeros_like(bm.mask[y0:y1, x0:x1]))
                bm.mask[y0:y1, x0:x1] = torch.clamp(bm.mask[y0:y1, x0:x1] + add, 0.0, 1.0)
            bm.active = bool((bm.mask > 1e-6).any().item())
            self.mask_dirty = True

    def paint_segment(self, bm_id: int, x0: int, y0: int, x1: int, y1: int,
                      radius: float, erase: bool = False):
        """Bresenham-rasterize a line and stamp a disk at every grid step."""
        with self.lock:
            bm = self.get_brush(bm_id)
            if bm is None:
                return
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            cx, cy = x0, y0
            # Cap iterations to prevent runaway on bad input
            max_iter = max(dx, dy) + 1
            for _ in range(max_iter):
                self._stamp_disk_unlocked(bm, cx, cy, radius, erase)
                if cx == x1 and cy == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx += sx
                if e2 < dx:
                    err += dx
                    cy += sy

    def _stamp_disk_unlocked(self, bm: BrushModel, cx: int, cy: int,
                             radius: float, erase: bool):
        """Same as stamp_disk body but assumes lock already held."""
        H, W = self.p.H, self.p.W
        r = max(0.5, radius)
        ri = int(math.ceil(r))
        x0, x1 = max(0, cx - ri), min(W, cx + ri + 1)
        y0, y1 = max(0, cy - ri), min(H, cy + ri + 1)
        if x0 >= x1 or y0 >= y1:
            return
        ys = torch.arange(y0, y1, device=self.device).reshape(-1, 1).float()
        xs = torch.arange(x0, x1, device=self.device).reshape(1, -1).float()
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        inside = d2 <= r * r
        sub = bm.mask[y0:y1, x0:x1]
        if erase:
            bm.mask[y0:y1, x0:x1] = torch.where(inside, torch.zeros_like(sub), sub)
        else:
            per_pixel = 0.08 + 0.12 / max(1.0, r)
            add = torch.where(inside, torch.full_like(sub, per_pixel), torch.zeros_like(sub))
            bm.mask[y0:y1, x0:x1] = torch.clamp(sub + add, 0.0, 1.0)
        bm.active = True
        self.mask_dirty = True

    def clear_brush_mask(self, bm_id: int):
        with self.lock:
            bm = self.get_brush(bm_id)
            if bm is None:
                return
            bm.mask.zero_()
            bm.active = False
            self.mask_dirty = True

    def clear_state(self):
        with self.lock:
            self.state.zero_()
            self.step_count = 0

    # ---------- NCA step ----------
    def count_loaded_models(self) -> int:
        n = sum(1 for m in self.base_models if m is not None)
        return n + len(self.brush_models)

    @torch.no_grad()
    def step(self):
        with self.lock:
            if self.count_loaded_models() == 0:
                return
            if self.mask_dirty or self.mask is None:
                self._rebuild_mask()
            if self.p.noise_z_speed > 0:
                self.p.noise_z += self.p.noise_z_speed
                self.altitude_dirty = True
                self.mask_dirty = True

            H, W = self.p.H, self.p.W
            x = self.state
            # Perception: depthwise 3x3 with periodic padding
            padded = periodic_pad(x, 1)
            raw = F.conv2d(padded, self.perception, groups=CH)  # [1, CH*4, H, W]

            # Split identity/dx/dy/lap and rotate dx/dy by per-pixel direction
            p4 = raw.reshape(1, CH, FILTER_N, H, W)
            p_id = p4[:, :, 0]
            p_dx = p4[:, :, 1]
            p_dy = p4[:, :, 2]
            p_lap = p4[:, :, 3]
            c = self.dir_cos  # [1,1,H,W]
            s = self.dir_sin
            rot_dx = p_dx * c - p_dy * s
            rot_dy = p_dx * s + p_dy * c
            perceived = torch.stack([p_id, rot_dx, rot_dy, p_lap], dim=2)  # [1,CH,4,H,W]
            perceived = perceived.reshape(1, CH * FILTER_N, H, W)

            # Snapshot model lists inside the lock so iteration is consistent
            base_snap = list(self.base_models)
            brush_snap = list(self.brush_models)
            n_brush = len(brush_snap)
            total = BASE_MODELS + n_brush
            combined = torch.zeros_like(x)
            for k in range(total):
                if k < BASE_MODELS:
                    m = base_snap[k]
                else:
                    bm = brush_snap[k - BASE_MODELS]
                    m = bm.weights
                    if not bm.active:
                        continue
                if m is None:
                    continue
                dxk = compute_dx(perceived, m)
                # mask was built with the same n_brush snapshot; channel k matches
                ms = self.mask[:, k:k + 1]
                combined = combined + dxk * ms

            update = (torch.rand(1, 1, H, W, device=self.device) < 0.5).float()
            self.state = torch.clamp(x + combined * update, -2.0, 2.0)
            self.step_count += 1

    # ---------- Render ----------
    @torch.no_grad()
    def render_rgb(self) -> np.ndarray:
        """Map state[:, 0:3] → uint8 [H, W, 3] for client display."""
        rgb = self.state[0, :3].clamp(-1, 1) * 0.5 + 0.5  # [3, H, W]
        if self.p.show_mask_tint and self.mask is not None:
            tint = self._compose_tint()  # [3, H, W] in 0..1
            rgb = rgb * 0.76 + tint * 0.24
        rgb = rgb.clamp(0, 1).permute(1, 2, 0).mul(255).to(torch.uint8)
        return rgb.cpu().numpy()

    def _compose_tint(self) -> torch.Tensor:
        """Weighted mix of band/brush palette colors using current mask."""
        H, W = self.p.H, self.p.W
        out = torch.zeros(3, H, W, device=self.device)
        if self.mask is None:
            return out
        for b in range(BASE_MODELS):
            col = torch.tensor(BASE_RGB[b] / 255.0, device=self.device).reshape(3, 1, 1)
            out = out + col * self.mask[0, b:b + 1]
        for k, bm in enumerate(self.brush_models):
            col = torch.tensor(bm.color / 255.0, device=self.device).reshape(3, 1, 1)
            out = out + col * self.mask[0, BASE_MODELS + k:BASE_MODELS + k + 1]
        return out

    # ---------- Disturbance ----------
    @torch.no_grad()
    def apply_disturbance(self, t_seconds: float):
        H, W = self.p.H, self.p.W
        cx = math.sin(t_seconds * 4.0) * W / 3 + W / 2
        cy = math.cos(t_seconds * 6.2) * H / 3 + H / 2
        ys = torch.arange(H, device=self.device).reshape(-1, 1).float()
        xs = torch.arange(W, device=self.device).reshape(1, -1).float()
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        keep = (d2 > 20 * 20).float().reshape(1, 1, H, W)
        self.state = self.state * keep


# TODO: port full brush evolution (drip propagation + heat-equation diffusion +
# life-decay aging) from JS evolveSingleBrushMask. The skeleton uses a static
# soft mask which is enough to validate end-to-end streaming.
