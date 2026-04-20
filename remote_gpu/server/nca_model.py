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
    # Per-pixel "wetness" — only the MAIN brush footprint contributes (not splatters).
    # Drips can only spawn where this exceeds threshold; the value above threshold
    # also controls drip length (heavily-wet → long & persistent, lightly-wet → short).
    accum: torch.Tensor  # [H, W] float32 on device
    # Drip particles (CPU bookkeeping, small N, evolved each step)
    drips: list = field(default_factory=list)
    active: bool = False


# Hard cap to keep evolve_drips bounded under heavy painting
MAX_DRIPS_PER_BRUSH = 256

# Hard cap on how many stamps a single paint segment can place. Without this,
# a fast pointer move (e.g. 600 px between two 60Hz events) would flood the
# GPU dispatch queue with 600 stamps and stall the sim loop for tens of ms.
# Each stamp issues ~10 dispatches on MPS, so this cap directly bounds the
# worst-case step time spent in painting. 20 keeps fast strokes well under
# the 33ms / 30fps budget while still feeling continuous (the splatter
# texture hides the slight gaps that appear at very high pointer speed).
MAX_STAMPS_PER_SEGMENT = 20


@dataclass
class Params:
    H: int = 270
    W: int = 480
    alignment: int = 2          # 0=cartesian, 1=polar, 2=bipolar
    rotation_deg: float = 0.0
    steps_per_frame: int = 1
    noise_scale: float = 5.0
    octaves: float = 1.0
    half_width: float = 0.02
    layer_freq_spread: float = 0.55
    noise_z: float = 0.0
    noise_z_speed: float = 0.0  # >0 → animate
    noise_z_scale: float = 1.0
    mask_threshold: float = 0.5
    mask_edge_sharpness: float = 33.0
    # Spray paint feel (matches background_multiPerlin_4move_highResol_cpugpu_rotate.html)
    spray_splatter_amount: int = 10     # # of small jittered dots around main disk
    spray_splatter_radius: float = 35.0 # max offset of splatter dots (in px)
    spray_drip_threshold: float = 0.40  # wet build-up before a drip can spawn
    spray_drip_speed: float = 0.40      # 0..1, higher = faster drips
    spray_drip_wobble: float = 0.25     # 0..1, sideways drift while dripping
    spray_drip_min_width: float = 1.0
    spray_drip_chance: float = 0.12     # per-stamp spawn prob once wet > threshold
    drip_gravity: int = 0               # 0=down, 1=up, 2=left, 3=right
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
        self.perlin_layers = self._make_perlin_layers(seed)
        self.altitude = self._compute_altitude()  # [4, H, W] on device
        self.altitude_dirty = False
        self.base_models: list[Optional[NCAWeights]] = [None, None, None, None]
        self.brush_models: list[BrushModel] = []
        # Per-pixel latest brush owner id (-1 = none). Latest stroke wins.
        self.stroke_owner = torch.full((params.H, params.W), -1, dtype=torch.int32, device=device)
        self.mask: Optional[torch.Tensor] = None
        self.mask_dirty = True
        self._next_brush_id = 1
        self.step_count = 0

    # ---------- Altitude / mask ----------
    @staticmethod
    def _make_perlin_layers(seed: int) -> list[Perlin3D]:
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        layers: list[Perlin3D] = []
        for i in range(BASE_MODELS):
            s = int(rng.integers(0, 2**32, dtype=np.uint32))
            s ^= int(np.uint32(i * 0x9E3779B9))
            layers.append(Perlin3D(s))
        return layers

    def _compute_altitude(self) -> torch.Tensor:
        wts = altitude_weights(
            self.p.H, self.p.W, self.perlin_layers,
            noise_scale=self.p.noise_scale,
            octaves=self.p.octaves,
            half_width=self.p.half_width,
            offset_x=0.0, offset_y=0.0,
            z=self.p.noise_z * self.p.noise_z_scale,
            threshold=self.p.mask_threshold,
            edge_sharpness=self.p.mask_edge_sharpness,
            layer_freq_spread=self.p.layer_freq_spread,
        )
        return torch.from_numpy(wts).permute(2, 0, 1).contiguous().to(self.device)  # [4, H, W]

    def reseed_noise(self, seed: int):
        with self.lock:
            self.perlin_layers = self._make_perlin_layers(seed)
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
            claims_list = []
            for bm in self.brush_models:
                owned = (self.stroke_owner == int(bm.id)).float()
                bm.active = bool(owned.any().item())
                claim = torch.sigmoid((bm.mask - thr) * sharp) * (bm.mask > 1e-6).float() * owned
                claims_list.append(claim)
            claims = torch.stack(claims_list, dim=0)  # [n_brush, H, W]
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
                accum=torch.zeros(self.p.H, self.p.W, device=self.device),
            )
            self.brush_models.append(bm)
            self.mask_dirty = True
            return bm_id

    def remove_brush_model(self, bm_id: int):
        with self.lock:
            self.stroke_owner = torch.where(
                self.stroke_owner == int(bm_id),
                torch.full_like(self.stroke_owner, -1),
                self.stroke_owner,
            )
            for bm in self.brush_models:
                if bm.id == bm_id:
                    bm.drips.clear()
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
            self._stamp_disk_unlocked(bm, cx, cy, radius, erase)

    def paint_segment(self, bm_id: int, x0: int, y0: int, x1: int, y1: int,
                      radius: float, erase: bool = False):
        """Stamp a disk at evenly-spaced points along (x0,y0)→(x1,y1).

        Inspired by the dripping-spray library: instead of rasterizing every
        pixel along the segment (which is O(distance) and floods the GPU
        dispatch queue on fast strokes), we space stamps by roughly the brush
        radius. Long fast strokes therefore become a *trail of spray clusters*
        with small gaps — exactly the spray-can feel — at constant cost.
        """
        with self.lock:
            bm = self.get_brush(bm_id)
            if bm is None:
                return
            distance = math.hypot(x1 - x0, y1 - y0)
            if distance < 0.5:
                self._stamp_disk_unlocked(bm, x0, y0, radius, erase)
                return
            # Spacing ≈ 80% of brush radius keeps the main disk visually
            # continuous (~20% overlap) under normal speeds. For very fast
            # strokes the hard cap kicks in and we accept slight gaps.
            target_spacing = max(1.0, float(radius) * 0.8)
            n = max(1, int(round(distance / target_spacing)))
            n = min(n, MAX_STAMPS_PER_SEGMENT)
            inv_n = 1.0 / n
            for i in range(n + 1):
                t = i * inv_n
                cx = int(round(x0 + (x1 - x0) * t))
                cy = int(round(y0 + (y1 - y0) * t))
                self._stamp_disk_unlocked(bm, cx, cy, radius, erase)

    def _stamp_disk_unlocked(self, bm: BrushModel, cx: int, cy: int,
                             radius: float, erase: bool):
        """Spray-paint stamp: main disk + N splatter dots, with drip spawn.

        Performance notes:
          * Splatter offsets are computed on CPU (small N, ~5).
          * The union of all dot footprints is computed once on GPU as a
            single bool mask; we then do one combined `where` write into
            the brush mask. This avoids N separate GPU dispatches.
          * Drip spawn uses a CPU `wet_estimate` accumulator on the brush —
            no per-stamp GPU→CPU sync.
        """
        H, W = self.p.H, self.p.W
        size = max(0.5, float(radius))

        if erase:
            # Erase: behave like the old hard-clear inside a slightly bigger disk.
            r = size + 2.0
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
            owner = self.stroke_owner[y0:y1, x0:x1]
            erasable = inside & (owner == int(bm.id))
            bm.mask[y0:y1, x0:x1] = torch.where(erasable, torch.zeros_like(sub), sub)
            sub_acc = bm.accum[y0:y1, x0:x1]
            bm.accum[y0:y1, x0:x1] = torch.where(erasable, torch.zeros_like(sub_acc), sub_acc)
            self.stroke_owner[y0:y1, x0:x1] = torch.where(
                erasable, torch.full_like(owner, -1), owner,
            )
            self.mask_dirty = True
            return

        # ---- Spray paint ----
        n_splat = max(0, int(self.p.spray_splatter_amount))
        splat_r = max(size, float(self.p.spray_splatter_radius))

        # Splatter offsets (small N; CPU-side random). The main disk is implicit
        # at (0,0,size) and is treated separately so only IT contributes to accum.
        splat_dots: list[tuple[float, float, float]] = []
        if n_splat > 0:
            two_pi = 2.0 * math.pi
            inv_splat_r = 1.0 / max(splat_r, 1e-6)
            for _ in range(n_splat):
                angle = np.random.random() * two_pi
                # Mild edge bias: average of uniform `rnd` and area-uniform
                # `sqrt(rnd)`. Keeps some clustering near the brush centre
                # while still spreading dots out a bit more than the original.
                u = np.random.random()
                rnd = 0.5 * (u + math.sqrt(u))
                ox = rnd * math.cos(angle) * splat_r
                oy = rnd * math.sin(angle) * splat_r
                dist = math.hypot(ox, oy)
                # Distance falloff (inner dots bigger). Slightly softened so
                # outer dots can occasionally be sizeable.
                falloff = 0.45 + 0.55 * (1.0 - dist * inv_splat_r)
                # Size jitter: most dots small, occasional bigger "blob".
                # Range ~0.55..1.75 — visible variety but no huge outliers.
                size_jitter = 0.55 + (np.random.random() ** 1.6) * 1.2
                dot_r = max(0.5, size * falloff * size_jitter)
                splat_dots.append((ox, oy, dot_r))

        # Combined bounding box that contains the main disk and all splatters
        outer = max(size, splat_r) + size + 2
        x0 = max(0, int(math.floor(cx - outer)))
        x1 = min(W, int(math.ceil(cx + outer)) + 1)
        y0 = max(0, int(math.floor(cy - outer)))
        y1 = min(H, int(math.ceil(cy + outer)) + 1)
        if x0 >= x1 or y0 >= y1:
            return

        ys = torch.arange(y0, y1, device=self.device).reshape(-1, 1).float()
        xs = torch.arange(x0, x1, device=self.device).reshape(1, -1).float()

        # Main disk footprint (separate so we can credit its wetness only).
        main_d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        main_inside = main_d2 <= size * size

        # Union of splatter footprints (excluding main disk).
        # Vectorized: stack all splatter centers and compute distance to all
        # pixels in ONE broadcasted op + ONE `any` reduction. This collapses
        # what was ~N (=spray_splatter_amount, up to 40) GPU dispatches into a
        # single dispatch — major win on MPS where dispatch latency dominates.
        if splat_dots:
            cxs = torch.tensor(
                [cx + d[0] for d in splat_dots],
                device=self.device, dtype=torch.float32,
            ).reshape(-1, 1, 1)
            cys = torch.tensor(
                [cy + d[1] for d in splat_dots],
                device=self.device, dtype=torch.float32,
            ).reshape(-1, 1, 1)
            r2s = torch.tensor(
                [d[2] * d[2] for d in splat_dots],
                device=self.device, dtype=torch.float32,
            ).reshape(-1, 1, 1)
            d2_all = (xs.unsqueeze(0) - cxs) ** 2 + (ys.unsqueeze(0) - cys) ** 2
            splat_inside = (d2_all <= r2s).any(dim=0)
        else:
            splat_inside = torch.zeros_like(main_inside)
        inside = main_inside | splat_inside

        per_pixel = 0.08 + 0.12 / max(1.0, size)
        sub = bm.mask[y0:y1, x0:x1]
        owner = self.stroke_owner[y0:y1, x0:x1]
        add = torch.where(inside, torch.full_like(sub, per_pixel), torch.zeros_like(sub))
        new_mask = torch.clamp(sub + add, 0.0, 1.0)
        bm.mask[y0:y1, x0:x1] = new_mask
        # Ownership rule: main disk always claims (you really are painting
        # there). Splatter pixels only claim if their mask value is actually
        # visible — otherwise an outer splatter dot with mask ≈ 0.1 would
        # steal ownership from the brush underneath without producing any
        # visible color of its own (the user sees a "hole").
        vis_thr = float(self.p.mask_threshold)
        claim_owner = main_inside | (splat_inside & (new_mask >= vis_thr))
        self.stroke_owner[y0:y1, x0:x1] = torch.where(
            claim_owner, torch.full_like(owner, int(bm.id)), owner,
        )
        # Wetness — only from the main disk. Splatters look wet visually but won't
        # ever drip on their own (matches the original ".accum only at main" feel).
        # `accum` is unbounded above 1.0 so heavily-overstamped pixels get long drips.
        sub_acc = bm.accum[y0:y1, x0:x1]
        acc_add = torch.where(main_inside, torch.full_like(sub_acc, per_pixel), torch.zeros_like(sub_acc))
        bm.accum[y0:y1, x0:x1] = sub_acc + acc_add
        bm.active = True
        self.mask_dirty = True
        # NOTE: drip spawning happens once per step in spawn_drips() (not per
        # stamp), to keep per-stamp cost free of GPU→CPU syncs.

    @torch.no_grad()
    def spawn_drips(self) -> None:
        """Probabilistically spawn new drips from over-threshold accum pixels.

        Called ONCE per step (after paint events have been applied), not per
        stamp. This batches all GPU→CPU syncs into ~2 per brush per step.

        Spawn position is sampled from `accum` weighted by wetness, so the
        heavily-painted core of the brush stroke is the natural source — not
        the sparse splatter pixels (which never accumulate wetness anyway,
        because only the main disk feeds into `accum`).

        Drip length scales with `over_ratio = accum / threshold` so:
          - a single light pass → short or no drip
          - heavy repeated stamping → long, persistent drips
        """
        thr = float(self.p.spray_drip_threshold)
        chance = float(self.p.spray_drip_chance)
        if thr <= 0.0 or chance <= 0.0:
            return
        drip_speed = float(self.p.spray_drip_speed)
        base_speed = max(2, int(round(6 - drip_speed * 8)))
        min_w = float(self.p.spray_drip_min_width)
        W = self.p.W

        for bm in self.brush_models:
            room = MAX_DRIPS_PER_BRUSH - len(bm.drips)
            if room <= 0:
                continue
            # Cheap per-step gate: only attempt with probability `chance`.
            # This keeps spawn rate predictable independent of stamp burst rate.
            if np.random.random() >= chance:
                continue

            # `topk` is much cheaper than `multinomial` on a 540×960 grid
            # (one O(N) reduction + small sort, no random-sampling cost).
            # We grab the K wettest pixels, then randomly pick a subset on the
            # CPU to keep spawn position varied across frames.
            K = min(8, room)
            top_vals_t, top_idx_t = torch.topk(bm.accum.view(-1), K)
            top_vals = top_vals_t.cpu().numpy()  # 1 sync (K floats)
            top_idx = top_idx_t.cpu().numpy()    # piggybacked on same sync
            eligible = [i for i in range(K) if top_vals[i] > thr]
            if not eligible:
                continue
            spawn_n = min(2, len(eligible), room)
            picked_local = np.random.choice(eligible, spawn_n, replace=False)
            zero_idx = top_idx_t[torch.as_tensor(picked_local, device=self.device)]
            bm.accum.view(-1)[zero_idx] = 0.0

            for li in picked_local:
                over_ratio = float(top_vals[li]) / max(thr, 1e-6)
                # 3..(3 + 10*over_ratio) — short to long depending on wetness.
                # Heavy spots (over_ratio≈3) → up to ~33 px persistent drips.
                drip_len = 3 + int(np.random.random() * 10.0 * over_ratio)
                drip_w = max(min_w, 1.5 + np.random.random() * 1.5)
                flat_i = int(top_idx[li])
                bm.drips.append({
                    "x": int(flat_i % W),
                    "y": int(flat_i // W),
                    "charge": drip_len,
                    "width": drip_w,
                    "speed": base_speed,
                    "max_speed": base_speed,
                })

    @torch.no_grad()
    def evolve_drips(self) -> bool:
        """Advance all drip particles one step; deposit a thin line each tick.

        Runs entirely under sim.lock. Cost ≈ O(total_drips) — capped per
        brush to MAX_DRIPS_PER_BRUSH so the worst case stays small.
        Returns True if any drip moved (so caller can mark mask dirty).
        """
        if not self.brush_models:
            return False
        H, W = self.p.H, self.p.W
        grav = int(self.p.drip_gravity)
        wobble = float(self.p.spray_drip_wobble)
        min_w = float(self.p.spray_drip_min_width)
        any_change = False
        for bm in self.brush_models:
            if not bm.drips:
                continue
            kept: list = []
            for d in bm.drips:
                d["speed"] -= 1
                if d["speed"] > 0:
                    kept.append(d)
                    continue
                d["speed"] = d["max_speed"]
                d["charge"] -= 1
                if d["charge"] <= 0:
                    continue
                ox, oy = int(round(d["x"])), int(round(d["y"]))
                r = np.random.random()
                drift = -1 if r < wobble * 0.5 else (1 if r > 1 - wobble * 0.5 else 0)
                if grav == 0:
                    nx, ny = max(0, min(W - 1, ox + drift)), oy + 1
                elif grav == 1:
                    nx, ny = max(0, min(W - 1, ox + drift)), oy - 1
                elif grav == 2:
                    nx, ny = ox - 1, max(0, min(H - 1, oy + drift))
                else:
                    nx, ny = ox + 1, max(0, min(H - 1, oy + drift))
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                is_vert = grav <= 1
                half_w = max(0, int(math.ceil(d["width"] * 0.5)))
                # Important: deposit must land above mask_threshold (default 0.5)
                # so the sigmoid claim in _rebuild_mask actually shows the drip.
                # 0.85~0.95 keeps the central column solid even after width-falloff.
                deposit = 0.85 + 0.10 * (d["charge"] / (d["charge"] + 5))
                if is_vert:
                    px0 = max(0, nx - half_w)
                    px1 = min(W, nx + half_w + 1)
                    py0, py1 = ny, ny + 1
                else:
                    px0, px1 = nx, nx + 1
                    py0 = max(0, ny - half_w)
                    py1 = min(H, ny + half_w + 1)
                if px0 >= px1 or py0 >= py1:
                    continue
                if is_vert:
                    wks = torch.arange(px0 - nx, px1 - nx, device=self.device).float()
                    falloff = (1.0 - torch.abs(wks) / (half_w + 1)).reshape(1, -1)
                else:
                    wks = torch.arange(py0 - ny, py1 - ny, device=self.device).float()
                    falloff = (1.0 - torch.abs(wks) / (half_w + 1)).reshape(-1, 1)
                target = bm.mask[py0:py1, px0:px1]
                # `maximum` (not additive) so a single drip tick is enough to put
                # the center pixel above mask_threshold — the drip shows on first
                # frame instead of slowly accumulating to threshold.
                new_v = torch.maximum(target, deposit * falloff)
                bm.mask[py0:py1, px0:px1] = new_v
                # Same visibility rule as stamps: only claim ownership where the
                # drip actually shows. Width-falloff edges that fall below the
                # mask threshold won't blank out the underlying brush.
                vis_thr = float(self.p.mask_threshold)
                visible = new_v >= vis_thr
                owner_sub = self.stroke_owner[py0:py1, px0:px1]
                self.stroke_owner[py0:py1, px0:px1] = torch.where(
                    visible, torch.full_like(owner_sub, int(bm.id)), owner_sub,
                )
                d["x"] = nx
                d["y"] = ny
                d["width"] += (np.random.randint(0, 3) - 1) * 0.12
                if d["width"] < min_w:
                    d["width"] = min_w
                kept.append(d)
                any_change = True
            bm.drips = kept
        if any_change:
            self.mask_dirty = True
        return any_change

    def clear_brush_mask(self, bm_id: int):
        with self.lock:
            bm = self.get_brush(bm_id)
            if bm is None:
                return
            bm.mask.zero_()
            bm.accum.zero_()
            bm.drips.clear()
            self.stroke_owner = torch.where(
                self.stroke_owner == int(bm_id),
                torch.full_like(self.stroke_owner, -1),
                self.stroke_owner,
            )
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
