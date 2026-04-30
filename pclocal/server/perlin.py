"""3D Perlin + layered priority-stack base weights (A/B/C/D).

Implements the same base-layer Perlin logic as the legacy
background_multiPerlin_4move_highResol_cpugpu_rotate.html:
- independent Perlin fields for B/C/D
- per-layer frequency spread + fixed decorrelating phase
- sigmoid claim per layer
- priority stack D -> C -> B, and A = remainder
"""
from __future__ import annotations

import numpy as np

LAYER_PHASE = np.array(
    [
        [0.0, 0.0, 0.0],
        [41.17, 109.3, 27.8],
        [203.6, 17.2, 91.4],
        [67.9, 251.1, 156.7],
    ],
    dtype=np.float64,
)


class Perlin3D:
    """Classic Perlin permutation table seeded by mulberry32-equivalent RNG."""

    def __init__(self, seed: int = 1):
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        order = np.arange(256, dtype=np.uint8)
        rng.shuffle(order)
        self.p = np.empty(512, dtype=np.int32)
        self.p[:256] = order
        self.p[256:] = order

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _grad3(h: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        h = h & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where((h & 1) != 0, -u, u) + np.where((h & 2) != 0, -v, v)

    def noise3(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255
        zi = np.floor(z).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        zf = z - np.floor(z)
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)
        p = self.p
        # All 8 cube-corner permutation hashes
        h000 = p[p[p[xi] + yi] + zi]
        h100 = p[p[p[xi + 1] + yi] + zi]
        h010 = p[p[p[xi] + yi + 1] + zi]
        h110 = p[p[p[xi + 1] + yi + 1] + zi]
        h001 = p[p[p[xi] + yi] + zi + 1]
        h101 = p[p[p[xi + 1] + yi] + zi + 1]
        h011 = p[p[p[xi] + yi + 1] + zi + 1]
        h111 = p[p[p[xi + 1] + yi + 1] + zi + 1]

        def lerp(t, a, b):
            return a + t * (b - a)

        x1 = lerp(u, self._grad3(h000, xf, yf, zf), self._grad3(h100, xf - 1, yf, zf))
        x2 = lerp(u, self._grad3(h010, xf, yf - 1, zf), self._grad3(h110, xf - 1, yf - 1, zf))
        y1 = lerp(v, x1, x2)
        x3 = lerp(u, self._grad3(h001, xf, yf, zf - 1), self._grad3(h101, xf - 1, yf, zf - 1))
        x4 = lerp(u, self._grad3(h011, xf, yf - 1, zf - 1), self._grad3(h111, xf - 1, yf - 1, zf - 1))
        y2 = lerp(v, x3, x4)
        return lerp(w, y1, y2)


def fbm3d(perlin: Perlin3D, x: np.ndarray, y: np.ndarray, z: np.ndarray, octaves: float) -> np.ndarray:
    oc = float(np.clip(octaves, 1, 14))
    full = int(np.floor(oc))
    frac = oc - full
    v = np.zeros_like(x)
    amp, freq, norm = 1.0, 1.0, 0.0
    for _ in range(full):
        v += amp * perlin.noise3(x * freq, y * freq, z * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    if frac > 1e-6:
        v += frac * amp * perlin.noise3(x * freq, y * freq, z * freq)
        norm += frac * amp
    return v / norm if norm > 1e-12 else v


def _sigmoid_claim(v01: np.ndarray, threshold: float, sharpness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(v01 - threshold) * sharpness))


def layered_base_weights(
    H: int,
    W: int,
    perlin_layers: list[Perlin3D],
    noise_scale: float = 1.5,
    octaves: float = 1.0,
    threshold: float = 0.5,
    edge_sharpness: float = 30.0,
    layer_freq_spread: float = 0.55,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    z: float = 0.0,
) -> np.ndarray:
    """Returns float32 [H, W, 4] base weights for A/B/C/D."""
    if len(perlin_layers) < 4:
        raise ValueError("perlin_layers must contain 4 entries for A/B/C/D")

    ys, xs = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    # Match legacy behavior: one reference span for x/y avoids anisotropic stretch.
    ref = float(max(1, max(W, H)))
    sx = (xs / ref) * float(noise_scale) + float(offset_x)
    sy = (ys / ref) * float(noise_scale) + float(offset_y)

    oc = float(np.clip(octaves, 1.0, 14.0))
    spread = float(np.clip(layer_freq_spread, 0.0, 1.0))
    thr = float(np.clip(threshold, 0.02, 0.98))
    sharp = float(np.clip(edge_sharpness, 0.5, 50.0))

    claims: list[np.ndarray] = []
    for li in (1, 2, 3):
        lmul = 1.0 + spread * li * 0.28
        ph = LAYER_PHASE[li]
        sz = np.full_like(sx, float(z) + ph[2] * 0.15, dtype=np.float64)
        raw = fbm3d(
            perlin_layers[li],
            sx * lmul + ph[0],
            sy * lmul + ph[1],
            sz,
            oc,
        )
        m01 = np.clip(raw * 0.5 + 0.5, 0.0, 1.0)
        claims.append(_sigmoid_claim(m01, thr, sharp))

    a1, a2, a3 = claims
    rem3 = 1.0 - a3
    rem23 = rem3 * (1.0 - a2)
    w3 = a3
    w2 = rem3 * a2
    w1 = rem23 * a1
    w0 = rem23 * (1.0 - a1)
    out = np.stack([w0, w1, w2, w3], axis=-1).astype(np.float32, copy=False)
    # Numerical safety: keep sum close to 1.
    s = out.sum(axis=-1, keepdims=True)
    out = np.divide(out, np.maximum(s, 1e-8), out=np.zeros_like(out), where=s > 0.0)
    return out


def altitude_weights(
    H: int,
    W: int,
    perlin_layers: list[Perlin3D],
    noise_scale: float = 1.5,
    octaves: float = 1.0,
    half_width: float = 0.02,
    edges: tuple[float, float, float] = (0.3, 0.5, 0.7),
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    z: float = 0.0,
    threshold: float = 0.5,
    edge_sharpness: float = 30.0,
    layer_freq_spread: float = 0.55,
) -> np.ndarray:
    """Compatibility wrapper now using layered base-weight logic.

    `half_width` / `edges` are retained for call compatibility with older code.
    """
    _ = (half_width, edges)
    return layered_base_weights(
        H=H,
        W=W,
        perlin_layers=perlin_layers,
        noise_scale=noise_scale,
        octaves=octaves,
        threshold=threshold,
        edge_sharpness=edge_sharpness,
        layer_freq_spread=layer_freq_spread,
        offset_x=offset_x,
        offset_y=offset_y,
        z=z,
    )
