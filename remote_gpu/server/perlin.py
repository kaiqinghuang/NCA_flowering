"""3D Perlin + fBm + altitude quartile weights.

Vectorized NumPy port of the JS implementation in
background_altitude_webgpu.html. Computes per-pixel altitude band weights
(A/B/C/D) from animated 3D Perlin noise.
"""
from __future__ import annotations

import numpy as np


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


def altitude_weights(
    H: int,
    W: int,
    perlin: Perlin3D,
    noise_scale: float = 1.5,
    octaves: float = 1.0,
    half_width: float = 0.02,
    edges: tuple[float, float, float] = (0.3, 0.5, 0.7),
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    z: float = 0.0,
) -> np.ndarray:
    """Returns float32 [H, W, 4] array of A/B/C/D blend weights summing to 1."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    sx = (xs / W) * noise_scale + offset_x
    sy = (ys / H) * noise_scale + offset_y
    sz = np.full_like(sx, z, dtype=np.float64)
    raw = fbm3d(perlin, sx.astype(np.float64), sy.astype(np.float64), sz, octaves)
    m01 = np.clip(raw * 0.5 + 0.5, 0.0, 1.0)

    e0, e1, e2 = edges
    h = float(np.clip(half_width, 0.0, 0.124))
    out = np.zeros((H, W, 4), dtype=np.float32)

    if h < 1e-6:
        # Hard quartile assignment
        out[..., 0] = (m01 < e0).astype(np.float32)
        out[..., 1] = ((m01 >= e0) & (m01 < e1)).astype(np.float32)
        out[..., 2] = ((m01 >= e1) & (m01 < e2)).astype(np.float32)
        out[..., 3] = (m01 >= e2).astype(np.float32)
        return out

    def smoothstep(a, b, x):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # Region masks
    in_pure_a = m01 < e0 - h
    in_ab = (m01 >= e0 - h) & (m01 < e0 + h)
    in_pure_b = (m01 >= e0 + h) & (m01 < e1 - h)
    in_bc = (m01 >= e1 - h) & (m01 < e1 + h)
    in_pure_c = (m01 >= e1 + h) & (m01 < e2 - h)
    in_cd = (m01 >= e2 - h) & (m01 < e2 + h)
    in_pure_d = m01 >= e2 + h

    out[..., 0] = np.where(in_pure_a, 1.0, 0.0)
    t_ab = smoothstep(e0 - h, e0 + h, m01)
    out[..., 0] = np.where(in_ab, 1.0 - t_ab, out[..., 0])
    out[..., 1] = np.where(in_ab, t_ab, 0.0)
    out[..., 1] = np.where(in_pure_b, 1.0, out[..., 1])
    t_bc = smoothstep(e1 - h, e1 + h, m01)
    out[..., 1] = np.where(in_bc, 1.0 - t_bc, out[..., 1])
    out[..., 2] = np.where(in_bc, t_bc, 0.0)
    out[..., 2] = np.where(in_pure_c, 1.0, out[..., 2])
    t_cd = smoothstep(e2 - h, e2 + h, m01)
    out[..., 2] = np.where(in_cd, 1.0 - t_cd, out[..., 2])
    out[..., 3] = np.where(in_cd, t_cd, 0.0)
    out[..., 3] = np.where(in_pure_d, 1.0, out[..., 3])
    return out
