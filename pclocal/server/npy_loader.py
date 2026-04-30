"""Load NCA model weights from .npy files.

The .npy files saved by the training pipeline contain a Python pickle of
two Float32 arrays (dense1: [perception_n+1, hidden] flattened with bias as
the last row; dense2: [hidden+1, ch] flattened with bias as the last row).

The JS client (background_altitude_webgpu.html parseNpyPickle) walks the
pickle bytes manually because it has no Python runtime. Server-side we just
np.load with allow_pickle=True; if that fails we fall back to a byte-walker
that mirrors the JS parser.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


CH = 12
FILTER_N = 4
PERCEPTION_N = CH * FILTER_N  # 48


class NCAWeights(NamedTuple):
    """One NCA model: two-layer MLP applied per-pixel via 1×1 conv."""
    name: str
    dense1k: torch.Tensor   # [1, 1, perception_n, hidden]
    dense1b: torch.Tensor   # [hidden]
    dense2k: torch.Tensor   # [1, 1, hidden, ch]
    dense2b: torch.Tensor   # [ch]
    hidden: int


def _walk_pickle_bytes(buf: bytes) -> list[np.ndarray]:
    """Mirror of JS parseNpyPickle: scan for BINBYTES (0x42) blocks > 100 bytes."""
    if buf[:6] != b"\x93NUMPY":
        raise ValueError("Not a .npy file (bad magic)")
    header_len = buf[8] | (buf[9] << 8)
    pickle_start = 10 + header_len
    blocks: list[np.ndarray] = []
    i = pickle_start
    n = len(buf)
    while i < n:
        op = buf[i]
        if op == 0x42:  # BINBYTES: 4-byte LE length + raw bytes
            if i + 5 > n:
                break
            length = int.from_bytes(buf[i + 1:i + 5], "little")
            if length > 100 and i + 5 + length <= n:
                blocks.append(np.frombuffer(buf[i + 5:i + 5 + length], dtype=np.float32).copy())
            i += 5 + length
        elif op == 0x43:  # SHORT_BINBYTES
            if i + 2 > n:
                break
            slen = buf[i + 1]
            i += 2 + slen
        else:
            i += 1
    return blocks


def _extract_two_arrays(obj) -> tuple[np.ndarray, np.ndarray]:
    """Pickle payload may be list/tuple/dict of arrays."""
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return np.asarray(obj[0], dtype=np.float32), np.asarray(obj[1], dtype=np.float32)
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if len(keys) >= 2:
            return np.asarray(obj[keys[0]], dtype=np.float32), np.asarray(obj[keys[1]], dtype=np.float32)
    raise ValueError(f"Unrecognized .npy payload structure: {type(obj)}")


def _parse_npy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    # Try standard numpy load first
    try:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size >= 2:
            return _extract_two_arrays(obj.tolist())
        if isinstance(obj, np.ndarray) and obj.ndim >= 1 and obj.dtype != object:
            # Single packed array — split via JS-style byte walker as fallback
            pass
        else:
            return _extract_two_arrays(obj)
    except Exception:
        pass
    # Fallback: byte walker matching the JS reference parser
    blocks = _walk_pickle_bytes(raw)
    if len(blocks) < 2:
        raise ValueError(f"Could not extract 2 weight matrices from {path.name} (found {len(blocks)})")
    return blocks[0], blocks[1]


def _split_weight_bias(arr: np.ndarray, in_dim: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Accept either 2D (in_dim+1, out) or flat (in_dim+1)*out and split off bias row."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if a.ndim == 2:
        rows, out = a.shape
        if rows != in_dim + 1:
            raise ValueError(f"{name}: expected {in_dim + 1} rows, got {rows}")
        return a[:in_dim], a[in_dim]
    a = a.ravel()
    if a.size % (in_dim + 1) != 0:
        raise ValueError(f"{name}: flat size {a.size} not divisible by in_dim+1={in_dim + 1}")
    out = a.size // (in_dim + 1)
    return a[: in_dim * out].reshape(in_dim, out), a[in_dim * out:]


def load_model(path: str | Path, device: torch.device, name: str | None = None) -> NCAWeights:
    """Load one .npy and return tensors ready for conv2d on device.

    Expected payload: 2 arrays — dense1 [PERCEPTION_N+1, hidden], dense2 [hidden+1, CH].
    Last row of each is the bias.
    """
    path = Path(path)
    name = name or path.stem
    d1_arr, d2_arr = _parse_npy_file(path)

    d1w, d1b = _split_weight_bias(d1_arr, PERCEPTION_N, "dense1")  # (pn, hidden), (hidden,)
    hidden = d1w.shape[1]
    d2w, d2b = _split_weight_bias(d2_arr, hidden, "dense2")        # (hidden, ch), (ch,)

    if d2w.shape[1] != CH:
        raise ValueError(f"dense2 out dim {d2w.shape[1]} != CH={CH}")

    # Stored as [in, out]; nca_model.compute_dx transposes + reshapes to conv2d weight.
    return NCAWeights(
        name=name,
        dense1k=torch.from_numpy(np.ascontiguousarray(d1w)).to(device),
        dense1b=torch.from_numpy(np.ascontiguousarray(d1b)).to(device),
        dense2k=torch.from_numpy(np.ascontiguousarray(d2w)).to(device),
        dense2b=torch.from_numpy(np.ascontiguousarray(d2b)).to(device),
        hidden=hidden,
    )
