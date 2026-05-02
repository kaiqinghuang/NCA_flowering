# PC-local NCA + Kinect demo

This is the **single-machine** build of the project. The NCA renderer,
the Kinect bridge, and the browser UI all live on the same Windows PC,
and **the rendered NCA frames travel as raw uncompressed RGBA bytes
over loopback** — no JPEG, no WebP, no compression artifacts. Every
pixel on screen is exactly what the PyTorch NCA simulator produced.

```
remote_gpu/  (multi-machine — render on a remote GPU host, bridge on
              the Kinect PC, two FastAPI apps, two ports, JPEG/WebP-
              compressed frames over WAN)

pclocal/     (this folder — one Windows PC, one FastAPI app, one port,
              raw RGBA frames over loopback, no compression)
```

If you don't have a Kinect plugged in (e.g. you're developing on
macOS / Linux), the NCA half still runs; the bridge half just reports
"Kinect not found" and the hand-tracking features stay disabled.

## Why raw RGBA over loopback?

`remote_gpu/` ships every frame across the open internet, so it has to
encode it (adaptive WebP/JPEG, ~30–60 KB/frame) and accept the visible
quality loss that comes with any lossy codec. On a single PC there's no
WAN to traverse — the kernel loopback path moves bytes between the
Python process and the browser at memory-bandwidth speeds (5+ GB/s on
modern hardware), so we can afford to send each frame as plain RGBA
bytes (`H × W × 4`) and skip the encoder entirely.

| | `remote_gpu` | `pclocal` |
|---|---|---|
| frame transport | adaptive WebP/JPEG over WAN WS | raw RGBA over loopback WS |
| bytes per frame (960×540) | ~30–60 KB (Q=68–92) | 2.07 MB (no compression) |
| bandwidth @ default fps | ~3 MB/s @ 30 fps | ~60 MB/s @ 30 fps (loopback can do ≫1 GB/s) |
| encode CPU on server | 5–10 ms / frame | 0 |
| client-side decode | `createImageBitmap` ~3–5 ms | `putImageData` ~1–2 ms |
| visible artifacts | yes (DCT blocks, chroma noise) | none — pixel-perfect |
| jitter buffer / adaptive quality | yes (needed over WAN) | removed (loopback has near-zero jitter) |

## Layout

```
pclocal/
  app.py              ← single-process entry point (this file glues server + bridge)
  requirements.txt    ← combined deps for both halves

  server/             ← NCA simulation + WebP frame stream  (PyTorch)
    nca_server.py        FastAPI app (mounted as the parent)
    nca_model.py         NCA simulator + brush evolution
    npy_loader.py        legacy .npy weight loader
    perlin.py            vectorized 3D Perlin / fBm

  bridge/             ← Kinect depth source + TV calibration  (PyKinect2)
    main.py              FastAPI app (mounted under /bridge of the parent)
    kinect_depth_source.py
    depth_processing.py
    tv_calibration.py
    tv_calibration.json  (created on first successful auto-calibration)

  client/
    index.html        ← canvas, paint UI, Kinect debug overlay
```

## Routes (single port)

| Path                       | Served by                    | What                                     |
|----------------------------|------------------------------|------------------------------------------|
| `/`                        | server/nca_server            | client (`index.html`)                    |
| `/static/...`              | server/nca_server            | client static assets                     |
| `/ws`                      | server/nca_server            | NCA WebSocket (frames, paint, params)    |
| `/bridge/`                 | bridge/main                  | bridge health JSON                       |
| `/bridge/ws`               | bridge/main                  | Kinect WebSocket (hand events, calib)    |
| `/bridge/debug/depth.jpg`  | bridge/main                  | live depth-debug overlay (when enabled)  |

The client (`index.html`) auto-points its main socket at `/ws` and its
bridge socket at `/bridge/ws` on the same host:port — no `?bridge=…`
override needed for the local build.

## Install (Windows w/ Kinect v2)

```powershell
cd pclocal
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt

# Install torch separately for your GPU/CPU. Pick the index that matches
# your Python + CUDA from https://pytorch.org/get-started/locally/  e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

You also need the **Kinect for Windows v2 SDK 2.0** runtime so PyKinect2
can talk to the device. See
<https://www.microsoft.com/en-us/download/details.aspx?id=44561>.

### One-time patch for `pykinect2` on 64-bit Python

`pykinect2 0.1.0` (last updated 2017) has two breakages against any
modern Python venv. `requirements.txt` already pins `comtypes==1.1.10`
to dodge one of them; the other (32-bit struct-size asserts) needs a
post-install patch. Run once per venv:

```powershell
.\scripts\patch_pykinect2.ps1
```

The script is idempotent — re-running it is safe. After it prints
`pykinect2 OK`, the bridge half can talk to the Kinect. Without the
patch you'll see `[kinect-depth] PyKinect2 import failed: AssertionError: 80`
in the bridge log. See `scripts/patch_pykinect2.ps1` for the exact
edits and rationale.

### Install (macOS / Linux — NCA only, no Kinect)

```bash
cd pclocal
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Skip pykinect2 / comtypes; they're already gated on sys_platform=="win32".

# torch for MPS (Mac Apple Silicon):
pip install torch
```

The bridge half will print "PyKinect2 import failed" and stay idle on
non-Windows platforms; the NCA half runs normally for paint-with-mouse
testing.

## Run

```powershell
cd pclocal
.\.venv\Scripts\activate
uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/> in a browser.

### Where the NCA models are loaded from

`server\nca_server.py` looks for `.npy` weight files in this order, and
picks the first directory that exists AND contains at least one `.npy`:

1. `$env:NCA_MODELS_DIR` (env var; absolute, or relative to
   `pclocal\server\`).
2. **Hardcoded paths for known dev machines** — see `_KNOWN_MODEL_DIRS`
   in `nca_server.py`. The demo Windows PC's path
   (`C:\Users\xiaoh\Documents\NCA_flowering\texture_model`) is already
   listed there, so on that PC you don't need to set anything — just
   `uvicorn` and go. To add another machine, append its path to that
   list and rebuild.
3. `<repo-root>\texture_model` (i.e. `pclocal\..\..\texture_model`).
4. `pclocal\texture_model`.

The startup log prints the chosen path and how many `.npy` files were
found:

```
[nca_server] models dir: C:\...\texture_model (37 found)
```

If `(0 found)` appears, the resolver fell all the way through without
finding any `.npy` — check that the folder really has the weight files
and either drop them under one of the candidate paths or set
`NCA_MODELS_DIR` explicitly. The browser "Base A/B/C/D" and "Add Brush"
dropdowns are populated from that list, so an empty models dir shows up
in the UI as "no models to load".

The first time you launch it on a new physical setup:

1. Click **Auto-Calibrate TV (Depth)** in the Kinect panel.
2. Confirm the magenta polygon hugs your TV in the depth-debug view
   (toggle **Debug View: On** under the Kinect panel).
3. Move your hand into the interaction box — a red dot tracks the
   fingertip and the active brush paints on the canvas.

The TV calibration is persisted to `pclocal/bridge/tv_calibration.json`
and reused on subsequent launches.

## Environment variables

All of `remote_gpu`'s environment knobs work identically here. The most
common ones:

### NCA server (frame stream, simulation)
| Var | Default | Effect |
|---|---|---|
| `NCA_W` / `NCA_H` | `960` / `540` | simulation grid size (also dictates the wire frame size: `W × H × 4` bytes) |
| `NCA_FPS` | `30` | render/broadcast rate — controls visual refresh smoothness (Kinect cursor liveness, paint stroke responsiveness). Cheap (~10ms render per frame), so this can stay high independent of `NCA_SPS`. |
| `NCA_SPS` | `10` | NCA steps per second — controls *how fast the simulation evolves* (pattern growth, drip flow). Each step costs ~20–25ms on Apple Silicon at 960×540, so this is the main GPU-load knob. Kept intentionally low so the sim looks calm and never overruns its budget. To bump evolution rate without changing the loop pacing, push the UI **Speed** slider (`steps_per_frame`) — Speed=3 + sps=10 ≈ 30 sps of NCA evolution. |
| `NCA_MODELS_DIR` | first match from `_KNOWN_MODEL_DIRS`, then `pclocal\..\..\texture_model`, then `pclocal\texture_model` | folder of `.npy` weights. Override with an **absolute** path if your machine isn't in the hardcoded list. |
| `NCA_PAINT_QUEUE_MAX` | `4096` | brush-event ring-buffer capacity |
| `NCA_PAINT_BATCH_LIMIT` | `96` | paint events drained per sim step |
| `NCA_DRIP_EVOLVE_EVERY` | `2` | spawn/evolve drips every Nth step |

> The `NCA_WEBP_Q` / `NCA_ADAPTIVE_*` variables from `remote_gpu/` are
> intentionally absent — there's no encoder left to tune.

### Kinect bridge (hand tracking, calibration)
See `bridge/README.md` for the full table — the same vars apply here.
Highlights:
| Var | Default | Effect |
|---|---|---|
| `BRIDGE_DEPTH_BAND_MIN_M` / `_MAX_M` | `0.02` / `0.45` | interaction box thickness |
| `BRIDGE_HAND_MIN_PX` | `200` | min hand blob to start tracking |
| `BRIDGE_AUTOFIT_COLOR_MODE` | `reject_yellow` | TV-detection colour filter |
| `BRIDGE_AUTOFIT_YELLOW_H_MIN` / `_MAX` | `10` / `40` | wood-yellow hue band |

## Differences from `remote_gpu/`

| Aspect | `remote_gpu/` | `pclocal/` |
|---|---|---|
| Processes | 2 (one per host) | 1 |
| Ports | 8000 (NCA) + 7000 (bridge) | 8000 only |
| Network | NCA over WAN/WiFi, bridge LAN | loopback only |
| Frame transport | adaptive WebP/JPEG over WS | **raw RGBA** over WS |
| Frame quality | lossy (DCT artifacts) | **pixel-perfect** (no encoder) |
| Server-side encode CPU | 5–10 ms/frame | 0 |
| Client-side decode CPU | `createImageBitmap` ~3–5 ms | `putImageData` ~1–2 ms |
| Default FPS / SPS | 30 / 30 | 30 / 10 (decoupled — visual refresh stays smooth, NCA evolution is intentionally calm) |
| Latency | 30–80 ms typical | 5–15 ms |
| Hand events | bridge → server → ⚠ user pastes URL | bridge → same app, automatic |
| Adaptive quality controller | yes | removed |
| Client jitter buffer | yes (8-frame buffer + paced draw) | removed |

## Differences in code vs `remote_gpu/`

The two service folders (`pclocal/server/` and `pclocal/bridge/`) start
as copies of the remote ones; the edits are:

* **`pclocal/app.py`** (new) — creates the single FastAPI app, mounts
  the bridge sub-app at `/bridge`, and forwards its lifespan hooks.
* **`pclocal/server/nca_server.py`** —
  * `_render_and_encode_blocking` (PIL→JPEG) replaced by
    `_render_raw_blocking` (pad alpha, GPU→CPU transfer, return RGBA
    bytes).
  * Adaptive-quality controller (`_aq`, `_aq_record`, `_aq_tick`) and
    its env vars (`NCA_WEBP_Q`, `NCA_ADAPTIVE_*`) deleted.
  * `hello` advertises `frame_format` + `frame_bytes`.
  * Default `NCA_FPS` / `NCA_SPS` decoupled to `30` / `10` (was both
    60). The render/broadcast loop stays at 30 fps so the canvas
    refreshes smoothly alongside the ~30Hz Kinect hand stream, while
    the NCA simulator only ticks 10×/sec so each step has plenty of
    GPU budget and never overruns. Result: smooth-feeling visuals with
    a calm, deliberate pattern-evolution pace. See the long comment
    above the constants in `nca_server.py`.
* **`pclocal/bridge/kinect_depth_source.py`** —
  `KinectDepthSource.start()` made idempotent (no-op if already
  running) so it's safe under the new lifespan chaining.
* **`pclocal/client/index.html`** —
  * `BRIDGE_URL` default now points at same-origin `/bridge/ws`
    instead of `:7000/ws`.
  * `ws.binaryType = 'arraybuffer'` (was `'blob'`).
  * Old jitter-buffered decode pipeline (`frameBuffer`,
    `pushFrameBitmap`, `renderLoop`, `drawFrameBlob`,
    `drawFrameViaImg`, ~120 LOC) deleted; replaced by a single
    `drawFrameRaw(buf)` that does `frameImageData.data.set(...)` →
    `ctx.putImageData(...)`.
  * FPS/bandwidth display now reports MB/s instead of kbps.

Everything else (calibration, depth pipeline, NCA simulator + brush
evolution + drips, debug overlay, paint protocol, env knobs, …) is
byte-identical to `remote_gpu/`.
