# NCA remote-GPU prototype

Splits the `background_altitude_webgpu.html` simulation into:

- **server/** — Python + PyTorch + FastAPI. Runs the NCA loop on a CUDA / MPS / CPU device, streams WebP frames over a WebSocket.
- **client/** — Thin HTML + canvas. Receives frames, sends paint events and parameter changes.

The skeleton mirrors the JS reference for: NCA step, perception kernel, periodic padding, direction map (cartesian / polar / bipolar), Perlin altitude bands, and priority-stack mask composition. The brush evolution (drip propagation, heat-equation diffusion, life decay) from `evolveSingleBrushMask` is **not** ported yet — the brush deposit is a static soft-mask disk. See the TODO at the bottom of `server/nca_model.py`.

## Layout
```
remote_gpu/
  server/
    nca_server.py       FastAPI app + WS handler + frame loop
    nca_model.py        PyTorch NCA + mask compositor
    npy_loader.py       Pickle .npy → tensor weights
    perlin.py           Vectorized 3D Perlin + fBm + altitude weights
    requirements.txt
  client/
    index.html          Canvas + WS + control panel
```

## Install

```bash
cd remote_gpu/server
pip install -r requirements.txt
# Then install torch from https://pytorch.org/get-started/locally/
# CUDA example:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
cd remote_gpu/server
NCA_MODELS_DIR=../../models uvicorn nca_server:app --host 0.0.0.0 --port 8000
```

Open `http://<host>:8000/` in a browser. The page connects to `/ws`, fetches the model inventory, and displays incoming frames.

### Environment variables
| Var | Default | Meaning |
|---|---|---|
| `NCA_W` / `NCA_H` | 480 / 270 | Simulation grid size |
| `NCA_FPS` | 30 | Frame broadcast rate |
| `NCA_SPS` | 60 | Target NCA steps per second |
| `NCA_WEBP_Q` | 70 | WebP encode quality |
| `NCA_MODELS_DIR` | `../../models` | Folder of `.npy` weight files |

## Protocol (WebSocket `/ws`)

Client → server (JSON text):

| op | fields |
|---|---|
| `list_models` | — |
| `load_base` | `slot` 0..3, `path` filename |
| `load_brush` | `path` filename |
| `remove_brush` | `id` int |
| `stamp` | `id`, `x`, `y`, `r`, `erase` |
| `clear_brush` | `id` |
| `clear_state` | — |
| `reseed` | — |
| `set_param` | `name`, `value` |

Settable params: `alignment`, `rotation_deg`, `noise_scale`, `octaves`, `layer_freq_spread`, `half_width`, `noise_z_scale`, `noise_z_speed`, `mask_threshold`, `mask_edge_sharpness`, `steps_per_frame`, `disturbance`, `show_mask_tint`, `active`.

Server → client:

- Binary frames: WebP-encoded RGB image at the configured FPS.
- JSON text: `hello`, `models`, `loaded_base`, `loaded_brush`, `error`.

## Notes / next steps

- **Brush evolution** — port `evolveSingleBrushMask` (drip + diffusion + life) from the JS file for the wet-paint look.
- **Latency** — for sub-30 ms feedback over slow links, mirror the brush-mask deposit on the client and overlay it locally before the next server frame arrives.
- **Bandwidth** — 480×270 WebP@70 sits around 30–60 KB/frame; 1080p will need WebRTC / H.264 (`aiortc`) for smooth streaming.
- **Multi-client** — the simulation is single-shared right now: every client sees the same canvas and can paint. Add per-session simulators if isolation is wanted.
