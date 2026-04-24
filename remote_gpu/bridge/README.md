# Kinect v2 Bridge (`remote_gpu/bridge`)

Local process that reads the Kinect's right-hand joint tracking and exposes it
as a WebSocket the NCA browser client can subscribe to. Converts `HandTipRight`
and `ThumbRight` into canvas (x, y) + pinch state, so the existing brush
protocol (`stamp` / `stroke`) gets driven by gesture instead of the mouse.

```
┌────────────────────── Windows laptop ──────────────────────┐
│  Kinect v2 ── USB3 ──▶ bridge (this pkg) ──ws:7000─▶ browser │
│                                                      │      │
│                                                      ▼      │
│                                                 main NCA ws │
│                                                   (RunPod)  │
└─────────────────────────────────────────────────────────────┘
```

Mouse input is **not** disabled — the bridge just adds gesture input in
parallel. If the bridge is unreachable the browser silently falls back to
mouse-only.

## Requirements

- **Windows 8.1 / 10 / 11** (Kinect v2 SDK is Windows-only)
- **Kinect for Windows v2** (sensor + 12V power adapter + USB 3.0 cable)
- **USB 3.0 type-A host port** (USB-C hubs rarely work reliably)
- Python 3.10+
- Kinect for Windows SDK 2.0 installed (ships the runtime drivers;
  <https://www.microsoft.com/en-us/download/details.aspx?id=44561>)

## Install

```powershell
# from remote_gpu/
python -m venv .venv
.venv\Scripts\activate
pip install -r bridge/requirements.txt
```

## Run

```powershell
# Still from remote_gpu/
python -m uvicorn bridge.main:app --host 0.0.0.0 --port 7000
```

### Default: depth mode (no body / hand skeleton)

By default the bridge uses the 512×424 depth map: band in front of the TV plane → largest blob → fingertip (furthest from centroid) + geometric pinch (local point-cloud elongation).

1. In the browser Kinect panel, click **Fit TV plane (depth)** with **hands out of view** (~1.2s capture).
2. Then run the usual **Calibrate** four corners (pinch-hold) so `(x,z)` maps to the canvas.

Expected boot log:

```
[kinect-depth] Runtime started (depth + color).
[bridge] Kinect mode: depth  ...
```

To use the **Kinect SDK body / joint** tracking instead, set:

```powershell
set BRIDGE_KINECT_MODE=body
python -m uvicorn bridge.main:app --host 0.0.0.0 --port 7000
```

Then you should see `[kinect] Runtime started; waiting for body frames.` and body-frame lines.

If you see `PyKinect2 import failed` the bridge keeps running but emits no
hand frames — install PyKinect2 and the Kinect SDK.

## Calibration

1. Open the NCA browser client (the main remote_gpu site).
2. Bridge status panel appears when `ws://localhost:7000/ws` connects.
3. Click **Calibrate** → the 4 canvas corners light up one at a time (TL → TR → BR → BL).
4. For each corner: move your right hand above that corner of the TV,
   **pinch and hold** (thumb + index touching or a firm fist) for ~½ second,
   then release. The bridge records the median 3D position during the hold.
5. After the 4th corner the homography saves to `calibration.json` next to
   this README. Until you reset, the bridge always applies this calibration.

### Re-calibrate

Browser button **Reset calibration** or delete `calibration.json`.

## Tuning (env vars)

| Var | Default | Effect |
|---|---|---|
| `BRIDGE_PORT` | 7000 | local WS port |
| `BRIDGE_CANVAS_W` / `_H` | 960 / 540 | must match NCA server `NCA_W/H` |
| `BRIDGE_EMA_ALPHA` | 0.35 | cursor smoothing (lower = smoother / laggier) |
| `BRIDGE_PINCH_DIST_M` | 0.03 | thumb-index distance threshold for pinch |
| `BRIDGE_DEBOUNCE` | 1 | extra frames of agreement before pinch state flips |
| `BRIDGE_BROADCAST_HZ` | 30 | outgoing event rate |

## Protocol (for reference)

Every broadcast frame:

```json
{
  "op": "hand",
  "tracked": true,
  "confident": true,
  "cal_ready": true,
  "cal_mode": "idle",
  "cal_status": {"state": "idle"},
  "cx": 481.2,
  "cy": 270.7,
  "pinch": false,
  "t": 1734123456.789
}
```

Calibration state machine messages: `cal_started`, `cal_cancelled`,
`cal_reset`, plus per-frame `cal_status` inside every `hand` message while
`cal_mode == "capturing"` (fields: `state`, `corner`, `label`, `samples`, …).
