# Kinect v2 Bridge (`remote_gpu/bridge`)

Local Windows process that turns a Kinect v2 into a fingertip cursor for the
NCA browser client. The runtime is **depth-first**:

* The TV lies flat (or at any orientation) and the Kinect is mounted at TV
  height, slightly tilted toward the screen.
* You manually point your **right hand (HandTipRight)** at each of the four
  TV corners and press **Confirm** for each — that single capture defines
  both the TV's 3D plane *and* the homography that maps real-world
  fingertip positions onto the canvas. No hidden RANSAC, no auto-fit.
* At runtime, the bridge ignores the SDK skeleton: the depth frame is
  filtered by *(inside the calibrated TV polygon) ∧ (signed distance to
  plane within `[0.02, 0.45]`m)*, and the **fingertip = pixel closest to
  the screen along the plane normal**. While a fingertip exists in the
  interaction box, the bridge emits `pinch=true` so the canvas paints
  continuously.

```
┌────────────────────── Windows laptop ──────────────────────┐
│  Kinect v2 ── USB3 ──▶ bridge (this pkg) ──ws:7000─▶ browser │
│                                                      │      │
│                                                      ▼      │
│                                                 main NCA ws │
│                                                   (RunPod)  │
└─────────────────────────────────────────────────────────────┘
```

The mouse path is unaffected; if the bridge is unreachable the page falls
back to mouse only.

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

Default mode is `depth` (Body+Depth opened together: Body for the
calibration wizard, Depth for the runtime fingertip). To run the legacy
SDK-only body mode (no depth, no debug image) set:

```powershell
set BRIDGE_KINECT_MODE=body
```

Expected boot log:

```
[bridge] Kinect mode: depth (body+depth, depth fingertip)
[kinect-depth] Runtime started (body + depth).
[bridge] canvas=960x540  mode=depth  box=[0.02,0.45]m  tv_ready=no  listening on ws://0.0.0.0:7000/ws
```

If you see `PyKinect2 import failed` the bridge keeps running but emits no
hand frames — install PyKinect2 and the Kinect SDK.

## TV Calibration (one-time)

Open the NCA page and connect to the bridge. In the **Kinect** sidebar:

1. Click **Start TV Calibration**. The wizard panel appears.
2. Stand in front of the Kinect so it sees your right arm — the wizard
   shows `SDK body OK` in green when `HandTipRight` is being tracked.
3. For each corner (order: **TL → TR → BR → BL**):
   * point your **right hand fingertip** at that physical corner of the TV,
   * hold steady,
   * click **Confirm Corner**.
   The wizard advances to the next corner automatically.
4. After the 4th confirm, the bridge fits a plane through the 4 captured
   points (SVD), builds an in-plane (u, v) basis, and solves a homography
   from the corner (u, v) to the canvas corners. The result is saved to
   `tv_calibration.json` next to this README.

While capturing, **Redo** discards the last corner. **Cancel** aborts
without saving. **Reset TV Calibration** deletes the saved file.

### Verifying the fit

Toggle **Debug View: On** in the sidebar (`/debug/depth.jpg`). You'll see:

| color | meaning |
|-------|---------|
| TURBO pseudo-color | raw depth |
| **magenta polygon outline** | the 4 captured corners projected to the depth image |
| **magenta fill** | depth pixels within `BRIDGE_DEBUG_SURFACE_EPS_M` of the fitted plane *and* inside the polygon — should align with the physical TV |
| **light green** | the interaction box: pixels in the plane-normal slab `[BRIDGE_DEPTH_BAND_MIN_M, BRIDGE_DEPTH_BAND_MAX_M]` *and* inside the polygon |
| **yellow circle** | live SDK `HandTipRight` (handy during calibration) |
| **white cross + ring** | the depth-derived fingertip used at runtime (closest-to-plane median of the K nearest pixels) |

## Tuning (env vars)

| Var | Default | Effect |
|---|---|---|
| `BRIDGE_PORT` | `7000` | local WS port |
| `BRIDGE_CANVAS_W` / `_H` | `960` / `540` | must match NCA server `NCA_W/H` |
| `BRIDGE_BROADCAST_HZ` | `30` | outgoing hand-event rate |
| `BRIDGE_KINECT_MODE` | `depth` | `depth` (default) or `body` (legacy SDK only) |
| `BRIDGE_DEPTH_BAND_MIN_M` | `0.02` | near edge of the interaction box (m above plane) |
| `BRIDGE_DEPTH_BAND_MAX_M` | `0.45` | far edge of the interaction box (m above plane) |
| `BRIDGE_DEBUG_SURFACE_EPS_M` | `0.03` | thickness of the magenta "TV slab" in the debug overlay |

## Protocol (for reference)

Every broadcast frame:

```json
{
  "op": "hand",
  "t": 1734123456.789,
  "tracked": true,
  "confident": true,
  "tv_ready": true,
  "tv_status": {"mode": "idle", "ready": true, "current_corner": 0,
                "label": "TL", "captured": 0, "total": 4,
                "labels": ["TL", "TR", "BR", "BL"]},
  "body_tip_xyz": [0.18, -0.12, 1.42],
  "body_tracked": true,
  "tip_signed_dist_m": 0.083,
  "cx": 481.2,
  "cy": 270.7,
  "pinch": true
}
```

Calibration wizard ops (browser → bridge): `tv_calib_start`,
`tv_calib_confirm`, `tv_calib_redo`, `tv_calib_cancel`,
`tv_calib_reset`. Bridge → browser broadcasts: `tv_calib_started`,
`tv_calib_progress`, `tv_calib_done`, `tv_calib_cancelled`,
`tv_calib_reset`.
