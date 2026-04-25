# Kinect v2 Bridge (`remote_gpu/bridge`)

Local Windows process that turns a Kinect v2 into a fingertip cursor for the
NCA browser client. The runtime is **depth-first**:

* The TV lies flat (or at any orientation) and the Kinect is mounted at TV
  height, slightly tilted toward the screen.
* TV calibration is **one-click automatic**: capture ~1.5 s of depth
  frames, RANSAC-fit the dominant 3D plane, isolate the largest
  co-planar blob (= the TV screen), wrap a min-area rectangle around
  it to get 4 corners, and assign them to canvas TL/TR/BR/BL by their
  (X, Z) in camera space (front=top, right=right). The result is saved
  to `tv_calibration.json`.
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

## TV Calibration (one-time, auto)

Open the NCA page and connect to the bridge. In the **Kinect** sidebar:

1. Make sure the **TV screen is fully visible** to the Kinect, and
   **clear hands / people** out of the depth view (so the largest
   co-planar blob really is the TV).
2. Click **Auto-Calibrate TV (Depth)**.
3. The bridge captures ~1.5 s of depth, then:
   * RANSAC-fits the dominant 3D plane,
   * keeps depth pixels within `BRIDGE_AUTOFIT_EPS_M` of that plane,
   * morph-opens that mask (`BRIDGE_AUTOFIT_OPEN_PX`) to break thin
     bridges to neighbouring co-planar wood strips,
   * picks the **largest connected component** as the TV blob,
   * wraps a `cv2.minAreaRect` around the blob's in-plane (u, v)
     points,
   * **density-trims** that rect along its long & short axes — co-planar
     fringes like parallel wood slats with gaps between them have
     visibly lower per-row/per-column occupancy than the solid TV
     screen, so the trim shrinks the rect until each axis only contains
     the dense core (`BRIDGE_AUTOFIT_TRIM_FRAC` controls the cutoff;
     `BRIDGE_AUTOFIT_TRIM_MIN_M` is a per-axis safety floor).
   * Reconstructs 4 trimmed corners in 3D, sorts into TL/TR/BR/BL by
     (X, Z): front (smaller Z) = top, right (larger X) = right,
   * SVD-refits the plane on the 4 sorted corners and solves the
     homography to the canvas. Result saved to `tv_calibration.json`.

If the wood strips around the TV are co-planar *and* contiguous, the
density trim should clip them off automatically — the status bar will
show "trimmed N×M cm of co-planar fringe". If trimming is too
aggressive (eats into the TV), raise `BRIDGE_AUTOFIT_TRIM_FRAC`
(e.g. 0.55 → 0.75 keeps less); if it is not aggressive enough (slats
remain), lower it (e.g. 0.65 → 0.50). You can also bump the morph-open
kernel (`BRIDGE_AUTOFIT_OPEN_PX=5`) or tighten the on-plane tolerance
(`BRIDGE_AUTOFIT_EPS_M=0.010`) to physically detach more of the slats
from the TV blob upstream of the trim.

**Reset TV Calibration** deletes the saved file. Re-run Auto-Calibrate
any time the camera moves.

### Verifying the fit

Toggle **Debug View: On** in the sidebar (`/debug/depth.jpg`). You'll see:

| color | meaning |
|-------|---------|
| TURBO pseudo-color | raw depth |
| **magenta polygon outline** | the 4 auto-derived corners projected to the depth image — should hug the actual TV |
| **magenta fill** | depth pixels within `BRIDGE_DEBUG_SURFACE_EPS_M` of the fitted plane *and* inside the polygon |
| **light green** | the interaction box: pixels in the plane-normal slab `[BRIDGE_DEPTH_BAND_MIN_M, BRIDGE_DEPTH_BAND_MAX_M]` *and* inside the polygon |
| **yellow circle** | live SDK `HandTipRight` (visual reference only — no longer used for calibration) |
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
| `BRIDGE_AUTOFIT_EPS_M` | `0.015` | on-plane tolerance during auto-calibration |
| `BRIDGE_AUTOFIT_OPEN_PX` | `3` | morph-open kernel (px) on the on-plane mask |
| `BRIDGE_AUTOFIT_TRIM_FRAC` | `0.65` | density-trim cutoff as fraction of peak (0 disables trim) |
| `BRIDGE_AUTOFIT_TRIM_MIN_M` | `0.10` | per-axis minimum kept extent before trim is reverted (m) |
| `BRIDGE_AUTOFIT_TRIM_BIN_M` | `0.01` | density-grid bin size (m) |

Other auto-calibration parameters are currently set in code defaults
(`auto_calibrate_tv_from_depth(...)`): RANSAC inlier threshold `2 cm`,
min blob `1500 px`, plane fit point limit `12 000`. Adjust by editing
`bridge/depth_processing.py` if your scene needs it.

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

Calibration ops (browser → bridge):

| op | effect |
|---|---|
| `tv_autocalib_capture` | start one-shot depth-plane fit + 4-corner derivation |
| `tv_calib_reset`       | delete saved `tv_calibration.json` |

Bridge → browser broadcasts:

| op | payload |
|---|---|
| `tv_autocalib_started` | (no fields) — capture begun |
| `tv_autocalib_done`    | `ready, n_blob_px, area_m2, edge_a_m, edge_b_m, corners_3d, tv_status` |
| `tv_autocalib_failed`  | `reason` (e.g. `RANSAC failed`, `largest blob too small`, `finalize: ...`) |
| `tv_calib_reset`       | calibration cleared |
