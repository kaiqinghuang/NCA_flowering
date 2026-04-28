# Kinect v2 Bridge (`remote_gpu/bridge`)

Local Windows process that turns a Kinect v2 into a fingertip cursor for the
NCA browser client. The runtime is **depth-first**:

* The TV lies flat (or at any orientation) and the Kinect is mounted at TV
  height, slightly tilted toward the screen.
* TV calibration is **one-click automatic**: capture ~1.5 s of depth
  frames, RANSAC-fit the dominant 3D plane, isolate the largest
  co-planar blob (= the TV screen), fit a tight rectangle around it
  to get 4 corners, and assign them to **A/B/C/D** (clockwise from
  front-left, mapped to canvas **TL/TR/BR/BL** respectively) by their
  (X, Z) in camera space (front=top, right=right). The result is saved
  to `tv_calibration.json`.
* At runtime, the bridge **does not use the SDK skeleton at all** — only
  the depth frame is processed. Pipeline per frame:
  1. Filter pixels by *(inside the calibrated TV polygon) ∧ (signed
     distance to plane within `[0.02, 0.45]` m)* → raw `in_box` mask.
  2. Morph-open + largest connected component → `hand_mask` (the
     actual hand silhouette, with single-pixel specks and edge
     filaments stripped out).
  3. **PCA on `hand_mask` in (u, v) image space** to find the hand's
     long axis, take the two extrema of the projection (top/bottom 5%),
     pick the end whose mean signed distance to the plane is smaller →
     that end is the **fingertip**, the other end is the wrist/forearm.
     Median (x, y, z) of the chosen end's pixels is the 3D position.
  4. Temporal EMA on the fingertip 3D position (and debug u, v) with
     factor `BRIDGE_TIP_EMA_ALPHA` (default `0.5`) to suppress
     sub-frame jitter.
  5. While a fingertip exists in the interaction box, emit
     `pinch=true` so the canvas paints continuously.

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

The bridge opens **Depth + Color** only — the SDK skeleton (Body)
source is not used. Color is sampled once during auto-calibration to
filter out non-screen co-planar surfaces (wooden frames, bezels) by
brightness; it is not processed during the live loop.

Expected boot log:

```
[bridge] Kinect runtime: depth-only (no SDK skeleton)
[kinect-depth] Runtime started (depth + color, no body).
[bridge] canvas=960x540  box=[0.02,0.45]m  tv_ready=no  listening on ws://0.0.0.0:7000/ws
```

If you see `PyKinect2 import failed` the bridge keeps running but emits no
hand frames — install PyKinect2 and the Kinect SDK.

## TV Calibration (one-time, auto)

Open the NCA page and connect to the bridge. In the **Kinect** sidebar:

1. Make sure the **TV screen is fully visible** to the Kinect, and
   **clear hands / people** out of the depth view (so the largest
   co-planar blob really is the TV).
2. Make sure the TV screen is **black or showing dark content**
   during calibration (the algorithm uses the screen's near-zero
   brightness to distinguish it from any surrounding wood/bezel).
3. Click **Auto-Calibrate TV (Depth)**.
4. The bridge captures ~1.5 s of depth + 1 colour frame, then:
   * RANSAC-fits the dominant 3D plane,
   * keeps depth pixels within `BRIDGE_AUTOFIT_EPS_M` of that plane,
   * morph-opens that mask (`BRIDGE_AUTOFIT_OPEN_PX`) to break thin
     depth bridges to neighbouring co-planar surfaces,
   * picks the **largest connected component** as the depth-only
     candidate blob,
   * **uses the SDK `CoordinateMapper` to project every blob pixel
     into the colour frame** and discards anything brighter than
     `BRIDGE_AUTOFIT_COLOR_MAX_V` (default `90`). Wood/bezel survive
     the depth pipeline because they sit on the TV plane, but they
     are visibly bright while the powered-off TV is near-black —
     this single test cleanly removes them.
   * Closes pinholes (`BRIDGE_AUTOFIT_COLOR_CLOSE_PX`) and takes the
     largest CC again as the final TV mask,
   * Projects the colour-refined blob to the plane (u, v) and runs a
     **PCA + percentile** rectangle fit: PCA finds the TV's long /
     short edges, then we identify the front-back axis (the PCA axis
     with the largest |Z| component in camera space) and apply
     **asymmetric** trim there:
     `BRIDGE_AUTOFIT_TRIM_FRONT_PCT` on the side facing the camera
     (smaller Z) and `BRIDGE_AUTOFIT_TRIM_BACK_PCT` on the side away
     from it, while the left-right axis uses the symmetric
     `BRIDGE_AUTOFIT_TRIM_PCT`. This rejects bezel / transition pixels
     that survive the colour pass — typically concentrated on the front
     edge — without pulling the (usually clean) back edge inward. Set
     `BRIDGE_AUTOFIT_TRIM_PCT=0` (and leave the front/back overrides
     unset) to fall back to a strict bounding box.
   * Sorts the 4 corners into **A/B/C/D** (clockwise from front-left,
     mapped to canvas **TL/TR/BR/BL**) by (X, Z): front (smaller Z)
     = top, right (larger X) = right,
   * SVD-refits the plane on the 4 sorted corners and solves the
     homography to the canvas. Result saved to `tv_calibration.json`.

The status bar will show e.g. `color: 8214→5103px (removed 37.9% as
bright)`, telling you exactly how much of the depth blob was wood/bezel.

**Tuning the colour pass:**
* If wood/bezel is still inside the rect → lower
  `BRIDGE_AUTOFIT_COLOR_MAX_V` (e.g. `90 → 60`).
* If too much of the TV is being eaten (status shows the algorithm fell
  back to depth-only because `<2 %` survived) → raise it (`90 → 130`)
  or check that the TV is actually showing dark content.
* To disable the colour pass entirely (e.g. debugging): set
  `BRIDGE_AUTOFIT_COLOR_ENABLE=0`.

**Tuning the rectangle fit:**
* The PCA fit applies trim **independently** on each side of the rectangle.
  `BRIDGE_AUTOFIT_TRIM_PCT` is the symmetric default; the front-back axis
  honours `BRIDGE_AUTOFIT_TRIM_FRONT_PCT` and `BRIDGE_AUTOFIT_TRIM_BACK_PCT`
  if those are set.
* **A/B drifting in front of the TV** (most common case — bezel transition
  pixels survive the colour pass on the edge closest to the camera):
  raise `BRIDGE_AUTOFIT_TRIM_FRONT_PCT` (e.g. `1.5`). Each unit trims one
  extra percent of the most-extreme PCA-projected points on the front side.
* **C/D drifting outside the TV**: raise `BRIDGE_AUTOFIT_TRIM_BACK_PCT`.
* **C/D shrinking inward** (back edge already clean from the colour pass,
  but trim is eating it): set `BRIDGE_AUTOFIT_TRIM_BACK_PCT=0`.
* **Whole rectangle eating into the TV**: lower `BRIDGE_AUTOFIT_TRIM_PCT`
  or set `0` to use a strict bounding box.

**Reset TV Calibration** deletes the saved file. Re-run Auto-Calibrate
any time the camera moves.

### Verifying the fit

Toggle **Debug View: On** in the sidebar (`/debug/depth.jpg`). You'll see:

| color | meaning |
|-------|---------|
| TURBO pseudo-color | raw depth |
| **magenta polygon outline** | the 4 auto-derived corners projected to the depth image — should hug the actual TV |
| **magenta fill** | depth pixels within `BRIDGE_DEBUG_SURFACE_EPS_M` of the fitted plane *and* inside the polygon |
| **orange wireframe** | the 3D interaction box: 4 top edges + 4 vertical struts rising `BRIDGE_DEPTH_BAND_MAX_M` above the TV plane. Always visible; shows where the detection volume sits in space |
| **orange fill** | the **cleaned hand silhouette**: in-box pixels are first morph-opened with a `BRIDGE_DEBUG_NOISE_FILTER_PX` kernel (kills speckle / edge filaments) and then reduced to the largest connected component, so only the real hand shows up |
| **red square (5×5 px)** | the depth-derived fingertip used at runtime — pixel inside the orange hand blob with the smallest signed distance to the TV plane (median of the K nearest pixels) |

## Tuning (env vars)

| Var | Default | Effect |
|---|---|---|
| `BRIDGE_PORT` | `7000` | local WS port |
| `BRIDGE_CANVAS_W` / `_H` | `960` / `540` | must match NCA server `NCA_W/H` |
| `BRIDGE_BROADCAST_HZ` | `30` | outgoing hand-event rate |
| `BRIDGE_DEPTH_BAND_MIN_M` | `0.02` | near edge of the interaction box (m above plane) |
| `BRIDGE_DEPTH_BAND_MAX_M` | `0.45` | far edge of the interaction box (m above plane) |
| `BRIDGE_DEBUG_SURFACE_EPS_M` | `0.03` | thickness of the magenta "TV slab" in the debug overlay |
| `BRIDGE_DEBUG_NOISE_FILTER_PX` | `3` | morph-open kernel (px) applied to the in-box mask before largest-CC. Stabilizes the fingertip pick by stripping single-pixel specks and ~1-px filaments that connect the hand to edge noise. Set to `0` or `1` to disable |
| `BRIDGE_TIP_EMA_ALPHA` | `0.5` | temporal EMA factor on the fingertip 3D position (and debug u,v). `1.0` = no smoothing (raw); lower = more smoothing. Try `0.3` if the red square still jitters, `0.7` if it feels laggy |
| `BRIDGE_AUTOFIT_EPS_M` | `0.015` | on-plane tolerance during auto-calibration |
| `BRIDGE_AUTOFIT_OPEN_PX` | `3` | morph-open kernel (px) on the on-plane mask |
| `BRIDGE_AUTOFIT_COLOR_ENABLE` | `1` | `0` disables the RGB refine pass (depth-only fit) |
| `BRIDGE_AUTOFIT_COLOR_MAX_V` | `90` | max brightness (max(B,G,R), 0–255) to count as TV-black |
| `BRIDGE_AUTOFIT_COLOR_CLOSE_PX` | `3` | morph-close kernel (px) on the colour-refined mask |
| `BRIDGE_AUTOFIT_TRIM_PCT` | `1.5` | per-side percentile trim on the **left-right** axis of the PCA rect fit. Also the default for front/back when their overrides are unset. `0` = strict min/max bounding box |
| `BRIDGE_AUTOFIT_TRIM_FRONT_PCT` | _(inherits TRIM_PCT)_ | trim on the **front** edge of the TV (closer to the camera). Raise to pull A/B inward when bezel pixels leak past the colour filter |
| `BRIDGE_AUTOFIT_TRIM_BACK_PCT` | _(inherits TRIM_PCT)_ | trim on the **back** edge of the TV (farther from the camera). Set to `0` to keep C/D pinned to the actual blob extent |

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
                "label": "A", "captured": 0, "total": 4,
                "labels": ["A", "B", "C", "D"]},
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
