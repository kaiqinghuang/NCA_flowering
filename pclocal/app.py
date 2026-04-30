"""Local single-process launcher for the NCA + Kinect demo.

This is the **PC-local** version of the project. It combines the two
services that the remote setup runs in separate processes
(`server/nca_server.py` and `bridge/main.py`) into one FastAPI app on a
single port — no separate bridge process, no port 7000, no remote GPU.

All NCA + Kinect technical logic is identical to the remote_gpu code
paths; this file only adds wiring so both halves live in one ASGI app.

Run from the `pclocal/` directory (Windows w/ Kinect plugged in):

    uvicorn app:app --host 127.0.0.1 --port 8000

Endpoints:
    /                            → client (index.html)
    /static/...                  → client static assets
    /ws                          → NCA WebSocket (frames, paint, params)
    /bridge/ws                   → Kinect WebSocket (hand events, calib)
    /bridge/debug/depth.jpg      → live depth-debug overlay (when on)

Notes:
    * `pclocal/server/` is added to ``sys.path`` so the server module's
      existing top-level imports (``from nca_model import ...``) keep
      working without renaming.
    * Mounted ASGI sub-apps in Starlette do **not** receive their
      `@app.on_event` handlers from the parent's lifespan, so we
      forward the bridge sub-app's startup/shutdown hooks explicitly
      below. (Calling them once each is safe; ``KinectDepthSource.start``
      also has a duplicate-call guard.)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Make the server modules importable. They use plain top-level imports
# (``from nca_model import …``), not relative imports, so they need to
# find their siblings on ``sys.path`` rather than as a package.
if str(ROOT / "server") not in sys.path:
    sys.path.insert(0, str(ROOT / "server"))
# Make the ``bridge`` package importable. It does use relative imports,
# so we add the parent directory (which contains ``bridge/__init__.py``).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Import order matters: server first (creates its FastAPI app + static
# mounts), bridge second (Kinect hardware kept dormant until startup).
from server import nca_server  # noqa: E402  pylint: disable=wrong-import-position
from bridge import main as bridge_main  # noqa: E402  pylint: disable=wrong-import-position


nca_app = nca_server.app
bridge_app = bridge_main.app

# Mount the bridge sub-app under /bridge. After this:
#   bridge_app's   /ws                 →  /bridge/ws
#   bridge_app's   /debug/depth.jpg    →  /bridge/debug/depth.jpg
#   bridge_app's   /                   →  /bridge/   (health JSON)
# The NCA app keeps its /ws, /, and /static routes unchanged.
nca_app.mount("/bridge", bridge_app)


async def _run_handlers(handlers) -> None:
    """Invoke a list of startup/shutdown handlers, awaiting any coroutines."""
    for h in handlers:
        try:
            result = h()
            if inspect.isawaitable(result):
                await result
        except Exception as e:  # noqa: BLE001
            # Don't let one handler's failure stop the others.
            print(f"[pclocal] bridge lifespan handler {h!r} failed: {e}")


@nca_app.on_event("startup")
async def _start_bridge_lifespan() -> None:
    await _run_handlers(bridge_app.router.on_startup)


@nca_app.on_event("shutdown")
async def _stop_bridge_lifespan() -> None:
    await _run_handlers(bridge_app.router.on_shutdown)


# Top-level ASGI app for ``uvicorn app:app``.
app = nca_app
