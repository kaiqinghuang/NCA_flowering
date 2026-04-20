import asyncio
from fastapi import WebSocket

class ClientConnection:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
        self.send_task = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        try:
            while True:
                payload = await self.queue.get()
                await self.ws.send_bytes(payload)
        except Exception:
            pass

    def cancel(self):
        self.send_task.cancel()
