import asyncio

from aiohttp import web
from PyQt5.QtNetwork import QNetworkReply

from ai_diffusion.backend.network import RequestManager

from .conftest import qtapp


@qtapp
async def test_http_deadline_aborts_underlying_request():
    async def slow(request):
        await asyncio.sleep(0.1)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/slow", slow)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    requests = RequestManager()
    try:
        future = requests.post(f"http://127.0.0.1:{runner.addresses[0][1]}/slow", {})
        reply = next(iter(requests._requests))
        future.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
        assert reply.isFinished()
        assert reply.error() == QNetworkReply.NetworkError.OperationCanceledError  # type: ignore (PyQt5-stubs)
    finally:
        await runner.cleanup()
