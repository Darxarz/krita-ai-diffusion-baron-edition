import asyncio
import hashlib
import json

import pytest
from aiohttp import web

from ai_diffusion.backend.orchestrion_inbox import InboxAPI, InboxError, InboxItem
from ai_diffusion.image import Extent, Image
from ai_diffusion.model.inbox import Inbox, parse_document_info, return_key
from ai_diffusion.settings import settings

from .conftest import qtapp

token = "ork_krita_device-key"


def png(width=4, height=3, fill=0xFF336699):
    return Image.create(Extent(width, height), fill=fill).to_bytes().data()


class Site:
    def __init__(self, features=("inbox:1", "model-arch:1")):
        self.features = list(features)
        self.items: dict[str, dict] = {}
        self.images: dict[str, bytes] = {}
        self.removed: list[str] = []
        self.revision = 0
        self.requests: list[tuple[str, str]] = []
        self.auth: set[str] = set()
        self.acks: list[dict] = []
        self.returns: list[dict] = []
        self.returned_keys: dict[str, dict] = {}
        self.opened_by_other: set[str] = set()
        self.corrupt: set[str] = set()
        self.unauthorized = False

    def add(self, item_id: str, data: bytes, **extra):
        self.revision += 1
        self.images[item_id] = data
        self.items[item_id] = {
            "id": item_id,
            "createdAt": f"2026-10-19T10:00:0{len(self.items)}Z",
            "title": f"Picture {item_id}",
            "source": "gallery",
            "filename": f"{item_id}.png",
            "mime": "image/png",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "width": 4,
            "height": 3,
            "imageUrl": f"/api/krita/inbox/{item_id}/image",
            "status": "new",
            "returnTarget": {"kind": "gallery"},
            "version": 0,
            "expiresAt": "2026-10-26T10:00:00Z",
            "meta": {"prompt": "a cat"},
            **extra,
        }

    def check(self, request: web.Request):
        self.requests.append((request.method, request.path_qs))
        self.auth.add(request.headers.get("Authorization", ""))
        if self.unauthorized:
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "Login expired", "code": "unauthorized"}),
                content_type="application/json",
            )

    async def info(self, request):
        self.check(request)
        return web.json_response({"version": 1, "features": self.features})

    async def inbox(self, request):
        self.check(request)
        since = request.query.get("since")
        if since is None:
            return web.json_response({
                "items": list(self.items.values()),
                "cursor": str(self.revision),
            })
        if since == str(self.revision):
            await asyncio.sleep(min(0.2, float(request.query.get("wait", 0))))
            return web.json_response({"items": [], "removed": [], "cursor": since})
        return web.json_response({
            "items": list(self.items.values()),
            "removed": self.removed,
            "cursor": str(self.revision),
        })

    async def image(self, request):
        self.check(request)
        item_id = request.match_info["id"]
        data = self.images[item_id]
        if item_id in self.corrupt:
            data = data[:-1] + bytes([data[-1] ^ 0xFF])
        return web.Response(body=data, content_type="image/png")

    async def ack(self, request):
        self.check(request)
        item_id = request.match_info["id"]
        body = await request.json()
        self.acks.append(body)
        if item_id in self.opened_by_other and not body.get("force"):
            return web.json_response(
                {
                    "error": "Already opened",
                    "code": "already_opened",
                    "openedBy": {"deviceName": "Tablet", "at": "now"},
                },
                status=409,
            )
        item = self.items[item_id]
        item["status"] = "opened"
        item["openedBy"] = {"deviceName": body["deviceName"], "at": "now"}
        self.revision += 1
        return web.json_response({"ok": True, "item": item})

    async def return_image(self, request):
        self.check(request)
        item_id = request.match_info["id"]
        key = request.headers.get("Idempotency-Key", "")
        body = await request.read()
        self.returns.append({
            "key": key,
            "type": request.headers.get("Content-Type"),
            "width": request.query.get("width"),
            "height": request.query.get("height"),
            "body": body,
        })
        if key in self.returned_keys:
            return web.json_response(
                self.returned_keys[key], headers={"Idempotent-Replayed": "true"}
            )
        item = self.items[item_id]
        item["version"] += 1
        item["status"] = "returned"
        result = {"ok": True, "generationId": 100 + item["version"], "version": item["version"]}
        if item["returnTarget"]["kind"] == "comicPanel":
            result["placedInto"] = {"projectId": 8, "panelId": "p1"}
        self.returned_keys[key] = result
        return web.json_response(result)

    async def delete(self, request):
        self.check(request)
        item_id = request.match_info["id"]
        self.items.pop(item_id)
        self.removed.append(item_id)
        self.revision += 1
        return web.json_response({"ok": True})


async def serve(site: Site):
    app = web.Application()
    app.router.add_get("/api/krita/info", site.info)
    app.router.add_get("/api/krita/inbox", site.inbox)
    app.router.add_get("/api/krita/inbox/{id}/image", site.image)
    app.router.add_post("/api/krita/inbox/{id}/ack", site.ack)
    app.router.add_post("/api/krita/inbox/{id}/return", site.return_image)
    app.router.add_delete("/api/krita/inbox/{id}", site.delete)
    runner = web.AppRunner(app)
    await runner.setup()
    server = web.TCPSite(runner, "127.0.0.1", 0)
    await server.start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


class Opened:
    def __init__(self):
        self.documents: list[tuple[Image, str, dict]] = []

    def __call__(self, image: Image, name: str, info):
        self.documents.append((image, name, parse_document_info(info) or {}))


async def until(condition, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.02)


@pytest.fixture(autouse=True)
def inbox_settings(monkeypatch):
    monkeypatch.setattr(settings, "orchestrion_device_id", "krita-test-device")
    monkeypatch.setattr(settings, "orchestrion_inbox_auto_open", True)


@qtapp
async def test_inbox_opens_new_picture_and_returns_it():
    site = Site()
    site.add("item-1", png(), returnTarget={"kind": "comicPanel", "projectId": 8, "panelId": "p1"})
    runner, url = await serve(site)
    opened = Opened()
    inbox = Inbox(opened)
    try:
        inbox.start(InboxAPI(url, token))
        await until(lambda: len(opened.documents) == 1)
        image, name, info = opened.documents[0]
        assert image.extent == Extent(4, 3)
        assert name == "Picture item-1"
        assert info["id"] == "item-1" and info["origin"] == url
        assert info["returnTarget"]["kind"] == "comicPanel"
        assert site.acks == [
            {
                "deviceId": "krita-test-device",
                "deviceName": site.acks[0]["deviceName"],
                "force": False,
            }
        ]
        assert site.auth == {f"Bearer {token}"}
        assert inbox.items["item-1"].status == "opened"

        result_png = png(fill=0xFFFF0000)
        result = await inbox.return_picture(info, result_png, 4, 3)
        assert result["version"] == 1 and result["placedInto"] == {"projectId": 8, "panelId": "p1"}
        sent = site.returns[0]
        assert sent["type"] == "image/png" and sent["body"] == result_png
        assert (sent["width"], sent["height"]) == ("4", "3")
        assert sent["key"] == return_key("item-1", result_png)

        # A retry with the same picture reuses the key: no new version on the site
        again = await inbox.return_picture(info, result_png, 4, 3)
        assert again == result and site.items["item-1"]["version"] == 1
        # A changed picture is a new version
        changed = await inbox.return_picture(info, png(fill=0xFF00FF00), 4, 3)
        assert changed["version"] == 2
    finally:
        inbox.stop()
        await runner.cleanup()


@qtapp
async def test_inbox_rejects_damaged_download():
    site = Site()
    site.add("broken", png())
    site.corrupt.add("broken")
    runner, url = await serve(site)
    opened = Opened()
    inbox = Inbox(opened)
    errors = []
    inbox.failed.connect(errors.append)
    try:
        inbox.start(InboxAPI(url, token))
        await until(lambda: len(errors) == 1)
        assert "checksum" in errors[0]
        assert opened.documents == []
    finally:
        inbox.stop()
        await runner.cleanup()


@qtapp
async def test_inbox_skips_pictures_open_elsewhere_until_forced():
    site = Site()
    site.add("taken", png())
    site.opened_by_other.add("taken")
    runner, url = await serve(site)
    opened = Opened()
    inbox = Inbox(opened)
    errors = []
    inbox.failed.connect(errors.append)
    try:
        inbox.start(InboxAPI(url, token))
        await until(lambda: len(site.acks) == 1)
        await asyncio.sleep(0.1)
        assert opened.documents == [] and errors == []
        with pytest.raises(InboxError) as e:
            await inbox.open("taken")
        assert e.value.code == "already_opened" and "Tablet" in e.value.message
        await inbox.open("taken", force=True)
        assert len(opened.documents) == 1 and site.acks[-1]["force"] is True
    finally:
        inbox.stop()
        await runner.cleanup()


@qtapp
async def test_inbox_follows_changes_and_removals():
    site = Site()
    site.add("a", png())
    runner, url = await serve(site)
    opened = Opened()
    inbox = Inbox(opened)
    try:
        inbox.start(InboxAPI(url, token))
        await until(lambda: len(opened.documents) == 1)
        site.add("b", png())
        await until(lambda: len(opened.documents) == 2)
        await inbox.delete("a")
        assert "a" not in inbox.items and "a" not in site.items
        assert [d[2]["id"] for d in opened.documents] == ["a", "b"]
        assert any("since=" in path for _, path in site.requests)
    finally:
        inbox.stop()
        await runner.cleanup()


@qtapp
async def test_inbox_disabled_without_feature():
    site = Site(features=["model-arch:1"])
    site.add("a", png())
    runner, url = await serve(site)
    inbox = Inbox(Opened())
    try:
        inbox.start(InboxAPI(url, token))
        await until(lambda: len(site.requests) >= 1)
        await asyncio.sleep(0.1)
        assert not inbox.enabled
        assert [path for _, path in site.requests] == ["/api/krita/info"]
    finally:
        inbox.stop()
        await runner.cleanup()


@qtapp
async def test_inbox_stops_on_expired_login():
    site = Site()
    runner, url = await serve(site)
    inbox = Inbox(Opened())
    try:
        inbox.start(InboxAPI(url, token))
        await until(lambda: inbox.enabled)
        site.unauthorized = True
        site.add("a", png())
        await until(lambda: inbox.error != "")
        count = len(site.requests)
        await asyncio.sleep(0.3)
        assert len(site.requests) == count
    finally:
        inbox.stop()
        await runner.cleanup()


@qtapp
async def test_return_is_limited_to_the_origin_website():
    inbox = Inbox(Opened())
    inbox._api = InboxAPI("http://127.0.0.1:1", token)
    with pytest.raises(InboxError) as e:
        await inbox.return_picture({"id": "x", "origin": "https://other.site"}, png(), 4, 3)
    assert e.value.code == "origin"
    with pytest.raises(InboxError) as e:
        await inbox._api.return_image("x", png(), 9000, 9000, "key-12345678")
    assert e.value.code == "too_many_pixels"


@pytest.mark.parametrize(
    "url",
    [
        "https://elsewhere.example/image.png",
        "//elsewhere.example/api/krita/inbox/a/image",
        "/api/krita/inbox/../../secret/image",
        "/api/krita/inbox/a/image?redirect=1",
    ],
)
def test_inbox_item_image_must_stay_on_site(url):
    data = {"id": "a", "imageUrl": url, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        InboxItem.from_dict(data)


def test_document_info_roundtrip():
    assert parse_document_info(b'{"id": "a", "origin": "x"}') == {"id": "a", "origin": "x"}
    assert parse_document_info(b"garbage") is None
    assert parse_document_info(b'{"id": 5}') is None
    assert parse_document_info(None) is None
