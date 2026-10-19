from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

from PyQt5.QtCore import QByteArray

from ..localization import translate as _
from .network import NetworkError, RequestManager

max_return_bytes = 64 * 1024 * 1024
max_return_pixels = 8192 * 8192
_image_path = re.compile(r"^/api/krita/inbox/(?!\.+/)[A-Za-z0-9._~-]+/image$")
_idempotency_key = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_device_id = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")


@dataclass
class InboxItem:
    id: str
    title: str
    source: str
    filename: str
    mime: str
    bytes: int
    sha256: str
    width: int
    height: int
    image_url: str
    status: str
    return_target: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    expires_at: str = ""
    created_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    opened_by: dict[str, Any] | None = None
    last_return: dict[str, Any] | None = None

    @staticmethod
    def from_dict(data: dict):
        image_url = str(data.get("imageUrl", ""))
        if not _image_path.match(image_url):
            raise ValueError(f"Inbox item has an unexpected image location: {image_url}")
        sha256 = str(data.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("Inbox item has no valid checksum")
        return InboxItem(
            id=str(data["id"]),
            title=str(data.get("title", ""))[:160],
            source=str(data.get("source", "")),
            filename=str(data.get("filename", "")),
            mime=str(data.get("mime", "")),
            bytes=int(data.get("bytes", 0)),
            sha256=sha256,
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            image_url=image_url,
            status=str(data.get("status", "new")),
            return_target=dict(data.get("returnTarget") or {}),
            version=int(data.get("version", 0)),
            expires_at=str(data.get("expiresAt", "")),
            created_at=str(data.get("createdAt", "")),
            meta=dict(data.get("meta") or {}),
            opened_by=data.get("openedBy"),
            last_return=data.get("lastReturn"),
        )

    @property
    def is_comic_panel(self):
        return self.return_target.get("kind") == "comicPanel"


@dataclass
class InboxChanges:
    items: list[InboxItem]
    removed: list[str]
    cursor: str
    full: bool  # True if this is a complete listing (no cursor was used)


class InboxError(Exception):
    def __init__(self, code: str, message: str, data: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}

    def __str__(self):
        return self.message


def _inbox_error(e: NetworkError):
    data = e.data or {}
    code = str(data.get("code", ""))
    match code, e.status:
        case "expired", _:
            msg = _("This picture is no longer in the website inbox (it expired after 7 days)")
        case "not_found", _:
            msg = _("This picture was removed from the website inbox")
        case "already_opened", _:
            device = (data.get("openedBy") or {}).get("deviceName") or "?"
            msg = _("This picture is already open on {device}", device=device)
        case "too_large", _:
            limit = int(data.get("maxBytes", max_return_bytes)) // (1024 * 1024)
            msg = _("The picture is too large for the website (limit {size} MB)", size=limit)
        case "too_many_pixels", _:
            msg = _("The picture is too large for the website (more than 8192×8192 pixels)")
        case _, 401 | 403:
            msg = _("Your website login expired. Sign in again.")
            code = "auth"
        case _:
            msg = data.get("error") or e.message
            code = code or "network"
    return InboxError(code, msg, data)


class InboxAPI:
    def __init__(self, url: str, token: str, requests: RequestManager | None = None):
        self.url = url.rstrip("/")
        self.requests = requests or RequestManager(same_origin_only=True)
        self.requests.set_auth(token)

    def _item_url(self, item_id: str, action: str = ""):
        path = f"{self.url}/api/krita/inbox/{quote(item_id, safe='')}"
        return f"{path}/{action}" if action else path

    async def _call(self, method: str, url: str, data=None, timeout=30.0, **kwargs):
        try:
            return await self.requests.http(method, url, data, timeout=timeout, **kwargs)
        except NetworkError as e:
            raise _inbox_error(e) from e

    async def supported(self):
        info = await self._call("GET", f"{self.url}/api/krita/info", timeout=15)
        features = info.get("features", []) if isinstance(info, dict) else []
        return "inbox:1" in features

    async def changes(self, cursor: str = "", wait: int = 0):
        query = {}
        if cursor:
            query = {"since": cursor, "wait": str(max(0, min(25, wait)))}
        url = f"{self.url}/api/krita/inbox"
        if query:
            url += "?" + urlencode(query)
        result = await self._call("GET", url, timeout=wait + 30)
        if not isinstance(result, dict):
            raise InboxError("invalid", "Invalid inbox response")
        items = []
        for data in result.get("items", []):
            try:
                items.append(InboxItem.from_dict(data))
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed entries instead of stopping the inbox
        removed = [str(i) for i in result.get("removed", [])]
        return InboxChanges(items, removed, str(result.get("cursor", "")), not cursor)

    async def image(self, item: InboxItem):
        try:
            data = await self.requests.download(
                self.url + item.image_url, timeout=300, max_bytes=max_return_bytes
            )
        except NetworkError as e:
            raise _inbox_error(e) from e
        raw = bytes(data)
        if len(raw) != item.bytes or hashlib.sha256(raw).hexdigest() != item.sha256:
            raise InboxError(
                "checksum",
                _("The picture from the website arrived damaged (checksum mismatch). Try again."),
            )
        return raw

    async def ack(self, item_id: str, device_id: str, device_name: str, force=False):
        assert _device_id.match(device_id), "invalid device id"
        body = {"deviceId": device_id, "deviceName": device_name[:120], "force": force}
        result = await self._call("POST", self._item_url(item_id, "ack"), body)
        return InboxItem.from_dict(result["item"])

    async def return_image(self, item_id: str, png: bytes, width: int, height: int, key: str):
        assert _idempotency_key.match(key), "invalid idempotency key"
        if len(png) > max_return_bytes:
            raise InboxError(
                "too_large",
                _(
                    "The picture is too large for the website (limit {size} MB)",
                    size=max_return_bytes // (1024 * 1024),
                ),
            )
        if width * height > max_return_pixels:
            raise InboxError(
                "too_many_pixels",
                _("The picture is too large for the website (more than 8192×8192 pixels)"),
            )
        url = (
            self._item_url(item_id, "return") + "?" + urlencode({"width": width, "height": height})
        )
        result = await self._call(
            "POST",
            url,
            QByteArray(png),
            timeout=300,
            content_type="image/png",
            headers=[("Idempotency-Key", key)],
        )
        if not isinstance(result, dict) or not result.get("ok"):
            raise InboxError("invalid", "Invalid response from the website")
        return result

    async def delete(self, item_id: str):
        await self._call("DELETE", self._item_url(item_id))
