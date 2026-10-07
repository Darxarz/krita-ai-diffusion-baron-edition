from __future__ import annotations

import asyncio
from urllib.parse import urljoin, urlsplit, urlunsplit

from PyQt5.QtCore import QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from .network import NetworkError

_preview_hosts = {"image.civitai.com", "blobs-b2.civitai.com"}
_max_bytes = 5 * 1024 * 1024


def _allowed(url: str):
    try:
        parts = urlsplit(url)
        return (
            parts.scheme == "https"
            and parts.hostname in _preview_hosts
            and parts.port in (None, 443)
            and not parts.username
            and not parts.password
        )
    except ValueError:
        return False


def preview_url(url: str):
    if not _allowed(url):
        raise ValueError("Unsupported preview address")
    parts = urlsplit(url)
    path = parts.path.split("/")
    if parts.hostname == "image.civitai.com" and len(path) == 5 and all(path[i] for i in (1, 2, 4)):
        path[3] = "width=320,quality=85,format=jpeg"
    return urlunsplit((parts.scheme, parts.netloc, "/".join(path), parts.query, ""))


def preview_redirect(source: str, target: str):
    result = urljoin(source, target)
    if not target or not _allowed(result):
        raise ValueError("Unsupported preview redirect")
    return result


class ModelPreview:
    def __init__(self, network: QNetworkAccessManager | None = None):
        self._network = network if network is not None else QNetworkAccessManager()

    async def download(self, url: str, timeout=15):
        url = preview_url(url)
        for redirects in range(4):
            status, target, data = await self._get(url, timeout)
            if 300 <= status < 400 and redirects < 3:
                url = preview_redirect(url, target)
            elif 200 <= status < 300:
                return data
            else:
                raise ValueError("Preview could not be loaded")
        raise ValueError("Too many preview redirects")

    async def _get(self, url: str, timeout: float):
        request = QNetworkRequest(QUrl(url))
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.ManualRedirectPolicy,
        )
        request.setRawHeader(b"Accept", b"image/jpeg,image/png,image/webp")
        request.setTransferTimeout(int(timeout * 1000))
        reply = self._network.get(request)
        assert reply is not None
        reply.setReadBufferSize(64 * 1024)
        future = asyncio.get_running_loop().create_future()
        data = bytearray()

        def read():
            data.extend(reply.readAll().data())
            if len(data) > _max_bytes:
                if not future.done():
                    future.set_exception(ValueError("Preview exceeds the size limit"))
                reply.abort()

        def finished():
            if future.done():
                return
            read()
            if future.done():
                return
            code = reply.error()  # type: ignore (bug in PyQt5-stubs)
            if code != QNetworkReply.NetworkError.NoError:
                future.set_exception(NetworkError.from_reply(reply))
            else:
                target = reply.attribute(QNetworkRequest.Attribute.RedirectionTargetAttribute)
                future.set_result((
                    reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute),
                    target.toString() if isinstance(target, QUrl) else "",
                    bytes(data),
                ))

        reply.readyRead.connect(read)
        reply.finished.connect(finished)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            if not reply.isFinished():
                reply.abort()
            reply.deleteLater()
