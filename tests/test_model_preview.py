import asyncio

import pytest
from PyQt5.QtCore import QIODevice, QTimer, QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from ai_diffusion.backend.model_preview import ModelPreview, preview_redirect, preview_url

from .conftest import qtapp

_original = "https://image.civitai.com/account/id/original=true/photo.png"
_blob = "https://blobs-b2.civitai.com/file/preview.jpg"


def test_thumbnail_url_uses_small_jpeg_and_allows_civitai_blob():
    assert preview_url(_original) == (
        "https://image.civitai.com/account/id/width=320,quality=85,format=jpeg/photo.png"
    )
    assert preview_url(_blob) == _blob
    assert preview_redirect(_original, _blob) == _blob
    assert preview_redirect(_blob, "/file/next.jpg") == (
        "https://blobs-b2.civitai.com/file/next.jpg"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://image.civitai.com/image",
        "https://evil.example/image",
        "https://image.civitai.com.evil.example/image",
        "https://user@image.civitai.com/image",
        "https://image.civitai.com:8188/image",
        "file:///private",
        "https://127.0.0.1/image",
    ],
)
def test_preview_rejects_untrusted_addresses_and_redirects(url):
    with pytest.raises(ValueError):
        preview_url(url)
    with pytest.raises(ValueError):
        preview_redirect(_original, url)


class PreviewReply(QNetworkReply):
    def __init__(self, request, response, parent):
        super().__init__(parent)
        self.aborted = False
        status, target, self.data = response
        self.setRequest(request)
        self.setUrl(request.url())
        self.setAttribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute, status)
        self.setAttribute(QNetworkRequest.Attribute.RedirectionTargetAttribute, QUrl(target))
        self.open(QIODevice.OpenModeFlag.ReadOnly)
        if not parent.hold:
            QTimer.singleShot(0, self.deliver)

    def deliver(self):
        if self.isFinished():
            return
        self.readyRead.emit()
        if not self.isFinished():
            self.setFinished(True)
            self.finished.emit()

    def abort(self):
        if not self.isFinished():
            self.aborted = True
            self.setError(QNetworkReply.NetworkError.OperationCanceledError, "Aborted")
            self.setFinished(True)
            self.finished.emit()

    def bytesAvailable(self):
        return len(self.data) + super().bytesAvailable()

    def readData(self, maxlen):
        result, self.data = self.data[:maxlen], self.data[maxlen:]
        return result


class PreviewNetwork(QNetworkAccessManager):
    def __init__(self, responses, hold=False):
        super().__init__()
        self.responses = list(responses)
        self.requests = []
        self.replies = []
        self.hold = hold

    def createRequest(self, op, request, device=None):
        self.requests.append(request)
        reply = PreviewReply(request, self.responses.pop(0), self)
        self.replies.append(reply)
        return reply


@qtapp
async def test_download_follows_civitai_redirect_without_credentials():
    network = PreviewNetwork([(301, _blob, b""), (200, "", b"jpeg bytes")])
    assert await ModelPreview(network).download(_original) == b"jpeg bytes"
    assert len(network.requests) == 2
    for request in network.requests:
        assert not request.hasRawHeader(b"Authorization")
        assert not request.hasRawHeader(b"Cookie")
        assert request.attribute(QNetworkRequest.Attribute.RedirectPolicyAttribute) == (
            QNetworkRequest.RedirectPolicy.ManualRedirectPolicy
        )


@qtapp
async def test_download_rejects_foreign_redirect_before_request():
    network = PreviewNetwork([(302, "https://evil.example/private", b"")])
    with pytest.raises(ValueError, match="redirect"):
        await ModelPreview(network).download(_original)
    assert len(network.requests) == 1


@qtapp
async def test_download_caps_redirects_and_payload():
    network = PreviewNetwork([(301, _original, b"")] * 4)
    with pytest.raises(ValueError):
        await ModelPreview(network).download(_original)
    assert len(network.requests) == 4
    network = PreviewNetwork([(200, "", b"x" * (5 * 1024 * 1024 + 1))])
    with pytest.raises(ValueError, match="size limit"):
        await ModelPreview(network).download(_original)


@qtapp
async def test_cancelled_preview_aborts_reply():
    network = PreviewNetwork([(200, "", b"jpeg bytes")], hold=True)
    task = asyncio.create_task(ModelPreview(network).download(_original))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert network.replies[0].aborted
