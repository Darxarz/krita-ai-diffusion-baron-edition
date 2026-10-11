import asyncio
import json

import pytest
from aiohttp import web
from PyQt5.QtCore import QByteArray
from PyQt5.QtNetwork import QNetworkReply, QNetworkRequest

from ai_diffusion.backend.api import ExtentInput, ImageInput, WorkflowInput, WorkflowKind
from ai_diffusion.backend.client import ClientEvent
from ai_diffusion.backend.comfy_client import ComfyClient, JobInfo
from ai_diffusion.backend.network import NetworkError, RequestManager
from ai_diffusion.backend.orchestrion_client import (
    DeviceRequests,
    OrchestrionAPI,
    OrchestrionClient,
    estimate_prompt,
    lora_compatible,
    protect_token,
    quote_text,
    service_url,
)
from ai_diffusion.backend.resources import Arch
from ai_diffusion.image import DummyImage, Extent, Image, ImageCollection
from ai_diffusion.settings import ServerMode, Settings
from ai_diffusion.settings import settings as global_settings

from .conftest import qtapp


@pytest.mark.parametrize(
    "url", ["https://orchestrion.ink", "http://192.168.50.191:3001", "http://127.0.0.1:3001"]
)
def test_service_url(url):
    assert service_url(url + "/") == url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org",
        "ftp://example.org",
        "https://secret@example.org",
        "https://example.org/krita/secret",
        "https://example.org?token=secret",
        "https://example.org/#secret",
    ],
)
def test_service_url_rejects_unsafe_address(url):
    with pytest.raises(ValueError):
        service_url(url)


def test_device_credential_is_encrypted_and_roundtrips():
    token = "ork_krita_test_credential"
    encrypted = protect_token(token)
    assert encrypted.startswith("dpapi:")
    assert token not in encrypted
    assert protect_token(encrypted, decrypt=True) == token
    with pytest.raises(ValueError):
        protect_token("dpapi:aW52YWxpZA==", decrypt=True)


def test_quote_whole_batch():
    assert (
        quote_text({"coins": 9, "fiat": {"amount": 22.5, "currency": "KZT"}}, 3)
        == "≈ 27 bleatbucks"
    )
    assert quote_text({"coins": 9}) == "≈ 9 bleatbucks"


def test_price_estimate_never_serializes_canvas_and_uses_rendering_extents():
    work = WorkflowInput(
        WorkflowKind.refine,
        images=ImageInput(
            ExtentInput(
                Extent(2048, 2048), Extent(768, 1024), Extent(768, 1024), Extent(1024, 1536)
            ),
            initial_image=DummyImage(Extent(2048, 2048)),
        ),
        batch_count=2,
    )
    inputs = estimate_prompt(work)["1"]["inputs"]
    assert inputs == {"width": 1024, "height": 1536, "batch_size": 2}
    work.crop_upscale_extent = Extent(2048, 1536)
    assert estimate_prompt(work)["1"]["inputs"]["width"] == 2048


def test_device_transport_credentials_are_separate():
    settings = Settings()
    settings.server_mode = ServerMode.orchestrion
    assert settings.access_token == ""
    client = OrchestrionClient("https://orchestrion.ink", "test-device-key")
    assert client.url == "https://orchestrion.ink/krita/connection"
    assert client._token == "test-device-key"
    assert client._requests._bearer_token == "test-device-key"
    assert client.api.requests._bearer_token == "test-device-key"
    assert "test-device-key" not in client.url
    request = client.api.requests._prepare_request(client.url)
    assert request.attribute(QNetworkRequest.Attribute.RedirectPolicyAttribute) == (
        QNetworkRequest.RedirectPolicy.SameOriginRedirectPolicy
    )


@pytest.mark.parametrize("custom_token", ["", "different-custom-key"])
def test_device_websocket_uses_browser_credential(monkeypatch, custom_token):
    monkeypatch.setattr(global_settings, "server_authorization", custom_token)
    client = OrchestrionClient("https://orchestrion.ink", "ork_krita_device-key")
    assert client._websocket_args()["additional_headers"] == {
        "Authorization": "Bearer ork_krita_device-key"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "running", "missing", "completed", "failed"])
async def test_waiting_job_reconciles_server_without_resubmission(monkeypatch, state):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    client._waiting_job.set(job)
    requests = []

    async def get(path, timeout=None):
        requests.append(path)
        if path == "queue":
            return {
                "queue_pending": [[1, job.id]] if state == "pending" else [],
                "queue_running": [[1, job.id]] if state == "running" else [],
            }
        assert path == f"history/{job.id}"
        if state == "missing":
            return {}
        return {
            job.id: {
                "status": {
                    "completed": True,
                    "status_str": "error" if state == "failed" else "success",
                },
                "outputs": {"output": {"images": [{"source": "http", "id": "result"}]}},
            }
        }

    async def transfer(msg):
        assert state == "completed"
        return [Image.create(Extent(2, 3), fill=0xFFFFFFFF)]

    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_transfer_result_images", transfer)
    await client._recover_waiting_job()
    if state == "missing":
        assert client._waiting_job.peek() is job
        await client._recover_waiting_job()
        assert client._waiting_job.peek() is job
        await client._recover_waiting_job()
    if state == "pending":
        assert client._waiting_job.peek() is job
        assert client._messages.empty()
    elif state == "running":
        assert client._waiting_job.peek() is None
        assert client._active_job is job
        assert client._messages.empty()
    else:
        assert client._waiting_job.peek() is None
        event = client._messages.get_nowait()
        assert event.job_id == job.id
        assert event.event is (ClientEvent.finished if state == "completed" else ClientEvent.error)
        if state == "completed":
            assert event.images is not None and event.images[0].extent == Extent(2, 3)
    assert requests[0] == "queue"


@pytest.mark.asyncio
async def test_waiting_job_does_not_race_inflight_submit_or_websocket(monkeypatch):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    client._waiting_job.set(job)
    requests = []

    async def get(path, timeout=None):
        requests.append(path)
        await client._start_job(job.id)
        return {"queue_running": [], "queue_pending": []}

    monkeypatch.setattr(client, "_get", get)
    await client._recover_waiting_job()
    assert not requests
    job.submitted = True
    await client._recover_waiting_job()
    assert requests == ["queue"]
    assert client._active_job is job
    assert client._messages.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["running", "completed", "failed", "empty"])
async def test_active_job_recovers_missing_completion(monkeypatch, state):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    client._active_job = job

    async def get(path, timeout=None):
        if path == "queue":
            return {
                "queue_running": [[1, job.id]] if state == "running" else [],
                "queue_pending": [],
            }
        return {
            job.id: {
                "status": {
                    "completed": True,
                    "status_str": "error" if state == "failed" else "success",
                    "messages": [["execution_error", {"exception_message": "GPU out of memory"}]],
                },
                "outputs": {}
                if state == "empty"
                else {"out": {"images": [{"source": "http", "id": "image"}]}},
            }
        }

    async def transfer(msg):
        return [Image.create(Extent(2, 3), fill=0xFFFFFFFF)]

    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_transfer_result_images", transfer)
    await client._recover_jobs()
    if state == "running":
        assert client._active_job is job
        assert client._messages.empty()
    else:
        assert client._active_job is None
        event = client._messages.get_nowait()
        assert event.event is (ClientEvent.finished if state == "completed" else ClientEvent.error)
        if state == "failed":
            assert event.error and "GPU out of memory" in event.error


@pytest.mark.asyncio
async def test_recovery_and_websocket_complete_only_once(monkeypatch):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    client._active_job = job

    async def get(path, timeout=None):
        if path == "queue":
            return {"queue_running": [], "queue_pending": []}
        return {
            job.id: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"out": {"images": [{"source": "http", "id": "result"}]}},
            }
        }

    async def transfer(msg):
        client._clear_job(job.id)
        await client._report(ClientEvent.finished, job.id, 1, images=[Image.create(Extent(2, 3))])
        return [Image.create(Extent(2, 3))]

    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_transfer_result_images", transfer)
    await client._recover_jobs()
    await client._report(ClientEvent.finished, job.id, 1, images=[Image.create(Extent(2, 3))])
    assert client._messages.qsize() == 1


@pytest.mark.asyncio
async def test_missing_image_frame_does_not_reuse_previous_result(monkeypatch):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    client._active_job = job
    previous = ImageCollection([Image.create(Extent(1, 1))])
    await client._finish_job(job.id, ImageCollection(), previous, None)
    assert client._messages.empty()
    assert client._active_job is job
    assert client._submitted_jobs[job.id] is job


@pytest.mark.asyncio
async def test_following_job_does_not_discard_result_with_lost_completion():
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    old, new = [JobInfo.create(WorkflowInput(WorkflowKind.generate)) for _ in range(2)]
    old.submitted = new.submitted = True
    client._active_job = old
    client._waiting_job.set(new)
    await client._start_job(new.id)
    assert client._active_job is new
    assert client._submitted_jobs[old.id] is old
    assert client._messages.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", ["running", "history", "absent"])
async def test_lost_prompt_acknowledgement_never_resubmits(monkeypatch, ack):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    calls = []

    async def run(self, requested_job):
        calls.append(requested_job.id)
        raise asyncio.TimeoutError()

    async def get(path, timeout=None):
        if path == "queue":
            return {"queue_running": [[1, job.id]] if ack == "running" else [], "queue_pending": []}
        return {job.id: {"outputs": {}}} if ack == "history" else {}

    monkeypatch.setattr(ComfyClient, "_run_job", run)
    monkeypatch.setattr(client, "_get", get)
    if ack == "absent":
        with pytest.raises(asyncio.TimeoutError):
            await client._run_job(job)
        assert job.id not in client._submitted_jobs
    else:
        await client._run_job(job)
        assert job.submitted
        assert client._waiting_job.peek() is job
    assert calls == [job.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("recovers", [True, False])
async def test_completed_image_download_retries_are_bounded(monkeypatch, recovers):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    client._active_job = job
    downloads = []

    async def get(path, timeout=None):
        if path == "queue":
            return {"queue_running": [], "queue_pending": []}
        return {
            job.id: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"out": {"images": [{"source": "http", "id": "result"}]}},
            }
        }

    async def transfer(msg):
        downloads.append(msg)
        if len(downloads) < 3 or not recovers:
            raise NetworkError(203, "Image expired", client.url, status=404)
        return [Image.create(Extent(2, 3))]

    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_transfer_result_images", transfer)
    for _ in range(2):
        await client._recover_jobs()
        assert client._active_job is job and client._messages.empty()
    await client._recover_jobs()
    event = client._messages.get_nowait()
    assert event.event is (ClientEvent.finished if recovers else ClientEvent.error)
    assert client._active_job is None
    assert len(downloads) == 3
    if not recovers:
        assert event.error and "Image expired" in event.error


@pytest.mark.asyncio
async def test_http_monitor_delivers_completion_when_websocket_is_silent(monkeypatch):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    client._queue_check_interval = 0.01
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    client._active_job = job
    stopped = asyncio.Event()

    async def websocket():
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    async def get(path, timeout=None):
        if path == "queue":
            return {"queue_running": [], "queue_pending": []}
        return {
            job.id: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"out": {"images": [{"source": "http", "id": "result"}]}},
            }
        }

    async def transfer(msg):
        return [Image.create(Extent(2, 3))]

    async def unsubscribe():
        pass

    monkeypatch.setattr(client, "_listen", websocket)
    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_transfer_result_images", transfer)
    monkeypatch.setattr(client, "_unsubscribe_workflows", unsubscribe)
    stream = client.listen()
    try:
        event = await asyncio.wait_for(anext(stream), 1)
        assert event.event is ClientEvent.finished and event.job_id == job.id
    finally:
        await stream.aclose()
        await client.disconnect()
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_progress_stays_below_complete_until_result():
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    await client._report(ClientEvent.progress, "job", 1.3)
    assert client._messages.get_nowait().progress == pytest.approx(0.99)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_revoked_login_does_not_leave_jobs_stuck(status):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    old, new = [JobInfo.create(WorkflowInput(WorkflowKind.generate)) for _ in range(2)]
    old.submitted = new.submitted = True
    client._active_job = old
    client._waiting_job.set(new)
    await client._fail_authentication(NetworkError(0, "Login revoked", client.url, status=status))
    assert client._active_job is None and client._waiting_job.peek() is None
    events = [client._messages.get_nowait(), client._messages.get_nowait()]
    assert {event.job_id for event in events} == {old.id, new.id}
    assert all(event.event is ClientEvent.error for event in events)


@pytest.mark.asyncio
async def test_post_deadline_is_visible_and_cancels_request(monkeypatch):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    client._request_timeout = 0.01
    cancelled = asyncio.Event()

    async def post(self, op, data):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(ComfyClient, "_post", post)
    with pytest.raises(NetworkError, match="Connection timed out"):
        await client._post("prompt", {})
    assert cancelled.is_set()


@pytest.mark.parametrize("lost", ["image_event", "completion_event", "both"])
@qtapp
async def test_real_http_recovers_dropped_websocket_messages(monkeypatch, lost):
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.submitted = True
    png = Image.create(Extent(3, 2), fill=0xFFFFFFFF).to_bytes().data()
    calls = []

    async def http(request):
        assert request.headers.get("Authorization") == "Bearer device-key"
        calls.append(request.path)
        if request.path.endswith("queue"):
            return web.json_response({"queue_running": [], "queue_pending": []})
        if "/history/" in request.path:
            return web.json_response({
                job.id: {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"out": {"images": [{"source": "http", "id": "result"}]}},
                }
            })
        return web.Response(body=png, content_type="image/png")

    async def websocket(request):
        assert request.headers.get("Authorization") == "Bearer device-key"
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        frames = [("status", {}), ("execution_start", {"prompt_id": job.id})]
        if lost == "completion_event":
            frames.append((
                "executed",
                {"prompt_id": job.id, "output": {"images": [{"source": "http", "id": "result"}]}},
            ))
        if lost == "image_event":
            frames.append(("executing", {"prompt_id": job.id, "node": None}))
        for kind, data in frames:
            await socket.send_json({"type": kind, "data": data})
        async for _ in socket:
            pass
        return socket

    async def noop():
        pass

    app = web.Application()
    app.router.add_get("/krita/connection/ws", websocket)
    app.router.add_get("/krita/connection/queue", http)
    app.router.add_get("/krita/connection/history/{id}", http)
    app.router.add_get("/krita/connection/api/etn/image/{id}", http)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    client = OrchestrionClient(f"http://127.0.0.1:{runner.addresses[0][1]}", "device-key")
    client._queue_check_interval = 0.03
    client._waiting_job.set(job)
    monkeypatch.setattr(client, "_subscribe_workflows", noop)
    monkeypatch.setattr(client, "_unsubscribe_workflows", noop)
    stream = client.listen()
    try:
        while True:
            event = await asyncio.wait_for(anext(stream), 3)
            if event.event is ClientEvent.finished:
                assert event.job_id == job.id
                assert event.images is not None and event.images[0].extent == Extent(3, 2)
                break
        assert f"/krita/connection/history/{job.id}" in calls
        assert client._active_job is None
        assert client._waiting_job.peek() is None
    finally:
        await stream.aclose()
        await client.disconnect()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_orphaned_slot_no_longer_blocks_following_requests(monkeypatch):
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    old = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    old.submitted = True
    client._waiting_job.set(old)
    following = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    client._queue.put(following)
    client._is_connected = True
    client._queue_check_interval = 0.01
    sent = asyncio.Event()

    async def get(path, timeout=None):
        return {"queue_running": [], "queue_pending": []} if path == "queue" else {}

    async def run_job(job):
        assert job is following
        sent.set()

    monkeypatch.setattr(client, "_get", get)
    monkeypatch.setattr(client, "_run_job", run_job)
    runner = asyncio.create_task(client._run())
    try:
        await asyncio.wait_for(sent.wait(), 1)
        event = client._messages.get_nowait()
        assert event.event is ClientEvent.error and event.job_id == old.id
    finally:
        runner.cancel()
        await runner


@pytest.mark.asyncio
async def test_live_listener_authenticates_with_device_key(monkeypatch):
    from ai_diffusion.backend import comfy_client

    monkeypatch.setattr(global_settings, "server_authorization", "legacy-custom-key")
    client = OrchestrionClient("https://orchestrion.ink", "ork_krita_device-key")
    connections = []

    async def connect(url, **kwargs):
        connections.append((url, kwargs))
        yield object()

    async def subscribe():
        pass

    async def listen(_websocket):
        return

    monkeypatch.setattr(comfy_client.websockets, "connect", connect)
    monkeypatch.setattr(client, "_subscribe_workflows", subscribe)
    monkeypatch.setattr(client, "_listen_websocket", listen)
    await client._listen()
    assert connections[0][0].startswith("wss://orchestrion.ink/krita/connection/ws?clientId=")
    assert connections[0][1]["additional_headers"] == {
        "Authorization": "Bearer ork_krita_device-key"
    }


@qtapp
async def test_device_transport_delivers_progress_and_image(monkeypatch):
    monkeypatch.setattr(global_settings, "server_authorization", "wrong-custom-key")
    png = Image.create(Extent(1, 1), fill=0xFFFFFFFF).to_bytes().data()
    job = JobInfo.create(WorkflowInput(WorkflowKind.generate))
    job.node_count, job.sample_count = 2, 10
    credentials = []
    uploaded = []

    async def image(request):
        credentials.append(request.headers.get("Authorization"))
        if request.method == "PUT":
            uploaded.append(await request.read())
            return web.Response(status=201)
        return web.Response(body=png, content_type="image/png")

    async def websocket(request):
        credentials.append(request.headers.get("Authorization"))
        assert request.query["clientId"] == client._id
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        for kind, data in (
            ("status", {}),
            ("execution_start", {"prompt_id": job.id}),
            ("progress", {"prompt_id": job.id, "value": 1, "max": 10}),
            (
                "executed",
                {"prompt_id": job.id, "output": {"images": [{"source": "http", "id": "result"}]}},
            ),
            ("executing", {"prompt_id": job.id, "node": None}),
        ):
            await socket.send_str(json.dumps({"type": kind, "data": data}))
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_route("*", "/krita/connection/api/etn/image/{id}", image)
    app.router.add_get("/krita/connection/ws", websocket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    address = runner.addresses[0]
    client = OrchestrionClient(f"http://127.0.0.1:{address[1]}", "ork_krita_device-key")

    async def subscribe():
        pass

    monkeypatch.setattr(client, "_subscribe_workflows", subscribe)
    task = None
    try:
        await asyncio.wait_for(client.upload_images({"input": png}), 5)
        client._waiting_job.set(job)
        task = asyncio.create_task(client._listen())
        events = []
        while not events or events[-1].event is not ClientEvent.finished:
            events.append(await asyncio.wait_for(client._messages.get(), 5))
        assert uploaded == [png]
        assert credentials == ["Bearer ork_krita_device-key"] * 3
        assert any(event.event is ClientEvent.progress and event.progress > 0 for event in events)
        finished = events[-1]
        assert finished.job_id == job.id
        assert finished.images is not None and len(finished.images) == 1
        assert finished.images[0].extent == Extent(1, 1)
        assert client._active_job is None and len(client._waiting_job) == 0
    finally:
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("as_byte_array", [False, True])
async def test_cached_image_closed_upload_requires_exact_match(monkeypatch, as_byte_array):
    client = DeviceRequests("device-key")
    url = "https://orchestrion.ink/krita/connection/api/etn/image/cached"
    payload = b"original input image"
    requests = []

    async def put(self, requested_url, data):
        raise NetworkError(
            QNetworkReply.NetworkError.RemoteHostClosedError, "Connection closed", requested_url
        )

    async def download(requested_url, timeout, max_bytes):
        requests.append((requested_url, timeout, max_bytes))
        return QByteArray(payload)

    monkeypatch.setattr(RequestManager, "put", put)
    monkeypatch.setattr(client, "download", download)
    result = await client.put(url, QByteArray(payload) if as_byte_array else payload)
    assert result == {"status": "cached"}
    assert requests == [(url, 30, len(payload))]


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [b"different image", None])
async def test_closed_upload_with_missing_or_different_cache_still_fails(monkeypatch, cached):
    client = DeviceRequests("device-key")
    url = "https://orchestrion.ink/krita/connection/api/etn/image/cached"
    original_error = NetworkError(
        QNetworkReply.NetworkError.RemoteHostClosedError, "Connection closed", url
    )

    async def put(self, requested_url, data):
        raise original_error

    async def download(requested_url, timeout, max_bytes):
        if cached is None:
            raise NetworkError(203, "Not found", requested_url, status=404)
        return QByteArray(cached)

    monkeypatch.setattr(RequestManager, "put", put)
    monkeypatch.setattr(client, "download", download)
    with pytest.raises(NetworkError) as error:
        await client.put(url, b"original input image")
    assert error.value is original_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,path",
    [
        (QNetworkReply.NetworkError.AuthenticationRequiredError, "image/input"),
        (QNetworkReply.NetworkError.RemoteHostClosedError, "upload/loras/model"),
    ],
)
async def test_other_upload_failures_are_not_accepted_as_cached(monkeypatch, code, path):
    client = DeviceRequests("device-key")
    url = f"https://orchestrion.ink/krita/connection/api/etn/{path}"

    async def put(self, requested_url, data):
        raise NetworkError(code, "Failed", requested_url)

    async def download(*args, **kwargs):
        pytest.fail("An unrelated failure must not read the image cache")

    monkeypatch.setattr(RequestManager, "put", put)
    monkeypatch.setattr(client, "download", download)
    with pytest.raises(NetworkError):
        await client.put(url, b"input")


@pytest.mark.parametrize(
    "family,arch,compatible",
    [
        ("Krea 2", Arch.krea2, True),
        ("Krea 2", Arch.qwen2, False),
        ("Qwen Image 2.1", Arch.qwen2, True),
        ("Qwen Image", Arch.qwen2, False),
        ("Qwen Image", Arch.qwen_e, True),
        ("Flux.1", Arch.krea2, False),
        ("SDXL", Arch.sdxl, True),
        ("unknown", Arch.krea2, None),
    ],
)
def test_gallery_lora_family_compatibility(family, arch, compatible):
    assert lora_compatible(family, arch) is compatible


@pytest.mark.asyncio
async def test_sign_in_pending_and_slow_down(monkeypatch):
    api = OrchestrionAPI("https://orchestrion.ink")
    polls = iter([
        {"error": "authorization_pending"},
        {"error": "slow_down"},
        {"access_token": "credential"},
    ])

    async def post(url, data):
        if url.endswith("/start"):
            return {
                "user_code": "ABCDE12345",
                "device_code": "device",
                "interval": 5,
                "expires_in": 600,
            }
        assert data == {"device_code": "device"}
        return next(polls)

    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(api.requests, "post", post)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    stream = api.sign_in()
    assert (await anext(stream))["user_code"] == "ABCDE12345"
    assert await anext(stream) == "credential"
    assert delays == [5, 5, 10]


@pytest.mark.asyncio
async def test_sign_in_denial(monkeypatch):
    api = OrchestrionAPI("https://orchestrion.ink")

    async def post(url, data):
        return (
            {"device_code": "device", "interval": 5, "expires_in": 600}
            if url.endswith("/start")
            else {"error": "access_denied"}
        )

    async def sleep(delay):
        return

    monkeypatch.setattr(api.requests, "post", post)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    stream = api.sign_in()
    await anext(stream)
    with pytest.raises(ValueError, match="denied"):
        await anext(stream)


@pytest.mark.parametrize(
    "old,new",
    [
        ("https://orchestrion.su", "https://orchestrion.ink"),
        ("https://orchestrion.su:443/", "https://orchestrion.ink"),
        ("https://custom.example", "https://custom.example"),
    ],
)
def test_official_domain_migration(old, new):
    assert service_url(old) == new


@pytest.mark.asyncio
async def test_catalog_refresh_adds_uploaded_lora_and_removes_expired_lora(monkeypatch):
    from ai_diffusion.files import File, FileCollection, FileFilter, FileFormat, FileSource
    from ai_diffusion.text import extract_loras

    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    client.models.loras = ["expired.safetensors"]
    items = [
        {"kind": "lora", "name": "action/ExcessiveAnimeNosebleed-000013.safetensors"},
        {"kind": "checkpoint", "name": "base.safetensors"},
    ]

    async def catalog():
        return items

    monkeypatch.setattr(client.api, "catalog", catalog)
    assert await client.refresh_loras()
    assert client.models.loras == [items[0]["name"]]
    assert not await client.refresh_loras()
    files = FileCollection()
    files.update(
        [File.remote(name, FileFormat.lora) for name in client.models.loras], FileSource.remote
    )
    completion = FileFilter(files)
    completion.available_only = True
    assert completion.rowCount() == 1
    assert completion.index(0, 0).data() == "action/ExcessiveAnimeNosebleed-000013"
    prompt, loras = extract_loras("portrait <lora:action/ExcessiveAnimeNosebleed-000013>", files)
    assert prompt == "portrait"
    assert loras[0].name == items[0]["name"]


def test_gallery_lora_appends_to_current_prompt_without_changing_style(monkeypatch):
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import Mock

    from ai_diffusion.model import model as _document_model
    from ai_diffusion.ui import orchestrion as ui

    assert _document_model.DocumentModel
    selected = {
        "kind": "lora",
        "name": "action/ExcessiveAnimeNosebleed-000013.safetensors",
        "title": "Nosebleed",
        "family": "Illustrious",
    }
    client = OrchestrionClient("https://orchestrion.ink", "device-key")
    client.models.loras = [selected["name"]]
    region = SimpleNamespace(positive="portrait")
    style = object()
    active = SimpleNamespace(
        style=style, arch=Arch.sdxl, active_regions=SimpleNamespace(active_or_root=region)
    )
    monkeypatch.setattr(
        ui,
        "root",
        SimpleNamespace(
            active_model=active, connection=SimpleNamespace(client_if_connected=client)
        ),
    )
    styles = Mock()
    monkeypatch.setattr(ui.Styles, "list", lambda: styles)
    gallery = SimpleNamespace(
        list=SimpleNamespace(currentItem=lambda: SimpleNamespace(data=lambda role: selected)),
        accept=Mock(),
    )
    ui.ModelGallery._apply(cast(ui.ModelGallery, gallery))
    assert region.positive == "portrait <lora:action/ExcessiveAnimeNosebleed-000013>"
    assert active.style is style
    styles.create.assert_not_called()
    ui.ModelGallery._apply(cast(ui.ModelGallery, gallery))
    assert region.positive.count("<lora:") == 1
