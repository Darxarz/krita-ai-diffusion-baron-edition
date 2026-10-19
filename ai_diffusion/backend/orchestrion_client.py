from __future__ import annotations

import asyncio
import base64
import ctypes
import ipaddress
import sys
from collections import deque
from urllib.parse import urlsplit, urlunsplit

from PyQt5.QtCore import QByteArray
from PyQt5.QtNetwork import QNetworkReply

from ..image import ImageCollection
from ..localization import translate as _
from ..orchestrion import migrate_url
from ..util import client_logger as log
from .api import WorkflowInput
from .client import ClientEvent, User
from .comfy_client import (
    ComfyClient,
    JobInfo,
    _extract_job_info_output,
    _extract_pose_json,
    _extract_text_output,
    websocket_args,
)
from .network import NetworkError, RequestManager
from .resources import Arch


def service_url(value: str):
    parts = urlsplit(value.strip().rstrip("/"))
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ValueError(_("Enter the website address: https://orchestrion.ink"))
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(_("Enter the website address without a token, password or parameters"))
    if parts.scheme == "http":
        try:
            local = ipaddress.ip_address(parts.hostname).is_private
        except ValueError:
            local = parts.hostname == "localhost"
        if not local:
            raise ValueError(_("Internet connections require HTTPS"))
    if parts.path not in ("", "/"):
        raise ValueError(_("Enter the website root address without /krita/TOKEN"))
    return migrate_url(urlunsplit((parts.scheme, parts.netloc, "", "", "")))


def protect_token(token: str, decrypt=False):
    if not token:
        return ""
    if sys.platform != "win32":
        raise ValueError(_("Persistent Orchestrion login currently requires Windows"))

    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    raw = base64.b64decode(token.removeprefix("dpapi:")) if decrypt else token.encode()
    buffer = ctypes.create_string_buffer(raw)
    source = Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    crypt = ctypes.windll.crypt32
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [
        ctypes.POINTER(Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(Blob),
    ]
    fn.restype = ctypes.c_int
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ValueError(_("Cannot open the saved login. Sign in through your browser again."))
    try:
        data = ctypes.string_at(result.data, result.size)
        return data.decode() if decrypt else "dpapi:" + base64.b64encode(data).decode()
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(result.data)


def quote_text(quote: dict, count=1):
    coins = float(quote["coins"]) * count
    return _("≈ {amount} bleatbucks", amount=f"{coins:g}")


def estimate_prompt(work: WorkflowInput):
    # Pricing uses the largest rendering/output extent. No canvas pixels or
    # user prompts are transferred just to display a preliminary price.
    extents = [work.extent.initial, work.extent.desired, work.extent.target]
    if work.crop_upscale_extent:
        extents.append(work.crop_upscale_extent)
    size = max(extents, key=lambda e: e.pixel_count)
    return {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": size.width, "height": size.height, "batch_size": work.batch_count},
        }
    }


def lora_compatible(family: str, arch: Arch) -> bool | None:
    value = family.lower().replace(" ", "").replace("-", "").replace("_", "")
    if "krea" in value:
        return arch is Arch.krea2
    if "moss" in value:
        return arch is Arch.moss
    if "qwen21" in value or "qwenimage2.1" in value:
        return arch is Arch.qwen2
    if "qwen" in value:
        return arch in (Arch.qwen, Arch.qwen_e, Arch.qwen_e_p, Arch.qwen_l)
    if any(name in value for name in ("sdxl", "illustrious", "noobai", "pony")):
        return arch.is_sdxl_like
    if "sd1.5" in value or "sd15" in value:
        return arch is Arch.sd15
    if "flux2" in value:
        return arch in (Arch.flux2_4b, Arch.flux2_9b)
    if "flux" in value:
        return arch in (Arch.flux, Arch.flux_k)
    if "zimage" in value:
        return arch is Arch.zimage
    if "anima" in value:
        return arch is Arch.anima
    return None


class OrchestrionAPI:
    def __init__(self, url: str, token=""):
        self.url = service_url(url)
        self.token = token
        self.requests = RequestManager(same_origin_only=True)
        self.requests.set_auth(token)

    async def sign_in(self):
        data = await self.requests.post(f"{self.url}/api/krita/device/start", {})
        yield data
        interval = max(5, int(data["interval"]))
        deadline = asyncio.get_running_loop().time() + min(600, int(data["expires_in"]))
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(interval)
            result = await self.requests.post(
                f"{self.url}/api/krita/device/poll", {"device_code": data["device_code"]}
            )
            error = result.get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error:
                raise ValueError(_("Connection denied or code expired. Start sign-in again."))
            yield result["access_token"]
            return
        raise ValueError(_("Sign-in timed out. Start again."))

    async def account(self):
        return await self.requests.get(f"{self.url}/api/krita/account", timeout=15)

    async def catalog(self):
        return (await self.requests.get(f"{self.url}/api/krita/models", timeout=30))["items"]

    async def quote(self, prompt: dict):
        return await self.requests.http(
            "POST",
            f"{self.url}/api/krita/quote",
            {"prompt": prompt},
            timeout=15,
        )

    async def logout(self):
        await self.requests.http("POST", f"{self.url}/api/krita/logout", {}, timeout=15)


class DeviceRequests(RequestManager):
    def __init__(self, token: str):
        super().__init__(same_origin_only=True)
        self.token = token
        self.set_auth(token)

    def put(self, url: str, data: QByteArray | bytes):
        return asyncio.create_task(self._put_image(url, data))

    async def _put_image(self, url: str, data: QByteArray | bytes):
        try:
            return await asyncio.wait_for(super().put(url, data), 120)
        except asyncio.TimeoutError:
            raise NetworkError(
                QNetworkReply.NetworkError.TimeoutError,
                "Connection timed out, the server took too long to respond",
                url,
            ) from None
        except NetworkError as error:
            if error.code != QNetworkReply.NetworkError.RemoteHostClosedError or not urlsplit(
                url
            ).path.startswith("/krita/connection/api/etn/image/"):
                raise
            # ComfyUI can return "cached" before consuming the repeated PUT body.
            # The proxy may close that upload: accept only an exact cache match.
            try:
                cached = await self.download(url, timeout=30, max_bytes=len(data))
            except Exception:
                raise error from None
            if bytes(cached) != bytes(data):
                raise
            return {"status": "cached"}

    async def upload(
        self,
        url: str,
        data: QByteArray | bytes,
        sha256: str | None = None,
        bearer: str | None = None,
    ):
        async for progress in super().upload(url, data, sha256, bearer=self.token):
            yield progress


class OrchestrionClient(ComfyClient):
    _queue_check_interval: float = 15
    _request_timeout: float = 120

    def __init__(self, url: str, token: str):
        self.api = OrchestrionAPI(url, token)
        self.account_data: dict = {}
        self._user: User | None = None
        super().__init__(self.api.url + "/krita/connection", token)
        # ComfyClient normally uses the custom server credential for HTTP only.
        # This transport uses the same dedicated device key for HTTP and WS.
        self._requests = DeviceRequests(token)
        self._submitted_jobs: dict[str, JobInfo] = {}
        self._completed_jobs: deque[str] = deque()
        self._completed_ids: set[str] = set()
        self._recovery_lock = asyncio.Lock()
        self._missing_checks: dict[str, int] = {}
        self._download_failures: dict[str, int] = {}
        self._catalog_lock = asyncio.Lock()
        self.catalog_items: list[dict] = []

    async def _run_job(self, job: JobInfo):
        self._submitted_jobs[job.id] = job
        try:
            await super()._run_job(job)
        except (NetworkError, asyncio.TimeoutError):
            try:
                queue = await self._get("queue", timeout=15)
                history = await self._get(f"history/{job.id}", timeout=15)
                acknowledged = job.id in history or any(
                    entry[1] == job.id
                    for key in ("queue_running", "queue_pending")
                    for entry in queue.get(key, [])
                )
                if acknowledged or self._active_job is job or job.id in self._completed_ids:
                    job.submitted = True
                    log.warning("Recovered acknowledgement for Orchestrion job %s", job.id)
                    if (
                        self._owns_job(job)
                        and self._waiting_job.peek() is None
                        and self._active_job is not job
                    ):
                        self._waiting_job.set(job)
                    return
            except Exception:
                log.exception("Could not verify Orchestrion submission %s", job.id)
            self._submitted_jobs.pop(job.id, None)
            raise
        except BaseException:
            self._submitted_jobs.pop(job.id, None)
            raise

    async def _start_job(self, job_id: str):
        next_job = self._waiting_job.peek()
        if next_job is not None and next_job.id == job_id and self._active_job is not None:
            # A missing completion frame must not discard the previous result.
            old = self._active_job
            self._submitted_jobs.setdefault(old.id, old)
            self._active_job = None
        return await super()._start_job(job_id)

    async def _report(self, event: ClientEvent, job_id: str, value: float = 0, **kwargs):
        if job_id and job_id in self._completed_ids:
            return
        if job_id and event in (ClientEvent.finished, ClientEvent.error, ClientEvent.interrupted):
            self._submitted_jobs.pop(job_id, None)
            self._missing_checks.pop(job_id, None)
            self._download_failures.pop(job_id, None)
            self._completed_ids.add(job_id)
            self._completed_jobs.append(job_id)
            if len(self._completed_jobs) > 512:
                self._completed_ids.remove(self._completed_jobs.popleft())
        if event is ClientEvent.progress:
            value = min(value, 0.99)
        await super()._report(event, job_id, value, **kwargs)

    async def listen(self):
        monitor = asyncio.create_task(self._monitor_jobs())
        stream = super().listen()
        try:
            async for message in stream:
                yield message
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            await stream.aclose()

    async def _monitor_jobs(self):
        while True:
            await asyncio.sleep(self._queue_check_interval)
            try:
                await self._recover_jobs()
            except NetworkError as e:
                if e.status in (401, 403):
                    await self._fail_authentication(e)
                else:
                    log.exception("Could not reconcile Orchestrion jobs")
            except Exception:
                log.exception("Could not reconcile Orchestrion jobs")

    async def _fail_authentication(self, error: NetworkError):
        jobs = dict(self._submitted_jobs)
        for job in (self._active_job, self._waiting_job.peek()):
            if job is not None:
                jobs[job.id] = job
        for job in jobs.values():
            if self._owns_job(job):
                self._clear_recovered_job(job)
                await self._report(ClientEvent.error, job.id, error=error.message)

    async def _post(self, op: str, data: dict):
        timeout = 15 if op.startswith("api/etn/workflow/") else self._request_timeout
        try:
            return await asyncio.wait_for(super()._post(op, data), timeout)
        except asyncio.TimeoutError:
            raise NetworkError(
                QNetworkReply.NetworkError.TimeoutError,
                "Connection timed out, the server took too long to respond",
                f"{self.url}/{op}",
            ) from None

    async def _get(self, op: str, timeout: float | None = 60):
        return await asyncio.wait_for(super()._get(op, timeout), timeout)

    async def _transfer_result_image(self, id: str):
        return await asyncio.wait_for(super()._transfer_result_image(id), 180)

    async def _finish_job(self, job_id, images, last_images, result):
        if not len(images):
            # Recover this job's output, never substitute an unrelated previous image.
            if self._active_job is not None and self._active_job.id == job_id:
                self._submitted_jobs.setdefault(job_id, self._active_job)
                log.warning(
                    "Orchestrion completed %s without images; waiting for HTTP recovery", job_id
                )
            return last_images
        return await super()._finish_job(job_id, images, last_images, result)

    def _websocket_args(self):
        return websocket_args(self._token)

    async def _wait_for_slot(self):
        while self._waiting_job.peek() is not None:
            try:
                await asyncio.wait_for(self._waiting_job.wait(), self._queue_check_interval)
            except asyncio.TimeoutError:
                try:
                    await self._recover_waiting_job()
                except NetworkError as e:
                    if e.status in (401, 403):
                        await self._fail_authentication(e)
                    else:
                        log.exception("Could not check Orchestrion waiting job")
                except Exception:
                    log.exception("Could not check Orchestrion waiting job")

    async def _recover_waiting_job(self):
        await self._recover_jobs(waiting_only=True)

    def _owns_job(self, job: JobInfo):
        return job.id not in self._completed_ids and (
            self._submitted_jobs.get(job.id) is job
            or self._active_job is job
            or self._waiting_job.peek() is job
        )

    async def _recover_jobs(self, waiting_only=False):
        async with self._recovery_lock:
            jobs = {} if waiting_only else dict(self._submitted_jobs)
            if not waiting_only and self._active_job is not None:
                jobs[self._active_job.id] = self._active_job
            if job := self._waiting_job.peek():
                jobs[job.id] = job
            jobs = {id: job for id, job in jobs.items() if job.submitted and self._owns_job(job)}
            if not jobs:
                return
            queue = await self._get("queue", timeout=15)
            if not isinstance(queue, dict) or not all(
                isinstance(queue.get(key), list) for key in ("queue_running", "queue_pending")
            ):
                raise ValueError("Invalid queue response")
            running = {entry[1] for entry in queue["queue_running"]}
            pending = {entry[1] for entry in queue["queue_pending"]}
            for job in jobs.values():
                if not self._owns_job(job):
                    continue
                if waiting_only and self._waiting_job.peek() is not job:
                    continue
                if job.id in running or job.id in pending:
                    self._missing_checks.pop(job.id, None)
                    if job.id in running and self._waiting_job.peek() is job:
                        await self._start_job(job.id)
                    continue
                try:
                    await self._recover_job(job)
                except NetworkError as e:
                    if e.status in (401, 403):
                        raise
                    log.exception("Could not recover Orchestrion job %s", job.id)
                except Exception:
                    log.exception("Could not recover Orchestrion job %s", job.id)

    async def _recover_job(self, job: JobInfo):
        history = await self._get(f"history/{job.id}", timeout=15)
        if not isinstance(history, dict):
            raise TypeError("Invalid history response")
        record = history.get(job.id, {})
        status = record.get("status", {})
        if record and not status.get("completed") and status.get("status_str") != "error":
            return
        if not self._owns_job(job):
            return
        if not record:
            count = self._missing_checks.get(job.id, 0) + 1
            self._missing_checks[job.id] = count
            if count < 3:
                return
        images = ImageCollection()
        outputs = record.get("outputs", {}) if status.get("status_str") != "error" else {}
        result = None
        try:
            for node, output in outputs.items():
                message = {"data": {"output": output, "node": node}}
                if output.get("images"):
                    images.append(await self._transfer_result_images(message))
                result = _extract_pose_json(message) or result
        except Exception as e:
            if not self._owns_job(job):
                return
            failures = self._download_failures.get(job.id, 0) + 1
            self._download_failures[job.id] = failures
            log.warning("Orchestrion result download failed for %s (%s): %s", job.id, failures, e)
            if failures < 3:
                return
            self._clear_recovered_job(job)
            await self._report(
                ClientEvent.error,
                job.id,
                error=_("Could not download the completed result: ") + (str(e) or type(e).__name__),
            )
            return
        if not self._owns_job(job):
            return
        self._clear_recovered_job(job)
        if len(images):
            for node, output in outputs.items():
                message = {"data": {"output": output, "node": node}}
                for extract in (_extract_text_output, _extract_job_info_output):
                    if event := extract(job.id, message):
                        await self._messages.put(event)
            log.info("Recovered Orchestrion result %s from HTTP history", job.id)
            await self._report(ClientEvent.finished, job.id, 1, images=images, result=result)
        else:
            error = _(
                "Connection was interrupted. The server no longer has this queued task or its images. Please try again."
            )
            if record:
                error = _("The server completed this task but returned no downloadable images.")
            if status.get("status_str") == "error":
                for kind, data in status.get("messages", []):
                    if kind == "execution_error":
                        error = data.get("exception_message") or error
                    elif kind == "execution_interrupted":
                        await self._report(ClientEvent.interrupted, job.id)
                        return
            await self._report(ClientEvent.error, job.id, error=error)

    def _clear_recovered_job(self, job: JobInfo):
        if self._waiting_job.peek() is job:
            self._waiting_job.get()
        self._clear_job(job.id)

    @property
    def user(self):
        return self._user

    async def refresh_account(self):
        self.account_data = await self.api.account()
        if self._user is None:
            self._user = User(self.account_data["id"], self.account_data["name"])
        self._user.credits = round(self.account_data["coins"])
        return self.account_data

    async def connect(self):
        await self.refresh_account()
        await super().connect()

    async def refresh_loras(self):
        async with self._catalog_lock:
            items = await self.api.catalog()
            names = list(
                dict.fromkeys(
                    item["name"]
                    for item in items
                    if item.get("kind") == "lora" and isinstance(item.get("name"), str)
                )
            )
            changed = names != self.models.loras
            self.models.loras = names
            self.catalog_items = items
            return changed

    async def discover_models(self, refresh: bool):
        async for status in super().discover_models(refresh):
            yield status
        try:
            await self.refresh_loras()
        except Exception as error:
            log.warning(f"Could not refresh Orchestrion LoRAs: {error}")

    async def refresh(self):
        await super().refresh()
        await self.refresh_account()
