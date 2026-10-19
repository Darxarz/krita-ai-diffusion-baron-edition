from __future__ import annotations

import asyncio
from collections.abc import Iterable
from enum import Enum

from PyQt5.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices

from .. import eventloop, util
from ..backend.client import (
    Client,
    ClientEvent,
    ClientMessage,
    DeviceInfo,
    MissingResources,
    SharedWorkflow,
)
from ..backend.cloud_client import CloudClient
from ..backend.comfy_client import ComfyClient
from ..backend.network import NetworkError
from ..backend.orchestrion_client import OrchestrionAPI, OrchestrionClient, protect_token
from ..backend.orchestrion_inbox import InboxAPI
from ..image import Image
from ..localization import Localization
from ..localization import translate as _
from ..settings import PerformancePreset, ServerMode, Settings, settings
from .inbox import Inbox
from .properties import ObservableProperties, Property


def _open_inbox_document(image: Image, name: str, info):
    from ..document import open_inbox_document  # requires Krita

    return open_inbox_document(image, name, info)


class ConnectionState(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3

    discover_models = 10

    auth_missing = 20
    auth_requesting = 21
    auth_pending = 22
    auth_error = 23


class Connection(QObject, ObservableProperties):
    state = Property(ConnectionState.disconnected)
    error = Property("")
    progress = Property((1, 1))

    state_changed = pyqtSignal(ConnectionState)
    error_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(tuple)
    models_changed = pyqtSignal()
    message_received = pyqtSignal(ClientMessage)
    workflow_published = pyqtSignal(str)
    sign_in_code_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._client: Client | None = None
        self._task: asyncio.Task | None = None
        self._workflows: dict[str, dict] = {}
        self._temporary_disconnect = False
        self._auth_task: asyncio.Task | None = None
        self._catalog_task: asyncio.Task | None = None
        self._catalog_timer = QTimer(self)
        self._catalog_timer.setInterval(60000)
        self._catalog_timer.timeout.connect(self._refresh_catalog)
        self._catalog_timer.start()
        self.inbox = Inbox(_open_inbox_document)
        self.sign_in_code = ""
        self.error_kind = ""
        self.missing_resources: MissingResources | None = None

        settings.changed.connect(self._handle_settings_changed)
        self._update_state()

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    async def _sign_in(self, url: str):
        client = CloudClient(url)
        self.state = ConnectionState.auth_requesting
        try:
            sign_in = client.sign_in()
            url = await anext(sign_in)
            self.state = ConnectionState.auth_pending
            QDesktopServices.openUrl(QUrl(url))

            settings.access_token = await anext(sign_in)
            self.state = ConnectionState.disconnected
            await self._connect(client)
            settings.save()

        except Exception as e:
            self.error = util.log_error(e)
            self.state = ConnectionState.auth_error

    def sign_in(self):
        eventloop.run(self._sign_in(CloudClient.default_api_url))

    def sign_in_orchestrion(self):
        self.cancel_sign_in()
        self._auth_task = eventloop.run(self._sign_in_orchestrion())

    def cancel_sign_in(self):
        if self._auth_task:
            self._auth_task.cancel()
            self._auth_task = None
        self.sign_in_code = ""
        self.sign_in_code_changed.emit("")
        if self.state in (ConnectionState.auth_pending, ConnectionState.auth_requesting):
            self.state = ConnectionState.disconnected
            self._update_state()

    async def _sign_in_orchestrion(self):
        self.state = ConnectionState.auth_requesting
        self.error = ""
        try:
            api = OrchestrionAPI(settings.orchestrion_url)
            stream = api.sign_in()
            data = await anext(stream)
            self.sign_in_code = data["user_code"]
            self.sign_in_code_changed.emit(self.sign_in_code)
            self.state = ConnectionState.auth_pending
            QDesktopServices.openUrl(
                QUrl(api.url + "/connect/krita?lang=" + Localization.current.id)
            )
            token = await anext(stream)
            settings.orchestrion_token = protect_token(token)
            settings.save()
            self.state = ConnectionState.disconnected
            await self._connect(OrchestrionClient(api.url, token))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.error = str(e)
            self.state = ConnectionState.auth_error
        finally:
            self.sign_in_code = ""
            self.sign_in_code_changed.emit("")

    async def _connect(self, client: Client):
        if self.state in (ConnectionState.connecting, ConnectionState.discover_models):
            return
        if self._client is not None or self._task is not None:
            await self.disconnect()
        self.error = ""
        self.error_kind = ""
        self.state = ConnectionState.connecting
        try:
            await client.connect()

            self.state = ConnectionState.discover_models
            async for status in client.discover_models(refresh=False):
                self.progress = (status.current, status.total)
            self.missing_resources = client.missing_resources

            apply_performance_preset(settings, client.device_info)

            self._client = client
            self._task = eventloop._loop.create_task(self._handle_messages(client))
            if isinstance(client, OrchestrionClient):
                self.inbox.start(InboxAPI(client.api.url, client.api.token))
            self.state = ConnectionState.connected
            self.models_changed.emit()
        except NetworkError as e:
            self.error = e.message
            self.error_kind = "network"
            self.state = ConnectionState.error
            if e.status == 401:  # Unauthorized
                if isinstance(client, OrchestrionClient):
                    settings.orchestrion_token = ""
                else:
                    settings.access_token = ""
                settings.save()
                self._update_state()
        except MissingResources as e:
            self.error = _(
                "Connection established, but the server is missing required custom nodes or models."
            )
            self.error_kind = "missing_resources"
            self.missing_resources = e
            self.state = ConnectionState.error
        except Exception as e:
            self.error = util.log_error(e)
            self.error_kind = "unknown"
            self.state = ConnectionState.error

    def create_client(self, settings: Settings) -> Client | None:
        match settings.server_mode:
            case ServerMode.cloud:
                if settings.access_token == "":
                    self.state = ConnectionState.auth_missing
                    return None
                return CloudClient(CloudClient.default_api_url, settings.access_token)
            case ServerMode.external | ServerMode.managed:
                return ComfyClient(settings.server_url, settings.access_token)
            case ServerMode.orchestrion:
                if not settings.orchestrion_token:
                    self.state = ConnectionState.auth_missing
                    return None
                try:
                    token = protect_token(settings.orchestrion_token, decrypt=True)
                    return OrchestrionClient(settings.orchestrion_url, token)
                except ValueError as e:
                    self.error = str(e)
                    self.state = ConnectionState.auth_error
                    return None
            case _:
                raise ValueError(f"Invalid server mode: {settings.server_mode}")

    def connect(self, client: Client):
        eventloop.run(self._connect(client))

    async def disconnect(self):  # type: ignore (hides QObject.disconnect)
        self.inbox.stop()
        if self._catalog_task is not None:
            self._catalog_task.cancel()
            await asyncio.gather(self._catalog_task, return_exceptions=True)
            self._catalog_task = None
        if self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self._client = None
        self.error = ""
        self.missing_resources = None
        self.state = ConnectionState.disconnected
        self._update_state()

    def interrupt(self):
        eventloop.run(self.client.interrupt())

    def cancel(self, job_ids: Iterable[str]):
        eventloop.run(self.client.cancel(job_ids))

    def _refresh_catalog(self):
        client = self.client_if_connected
        if not isinstance(client, OrchestrionClient) or (
            self._catalog_task and not self._catalog_task.done()
        ):
            return

        async def refresh():
            try:
                changed = await client.refresh_loras()
                if changed and self.client_if_connected is client:
                    self.models_changed.emit()
            except Exception as error:
                util.client_logger.warning(f"Could not refresh Orchestrion LoRAs: {error}")

        self._catalog_task = eventloop.run(refresh())

    def refresh(self):
        async def _refresh():
            await self.client.refresh()
            self.models_changed.emit()

        if self.state is ConnectionState.connected:
            eventloop.run(_refresh())

    @property
    def client(self):
        assert self.state is ConnectionState.connected and self._client is not None
        return self._client

    @property
    def client_if_connected(self):
        return self._client

    @property
    def user(self):
        if client := self.client_if_connected:
            return client.user

    @property
    def workflows(self):
        return self._workflows

    async def _handle_messages(self, client: Client):
        try:
            self._temporary_disconnect = False

            async with client:
                async for msg in client.listen():
                    try:
                        self._handle_message(msg)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        util.client_logger.exception(e)
                        self.error = _("Error handling server message: ") + str(e)
        except asyncio.CancelledError:
            pass  # shutdown
        except Exception as e:
            util.client_logger.exception("Server message listener stopped")
            self.error = _("Error communicating with server: ") + str(e)
            self.state = ConnectionState.error

    def _handle_message(self, msg: ClientMessage):
        match msg:
            case (ClientEvent.error, "", *_):
                self.error = _("Error communicating with server: ") + str(msg.error)
            case (ClientEvent.disconnected, *_):
                self._temporary_disconnect = True
                self.error = _("Disconnected from server, trying to reconnect...")
            case (ClientEvent.connected, *_):
                if self._temporary_disconnect:
                    self._temporary_disconnect = False
                    self.error = ""
            case (ClientEvent.published, *_):
                assert isinstance(msg.result, SharedWorkflow)
                self._workflows[msg.result.publisher] = msg.result.workflow
                self.workflow_published.emit(msg.result.publisher)
            case _:
                self.message_received.emit(msg)

    def _update_state(self):
        if self.state in [ConnectionState.disconnected, ConnectionState.error] and (
            (settings.server_mode is ServerMode.cloud and not settings.access_token)
            or (settings.server_mode is ServerMode.orchestrion and not settings.orchestrion_token)
        ):
            self.state = ConnectionState.auth_missing
        elif self.state in [
            ConnectionState.auth_missing,
            ConnectionState.auth_error,
        ] and settings.server_mode not in (ServerMode.cloud, ServerMode.orchestrion):
            self.state = ConnectionState.disconnected

    def _handle_settings_changed(self, key: str, value: object):
        if key == "server_mode":
            self.cancel_sign_in()
            client_is_cloud = isinstance(self._client, CloudClient)
            mode_is_cloud = settings.server_mode is ServerMode.cloud
            client_is_orchestrion = isinstance(self._client, OrchestrionClient)
            mode_is_orchestrion = settings.server_mode is ServerMode.orchestrion
            if client_is_cloud != mode_is_cloud or client_is_orchestrion != mode_is_orchestrion:
                self.error = ""
                eventloop.run(self.disconnect())
            self._update_state()

        elif key in ("access_token", "orchestrion_token"):
            self._update_state()


def apply_performance_preset(settings: Settings, device: DeviceInfo):
    if settings.performance_preset is PerformancePreset.auto:
        if device.type == "cpu":
            settings.apply_performance_preset(PerformancePreset.cpu)
        elif device.type.lower() == "cloud":
            settings.apply_performance_preset(PerformancePreset.cloud)
        elif device.vram <= 6:
            settings.apply_performance_preset(PerformancePreset.low)
        elif device.vram <= 12:
            settings.apply_performance_preset(PerformancePreset.medium)
        else:
            settings.apply_performance_preset(PerformancePreset.high)
