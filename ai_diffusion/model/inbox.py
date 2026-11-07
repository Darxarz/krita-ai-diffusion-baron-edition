from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import re
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from uuid import uuid4

from PyQt5.QtCore import QByteArray, QObject, pyqtSignal

from ..backend.client import resolve_arch
from ..backend.orchestrion_inbox import InboxAPI, InboxChanges, InboxError, InboxItem
from ..image import Image
from ..localization import translate as _
from ..settings import settings
from ..style import SamplerPresets, Styles
from ..text import pattern_lora
from ..util import client_logger as log

# Called with the downloaded picture, a document name and the annotation which links the
# document to the inbox item. Implemented with the Krita API in document.py.
DocumentOpener = Callable[[Image, str, QByteArray], object]

_full_refresh_interval = 30 * 60
_auth_errors = ("auth",)


def device_id():
    if not settings.orchestrion_device_id:
        settings.orchestrion_device_id = "krita-" + uuid4().hex
        settings.save()
    return settings.orchestrion_device_id


def device_name():
    return (platform.node() or "Krita")[:120]


def return_key(item_id: str, png: bytes):
    # Same picture for the same item -> same key, so a retry after a lost response never
    # creates a duplicate version. A changed picture gets a new key and a new version.
    digest = hashlib.sha256(item_id.encode("utf-8") + b"\0" + png).hexdigest()
    return digest[:48]


def document_info(item: InboxItem, origin: str):
    info = {
        "id": item.id,
        "title": item.title,
        "source": item.source,
        "returnTarget": item.return_target,
        "origin": origin,
    }
    if item.meta:
        info["meta"] = item.meta  # generation settings, applied once to the new document
    return QByteArray(json.dumps(info, ensure_ascii=False).encode("utf-8"))


def _lora_key(name: str):
    name = name.replace("\\", "/").strip()
    return re.sub(r"\.(safetensors|ckpt|pt)$", "", name, flags=re.IGNORECASE).casefold()


def append_weighted_lora(prompt: str, name: str, strength: float = 1.0):
    key = _lora_key(name)
    if not key or any(char in key for char in "<>:\r\n"):
        return prompt
    if any(_lora_key(match[1]) == key for match in pattern_lora.finditer(prompt)):
        return prompt
    tag = re.sub(
        r"\.(safetensors|ckpt|pt)$", "", name.replace("\\", "/").strip(), flags=re.IGNORECASE
    )
    weight = "" if abs(strength - 1.0) < 1e-6 else f":{strength:g}"
    return prompt + (" " if prompt and not prompt[-1].isspace() else "") + f"<lora:{tag}{weight}>"


def _sampler_preset(sampler: str, scheduler: str):
    for name, preset in SamplerPresets.instance().items():
        if preset.sampler == sampler and preset.scheduler == scheduler and not preset.lora:
            return name
    return None


def apply_inbox_settings(model, client) -> bool:
    """Puts the prompt, negative, LoRAs, seed and model/sampler of a picture received from the
    website into its new document, once. Returns True when something was applied."""
    find = getattr(model.document, "find_annotation", None)
    info = parse_document_info(find("orchestrion_inbox") if find else None)
    meta = info.get("meta") if info else None
    if not info or info.get("applied") or not isinstance(meta, dict):
        return False
    models = getattr(client, "models", None)
    available = {_lora_key(name): name for name in (models.loras if models else [])}
    prompt = str(meta.get("prompt") or "")
    for lora in meta.get("loras") or []:
        if not isinstance(lora, dict) or not isinstance(lora.get("name"), str):
            continue
        if models is not None and _lora_key(lora["name"]) not in available:
            log.info(f"Inbox: LoRA {lora['name']} is not installed, skipped")
            continue
        try:
            strength = float(lora.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        name = available.get(_lora_key(lora["name"]), lora["name"])
        prompt = append_weighted_lora(prompt, name, strength)
    model.regions.positive = prompt
    if isinstance(meta.get("negative"), str):
        model.regions.negative = meta["negative"]
    if isinstance(meta.get("seed"), int) and meta["seed"] >= 0:
        model.seed = meta["seed"]
    checkpoint = meta.get("checkpoint") or meta.get("model")
    if models is not None and isinstance(checkpoint, str) and checkpoint in models.checkpoints:
        arch = models.checkpoints[checkpoint].arch
        if client.supports_arch(arch):
            styles = Styles.list()
            current = model.style
            source = (
                current
                if resolve_arch(current, client) is arch
                else next((s for s in styles if resolve_arch(s, client) is arch), current)
            )
            style = styles.create("orchestrion.json", copy_from=source)
            style.checkpoints = [checkpoint]
            style.architecture = arch
            style.name = Path(checkpoint.replace("\\", "/")).stem
            if isinstance(meta.get("steps"), int):
                style.sampler_steps = meta["steps"]
            if isinstance(meta.get("cfg"), (int, float)):
                style.cfg_scale = float(meta["cfg"])
            if preset := _sampler_preset(
                str(meta.get("sampler", "")), str(meta.get("scheduler", ""))
            ):
                style.sampler = preset
            style.save()
            styles.changed.emit()
            model.style = style
    info["applied"] = True
    model.document.annotate(
        "orchestrion_inbox", QByteArray(json.dumps(info, ensure_ascii=False).encode("utf-8"))
    )
    return True


def parse_document_info(data: QByteArray | bytes | None) -> dict | None:
    if data is None:
        return None
    try:
        info = json.loads(bytes(data).decode("utf-8"))
        return info if isinstance(info, dict) and isinstance(info.get("id"), str) else None
    except (ValueError, UnicodeDecodeError):
        return None


class Inbox(QObject):
    """Pictures sent from the Orchestrion website to this Krita ("Send to Krita")."""

    changed = pyqtSignal()
    opened = pyqtSignal(str)  # item id
    failed = pyqtSignal(str)  # error message

    def __init__(self, opener: DocumentOpener | None = None):
        super().__init__()
        self.items: dict[str, InboxItem] = {}
        self.enabled = False
        self.error = ""
        self._api: InboxAPI | None = None
        self._task: asyncio.Task | None = None
        self._opener = opener
        self._handled: set[str] = set()
        self._opening: set[str] = set()

    @property
    def api(self):
        return self._api

    @property
    def available(self):
        return sorted(
            (i for i in self.items.values() if i.status != "expired"),
            key=lambda i: i.created_at,
            reverse=True,
        )

    @property
    def unopened_count(self):
        return sum(1 for i in self.items.values() if i.status == "new")

    def start(self, api: InboxAPI, opener: DocumentOpener | None = None):
        self.stop()
        self._api = api
        if opener is not None:
            self._opener = opener
        self._task = asyncio.get_running_loop().create_task(self._run(api))

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._api = None
        self.items = {}
        self._handled = set()
        self.enabled = False
        self.error = ""
        self.changed.emit()

    async def _run(self, api: InboxAPI):
        try:
            if not await api.supported():
                log.info("Orchestrion inbox: not supported by this website")
                return
        except Exception as e:
            log.warning(f"Orchestrion inbox: could not query website features: {e}")
            return
        self.enabled = True
        self.changed.emit()
        cursor = ""
        last_full = 0.0
        delay = 0
        while self._api is api:
            try:
                if monotonic() - last_full > _full_refresh_interval:
                    cursor = ""
                changes = await api.changes(cursor, wait=25)
                if changes.full:
                    last_full = monotonic()
                self._apply(changes)
                self.error = ""
                delay = 0
                if self._api is not api:
                    return
                await self._open_new(api)
                if not changes.cursor:
                    await asyncio.sleep(25)
                cursor = changes.cursor
            except asyncio.CancelledError:
                raise
            except InboxError as e:
                log.warning(f"Orchestrion inbox: {e.message}")
                if e.code in _auth_errors:
                    self.error = e.message
                    self.changed.emit()
                    return
                cursor = ""  # re-read everything after any problem
                delay = min(120, delay * 2 or 5)
                await asyncio.sleep(delay)
            except Exception as e:
                log.warning(f"Orchestrion inbox: {e}")
                cursor = ""
                delay = min(120, delay * 2 or 5)
                await asyncio.sleep(delay)

    def _apply(self, changes: InboxChanges):
        if changes.full:
            self.items = {}
        for item in changes.items:
            if item.status == "expired":
                self.items.pop(item.id, None)
            else:
                self.items[item.id] = item
        for removed in changes.removed:
            self.items.pop(removed, None)
        self.changed.emit()

    async def _open_new(self, api: InboxAPI):
        if not settings.orchestrion_inbox_auto_open:
            return
        pending = [
            i for i in self.items.values() if i.status == "new" and i.id not in self._handled
        ]
        for item in sorted(pending, key=lambda i: i.created_at):
            self._handled.add(item.id)
            try:
                await self._open(api, item, force=False)
            except InboxError as e:
                if e.code != "already_opened":
                    self.failed.emit(e.message)

    async def open(self, item_id: str, force=False):
        api = self._api
        item = self.items.get(item_id)
        if api is None or item is None:
            return
        self._handled.add(item_id)
        await self._open(api, item, force)

    async def _open(self, api: InboxAPI, item: InboxItem, force: bool):
        if item.id in self._opening:
            return
        self._opening.add(item.id)
        try:
            claimed = await api.ack(item.id, device_id(), device_name(), force=force)
            if self._api is api and item.id in self.items:
                self.items[item.id] = claimed
                self.changed.emit()
            data = await api.image(claimed)
            image = Image.from_bytes(data)
            if image.width == 0 or image.height == 0:
                raise InboxError("invalid", _("The picture from the website could not be read"))
            if self._api is not api:
                return  # disconnected while downloading
            assert self._opener is not None, "No document opener"
            self._opener(image, claimed.title or claimed.filename, document_info(claimed, api.url))
            self.opened.emit(claimed.id)
        finally:
            self._opening.discard(item.id)

    async def return_picture(self, info: dict, png: bytes, width: int, height: int):
        """Sends a picture back to the website item the document was opened from."""
        api = self._api
        if api is None:
            raise InboxError("offline", _("Not connected to the website"))
        if info.get("origin") != api.url:
            raise InboxError("origin", _("This document was opened from a different website"))
        key = return_key(info["id"], png)
        result = await api.return_image(info["id"], png, width, height, key)
        if item := self.items.get(info["id"]):
            item.status = "returned"
            item.version = int(result.get("version", item.version))
            self.changed.emit()
        return result

    async def delete(self, item_id: str):
        if api := self._api:
            await api.delete(item_id)
            self.items.pop(item_id, None)
            self.changed.emit()
