from __future__ import annotations

import asyncio

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from .. import eventloop
from ..backend.orchestrion_client import OrchestrionAPI, OrchestrionClient, protect_token
from ..backend.prompt_assistant import editor_context, prepare_images
from ..image import Bounds
from ..localization import Localization
from ..localization import translate as _
from ..model.model import DocumentModel
from ..model.root import root
from ..settings import settings


class KadanceDialog(QDialog):
    def __init__(self, model: DocumentModel, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Kadance · Prompt assistant"))
        self.setObjectName("kadanceDialog")
        self.resize(680, 680)
        self._model = model
        self._regions = model.active_regions
        self._target = self._regions.active_or_root
        self._task: asyncio.Task | None = None
        self._original = self._target.positive
        self._applied = ""
        self._undo = ""
        self._suggested_for: str | None = None
        self._context = self._editor_context()
        layout = QVBoxLayout(self)
        context_label = QLabel(self)
        context_label.setTextFormat(Qt.TextFormat.PlainText)
        context_label.setText(
            f"{self._context['model']} · {self._context['architecture']} · {self._context['mode']}"
        )
        context_label.setWordWrap(True)
        layout.addWidget(context_label)
        description = QLabel(
            _(
                "Describe what to change and what to preserve. Kadance proposes a prompt; you decide when to insert it."
            ),
            self,
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.instruction = QPlainTextEdit(self)
        self.instruction.setObjectName("kadanceInstruction")
        self.instruction.setPlaceholderText(
            _(
                "For example: keep image 1's pose and scene, but match the character's fur colors and markings to image 2."
            )
        )
        self.instruction.setMaximumHeight(120)
        layout.addWidget(self.instruction)
        presets = QHBoxLayout()
        for title, instruction in [
            (
                _("Improve prompt"),
                _("Improve my current prompt for the selected model, preserving my intent."),
            ),
            (
                _("Match reference"),
                _(
                    "Correct the existing character in image 1 to match the appearance, fur colors and markings in image 2. Keep image 1's pose, composition, background and lighting."
                ),
            ),
        ]:
            button = QPushButton(title, self)
            button.clicked.connect(
                lambda _checked=False, text=instruction: self.instruction.setPlainText(text)
            )
            presets.addWidget(button)
        layout.addLayout(presets)
        self.images = QCheckBox(_("Include the edit canvas and configured reference layers"), self)
        self.images.setChecked(True)
        layout.addWidget(self.images)
        note = QLabel(
            _(
                "Images are reduced and sent to Orchestrion's AI provider only when you press Ask Kadance. In Edit mode the canvas is image 1; reference layers follow in order."
            ),
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.ask = QPushButton(_("Ask Kadance"), self)
        self.ask.setObjectName("kadanceAsk")
        self.ask.clicked.connect(self.review)
        layout.addWidget(self.ask)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.proposal = QPlainTextEdit(self)
        self.proposal.setObjectName("kadanceSuggestion")
        self.proposal.setPlaceholderText(
            _("The proposed prompt will appear here. You can edit it before inserting.")
        )
        layout.addWidget(self.proposal, 1)
        actions = QHBoxLayout()
        self.apply_button = QPushButton(_("Use this prompt"), self)
        self.apply_button.setObjectName("kadanceApply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply)
        actions.addWidget(self.apply_button)
        self.undo_button = QPushButton(_("Undo insertion"), self)
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo)
        actions.addWidget(self.undo_button)
        close = QPushButton(_("Close"), self)
        close.clicked.connect(self.reject)
        actions.addWidget(close)
        layout.addLayout(actions)
        self.finished.connect(self.cancel)

    def _current(self):
        return (
            root.active_model is self._model
            and self._model.active_regions is self._regions
            and self._regions.active_or_root is self._target
            and self._editor_context() == self._context
        )

    def _editor_context(self):
        client = root.connection.client_if_connected
        available = client.models.checkpoints if client else []
        return editor_context(self._model, self._model.active_style.preferred_checkpoint(available))

    def _image_context(self):
        images = []
        if not self.images.isChecked():
            return images
        if self._context["mode"] != "generate":
            bounds = Bounds.from_extent(self._model.document.extent)
            images.append(("canvas", _("Canvas"), self._model._get_current_image(bounds)))
        controls = list(self._regions.control)
        for region in self._regions:
            controls.extend(region.control)
        for control in controls:
            if control.mode.is_ip_adapter and control.is_supported:
                layer = control.layer
                images.append((
                    "reference",
                    f"{layer.name} ({control.mode.name})",
                    layer.get_pixels(),
                ))
        return images

    def review(self):
        if self._task and not self._task.done():
            return
        self._task = eventloop.run(self._review())

    async def _review(self):
        self.ask.setEnabled(False)
        self.apply_button.setEnabled(False)
        self._suggested_for = None
        self.status.setText(_("Kadance is reviewing your request…"))
        try:
            if not self._current():
                raise ValueError(
                    _("The document, mode or active region changed. Open Kadance again.")
                )
            client = root.connection.client_if_connected
            if isinstance(client, OrchestrionClient):
                api = client.api
            elif settings.orchestrion_token:
                api = OrchestrionAPI(
                    settings.orchestrion_url,
                    protect_token(settings.orchestrion_token, decrypt=True),
                )
            else:
                raise ValueError(
                    _("Sign in to Orchestrion in the connection settings to use Kadance.")
                )
            self._original = self._target.positive
            images = self._image_context()
            encoded = await asyncio.to_thread(prepare_images, images)
            if self._context["model"] == "not-found":
                raise ValueError(_("Select an available model before asking Kadance"))
            suggestion = await api.prompt_assist({
                **self._context,
                "instruction": self.instruction.toPlainText(),
                "prompt": self._original,
                "language": Localization.current.id,
                "images": encoded,
            })
            if not self._current():
                raise ValueError(
                    _("The document, mode or active region changed. Open Kadance again.")
                )
            self.proposal.setPlainText(suggestion.prompt)
            self.status.setText("\n".join([suggestion.summary, *suggestion.tips]))
            self._suggested_for = self._original
            self.apply_button.setEnabled(True)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self.status.setText(str(error))
        finally:
            self.ask.setEnabled(True)

    def apply(self):
        if not self._current() or self._suggested_for != self._target.positive:
            self.status.setText(
                _("The prompt changed while Kadance was working. Ask again before inserting.")
            )
            self.apply_button.setEnabled(False)
            return
        value = self.proposal.toPlainText().strip()
        if not value:
            return
        self._undo, self._applied = self._target.positive, value
        self._target.positive = value
        self.apply_button.setEnabled(False)
        self.undo_button.setEnabled(True)

    def undo(self):
        if self._current() and self._target.positive == self._applied:
            self._target.positive = self._undo
        self.undo_button.setEnabled(False)

    def cancel(self):
        if self._task and not self._task.done():
            self._task.cancel()
