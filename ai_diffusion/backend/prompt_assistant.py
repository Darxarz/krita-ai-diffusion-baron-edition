from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..image import Extent, Image
from ..localization import translate as _
from ..settings import ImageFileFormat

if TYPE_CHECKING:
    from ..model.model import DocumentModel


def editor_context(model: DocumentModel, checkpoint: str):
    selection = model.document.selection_bounds is not None
    regional = model.active_regions.active_or_root is not model.active_regions
    mode = "edit" if model.is_editing else "refine" if model.strength < 1 else "generate"
    if not model.is_editing and (selection or regional):
        mode = "inpaint"
    return {
        "model": checkpoint,
        "architecture": model.arch.name,
        "mode": mode,
        "has_selection": selection,
        "scope": "region" if regional else "global",
        "strength": model.strength,
        "prompt_mode": model.active_style.prompt_mode,
        "reference_purpose": model.active_style.krea2_reference_mode,
    }


@dataclass
class PromptSuggestion:
    prompt: str
    summary: str
    tips: list[str]
    images_used: int

    @staticmethod
    def from_dict(data: dict):
        prompt, summary, tips = data.get("prompt"), data.get("summary"), data.get("tips")
        count = data.get("images_used")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > 32000
            or not isinstance(summary, str)
            or len(summary) > 4000
            or not isinstance(tips, list)
            or len(tips) > 4
            or any(not isinstance(t, str) or len(t) > 1500 for t in tips)
            or type(count) is not int
            or not 0 <= count <= 4
        ):
            raise ValueError(_("Kadance returned an invalid answer. Please try again."))
        return PromptSuggestion(prompt, summary, tips, count)


def prepare_images(images: list[tuple[str, str, Image]]):
    if len(images) > 4:
        raise ValueError(_("Kadance can review at most four images, including the canvas."))
    result = []
    for role, label, image in images:
        if image.extent.longest_side > 1024:
            image = Image.scale_to_fit(image, Extent(1024, 1024))
        encoded = image.to_base64(ImageFileFormat.jpeg)
        if len(encoded) > 768 * 1024 * 4 // 3:
            raise ValueError(_("An assistant image is too large. Try a smaller reference."))
        result.append({
            "role": role,
            "label": label[:200],
            "data": "data:image/jpeg;base64," + encoded,
        })
    return result
