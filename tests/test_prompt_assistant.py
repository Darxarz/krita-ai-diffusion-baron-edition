import base64
from types import SimpleNamespace

import pytest

from ai_diffusion.backend.orchestrion_client import OrchestrionAPI
from ai_diffusion.backend.prompt_assistant import PromptSuggestion, editor_context, prepare_images
from ai_diffusion.backend.resources import Arch
from ai_diffusion.image import Extent, Image

from .conftest import qtapp


def reply(**changes):
    return {
        "prompt": "Match the character's fur to image 2.",
        "summary": "Сохранена поза.",
        "tips": [],
        "images_used": 2,
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [{"prompt": ""}, {"tips": [None]}, {"images_used": True}, {"images_used": 5}, {"summary": 4}],
)
def test_rejects_unusable_answers(changes):
    with pytest.raises(ValueError, match="invalid answer"):
        PromptSuggestion.from_dict(reply(**changes))


def test_order_and_image_budget_preserve_reference_identity():
    encoded = prepare_images([
        ("canvas", "target", Image.create(Extent(1600, 800))),
        ("reference", "fur reference", Image.create(Extent(400, 700))),
    ])
    assert [(item["role"], item["label"]) for item in encoded] == [
        ("canvas", "target"),
        ("reference", "fur reference"),
    ]
    first = Image.from_bytes(base64.b64decode(encoded[0]["data"].split(",")[1]))
    assert first.extent == Extent(1024, 512)
    with pytest.raises(ValueError, match="at most four"):
        prepare_images([("reference", "x", first)] * 5)


@pytest.mark.parametrize(
    "arch,editing,strength,selection,regional,mode",
    [
        (Arch.flux3, True, 1.0, False, False, "edit"),
        (Arch.qwen2, True, 0.7, True, False, "edit"),
        (Arch.krea2, False, 1.0, False, False, "generate"),
        (Arch.sdxl, False, 0.4, False, False, "refine"),
        (Arch.moss, False, 1.0, False, True, "inpaint"),
        (Arch.anima, False, 1.0, True, False, "inpaint"),
    ],
)
def test_context_matches_actual_checkpoint_architecture_and_editor_operation(
    arch, editing, strength, selection, regional, mode
):
    regions = SimpleNamespace()
    regions.active_or_root = object() if regional else regions
    model = SimpleNamespace(
        arch=arch,
        is_editing=editing,
        strength=strength,
        document=SimpleNamespace(selection_bounds=object() if selection else None),
        active_regions=regions,
        active_style=SimpleNamespace(prompt_mode="a1111", krea2_reference_mode="general_v2"),
    )
    context = editor_context(model, "opaque-checkpoint.safetensors")  # type: ignore[arg-type]
    assert context["architecture"] == arch.name
    assert context["model"] == "opaque-checkpoint.safetensors"
    assert context["mode"] == mode
    assert context["has_selection"] == selection
    assert context["scope"] == ("region" if regional else "global")
    assert context["prompt_mode"] == "a1111"
    assert context["reference_purpose"] == "general_v2"


@qtapp
async def test_assistant_uses_device_transport_and_does_not_submit_generation():
    class Requests:
        def __init__(self):
            self.calls = []

        async def http(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return reply()

    api = OrchestrionAPI("https://orchestrion.ink", "device-test")
    requests = Requests()
    api.requests = requests  # type: ignore[assignment]
    suggestion = await api.prompt_assist({"instruction": "recolor fur", "mode": "edit"})
    assert suggestion.images_used == 2
    assert len(requests.calls) == 1
    assert requests.calls[0][0][:2] == ("POST", "https://orchestrion.ink/api/krita/prompt-assist")
    assert requests.calls[0][1]["timeout"] == 100
