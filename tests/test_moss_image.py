import json
import math
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter

from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    LoraInput,
    RegionInput,
    SamplingInput,
    UpscaleInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.comfy_client import (
    ComfyClient,
    _find_text_encoder_models,
    _find_vae_models,
)
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.orchestrion_client import lora_compatible
from ai_diffusion.backend.resolution import TileLayout
from ai_diffusion.backend.resources import Arch, ResourceKind, UpscalerName, resource_id
from ai_diffusion.backend.workflow import (
    ImageReshape,
    LayoutBox,
    create,
    moss_layout_items,
    prepare,
)
from ai_diffusion.files import FileCollection, FileFormat, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import SamplerPresets, Style

model_name = "moss\\moss_image_sft_int8_convrot.safetensors"
encoder_name = "qwen3.5_9b_w4a8_gptq.safetensors"
vae_name = "flux2-vae.safetensors"
qwen21_enhancer = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
lora_name = "moss\\characters\\Baron_Satyr_MOSS.safetensors"


@pytest.mark.parametrize("string", ["moss-image", "moss_image", "moss"])
def test_moss_from_string(string):
    assert Arch.from_string(string, filename=model_name) is Arch.moss


def test_qwen21_etn_alias():
    assert Arch.from_string("qwen-image-2.1") is Arch.qwen2


def loader_nodes(clip_types=("flux2", "qwen_image", "moss_image")):
    return ComfyObjectInfo({
        "UNETLoader": {"input": {"required": {"unet_name": [[model_name]]}}},
        "CLIPLoader": {
            "input": {"required": {"clip_name": [[encoder_name]], "type": [list(clip_types)]}}
        },
        "VAELoader": {"input": {"required": {"vae_name": [[vae_name]]}}},
        "LoraLoader": {"input": {"required": {"lora_name": [[lora_name]]}}},
    })


def moss_client(clip_types=("flux2", "qwen_image", "moss_image")):
    client = ComfyClient("http://localhost:8188")
    metadata = {model_name: {"base_model": "moss-image"}}
    client._refresh_models(loader_nodes(clip_types), {}, metadata)
    client.models.resources.update(_find_text_encoder_models([encoder_name, qwen21_enhancer]))
    client.models.resources.update(_find_vae_models([vae_name]))
    for upscaler in UpscalerName:
        client.models.resources[resource_id(ResourceKind.upscaler, Arch.all, upscaler)] = (
            upscaler.value
        )
    return client


def test_moss_discovery():
    client = moss_client()
    info = client.models.checkpoints[model_name]
    assert info.arch is Arch.moss
    assert info.format is FileFormat.diffusion
    missing = client._check_workload(Arch.moss)
    assert not [m for m in missing if m.arch is Arch.moss or m.kind is ResourceKind.text_encoder]


def test_moss_requires_new_comfyui():
    client = moss_client(clip_types=("flux2", "qwen_image"))
    assert model_name not in client.models.checkpoints


@pytest.mark.parametrize(
    "files,expected",
    [
        ([encoder_name, qwen21_enhancer], encoder_name),
        (
            [qwen21_enhancer, "qwen3.5_9b_int8_convrot.safetensors"],
            "qwen3.5_9b_int8_convrot.safetensors",
        ),
        ([qwen21_enhancer], None),
    ],
)
def test_moss_text_encoder_search(files, expected):
    found = _find_text_encoder_models(files)
    assert found[resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3_5_9b")] == expected


def test_moss_encoder_is_not_claimed_by_other_archs():
    found = _find_text_encoder_models([encoder_name])
    for te in ["qwen_3_4b", "qwen_3_8b", "qwen_3_06b", "qwen_3vl_4b", "qwen_3vl_8b"]:
        assert found[resource_id(ResourceKind.text_encoder, Arch.all, te)] is None


def graph_nodes(graph: dict):
    return [node for node in graph.values()]


def assert_no_model_sampling(nodes: list[dict]):
    assert not any(n["class_type"].startswith("ModelSampling") for n in nodes)


def test_moss_generate_workflow():
    client = moss_client()
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.moss),
        images=ImageInput.from_extent(Extent(1024, 1024)),
        conditioning=ConditioningInput('a coffee shop sign that says "MOSS"', "blurry"),
        sampling=SamplingInput("euler", "sgm_uniform", 3.0, 30),
    )
    nodes = graph_nodes(create(work, client.models).root)
    by_type = {n["class_type"]: n["inputs"] for n in nodes}
    assert by_type["UNETLoader"]["unet_name"] == model_name
    assert by_type["CLIPLoader"] == {"clip_name": encoder_name, "type": "moss_image"}
    assert by_type["VAELoader"]["vae_name"] == vae_name
    assert by_type["SkipLayerGuidanceDiTSimple"]["single_layers"] == "22"
    assert by_type["SkipLayerGuidanceDiTSimple"]["double_layers"] == ""
    assert "EmptyFlux2LatentImage" in by_type
    assert by_type["KSamplerSelect"]["sampler_name"] == "euler"
    assert by_type["BasicScheduler"]["scheduler"] == "sgm_uniform"
    assert by_type["BasicScheduler"]["steps"] == 30
    assert by_type["CFGGuider"]["cfg"] == pytest.approx(3.0)
    texts = [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]
    assert 'a coffee shop sign that says "MOSS"' in texts and "blurry" in texts
    assert_no_model_sampling(nodes)
    assert "EasyCache" not in by_type


def test_moss_lora_is_model_only():
    client = moss_client()
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.moss, loras=[LoraInput(lora_name, 0.8)]),
        images=ImageInput.from_extent(Extent(1024, 1024)),
        conditioning=ConditioningInput("baronsatyr playing a flute"),
        sampling=SamplingInput("euler", "sgm_uniform", 3.0, 30),
    )
    nodes = graph_nodes(create(work, client.models).root)
    loras = [n for n in nodes if n["class_type"] == "LoraLoaderModelOnly"]
    assert [n["inputs"]["lora_name"] for n in loras] == [lora_name]
    assert loras[0]["inputs"]["strength_model"] == pytest.approx(0.8)
    assert not any(n["class_type"] == "LoraLoader" for n in nodes)


def moss_style():
    style = Style.load(Path(__file__).parents[1] / "ai_diffusion/styles/moss-image.json")
    assert style is not None
    return style


def test_moss_style_and_preset():
    client = moss_client()
    style = moss_style()
    assert style.preferred_checkpoint(client.models.checkpoints) == model_name
    preset = SamplerPresets.instance()["MOSS - Euler uniform"]
    assert (preset.sampler, preset.scheduler, preset.steps, preset.cfg) == (
        "euler",
        "sgm_uniform",
        30,
        3.0,
    )


@pytest.mark.parametrize(
    "kind,strength", [(WorkflowKind.inpaint, 1.0), (WorkflowKind.refine_region, 0.6)]
)
def test_moss_inpaint_uses_differential_diffusion(kind, strength):
    client = moss_client()
    style = moss_style()
    image = Image.create(Extent(1024, 1024))
    mask = Mask.rectangle(Bounds(384, 384, 256, 256), Bounds(0, 0, 1024, 1024))
    work = prepare(
        kind,
        image,
        ConditioningInput("a small red bird"),
        style,
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        mask=mask,
        strength=strength,
        inpaint=InpaintParams(InpaintMode.fill, mask.bounds),
    )
    nodes = graph_nodes(create(work, client.models).root)
    types = {n["class_type"] for n in nodes}
    assert "DifferentialDiffusion" in types
    assert "SetLatentNoiseMask" in types
    assert "SkipLayerGuidanceDiTSimple" in types
    assert not any(t.startswith("INPAINT_LoadFooocus") or "ControlNet" in t for t in types)
    assert_no_model_sampling(nodes)


@pytest.mark.parametrize("canvas", [Extent(1000, 1300), Extent(300, 300), Extent(4096, 4096)])
def test_moss_resolution(canvas):
    client = moss_client()
    work = prepare(
        WorkflowKind.generate,
        canvas,
        ConditioningInput("a lighthouse"),
        moss_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
    )
    assert work.images is not None
    initial = work.images.extent.initial
    assert initial.width % 16 == 0 and initial.height % 16 == 0
    assert min(initial.width, initial.height) >= 512
    assert initial.pixel_count <= 2048**2


def test_moss_arch_capabilities():
    assert Arch.moss in Arch.list()
    assert Arch.moss.text_encoders == ["qwen_3_5_9b"]
    assert Arch.moss.supports_cfg
    assert not Arch.moss.supports_edit
    assert Arch.moss.supports_regions
    assert not Arch.moss.supports_clip_skip
    assert not Arch.moss.has_controlnet_inpaint
    assert not Arch.moss.is_flux2
    assert not Arch.is_compatible(Arch.moss, Arch.ernie)


def test_moss_gallery_lora_family():
    assert lora_compatible("MOSS-Image", Arch.moss) is True
    assert lora_compatible("MOSS-Image", Arch.flux2_9b) is False


def rect_mask(extent: Extent, *rects: Bounds):
    qimg = QImage(extent.width, extent.height, QImage.Format.Format_Grayscale8)
    qimg.fill(0)
    painter = QPainter(qimg)
    for r in rects:
        painter.fillRect(r.x, r.y, r.width, r.height, Qt.GlobalColor.white)
    painter.end()
    return Image(qimg)


def centred(x0: float, y0: float, x1: float, y1: float, extent: Extent):
    w, h = extent
    return [
        math.floor(y0) - h // 2,
        math.floor(x0) - w // 2,
        math.ceil(y1) - h // 2,
        math.ceil(x1) - w // 2,
    ]


def test_mask_bounds():
    extent = Extent(300, 200)
    assert rect_mask(extent).mask_bounds() is None
    mask = rect_mask(extent, Bounds(10, 20, 30, 40), Bounds(100, 5, 1, 1))
    assert mask.mask_bounds() == Bounds(10, 5, 91, 55)
    rgba = Image.create(extent, fill=Qt.GlobalColor.black)
    assert rgba.mask_bounds() is None


def test_layout_box_crop_and_scale():
    mask = rect_mask(Extent(1000, 800), Bounds(150, 100, 300, 250))
    box = LayoutBox.from_mask(mask)
    assert box is not None and box == LayoutBox(Extent(1000, 800), 150, 100, 450, 350)
    assert box.to_yxyx() == [100 - 400, 150 - 500, 350 - 400, 450 - 500]

    scaled = LayoutBox.from_mask(mask, Extent(640, 512))
    assert scaled is not None and scaled.extent == Extent(640, 512)
    assert scaled.to_yxyx() == centred(96, 64, 288, 224, Extent(640, 512))

    # Crop to the inpaint target bounds first, then scale the crop
    crop = Bounds(300, 200, 400, 400)
    reshape = ImageReshape(Extent(1024, 1024), crop=(Extent(1000, 800), crop))
    cropped = LayoutBox.from_mask(mask, reshape)
    assert cropped is not None and cropped.extent == Extent(1024, 1024)
    assert cropped.to_yxyx() == centred(0, 0, 150 * 2.56, 150 * 2.56, Extent(1024, 1024))
    # Mask outside of the crop
    outside = ImageReshape(Extent(1024, 1024), crop=(Extent(1000, 800), Bounds(600, 500, 64, 64)))
    assert LayoutBox.from_mask(mask, outside) is None
    # A crop is only applied to masks of the trigger extent
    other = ImageReshape(Extent(500, 400), crop=(Extent(2000, 1600), crop))
    assert LayoutBox.from_mask(mask, other) == LayoutBox(Extent(500, 400), 75, 50, 225, 175)


def test_layout_box_rounds_outwards():
    box = LayoutBox(Extent(1001, 999), 10.2, 20.7, 30.2, 40.0000000001)
    assert box.to_yxyx() == [20 - 499, 10 - 500, 40 - 499, 31 - 500]


def test_layout_items_labels_and_quoted_text():
    bbox = [-10, -20, 30, 40]
    assert moss_layout_items("a red apple", bbox) == [
        {"label": "object", "caption": "a red apple", "bbox": bbox}
    ]
    assert moss_layout_items("Character: a knight", bbox) == [
        {"label": "character", "caption": "a knight", "bbox": bbox}
    ]
    sign = 'a neon sign "Open 24/7" and «BAR»'
    assert moss_layout_items(sign, bbox) == [
        {"label": "object", "caption": sign, "bbox": bbox},
        {"label": "text", "caption": sign, "text": "Open 24/7", "bbox": bbox},
        {"label": "text", "caption": sign, "text": "BAR", "bbox": bbox},
    ]
    assert moss_layout_items("text: EXIT", bbox) == [
        {"label": "text", "caption": "EXIT", "text": "EXIT", "bbox": bbox}
    ]


global_prompt = "a fantasy street at dusk"
character_prompt = "character: a red-haired girl in a green cloak"
sign_prompt = 'a wooden tavern sign that says "TAVERN"'
character_rect = Bounds(40, 100, 420, 800)
sign_rect = Bounds(560, 50, 400, 300)


def two_regions(extent: Extent, merged=False):
    full = Bounds(0, 0, *extent)
    suffix = f", {global_prompt}" if merged else ""  # Krita appends the root prompt
    background = rect_mask(extent, full)
    return [
        RegionInput(background, full, global_prompt),
        RegionInput(rect_mask(extent, character_rect), full, character_prompt + suffix),
        RegionInput(rect_mask(extent, sign_rect), full, sign_prompt + suffix),
    ]


def layout_prompts(work: WorkflowInput):
    client = moss_client()
    nodes = graph_nodes(create(work, client.models).root)
    types = {n["class_type"] for n in nodes}
    assert not types & {"ETN_AttentionMask", "ETN_DefineRegion", "ETN_BackgroundRegion"}
    assert not types & {"ConditioningSetMask", "ETN_ListRegionMasks"}
    texts = [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]
    return [json.loads(t) for t in texts if t.startswith("{")], texts


def expected_items(char_box: list[int], sign_box: list[int]):
    caption = character_prompt.removeprefix("character: ")
    return [
        {"label": "character", "caption": caption, "bbox": char_box},
        {"label": "object", "caption": sign_prompt, "bbox": sign_box},
        {"label": "text", "caption": sign_prompt, "text": "TAVERN", "bbox": sign_box},
    ]


@pytest.mark.parametrize("merged", [False, True], ids=["tablet", "krita"])
def test_moss_regions_layout_prompt(merged):
    extent = Extent(1024, 1024)
    client = moss_client()
    work = prepare(
        WorkflowKind.generate,
        extent,
        ConditioningInput(global_prompt, "blurry", regions=two_regions(extent, merged)),
        moss_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
    )
    layouts, texts = layout_prompts(work)
    assert len(layouts) == 1
    layout = layouts[0]
    assert list(layout) == ["summary", "items", "bbox_coord_space", "bbox_format"]
    assert layout["summary"] == {"caption": global_prompt}
    assert layout["bbox_coord_space"] == "abs_centralized"
    assert layout["bbox_format"] == "yxyx"
    assert layout["items"] == expected_items(
        [100 - 512, 40 - 512, 900 - 512, 460 - 512], [50 - 512, 560 - 512, 350 - 512, 960 - 512]
    )
    assert "blurry" in texts  # the negative prompt stays plain text
    # Serialized like ComfyUI's MOSSImageLayoutPrompt
    assert json.dumps(layout, ensure_ascii=False, separators=(",", ":")) in texts


def test_moss_regions_layout_inpaint_crop():
    canvas = Extent(3072, 3072)
    full = Bounds(0, 0, *canvas)
    target = Bounds(1024, 1024, 1024, 1024)
    region_rect = Bounds(1200, 900, 400, 900)
    regions = [
        RegionInput(rect_mask(canvas, full), full, global_prompt),
        RegionInput(rect_mask(canvas, region_rect), full, "a cat"),
    ]
    client = moss_client()
    work = prepare(
        WorkflowKind.inpaint,
        Image.create(canvas),
        ConditioningInput(global_prompt, regions=regions),
        moss_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        mask=Mask.rectangle(Bounds(256, 256, 512, 512), target),
        inpaint=InpaintParams(InpaintMode.fill, target),
    )
    assert work.images is not None and work.inpaint is not None
    initial = work.images.extent.initial
    assert initial != canvas
    layouts, _ = layout_prompts(work)
    boxes = [layout["items"][0]["bbox"] for layout in layouts]
    assert len(boxes) == 2

    # Initial pass: the whole canvas, scaled to the initial resolution
    sx, sy = initial.width / canvas.width, initial.height / canvas.height
    assert boxes[0] == centred(1200 * sx, 900 * sy, 1600 * sx, 1800 * sy, initial)

    # Refinement pass: cropped to the target bounds, scaled to the upscale resolution
    tb, up = work.inpaint.target_bounds, work.crop_upscale_extent
    assert up is not None
    ux, uy = up.width / tb.width, up.height / tb.height
    x0, y0 = max(1200 - tb.x, 0), max(900 - tb.y, 0)
    x1, y1 = min(1600 - tb.x, tb.width), min(1800 - tb.y, tb.height)
    assert boxes[1] == centred(x0 * ux, y0 * uy, x1 * ux, y1 * uy, up)


def test_moss_regions_layout_tiles():
    canvas = Extent(1024, 1024)
    client = moss_client()
    work = prepare(
        WorkflowKind.upscale_tiled,
        Image.create(canvas),
        ConditioningInput(global_prompt, regions=two_regions(canvas)),
        moss_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        strength=0.4,
        upscale_factor=2.0,
        upscale=UpscaleInput("", tile_overlap=64),
    )
    assert work.images is not None
    extent = work.images.extent
    layout = TileLayout(extent.initial, extent.desired.width, 64, 16)
    assert layout.total_tiles > 1
    layouts, texts = layout_prompts(work)

    def tile_boxes(rect: Bounds):
        # Region boxes are relative to each tile, regions outside a tile are left out
        rect = Bounds.scale(rect, 2)
        result = []
        for i in range(layout.total_tiles):
            tile = layout.bounds(i)
            x0, y0 = max(rect.x - tile.x, 0), max(rect.y - tile.y, 0)
            x1 = min(rect.x + rect.width - tile.x, tile.width)
            y1 = min(rect.y + rect.height - tile.y, tile.height)
            result.append(centred(x0, y0, x1, y1, tile.extent) if x1 > x0 and y1 > y0 else None)
        return result

    expected = [
        [b for b in (char, sign) if b is not None]
        for char, sign in zip(tile_boxes(character_rect), tile_boxes(sign_rect))
    ]
    assert [] in expected and len([e for e in expected if e]) == len(layouts)
    assert global_prompt in texts  # tile without regions uses the plain prompt
    actual = [[i["bbox"] for i in layout["items"] if i["label"] != "text"] for layout in layouts]
    assert actual == [e for e in expected if e]


def test_moss_regional_lora_is_rejected():
    extent = Extent(1024, 1024)
    regions = two_regions(extent)
    regions[1].loras = [LoraInput(lora_name, 1.0)]
    client = moss_client()
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.moss),
        images=ImageInput.from_extent(extent),
        conditioning=ConditioningInput(global_prompt, regions=regions),
        sampling=SamplingInput("euler", "sgm_uniform", 3.0, 30),
    )
    with pytest.raises(RuntimeError, match="LoRA"):
        create(work, client.models)
