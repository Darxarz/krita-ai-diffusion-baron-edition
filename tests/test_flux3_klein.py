import json
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter

from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ControlInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    LoraInput,
    RegionInput,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.comfy_client import (
    ComfyClient,
    _find_text_encoder_models,
    _find_upscalers,
    _find_vae_models,
)
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.orchestrion_client import lora_compatible
from ai_diffusion.backend.resources import (
    Arch,
    ControlMode,
    ResourceKind,
    UpscalerName,
    resource_id,
)
from ai_diffusion.backend.workflow import LayoutBox, build_instructions, create, prepare
from ai_diffusion.files import FileCollection, FileFormat, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import SamplerPresets, Style

model_name = "flux3\\flux3-image-klein-int8-convrot.safetensors"
encoder_name = "qwen3vl_4b_fp8_scaled.safetensors"
vae_name = "flux2-vae.safetensors"
lora_name = "flux3\\Baron_Satyr_FLUX3.safetensors"


@pytest.mark.parametrize("string", ["flux3-klein", "flux3_klein", "flux3"])
def test_flux3_from_string(string):
    assert Arch.from_string(string, filename=model_name) is Arch.flux3


def loader_nodes(clip_types=("flux2", "krea2", "flux3")):
    return ComfyObjectInfo({
        "UNETLoader": {"input": {"required": {"unet_name": [[model_name]]}}},
        "CLIPLoader": {
            "input": {"required": {"clip_name": [[encoder_name]], "type": [list(clip_types)]}}
        },
        "VAELoader": {"input": {"required": {"vae_name": [[vae_name]]}}},
        "LoraLoader": {"input": {"required": {"lora_name": [[lora_name]]}}},
    })


def flux3_client(clip_types=("flux2", "krea2", "flux3")):
    client = ComfyClient("http://localhost:8188")
    # As reported by comfyui-tooling-nodes on the Orchestrion server
    metadata = {model_name: {"base_model": "flux3-klein", "is_inpaint": False}}
    client._refresh_models(loader_nodes(clip_types), {}, metadata)
    client.models.resources.update(_find_text_encoder_models([encoder_name]))
    client.models.resources.update(_find_vae_models([vae_name]))
    client.models.resources.update(_find_upscalers([u.value for u in UpscalerName]))
    return client


def test_flux3_discovery():
    client = flux3_client()
    info = client.models.checkpoints[model_name]
    assert info.arch is Arch.flux3
    assert info.format is FileFormat.diffusion
    missing = client._check_workload(Arch.flux3)
    assert not [m for m in missing if m.arch is Arch.flux3 or m.kind is ResourceKind.text_encoder]
    te = _find_text_encoder_models([encoder_name])
    assert te[resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3vl_4b")] == encoder_name


def test_flux3_requires_new_comfyui():
    client = flux3_client(clip_types=("flux2", "krea2"))
    assert model_name not in client.models.checkpoints


def graph_nodes(graph: dict):
    return [node for node in graph.values()]


def assert_no_model_sampling(nodes: list[dict]):
    assert not any(n["class_type"].startswith("ModelSampling") for n in nodes)


def node_by_id(graph: dict, link: list):
    return graph[link[0]]


def generate_graph(cfg: float):
    client = flux3_client()
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.flux3),
        images=ImageInput.from_extent(Extent(1024, 1024)),
        conditioning=ConditioningInput("a red fox in fresh snow at sunrise", "blurry"),
        sampling=SamplingInput("euler", "simple", cfg, 16),
    )
    return create(work, client.models).root


def test_flux3_generate_workflow():
    graph = generate_graph(1.0)
    nodes = graph_nodes(graph)
    by_type = {n["class_type"]: n["inputs"] for n in nodes}
    assert by_type["UNETLoader"]["unet_name"] == model_name
    assert by_type["CLIPLoader"] == {"clip_name": encoder_name, "type": "flux3"}
    assert by_type["VAELoader"]["vae_name"] == vae_name
    assert by_type["EmptyFlux2LatentImage"]["width"] == 1024
    assert by_type["KSamplerSelect"]["sampler_name"] == "euler"
    assert by_type["BasicScheduler"]["scheduler"] == "simple"
    assert by_type["BasicScheduler"]["steps"] == 16
    assert "CFGGuider" not in by_type  # CFG 1: no unconditional pass
    assert (
        node_by_id(graph, by_type["BasicGuider"]["conditioning"])["class_type"] == "CLIPTextEncode"
    )
    texts = [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]
    assert texts == ["a red fox in fresh snow at sunrise"]
    assert_no_model_sampling(nodes)
    assert "ReferenceMultiLatent" not in by_type
    assert "SkipLayerGuidanceDiTSimple" not in by_type


def test_flux3_negative_is_zeroed_positive():
    graph = generate_graph(2.0)
    nodes = graph_nodes(graph)
    by_type = {n["class_type"]: n["inputs"] for n in nodes}
    assert by_type["CFGGuider"]["cfg"] == pytest.approx(2.0)
    # The negative text is not encoded, the unconditional is the zeroed positive
    texts = [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]
    assert texts == ["a red fox in fresh snow at sunrise"]
    negative = node_by_id(graph, by_type["CFGGuider"]["negative"])
    assert negative["class_type"] == "ConditioningZeroOut"
    zeroed = node_by_id(graph, negative["inputs"]["conditioning"])
    assert zeroed["class_type"] == "CLIPTextEncode"


def test_flux3_lora_is_model_only():
    client = flux3_client()
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.flux3, loras=[LoraInput(lora_name, 0.8)]),
        images=ImageInput.from_extent(Extent(1024, 1024)),
        conditioning=ConditioningInput("baronsatyr playing a flute"),
        sampling=SamplingInput("euler", "simple", 1.0, 16),
    )
    nodes = graph_nodes(create(work, client.models).root)
    loras = [n for n in nodes if n["class_type"] == "LoraLoaderModelOnly"]
    assert [n["inputs"]["lora_name"] for n in loras] == [lora_name]
    assert loras[0]["inputs"]["strength_model"] == pytest.approx(0.8)
    assert not any(n["class_type"] == "LoraLoader" for n in nodes)


def flux3_style():
    style = Style.load(Path(__file__).parents[1] / "ai_diffusion/styles/flux3-klein.json")
    assert style is not None
    return style


def test_flux3_style_and_preset():
    client = flux3_client()
    style = flux3_style()
    assert style.preferred_checkpoint(client.models.checkpoints) == model_name
    preset = SamplerPresets.instance()["FLUX 3 Klein - Euler"]
    assert (preset.sampler, preset.scheduler, preset.steps, preset.cfg) == (
        "euler",
        "simple",
        16,
        1.0,
    )
    assert preset.minimum_steps == 8
    assert (style.sampler_steps, style.cfg_scale) == (16, 1.0)


def edit_work(
    control: list[ControlInput],
    edit_reference=True,
    prompt="make it night",
    canvas: Extent | None = None,
    strength=1.0,
):
    client = flux3_client()
    cond = ConditioningInput(prompt, control=control, edit_reference=edit_reference)
    cond.positive = build_instructions(cond, Arch.flux3, None)  # done by prepare_prompts
    work = prepare(
        WorkflowKind.refine,
        Image.create(canvas or Extent(1024, 768)),
        cond,
        flux3_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        strength=strength,
    )
    return work, client


def test_flux3_edit_uses_reference_multi_latent():
    style_image = Image.create(Extent(512, 512))
    work, client = edit_work([ControlInput(ControlMode.style, style_image)])
    assert work.conditioning is not None
    assert work.conditioning.positive.startswith("Apply the style from <ref_image_1>.")
    graph = create(work, client.models).root
    nodes = graph_nodes(graph)
    canvas_ref, extra_ref = [
        n["inputs"] for n in nodes if n["class_type"] == "ReferenceMultiLatent"
    ]
    # The canvas (<ref_image_0>) keeps the sampled size, so output and reference match
    assert canvas_ref["resolution"] == "keep resolution"
    assert "resolution.megapixels" not in canvas_ref
    assert [k for k in canvas_ref if k.startswith("images.")] == ["images.image0"]
    sampled = node_by_id(graph, by_type(nodes, "VAEEncode")["pixels"])
    assert node_by_id(graph, canvas_ref["images.image0"]) == sampled
    # Other references (<ref_image_1>...) are resized to about 1 MP, appended after the canvas
    assert extra_ref["resolution"] == "resize"
    assert extra_ref["resolution.megapixels"] == pytest.approx(1.0)
    assert [k for k in extra_ref if k.startswith("images.")] == ["images.image0"]
    assert node_by_id(graph, extra_ref["positive"])["inputs"] == canvas_ref
    assert node_by_id(graph, canvas_ref["vae"])["class_type"] == "VAELoader"
    # References go to the positive and the zeroed negative
    assert node_by_id(graph, canvas_ref["negative"])["class_type"] == "ConditioningZeroOut"
    guider = by_type(nodes, "BasicGuider")
    assert node_by_id(graph, guider["conditioning"])["inputs"] == extra_ref
    assert not any(n["class_type"] == "ReferenceLatent" for n in nodes)
    assert len([n for n in nodes if n["class_type"] == "SamplerCustomAdvanced"]) == 1


def by_type(nodes: list[dict], class_type: str):
    matches = [n["inputs"] for n in nodes if n["class_type"] == class_type]
    assert len(matches) == 1, f"{class_type}: {len(matches)} nodes"
    return matches[0]


def samplers(graph: dict):
    """Inputs of each SamplerCustomAdvanced with its guider conditioning and sigmas."""
    result = []
    for node in graph.values():
        if node["class_type"] == "SamplerCustomAdvanced":
            guider = node_by_id(graph, node["inputs"]["guider"])
            sigmas = node_by_id(graph, node["inputs"]["sigmas"])
            result.append((node["inputs"], guider["inputs"]["conditioning"], sigmas))
    return result


def scale_extents(nodes: list[dict]):
    return [
        Extent(n["inputs"]["width"], n["inputs"]["height"])
        for n in nodes
        if n["class_type"] == "ImageScale"
    ]


def assert_second_pass(graph: dict, base: Extent, canvas: Extent):
    nodes = graph_nodes(graph)
    (_, first_cond, first_sigmas), (second, second_cond, second_sigmas) = samplers(graph)
    # Base pass at about 1 MP, full denoise
    assert base.pixel_count <= 1.2 * 2**20
    assert first_sigmas["class_type"] == "BasicScheduler"
    # Upscale with the upscale model, resize to the exact canvas size
    assert any(n["class_type"] == "ImageUpscaleWithModel" for n in nodes)
    assert canvas in scale_extents(nodes)
    second_latent = node_by_id(graph, second["latent_image"])
    assert second_latent["class_type"] == "VAEEncode"
    upscaled = node_by_id(graph, second_latent["inputs"]["pixels"])
    assert upscaled["class_type"] == "ImageScale"
    assert Extent(upscaled["inputs"]["width"], upscaled["inputs"]["height"]) == canvas
    # Second pass: same conditioning, 0.3 of the 16 steps
    assert second_cond == first_cond
    assert second_sigmas["class_type"] == "SplitSigmas"
    assert second_sigmas["inputs"]["step"] == 11
    scheduler = node_by_id(graph, second_sigmas["inputs"]["sigmas"])
    assert scheduler["inputs"]["steps"] == 16
    return first_cond


def test_flux3_large_edit_uses_two_passes():
    canvas = Extent(1408, 1408)
    reference = Image.create(Extent(900, 1400))
    work, client = edit_work([ControlInput(ControlMode.reference, reference)], canvas=canvas)
    assert work.images is not None
    extent = work.images.extent
    assert extent.initial == Extent(1024, 1024)
    assert extent.desired == extent.target == canvas
    assert work.images.initial_image is not None
    assert work.images.initial_image.extent == extent.initial  # canvas scaled down once
    graph = create(work, client.models).root
    cond = assert_second_pass(graph, extent.initial, canvas)
    # The conditioning of both passes holds the references: canvas then reference layer
    nodes = graph_nodes(graph)
    refs = [n["inputs"] for n in nodes if n["class_type"] == "ReferenceMultiLatent"]
    assert len(refs) == 2
    assert node_by_id(graph, cond)["inputs"] == refs[1]
    canvas_image = node_by_id(graph, refs[0]["images.image0"])
    assert canvas_image["class_type"] == "ETN_LoadImageCache"  # 1024x1024, the sampled size
    first_latent = node_by_id(graph, samplers(graph)[0][0]["latent_image"])
    assert node_by_id(graph, first_latent["inputs"]["pixels"]) == canvas_image
    assert texts_of(nodes) == ["make it night"]  # encoded once


def texts_of(nodes: list[dict]):
    return [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]


def test_flux3_large_generate_uses_two_passes():
    canvas = Extent(1408, 1408)
    client = flux3_client()
    work = prepare(
        WorkflowKind.generate,
        canvas,
        ConditioningInput("a red fox in fresh snow"),
        flux3_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
    )
    assert work.images is not None
    assert work.images.extent.initial == Extent(1024, 1024)
    graph = create(work, client.models).root
    nodes = graph_nodes(graph)
    assert by_type(nodes, "EmptyFlux2LatentImage")["width"] == 1024
    assert_second_pass(graph, work.images.extent.initial, canvas)
    assert texts_of(nodes) == ["a red fox in fresh snow"]


def test_flux3_large_selection_edit_uses_two_passes():
    canvas = Extent(1408, 1408)
    client = flux3_client()
    mask = Mask.rectangle(Bounds(300, 300, 600, 600), Bounds(0, 0, *canvas))
    work = prepare(
        WorkflowKind.refine_region,
        Image.create(canvas),
        ConditioningInput("make the cloak red", edit_reference=True),
        flux3_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        mask=mask,
        strength=1.0,
        inpaint=InpaintParams(InpaintMode.custom, mask.bounds),
    )
    assert work.images is not None
    graph = create(work, client.models).root
    cond = assert_second_pass(graph, work.images.extent.initial, canvas)
    ref = node_by_id(graph, cond)
    assert ref["class_type"] == "ReferenceMultiLatent"
    assert ref["inputs"]["resolution"] == "keep resolution"


def test_flux3_small_canvas_and_weak_refine_use_one_pass():
    for canvas, strength in [(Extent(1024, 1024), 1.0), (Extent(1408, 1408), 0.5)]:
        work, client = edit_work([], canvas=canvas, strength=strength)
        assert work.images is not None
        assert work.images.extent.initial == work.images.extent.desired == canvas
        graph = create(work, client.models).root
        assert len(samplers(graph)) == 1


def test_flux3_reference_names_start_at_zero():
    image = Image.create(Extent(64, 64))
    cond = ConditioningInput("a portrait", control=[ControlInput(ControlMode.face, image)])
    assert build_instructions(cond, Arch.flux3, None).startswith(
        "Keep the face from <ref_image_0>."
    )


def test_flux3_reference_limit():
    image = Image.create(Extent(64, 64))
    control = [ControlInput(ControlMode.reference, image) for _ in range(10)]
    work, client = edit_work(control)
    with pytest.raises(RuntimeError, match="10 reference images"):
        create(work, client.models)


@pytest.mark.parametrize(
    "kind,strength", [(WorkflowKind.inpaint, 1.0), (WorkflowKind.refine_region, 0.9)]
)
def test_flux3_inpaint_uses_differential_diffusion(kind, strength):
    client = flux3_client()
    image = Image.create(Extent(1024, 1024))
    mask = Mask.rectangle(Bounds(384, 384, 256, 256), Bounds(0, 0, 1024, 1024))
    work = prepare(
        kind,
        image,
        ConditioningInput("a small red bird"),
        flux3_style(),
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
    assert not any(t.startswith("INPAINT_LoadFooocus") or "ControlNet" in t for t in types)
    assert "ReferenceMultiLatent" not in types
    assert_no_model_sampling(nodes)


@pytest.mark.parametrize("canvas", [Extent(1000, 1300), Extent(300, 300), Extent(4096, 4096)])
def test_flux3_resolution(canvas):
    client = flux3_client()
    work = prepare(
        WorkflowKind.generate,
        canvas,
        ConditioningInput("a lighthouse"),
        flux3_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
    )
    assert work.images is not None
    initial = work.images.extent.initial
    assert initial.width % 16 == 0 and initial.height % 16 == 0
    assert min(initial.width, initial.height) >= 512
    assert initial.pixel_count <= 1.2 * 2**20
    assert work.images.extent.desired.pixel_count >= initial.pixel_count


def test_flux3_arch_capabilities():
    assert Arch.flux3 in Arch.list()
    assert Arch.flux3.text_encoders == ["qwen_3vl_4b"]
    assert Arch.flux3.supports_cfg
    assert Arch.flux3.supports_edit
    assert not Arch.flux3.is_edit
    assert Arch.flux3.supports_regions
    assert not Arch.flux3.supports_clip_skip
    assert not Arch.flux3.has_controlnet_inpaint
    assert not Arch.flux3.is_flux2
    assert not Arch.is_compatible(Arch.flux3, Arch.flux2_4b)
    assert ControlMode.depth.can_substitute_instruction(Arch.flux3)


def test_flux3_gallery_lora_family():
    assert lora_compatible("FLUX 3 Klein", Arch.flux3) is True
    assert lora_compatible("flux3-klein", Arch.flux3) is True
    assert lora_compatible("FLUX 3 Klein", Arch.flux2_4b) is False
    assert lora_compatible("Flux 2 Klein", Arch.flux3) is False


def rect_mask(extent: Extent, *rects: Bounds):
    qimg = QImage(extent.width, extent.height, QImage.Format.Format_Grayscale8)
    qimg.fill(0)
    painter = QPainter(qimg)
    for r in rects:
        painter.fillRect(r.x, r.y, r.width, r.height, Qt.GlobalColor.white)
    painter.end()
    return Image(qimg)


def test_layout_box_grid():
    box = LayoutBox(Extent(1000, 800), 150, 100, 450, 350)
    assert box.to_grid() == [125, 150, 438, 450]
    assert LayoutBox(Extent(64, 64), -1, 0, 70, 64).to_grid() == [0, 0, 1000, 1000]


global_prompt = "a fantasy street at dusk, <obj_1> waves at <obj_2>"
character_prompt = "a red-haired girl in a green cloak"
sign_prompt = 'text: a wooden tavern sign that says "TAVERN"'
character_rect = Bounds(40, 100, 420, 800)
sign_rect = Bounds(560, 50, 400, 300)


@pytest.mark.parametrize("merged", [False, True], ids=["tablet", "krita"])
def test_flux3_regions_layout_prompt(merged):
    extent = Extent(1024, 1024)
    full = Bounds(0, 0, *extent)
    suffix = f", {global_prompt}" if merged else ""  # Krita appends the root prompt
    regions = [
        RegionInput(rect_mask(extent, full), full, global_prompt),
        RegionInput(rect_mask(extent, character_rect), full, character_prompt + suffix),
        RegionInput(rect_mask(extent, sign_rect), full, sign_prompt + suffix),
    ]
    client = flux3_client()
    work = prepare(
        WorkflowKind.generate,
        extent,
        ConditioningInput(global_prompt, regions=regions),
        flux3_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
    )
    nodes = graph_nodes(create(work, client.models).root)
    types = {n["class_type"] for n in nodes}
    assert not types & {"ETN_AttentionMask", "ETN_DefineRegion", "ETN_BackgroundRegion"}
    texts = [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]
    assert len(texts) == 1
    rows = [
        {"id": "obj_1", "bbox": [98, 39, 879, 449], "desc": character_prompt},
        {"id": "obj_2", "bbox": [49, 547, 342, 938], "desc": sign_prompt.removeprefix("text: ")},
    ]
    # Serialized like ComfyUI's Flux3LayoutPrompt, the background region is the global prompt
    assert texts[0] == f"{global_prompt} {json.dumps(rows, ensure_ascii=False)}"


def test_flux3_regions_reject_loras():
    extent = Extent(1024, 1024)
    full = Bounds(0, 0, *extent)
    regions = [
        RegionInput(rect_mask(extent, full), full, global_prompt),
        RegionInput(
            rect_mask(extent, character_rect), full, "a cat", loras=[LoraInput(lora_name, 1)]
        ),
        RegionInput(rect_mask(extent, sign_rect), full, "a dog"),
    ]
    client = flux3_client()
    work = prepare(
        WorkflowKind.generate,
        extent,
        ConditioningInput(global_prompt, regions=regions),
        flux3_style(),
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
    )
    with pytest.raises(RuntimeError, match="LoRAs"):
        create(work, client.models)
