from pathlib import Path

import pytest

from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ControlInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.client import CheckpointInfo
from ai_diffusion.backend.comfy_client import (
    ComfyClient,
    _find_text_encoder_models,
    _find_vae_models,
)
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow, Output
from ai_diffusion.backend.resources import (
    Arch,
    ControlMode,
    ResourceKind,
    UpscalerName,
    resource_id,
)
from ai_diffusion.backend.workflow import create, prepare
from ai_diffusion.files import FileCollection, FileFormat, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.model.updates import AutoUpdate
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Style

model_name = "Qwen21\\qwen_image_2.1_int8_convrot.safetensors"
encoder_name = "qwen3vl_8b_int8_convrot.safetensors"
vae_name = "qwen_image_2.1_vae_bf16.safetensors"


@pytest.mark.parametrize(
    "filename",
    [model_name, "qwen_image_2.1_bf16.safetensors", "Qwen-Image-2.1-Q4_K_M.gguf"],
)
def test_qwen21_unknown_metadata(filename):
    assert Arch.from_string("unknown", filename=filename) is Arch.qwen2
    assert Arch.from_string("qwen-image21", filename=filename) is Arch.qwen2
    assert Arch.from_string("sdxl", filename=filename) is Arch.sdxl


@pytest.mark.parametrize(
    "filename",
    [
        "qwen_image_bf16.safetensors",
        "qwen_image_edit_2511.safetensors",
        "qwen_image_layered.safetensors",
        "qwen3vl_8b_int8_convrot.safetensors",
        "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
        "qwen_image_2.1_vae_bf16.safetensors",
        "qwen_image_2.1_fun_controlnet_union_bf16.safetensors",
        "Qwen21/unrelated.safetensors",
        "qwen_image_2.10.safetensors",
    ],
)
def test_qwen21_does_not_claim_other_models(filename):
    assert Arch.from_string("unknown", filename=filename) is None


def loader_nodes():
    return ComfyObjectInfo({
        "UNETLoader": {"input": {"required": {"unet_name": [[model_name]]}}},
        "UnetLoaderGGUF": {"input": {"required": {"unet_name": [["Qwen-Image-2.1-Q4_K_M.gguf"]]}}},
        "CLIPLoader": {"input": {"required": {"clip_name": [[encoder_name]]}}},
        "VAELoader": {"input": {"required": {"vae_name": [[vae_name]]}}},
    })


@pytest.mark.parametrize("metadata", [None, {}, {model_name: {"base_model": "unknown"}}])
def test_qwen21_discovery(metadata):
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, metadata)
    assert client.models.checkpoints[model_name].arch is Arch.qwen2
    assert client.models.checkpoints[model_name].format is FileFormat.diffusion
    assert client.models.checkpoints["Qwen-Image-2.1-Q4_K_M.gguf"].arch is Arch.qwen2


def test_qwen21_preserves_server_metadata():
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, {model_name: {"base_model": "sdxl"}})
    assert client.models.checkpoints[model_name].arch is Arch.sdxl
    client._refresh_models(
        loader_nodes(), {}, {model_name: {"base_model": "unknown", "is_refiner": True}}
    )
    assert model_name not in client.models.checkpoints


def test_qwen21_checkpoint_serialization():
    info = CheckpointInfo(model_name, Arch.qwen2, FileFormat.diffusion)
    assert CheckpointInfo.from_dict(info.to_dict()) == info


def test_qwen21_edit_request_serialization():
    work = WorkflowInput(
        WorkflowKind.refine,
        models=CheckpointInput(model_name, Arch.qwen2),
        images=ImageInput.from_extent(Extent(512, 512)),
        conditioning=ConditioningInput("edit", edit_reference=True),
        sampling=SamplingInput("euler", "simple", 1.0, 25),
    )
    assert WorkflowInput.from_dict(work.to_dict()) == work


@pytest.mark.parametrize("cfg", [1.0, 2.5])
def test_qwen21_workflow_and_preset(cfg):
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, {model_name: {"base_model": "unknown"}})
    models = client.models
    models.resources.update(_find_text_encoder_models([encoder_name]))
    models.resources.update(_find_vae_models([vae_name]))
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.qwen2),
        images=ImageInput.from_extent(Extent(512, 512)),
        conditioning=ConditioningInput("a red apple", "blurry"),
        sampling=SamplingInput("euler", "simple", cfg, 25),
    )
    graph = create(work, models).root
    nodes = {node["class_type"]: node["inputs"] for node in graph.values()}
    assert nodes["UNETLoader"]["unet_name"] == model_name
    assert nodes["CLIPLoader"]["clip_name"] == encoder_name
    assert nodes["CLIPLoader"]["type"] == "qwen_image"
    assert nodes["VAELoader"]["vae_name"] == vae_name
    assert nodes["TextEncodeQwenImage21"]["prompt"] == "a red apple"
    assert nodes["TextEncodeQwenImage21"]["negative_prompt"] == ("blurry" if cfg > 1 else "")
    style = Style.load(Path(__file__).parents[1] / "ai_diffusion/styles/qwen21.json")
    assert style is not None
    assert style.preferred_checkpoint(models.checkpoints) == model_name


@pytest.mark.parametrize("start_step", [0, 12])
def test_qwen21_edit_reference(start_step):
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, {model_name: {"base_model": "unknown"}})
    models = client.models
    models.resources.update(_find_text_encoder_models([encoder_name]))
    models.resources.update(_find_vae_models([vae_name]))
    image_input = ImageInput.from_extent(Extent(512, 512))
    image_input.initial_image = Image.create(Extent(512, 512))
    work = WorkflowInput(
        WorkflowKind.refine,
        models=CheckpointInput(model_name, Arch.qwen2),
        images=image_input,
        conditioning=ConditioningInput("change the apple to green", edit_reference=True),
        sampling=SamplingInput("euler", "simple", 1.0, 25, start_step=start_step),
    )
    nodes = [node for node in create(work, models).root.values()]
    encoder = next(node for node in nodes if node["class_type"] == "TextEncodeQwenImage21")
    assert "images.image_1" in encoder["inputs"]
    assert encoder["inputs"]["prompt"] == "change the apple to green"
    assert sum(node["class_type"] == "VAEEncode" for node in nodes) == (1 if start_step else 0)


def test_qwen21_transparent_extra_reference():
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, {model_name: {"base_model": "unknown"}})
    models = client.models
    models.resources.update(_find_text_encoder_models([encoder_name]))
    models.resources.update(_find_vae_models([vae_name]))
    reference = ControlInput(ControlMode.reference, Image.create(Extent(512, 512)))
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(model_name, Arch.qwen2),
        images=ImageInput.from_extent(Extent(512, 512)),
        conditioning=ConditioningInput("use this reference", control=[reference]),
        sampling=SamplingInput("euler", "simple", 1.0, 25),
    )
    graph = create(work, models).root
    encoder = next(node for node in graph.values() if node["class_type"] == "TextEncodeQwenImage21")
    image_node_id = encoder["inputs"]["images.image_1"][0]
    assert graph[image_node_id]["class_type"] == "ETN_ApplyMaskToImage"


def test_qwen21_reference_limit():
    graph = ComfyWorkflow()
    graph.text_encode_qwen2(Output(1, 0), Output(2, 0), [Output(3, 0)] * 10, "edit", "")
    with pytest.raises(ValueError, match="10 reference images"):
        graph.text_encode_qwen2(Output(1, 0), Output(2, 0), [Output(3, 0)] * 11, "edit", "")


@pytest.mark.parametrize("kind", [WorkflowKind.inpaint, WorkflowKind.refine_region])
def test_qwen21_masked_edit_stays_single_pass(kind):
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, {model_name: {"base_model": "unknown"}})
    client.models.resources.update(_find_text_encoder_models([encoder_name]))
    client.models.resources.update(_find_vae_models([vae_name]))
    style = Style(Path("qwen.json"))
    style.checkpoints = [model_name]
    style.architecture = Arch.qwen2
    style.preferred_resolution = 512
    style.sampler = "Qwen Image 2.1 - Euler"
    image = Image.create(Extent(1024, 768))
    mask = Mask.rectangle(Bounds(256, 256, 256, 256), Bounds(0, 0, 1024, 768))
    work = prepare(
        kind,
        image,
        ConditioningInput("edit the selected area", edit_reference=True),
        style,
        1234,
        client.models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        mask=mask,
        inpaint=InpaintParams(InpaintMode.custom, mask.bounds),
    )
    assert work.images is not None
    assert work.images.extent.initial == work.images.extent.desired
    work.color_match = 1.0
    graph = create(work, client.models).root
    color_match = next(n for n in graph.values() if n["class_type"] == "INPAINT_ColorMatch")
    for input_name in ("target", "reference"):
        source_id, source_output = color_match["inputs"][input_name]
        assert graph[source_id]["class_type"] == "SplitImageWithAlpha"
        assert source_output == 0
    apply = next(
        n
        for n in graph.values()
        if n["class_type"] == "ETN_ApplyMaskToImage"
        and graph[n["inputs"]["mask"][0]]["class_type"] == "MaskComposite"
    )
    assert graph[apply["inputs"]["mask"][0]]["inputs"]["operation"] == "multiply"
    join = next(n for n in graph.values() if n["class_type"] == "JoinImageWithAlpha")
    assert join["inputs"]["alpha"] == [color_match["inputs"]["target"][0], 1]


@pytest.mark.parametrize("preserve_alpha", [False, True])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_color_match_preserves_alpha_only_when_requested(preserve_alpha, strength):
    graph = ComfyWorkflow()
    target, reference, mask = Output(100, 0), Output(101, 0), Output(102, 0)
    result = graph.color_match(target, reference, mask, strength, preserve_alpha=preserve_alpha)
    if strength == 0:
        assert result == target
        assert not graph.root
        return
    color_match = next(n for n in graph.root.values() if n["class_type"] == "INPAINT_ColorMatch")
    assert color_match["inputs"]["exclude_mask"] == ["102", 0]
    assert color_match["inputs"]["strength"] == strength
    if preserve_alpha:
        assert len(graph.root) == 4
        assert graph.root[str(result.node)]["class_type"] == "JoinImageWithAlpha"
    else:
        assert len(graph.root) == 1
        assert color_match["inputs"]["target"] == ["100", 0]
        assert color_match["inputs"]["reference"] == ["101", 0]


@pytest.mark.parametrize("preserve_alpha", [False, True])
def test_upscaler_keeps_alpha_out_of_rgb_model(preserve_alpha):
    graph = ComfyWorkflow()
    result = graph.upscale_image(Output(100, 0), Output(101, 0), preserve_alpha=preserve_alpha)
    upscale = next(n for n in graph.root.values() if n["class_type"] == "ImageUpscaleWithModel")
    if preserve_alpha:
        source_id, source_output = upscale["inputs"]["image"]
        assert graph.root[source_id]["class_type"] == "SplitImageWithAlpha"
        assert source_output == 0
        join = graph.root[str(result.node)]
        assert join["class_type"] == "JoinImageWithAlpha"
        assert join["inputs"]["alpha"] == [source_id, 1]
    else:
        assert len(graph.root) == 1
        assert upscale["inputs"]["image"] == ["101", 0]


@pytest.mark.parametrize("preserve_alpha", [False, True])
def test_resize_does_not_quantize_rgb_through_nearly_transparent_alpha(preserve_alpha):
    graph = ComfyWorkflow()
    result = graph.scale_image(Output(101, 0), Extent(1440, 1536), preserve_alpha=preserve_alpha)
    resize = next(n for n in graph.root.values() if n["class_type"] == "ImageScale")
    if preserve_alpha:
        source_id, source_output = resize["inputs"]["image"]
        assert graph.root[source_id]["class_type"] == "SplitImageWithAlpha"
        assert source_output == 0
        join = graph.root[str(result.node)]
        assert join["class_type"] == "JoinImageWithAlpha"
        assert join["inputs"]["alpha"] == [source_id, 1]
    else:
        assert len(graph.root) == 1
        assert resize["inputs"]["image"] == ["101", 0]


@pytest.mark.parametrize("preserve_alpha", [False, True])
def test_selection_mask_preserves_generated_transparency(preserve_alpha):
    graph = ComfyWorkflow()
    result = graph.apply_mask(Output(100, 0), Output(101, 0), preserve_alpha=preserve_alpha)
    apply = graph.root[str(result.node)]
    if preserve_alpha:
        mask = graph.root[apply["inputs"]["mask"][0]]
        assert mask["class_type"] == "MaskComposite"
        assert mask["inputs"]["operation"] == "multiply"
        assert mask["inputs"]["source"] == ["101", 0]
        opacity = graph.root[mask["inputs"]["destination"][0]]
        assert opacity["class_type"] == "InvertMask"
        assert opacity["inputs"]["mask"] == [apply["inputs"]["image"][0], 1]
    else:
        assert len(graph.root) == 1
        assert apply["inputs"]["mask"] == ["101", 0]


@pytest.mark.parametrize("kind", [WorkflowKind.generate, WorkflowKind.refine])
@pytest.mark.parametrize("multiplier", [0.5, 0.7])
def test_qwen21_lower_resolution_returns_rgba_to_canvas(kind, multiplier):
    client = ComfyClient("http://localhost:8188")
    client._refresh_models(loader_nodes(), {}, {model_name: {"base_model": "unknown"}})
    models = client.models
    models.resources.update(_find_text_encoder_models([encoder_name]))
    models.resources.update(_find_vae_models([vae_name]))
    for upscaler in UpscalerName:
        models.resources[resource_id(ResourceKind.upscaler, Arch.all, upscaler)] = upscaler.value
    style = Style(Path("qwen.json"))
    style.checkpoints = [model_name]
    style.architecture = Arch.qwen2
    style.preferred_resolution = 512
    style.sampler = "Qwen Image 2.1 - Euler"
    canvas = Extent(1536, 1536)
    work = prepare(
        kind,
        Image.create(canvas) if kind is WorkflowKind.refine else canvas,
        ConditioningInput("edit the image", edit_reference=kind is WorkflowKind.refine),
        style,
        1234,
        models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1, resolution_multiplier=multiplier),
    )
    assert work.images is not None
    assert work.images.extent.desired.width < canvas.width
    assert work.images.extent.target == canvas
    graph = create(work, models).root
    for node in graph.values():
        if node["class_type"] == "ImageScale":
            source_id, source_output = node["inputs"]["image"]
            assert graph[source_id]["class_type"] == "SplitImageWithAlpha"
            assert source_output == 0
    upscalers = [n for n in graph.values() if n["class_type"] == "ImageUpscaleWithModel"]
    assert upscalers
    for node in upscalers:
        source_id, source_output = node["inputs"]["image"]
        assert graph[source_id]["class_type"] == "SplitImageWithAlpha"
        assert source_output == 0
        assert any(
            n["class_type"] == "JoinImageWithAlpha" and n["inputs"]["alpha"] == [source_id, 1]
            for n in graph.values()
        )


@pytest.mark.asyncio
async def test_baron_updater_does_not_replace_fork(tmp_path):
    updater = AutoUpdate(plugin_dir=tmp_path, current_version="1.53.0-baron.1")

    class Feed:
        async def get(self, url, timeout):
            return {
                "schema": 1,
                "edition": "baron",
                "channel": "stable",
                "windows": {"version": "1.53.1"},
            }

    updater._request_manager = Feed()  # type: ignore
    with pytest.raises(ValueError, match="Baron version"):
        await updater._check()
    assert updater._package is None
