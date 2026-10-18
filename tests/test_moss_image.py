from pathlib import Path

import pytest

from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    LoraInput,
    SamplingInput,
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
from ai_diffusion.backend.resources import Arch, ResourceKind, resource_id
from ai_diffusion.backend.workflow import create, prepare
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
    assert not Arch.moss.supports_regions
    assert not Arch.moss.supports_clip_skip
    assert not Arch.moss.has_controlnet_inpaint
    assert not Arch.moss.is_flux2
    assert not Arch.is_compatible(Arch.moss, Arch.ernie)


def test_moss_gallery_lora_family():
    assert lora_compatible("MOSS-Image", Arch.moss) is True
    assert lora_compatible("MOSS-Image", Arch.flux2_9b) is False
