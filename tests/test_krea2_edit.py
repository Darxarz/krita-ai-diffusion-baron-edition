from pathlib import Path

import pytest

from ai_diffusion.backend.api import (
    ConditioningInput,
    ControlInput,
    InpaintMode,
    InpaintParams,
    LoraInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_client import _find_text_encoder_models, _find_vae_models
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.backend.workflow import create, prepare, prepare_prompts
from ai_diffusion.files import FileCollection, FileFormat, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Style

model_name = "Krea2\\krea2Turbo18For_v2.safetensors"
identity_name = "krea\\krea2_identity_edit_v1_2_r128.safetensors"
style_name = "krea\\krea2_style_reference.safetensors"


def models_and_style(mode="identity"):
    models = ClientModels()
    models.checkpoints[model_name] = CheckpointInfo(model_name, Arch.krea2, FileFormat.diffusion)
    models.loras = [identity_name, style_name, "krea\\my_character.safetensors"]
    models.node_inputs = ComfyObjectInfo({
        name: {}
        for name in [
            "Krea2EditGroundedEncode",
            "Krea2EditModelPatch",
            "TextEncodeQwenImageEditPlus",
            "FluxKontextMultiReferenceLatentMethod",
            "ModelSamplingFlux",
        ]
    })
    models.resources.update(_find_text_encoder_models(["qwen3vl_4b_fp8_scaled.safetensors"]))
    models.resources.update(_find_vae_models(["qwen_image_vae.safetensors"]))
    style = Style(Path("krea.json"))
    style.architecture = Arch.krea2
    style.checkpoints = [model_name]
    style.style_prompt = ""
    style.krea2_reference_mode = mode
    style.sampler = "Flux - Euler simple"
    style.sampler_steps = 10
    style.cfg_scale = 1.0
    return models, style


def request(models, style, *, kind=WorkflowKind.refine, refs=0, edit=True, strength=1.0):
    cond = ConditioningInput(
        "Change the lighting to sunset",
        edit_reference=edit,
        control=[
            ControlInput(ControlMode.reference, Image.create(Extent(384, 512))) for _ in range(refs)
        ],
    )
    canvas = Extent(768, 512) if kind is WorkflowKind.generate else Image.create(Extent(768, 512))
    mask = (
        Mask.rectangle(Bounds(448, 320, 256, 128), Bounds(0, 0, 768, 512))
        if kind in [WorkflowKind.inpaint, WorkflowKind.refine_region]
        else None
    )
    return prepare(
        kind,
        canvas,
        cond,
        style,
        1234,
        models,
        FileLibrary(FileCollection(), FileCollection()),
        PerformanceSettings(batch_size=1),
        strength=strength,
        mask=mask,
        inpaint=InpaintParams(InpaintMode.custom, mask.bounds) if mask else None,
    )


@pytest.mark.parametrize(
    "kind", [WorkflowKind.refine, WorkflowKind.refine_region, WorkflowKind.inpaint]
)
@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_edit_wiring(kind, strength):
    models, style = models_and_style()
    work = request(models, style, kind=kind, refs=1, strength=strength)
    assert work.models is not None and work.models.loras == [LoraInput(identity_name, 1.0)]
    graph = create(work, models).root
    patch_id, patch = next(
        (key, n["inputs"]) for key, n in graph.items() if n["class_type"] == "Krea2EditModelPatch"
    )
    encoders = [n["inputs"] for n in graph.values() if n["class_type"] == "Krea2EditGroundedEncode"]
    assert len(encoders) == 2
    assert {n["prompt"] for n in encoders} == {"Change the lighting to sunset", ""}
    for encoder in encoders:
        assert encoder["image"] == patch["source_image"]
        assert encoder["image_b"] == patch["source_image_b"]
        assert encoder["grounding_px"] == 768
    assert patch["fit_mode"] == "fit"
    assert patch["ref_boost"] == pytest.approx(4.0)
    sampler = next(
        n["inputs"] for n in graph.values() if n["class_type"] == "SamplerCustomAdvanced"
    )
    assert patch["target_latent"] == sampler["latent_image"]
    guider = graph[sampler["guider"][0]]["inputs"]
    assert guider["model"] == [patch_id, 0]
    if kind is WorkflowKind.refine and work.sampling is not None and work.sampling.start_step == 0:
        assert not any(n["class_type"] == "VAEEncode" for n in graph.values())
    else:
        assert any(n["class_type"] == "VAEEncode" for n in graph.values())
    assert work.models == WorkflowInput.from_dict(work.to_dict()).models


def test_plain_generation_does_not_add_edit_lora():
    models, style = models_and_style()
    work = request(models, style, kind=WorkflowKind.generate, edit=False)
    assert work.models is not None and work.models.loras == []
    graph = create(work, models).root
    assert not any(n["class_type"].startswith("Krea2Edit") for n in graph.values())


def test_reference_generation_allows_new_composition():
    models, style = models_and_style()
    work = request(models, style, kind=WorkflowKind.generate, edit=False, refs=2)
    graph = create(work, models).root
    patch = next(n["inputs"] for n in graph.values() if n["class_type"] == "Krea2EditModelPatch")
    assert patch["ref_boost"] == pytest.approx(1.0)


def test_explicit_rank_and_additional_character_lora():
    models, style = models_and_style()
    full_rank = "krea\\krea2_identity_edit_v1_2.safetensors"
    models.loras.append(full_rank)
    style.loras = [
        {"name": "krea\\my_character.safetensors", "strength": 0.7},
        {"name": full_rank, "strength": 1.0},
    ]
    work = request(models, style)
    graph = create(work, models).root
    identity = next(n["inputs"] for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert identity["lora_name"] == full_rank
    extra = next(n["inputs"] for n in graph.values() if n["class_type"] == "LoraLoader")
    assert graph[extra["model"][0]]["inputs"] == identity


@pytest.mark.parametrize("mode,refs", [("identity", 2), ("style", 3)])
def test_reference_limit_counts_canvas(mode, refs):
    models, style = models_and_style(mode)
    with pytest.raises(RuntimeError, match="including the source canvas"):
        request(models, style, refs=refs)


def test_missing_resources_fail_before_submission():
    models, style = models_and_style()
    models.loras = []
    with pytest.raises(RuntimeError, match="Identity Edit LoRA"):
        request(models, style)
    models.loras = [identity_name]
    models.node_inputs = ComfyObjectInfo({})
    with pytest.raises(RuntimeError, match="node Krea2EditGroundedEncode"):
        request(models, style)


def test_style_reference_matches_official_graph():
    models, style = models_and_style("style")
    work = request(models, style, kind=WorkflowKind.generate, edit=False, refs=3)
    assert work.models is not None and work.models.loras == [LoraInput(style_name, 1.0)]
    graph = create(work, models).root
    types = {n["class_type"]: n["inputs"] for n in graph.values()}
    assert "Krea2EditModelPatch" not in types
    assert types["TextEncodeQwenImageEditPlus"]["vae"] is not None
    assert all(types["TextEncodeQwenImageEditPlus"][f"image{i}"] for i in range(1, 4))
    assert (
        types["FluxKontextMultiReferenceLatentMethod"]["reference_latents_method"]
        == "index_timestep_zero"
    )
    assert types["ModelSamplingFlux"]["max_shift"] == pytest.approx(1.15)
    assert types["ModelSamplingFlux"]["width"] == work.extent.initial.width


def test_named_reference_and_face_instruction():
    _models, style = models_and_style()
    cond = ConditioningInput("Use <layer:Person> for the face", edit_reference=True)
    prepared = prepare_prompts(cond, style, 1, Arch.krea2, ref_layers={"Person": 2})
    assert prepared.conditioning.positive == "Use image 2 for the face"
    cond = ConditioningInput("new outfit", control=[ControlInput(ControlMode.face)])
    prepared = prepare_prompts(cond, style, 1, Arch.krea2)
    assert prepared.conditioning.control[0].mode is ControlMode.reference
    assert "Keep the face from image 1." in prepared.conditioning.positive
