import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai_diffusion.backend.a1111 import apply_prompt_mode, validate_prompt_mode
from ai_diffusion.backend.api import CheckpointInput, SamplingInput, WorkflowInput, WorkflowKind
from ai_diffusion.backend.comfy_workflow import (
    ComfyObjectInfo,
    ComfyWorkflow,
    ConditioningOutput,
    Output,
)
from ai_diffusion.backend.resources import Arch
from ai_diffusion.style import Style


def request(arch=Arch.sdxl, noise=False):
    return WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput("model", arch),
        sampling=SamplingInput("euler", "normal", 7, 20, seed=123),
        prompt_mode="a1111",
        a1111_gpu_noise=noise,
        a1111_ensd=31337,
    )


def test_website_encoding_and_noise_golden():
    data = json.loads((Path(__file__).parent / "data/a1111_site.json").read_text())
    for noise in (False, True):
        w = ComfyWorkflow.from_dict(deepcopy(data["graph"]))
        apply_prompt_mode(w, request(noise=noise))
        expected = data["gpu" if noise else "cpu"]
        for id in data["graph"]:
            actual = w.root[id]
            if actual["class_type"] == "KSamplerAdvanced" and noise:
                assert isinstance(actual["inputs"]["model"][0], str)
                actual = deepcopy(actual)
                actual["inputs"]["model"] = ["a1111_settings_0", 0]
            assert actual["class_type"] == expected[id]["class_type"]
            assert actual["inputs"] == expected[id]["inputs"]
        settings = [n for n in w.root.values() if n["class_type"] == "smZ Settings"]
        assert len(settings) == int(noise)
        if noise:
            assert settings[0]["inputs"] == expected["a1111_settings_0"]["inputs"]


def test_a1111_uses_ksampler_and_preserves_refine_steps():
    cond = ConditioningOutput(Output(2, 0), Output(3, 0))
    w = ComfyWorkflow()
    w.a1111_mode = True
    result = w.sampler_custom_advanced(
        Output(1, 0),
        cond,
        Output(4, 0),
        Arch.sdxl,
        steps=30,
        start_at_step=18,
        seed=123,
        sampler="dpmpp_2m",
        scheduler="karras",
    )
    assert result.output == 0
    node = w.root[str(result.node)]
    assert node["class_type"] == "KSamplerAdvanced"
    assert node["inputs"]["start_at_step"] == 18
    assert node["inputs"]["end_at_step"] == 30
    assert node["inputs"]["noise_seed"] == 123
    assert w.sample_count == 12
    normal = ComfyWorkflow()
    normal.sampler_custom_advanced(Output(1, 0), cond, Output(4, 0), Arch.sdxl)
    assert any(n["class_type"] == "SamplerCustomAdvanced" for n in normal.root.values())


@pytest.mark.parametrize("arch", [Arch.qwen2, Arch.flux, Arch.krea2])
def test_reject_unsupported_architectures(arch):
    with pytest.raises(ValueError, match=r"SD 1\.5 and SDXL"):
        validate_prompt_mode(request(arch), ComfyObjectInfo({}))


def test_capabilities_and_default_request():
    nodes = ComfyObjectInfo({
        "smZ CLIPTextEncode": {},
        "KSampler": {"input": {"required": {"scheduler": [["normal", "karras"]]}}},
    })
    validate_prompt_mode(request(), nodes)
    with pytest.raises(ValueError, match="smZ Settings"):
        validate_prompt_mode(request(noise=True), nodes)
    nodes.nodes["smZ Settings"] = {}
    validate_prompt_mode(request(noise=True), nodes)
    work = request()
    assert work.sampling
    work.sampling.scheduler = "align_your_steps"
    with pytest.raises(ValueError, match="KSampler scheduler"):
        validate_prompt_mode(work, nodes)
    default = WorkflowInput(WorkflowKind.generate)
    validate_prompt_mode(default, ComfyObjectInfo({}))
    assert "prompt_mode" not in default.to_dict()
    assert WorkflowInput.from_dict(request(noise=True).to_dict()) == request(noise=True)


def test_style_roundtrip(tmp_path):
    style = Style(tmp_path / "a1111.json")
    style.prompt_mode = "a1111"
    style.a1111_gpu_noise = True
    style.a1111_ensd = 31337
    style.save()
    restored = Style.load(style.filepath)
    assert restored and restored.prompt_mode == "a1111"
    assert restored.a1111_gpu_noise and restored.a1111_ensd == 31337
