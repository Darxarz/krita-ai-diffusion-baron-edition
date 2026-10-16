import pytest

from ai_diffusion.backend.api import InpaintMode, InpaintParams
from ai_diffusion.backend.comfy_workflow import ComfyWorkflow, Output
from ai_diffusion.backend.workflow import denoise_to_compositing_mask
from ai_diffusion.image import Bounds


@pytest.mark.parametrize("preserve_alpha", [False, True])
def test_manual_feather_reaches_output_alpha_without_thresholding(preserve_alpha):
    graph = ComfyWorkflow()
    mask, image = Output(100, 0), Output(101, 0)
    params = InpaintParams(InpaintMode.custom, Bounds(0, 0, 64, 64))
    params.blend = 0
    compositing = denoise_to_compositing_mask(graph, mask, params)
    graph.apply_mask(image, compositing, preserve_alpha=preserve_alpha)
    assert compositing is mask
    assert not any(
        node["class_type"] in ("ThresholdMask", "INPAINT_ShrinkMask")
        for node in graph.root.values()
    )
    applied = next(
        node for node in graph.root.values() if node["class_type"] == "ETN_ApplyMaskToImage"
    )
    if preserve_alpha:
        multiplied = graph.root[applied["inputs"]["mask"][0]]
        assert multiplied["class_type"] == "MaskComposite"
        assert multiplied["inputs"]["operation"] == "multiply"
        assert ["100", 0] in multiplied["inputs"].values()
    else:
        assert applied["inputs"]["mask"] == ["100", 0]


@pytest.mark.parametrize("blend", [1, 5, 20, 21])
def test_plugin_blending_retains_original_threshold_and_shrink(blend):
    graph = ComfyWorkflow()
    params = InpaintParams(InpaintMode.custom, Bounds(0, 0, 64, 64))
    params.blend = blend
    result = denoise_to_compositing_mask(graph, Output(100, 0), params)
    assert graph.root == {
        "1": {"class_type": "ThresholdMask", "inputs": {"mask": ["100", 0], "value": 0.0}},
        "2": {
            "class_type": "INPAINT_ShrinkMask",
            "inputs": {
                "mask": ["1", 0],
                "shrink": blend // 2,
                "blur": blend,
                "blur_type": "gaussian",
            },
        },
    }
    assert result == Output(2, 0)
