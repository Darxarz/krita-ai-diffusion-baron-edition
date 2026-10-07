import json
from dataclasses import replace
from pathlib import Path

import pytest
from krita import Krita
from PyQt5.QtGui import QColor

from ai_diffusion.backend import background, workflow
from ai_diffusion.backend.api import (
    BackgroundRemovalInput,
    CustomWorkflowInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.client import ClientEvent, ClientMessage, ClientModels
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.model.background import result_mask
from ai_diffusion.model.jobs import JobKind
from ai_diffusion.model.model import Workspace
from ai_diffusion.persistence import ModelSync
from ai_diffusion.settings import ImageFileFormat

from .conftest import qtapp
from .test_model import _model_env, _wait_for_enqueue


@pytest.fixture
def nodes():
    names = [
        "SplitImageWithAlpha",
        "MaskToImage",
        "ImageBatch",
        "ImageAddNoise",
        "MaskComposite",
        "GrowMask",
        "SegmDetectorSEGS",
        "SegsToCombinedMask",
        "Segment",
        "MaskBlur+",
    ]
    data = {n: {} for n in names}
    data["RMBG"] = {"input": {"required": {"model": [["RMBG-2.0"]]}}}
    data["BiRefNetRMBG"] = {
        "input": {"required": {"model": [["BiRefNet-HR-matting", "BiRefNet_toonout"]]}}
    }
    data["UltralyticsDetectorProvider"] = {
        "input": {"required": {"model_name": [["segm/person_yolov8m-seg.pt"]]}}
    }
    data["SAMLoader"] = {"input": {"required": {"model_name": [["sam_vit_b_01ec64.pth"]]}}}
    data["SAMDetectorCombined"] = {}
    return ComfyObjectInfo(data)


@pytest.fixture
def workflows_dir(tmp_path: Path):
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


def test_serialization_and_pure_workflow(nodes):
    image = Image.create(Extent(73, 91), QColor(120, 60, 200))
    params = BackgroundRemovalInput(edge_blur=2, edge_offset=-1)
    request = background.prepare(image, params, nodes)
    assert params.output_seed == 0
    restored = WorkflowInput.from_dict(request.to_dict(image_format=ImageFileFormat.png))
    assert restored.background_removal == request.background_removal
    assert Image.compare(restored.image, image) == 0
    models = ClientModels()
    models.node_inputs = nodes
    before = request.to_dict()
    first, second = workflow.create(request, models), workflow.create(request, models)
    assert first.root == second.root
    assert request.to_dict() == before
    assert request.cost == 0
    classes = [n["class_type"] for n in first.root.values()]
    assert "KSampler" not in classes
    assert "SegmDetectorSEGS" in classes
    assert "MaskComposite" in classes
    assert "MaskBlur+" in classes
    transport = next(n["inputs"] for n in first.root.values() if n["class_type"] == "ImageAddNoise")
    assert transport["strength"] == 0
    assert request.background_removal is not None
    assert transport["seed"] == request.background_removal.output_seed


def test_foreground_mode_does_not_require_detector(nodes):
    del nodes.nodes["UltralyticsDetectorProvider"]
    params = BackgroundRemovalInput(model="RMBG-2.0", isolate_subject=False)
    request = background.prepare(Image.create(Extent(41, 59)), params, nodes)
    models = ClientModels()
    models.node_inputs = nodes
    graph = workflow.create(request, models)
    assert not any(n["class_type"] == "SegmDetectorSEGS" for n in graph.root.values())
    rmbg = next(n["inputs"] for n in graph.root.values() if n["class_type"] == "RMBG")
    assert rmbg["process_res"] == 1536
    assert rmbg["mask_blur"] == 0
    assert rmbg["mask_offset"] == 0


def test_text_exclusion_is_subtracted_after_matte(nodes):
    request = background.prepare(
        Image.create(Extent(50, 60)), BackgroundRemovalInput(subject_detector="text"), nodes
    )
    models = ClientModels()
    models.node_inputs = nodes
    graph = workflow.create(request, models)
    segments = [n for n in graph.root.values() if n["class_type"] == "Segment"]
    assert [n["inputs"]["prompt"] for n in segments] == ["person", "chair"]
    operations = [
        n["inputs"]["operation"] for n in graph.root.values() if n["class_type"] == "MaskComposite"
    ]
    assert operations == ["multiply", "subtract"]


def test_optional_sam_refinement(nodes):
    request = background.prepare(
        Image.create(Extent(50, 60)), BackgroundRemovalInput(refine_subject=True), nodes
    )
    models = ClientModels()
    models.node_inputs = nodes
    graph = workflow.create(request, models)
    sam = next(n["inputs"] for n in graph.root.values() if n["class_type"] == "SAMDetectorCombined")
    assert sam["detection_hint"] == "mask-area"
    assert sam["mask_hint_use_negative"] == "Outter"


def test_exported_custom_workflow_compiles_with_alpha(nodes):
    image = Image.create(Extent(37, 51), QColor(80, 120, 60, 128))
    inputs = ImageInput.from_extent(image.extent)
    inputs.initial_image = image
    file = Path(__file__).parents[1] / "ai_diffusion/workflows/background_remover_baron.json"
    custom = CustomWorkflowInput(
        json.loads(file.read_text(encoding="utf-8")),
        {
            "detection threshold": 0.3,
            "hair detail margin": 8,
            "clean edge colors": True,
        },
    )
    request = WorkflowInput(
        WorkflowKind.custom,
        inputs,
        sampling=SamplingInput("euler", "simple", 1, 1, seed=42),
        inpaint=InpaintParams(InpaintMode.custom, Bounds(0, 0, *image.extent)),
        custom_workflow=custom,
    )
    models = ClientModels()
    models.node_inputs = nodes
    graph = workflow.create(request, models)
    assert sum(n["class_type"] == "ETN_ApplyMaskToImage" for n in graph.root.values()) == 1
    assert (
        sum(
            n["class_type"] in ("ETN_SaveImageCache", "ETN_SendImageWebSocket", "ETN_KritaOutput")
            for n in graph.root.values()
        )
        == 1
    )
    assert not any(
        n["class_type"] in ("ETN_KritaCanvas", "ETN_Parameter") for n in graph.root.values()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model": "unavailable"},
        {"subject_detector": "bbox/faces.pt"},
        {"subject": "", "subject_detector": "text"},
        {"threshold": 1.0},
        {"detail_margin": 33},
        {"process_resolution": 128},
        {"edge_offset": -21},
        {"edge_blur": 9},
    ],
)
def test_invalid_request_fails_before_submission(nodes, change):
    with pytest.raises((ValueError, RuntimeError)):
        background.prepare(
            Image.create(Extent(10, 10)), replace(BackgroundRemovalInput(), **change), nodes
        )


def test_source_transparency_is_preserved_without_mutation():
    source = Image.create(Extent(3, 1), QColor(40, 100, 200, 255))
    source.set_pixel(0, 0, (40, 100, 200, 0))
    source.set_pixel(1, 0, (40, 100, 200, 128))
    matte = Image.create(Extent(3, 1), QColor(128, 128, 128))
    original = source._qimage.copy()
    mask = result_mask(matte, source)
    assert mask.pixel(0, 0) == 0
    value = mask.pixel(1, 0)
    assert isinstance(value, int)
    assert abs(value - 64) <= 1
    assert mask.pixel(2, 0) == 128
    assert source._qimage == original
    assert matte.pixel(0, 0) == (128, 128, 128, 255)


@qtapp
async def test_workspace_creates_editable_layers_and_preserves_original(workflows_dir, nodes):
    doc = Krita.instance().openDocument("test")
    async with _model_env(doc, workflows_dir) as (model, client):
        client.models.node_inputs = nodes
        image = Image.create(doc_extent := model.document.extent, QColor(90, 60, 30))
        original = model.layers.create("Original", image, Bounds(0, 0, *model.document.extent))
        model.layers.active = original
        model.workspace = Workspace.background
        model.background.source = "layer"
        model.background.remove_background()
        request = (await _wait_for_enqueue(client))[0]
        assert request.kind is WorkflowKind.remove_background
        assert request.sampling is None
        assert model.background.in_progress
        job = model.jobs.find("mock-job-0")
        assert job is not None and job.kind is JobKind.background_removal
        mask_image = Image.create(doc_extent, QColor(128, 128, 128))
        model.handle_message(
            ClientMessage(
                ClientEvent.finished, job.id or "", images=ImageCollection([image, mask_image])
            )
        )
        assert not model.background.in_progress
        assert not original.is_visible
        assert original.get_pixels(Bounds(0, 0, *model.document.extent)) == image
        group = next(l for l in model.layers.root.child_layers if l.name == "Separated character")
        foreground = next(l for l in group.child_layers if l.name == "Character")
        assert foreground.node.childNodes()[0].type() == "transparencymask"
        bg = next(l for l in group.child_layers if l.name.startswith("Background"))
        assert not bg.is_visible
        assert model.layers.active == foreground


@qtapp
async def test_empty_mask_keeps_source_visible_and_clears_busy_state(workflows_dir, nodes):
    doc = Krita.instance().openDocument("test")
    async with _model_env(doc, workflows_dir) as (model, client):
        client.models.node_inputs = nodes
        image = Image.create(model.document.extent, QColor(60, 90, 30))
        source = model.layers.create("Original", image, Bounds(0, 0, *model.document.extent))
        model.background.remove_background()
        await _wait_for_enqueue(client)
        model.handle_message(
            ClientMessage(
                ClientEvent.finished,
                "mock-job-0",
                images=ImageCollection([image, Image.create(image.extent, QColor(0, 0, 0))]),
            )
        )
        assert not model.background.in_progress
        assert source.is_visible
        assert model.error
        assert not any(l.name == "Separated character" for l in model.layers.root.child_layers)


@qtapp
async def test_cancel_pending_job_and_persist_settings(workflows_dir, nodes):
    doc = Krita.instance().openDocument("test")
    async with _model_env(doc, workflows_dir) as (model, client):
        client.models.node_inputs = nodes
        image = Image.create(model.document.extent, QColor(100, 150, 40))
        source = model.layers.create("Original", image, Bounds(0, 0, *model.document.extent))
        model.layers.active = source
        model.background.edge_offset = -2
        model.background.source = "layer"
        sync = ModelSync(model)
        sync._save()
        state = bytes(model.document.find_annotation("ui.json"))
        model.background.edge_offset = 5
        sync._load(model, state)
        assert model.background.edge_offset == -2
        model.background.remove_background()
        await _wait_for_enqueue(client)
        assert model.clear_queued() == ["mock-job-0"]
        assert not model.background.in_progress
        assert source.is_visible
