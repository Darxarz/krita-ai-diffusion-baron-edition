"""Tests for ai_diffusion/persistence.py - the layer that persists DocumentModel state to Krita
document annotations and the plugin's settings.json file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from krita import Document as MockKritaDocument
from krita import Krita
from PyQt5.QtCore import QByteArray

from ai_diffusion.backend.api import FillMode, InpaintMode
from ai_diffusion.document import KritaDocument
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.model.connection import Connection
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.jobs import Job, JobKind, JobParams, JobState
from ai_diffusion.model.model import DocumentModel, InpaintContext, QueueMode
from ai_diffusion.persistence import ModelSync, RecentlyUsedSync
from ai_diffusion.settings import Settings, settings
from ai_diffusion.style import Style

from .conftest import qtapp

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def workflows_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


def _make_model(krita_doc: MockKritaDocument, workflows_dir: Path) -> DocumentModel:
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None
    conn = Connection()
    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    return DocumentModel(doc, conn, wf_coll)


def _make_style(filename: str = "test.json", checkpoint: str = "test_sd15.safetensors") -> Style:
    style = Style(Path(filename))
    style.checkpoints = [checkpoint]
    return style


# ---------------------------------------------------------------------------
# test_recently_used
# ---------------------------------------------------------------------------


def test_recently_used(workflows_dir: Path, tmp_path: Path):
    """RecentlyUsedSync writes recently used parameters to settings and re-applies them to new
    models that have no stored annotation yet."""
    settings_path = tmp_path / "settings.json"

    # Use a private Settings instance so the test doesn't pollute the global one.
    local_settings = Settings()
    local_settings.default_path = settings_path

    # Patch the module-level settings reference used inside persistence.py
    import ai_diffusion.persistence as persistence_mod
    import ai_diffusion.settings as settings_mod

    original_settings = settings_mod.settings
    persistence_mod.settings = local_settings  # type: ignore[assignment]
    settings_mod.settings = local_settings  # type: ignore[assignment]
    try:
        recently_used = RecentlyUsedSync()

        # ── model 1: set several tracked properties ──────────────────────
        krita_doc1 = Krita.instance().openDocument("test1")
        model1 = _make_model(krita_doc1, workflows_dir)
        recently_used.track(model1)

        style1 = _make_style("custom.json", "custom_sd15.safetensors")
        model1.style = style1
        model1.batch_count = 4
        model1.translation_enabled = False
        model1.inpaint.mode = InpaintMode.fill
        model1.inpaint.fill = FillMode.blur
        model1.inpaint.use_inpaint = False
        model1.inpaint.use_prompt_focus = True
        model1.inpaint.context = InpaintContext.entire_image
        model1.upscale.upscaler = "RealESRGAN_x4plus.pth"

        # ── model 2: fresh doc, load recently used ────────────────────────
        recently_used2 = RecentlyUsedSync.from_settings()
        krita_doc2 = Krita.instance().openDocument("test2")
        model2 = _make_model(krita_doc2, workflows_dir)
        recently_used2.track(model2)

        # model2 has no annotation → values are taken from recently_used
        assert model2.batch_count == 4
        assert model2.translation_enabled is False
        assert model2.inpaint.mode is InpaintMode.fill
        assert model2.inpaint.fill is FillMode.blur
        assert model2.inpaint.use_inpaint is False
        assert model2.inpaint.use_prompt_focus is True
        assert model2.inpaint.context is InpaintContext.entire_image
        assert model2.upscale.upscaler == "RealESRGAN_x4plus.pth"
    finally:
        persistence_mod.settings = original_settings  # type: ignore[assignment]
        settings_mod.settings = original_settings  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# test_sync
# ---------------------------------------------------------------------------


@qtapp
async def test_sync(workflows_dir: Path):
    """ModelSync persists DocumentModel state as a document annotation and restores it when a new
    ModelSync is created for the same document."""

    krita_doc = Krita.instance().openDocument("test")
    model1 = _make_model(krita_doc, workflows_dir)

    style = _make_style("synced.json", "synced_sd15.safetensors")
    model1.style = style
    model1.batch_count = 3
    model1.strength = 0.75
    model1.seed = 12345
    model1.fixed_seed = True
    model1.translation_enabled = False
    model1.queue_mode = QueueMode.front
    model1.regions.positive = "a majestic mountain"
    model1.regions.negative = "blurry"
    model1.inpaint.mode = InpaintMode.fill
    model1.inpaint.fill = FillMode.blur
    model1.inpaint.use_inpaint = False
    model1.inpaint.use_prompt_focus = True
    model1.live.strength = 0.55
    model1.upscale.upscaler = "4x-UltraSharp.pt"
    model1.upscale.strength = 0.42
    model1.upscale.use_diffusion = False

    # Create sync and let it flush (it saves immediately on construction and on demand)
    sync1 = ModelSync(model1)
    sync1._save()  # force synchronous write so annotation is populated before we read it

    # ── restore from the same krita_doc ──────────────────────────────────
    # Build a second DocumentModel backed by the same MockKritaDocument (same annotation store)
    Krita.instance().setActiveDocument(krita_doc)
    doc2 = KritaDocument(krita_doc, None)
    conn2 = Connection()
    wf_coll2 = WorkflowCollection(conn2, folder=workflows_dir)
    model2 = DocumentModel(doc2, conn2, wf_coll2)
    _sync2 = ModelSync(model2)  # triggers _load() from annotation

    assert model2.batch_count == model1.batch_count
    assert model2.strength == pytest.approx(model1.strength)
    assert model2.seed == model1.seed
    assert model2.fixed_seed == model1.fixed_seed
    assert model2.translation_enabled == model1.translation_enabled
    assert model2.queue_mode is model1.queue_mode
    assert model2.regions.positive == model1.regions.positive
    assert model2.regions.negative == model1.regions.negative
    assert model2.inpaint.mode is model1.inpaint.mode
    assert model2.inpaint.fill is model1.inpaint.fill
    assert model2.inpaint.use_inpaint == model1.inpaint.use_inpaint
    assert model2.inpaint.use_prompt_focus == model1.inpaint.use_prompt_focus
    assert model2.live.strength == pytest.approx(model1.live.strength)
    assert model2.upscale.upscaler == model1.upscale.upscaler
    assert model2.upscale.strength == pytest.approx(model1.upscale.strength)
    assert model2.upscale.use_diffusion == model1.upscale.use_diffusion


# ---------------------------------------------------------------------------
# test_history
# ---------------------------------------------------------------------------


@qtapp
async def test_android_integer_numbers_restore_prompts_and_history(workflows_dir: Path):
    document = Krita.instance().openDocument("android integer JSON")
    images = ImageCollection([Image.create(Extent(32, 24), fill=0x46112233)])
    blob, offsets = images.to_bytes()
    state = {
        "version": 1,
        "strength": 1,
        "resolution_multiplier": 1,
        "upscale": {"factor": 2, "strength": 1, "unblur_strength": 0},
        "live": {"strength": 1},
        "root": {"positive": "Android generation prompt", "negative": "blurry"},
        "edit": {"positive": "Android edit prompt"},
        "history": [
            {
                "id": "android-job",
                "slot": 0,
                "offsets": offsets,
                "params": {
                    "bounds": [11, 29, 32, 24],
                    "name": "Android generation prompt",
                    "seed": 4294967295,
                },
                "in_use": {"0": True},
            }
        ],
    }
    document.setAnnotation(
        "ai_diffusion/ui.json", "AI Diffusion", QByteArray(json.dumps(state).encode())
    )
    document.setAnnotation("ai_diffusion/result0.webp", "AI Diffusion", blob)
    model = _make_model(document, workflows_dir)
    sync = ModelSync(model)

    assert model.upscale.factor == pytest.approx(2.0) and type(model.upscale.factor) is float
    assert model.upscale.unblur_strength == pytest.approx(0.0)
    assert model.regions.positive == "Android generation prompt"
    assert model.edit_regions.positive == "Android edit prompt"
    assert model.regions.negative == "blurry"
    job = model.jobs.find("android-job")
    assert job is not None and job.state is JobState.finished
    assert job.params.seed == 4294967295 and job.result_was_used(0)
    assert job.results[0]._qimage.pixelColor(0, 0).alpha() == 70
    assert job.params.bounds == Bounds(11, 29, 32, 24)
    sync._save()
    saved = json.loads(bytes(document.annotation("ai_diffusion/ui.json")))
    assert type(saved["upscale"]["factor"]) is float
    assert len(saved["history"]) == 1


@qtapp
async def test_legacy_android_prompt_banks_recovered_once(workflows_dir: Path):
    document = Krita.instance().openDocument("legacy android prompt banks")
    state = {"version": 1, "upscale": {"factor": 2}, "root": {"positive": "old desktop prompt"}}
    native = {
        "schema": 1,
        "mode": "edit",
        "prompt_banks": {
            "generate": {"prompt": "Android generation", "negative": "bad generation"},
            "edit": {"prompt": "Android edit", "negative": "bad edit"},
        },
    }
    document.setAnnotation(
        "ai_diffusion/ui.json", "AI Diffusion", QByteArray(json.dumps(state).encode())
    )
    document.setAnnotation("baron-native-state", "Baron", QByteArray(json.dumps(native).encode()))
    model = _make_model(document, workflows_dir)
    sync = ModelSync(model)
    assert model.regions.positive == "Android generation"
    assert model.regions.negative == "bad generation"
    assert model.edit_regions.positive == "Android edit"
    model.regions.positive = "new desktop prompt"
    sync._save()
    reopened = _make_model(document, workflows_dir)
    restored = ModelSync(reopened)
    assert reopened.regions.positive == "new desktop prompt"
    assert reopened.edit_regions.positive == "Android edit"
    assert len(restored._history) == 0


@qtapp
async def test_history(workflows_dir: Path):
    """ModelSync encodes finished job result images into the document annotations and
    restores them (with correct images and metadata) when loaded by a new ModelSync."""

    krita_doc = Krita.instance().openDocument("test")
    model1 = _make_model(krita_doc, workflows_dir)
    sync1 = ModelSync(model1)

    # Build a finished diffusion job with two result images
    bounds = Bounds(0, 0, 512, 512)
    params = JobParams(bounds=bounds, name="history test", seed=99)
    job1 = model1.jobs.add_job(Job("history-job-1", JobKind.diffusion, params))

    img_a = Image.create(Extent(512, 512), fill=0xFFFF0000)  # red
    img_b = Image.create(Extent(512, 512), fill=0xFF0000FF)  # blue
    results = ImageCollection([img_a, img_b])
    model1.jobs.set_results(job1, results)
    model1.jobs.notify_finished(job1)

    # Wait for the async image-saving task to complete
    if sync1._image_task is not None:
        await sync1._image_task

    # Force the state JSON to be written as well
    sync1._save()

    # ── restore ───────────────────────────────────────────────────────────
    Krita.instance().setActiveDocument(krita_doc)
    doc2 = KritaDocument(krita_doc, None)
    conn2 = Connection()
    wf_coll2 = WorkflowCollection(conn2, folder=workflows_dir)
    model2 = DocumentModel(doc2, conn2, wf_coll2)
    _sync2 = ModelSync(model2)

    # The finished job should have been restored in model2's queue
    restored_job = model2.jobs.find("history-job-1")
    assert restored_job is not None, "finished job not found in restored model"
    assert restored_job.state is JobState.finished
    assert restored_job.params.name == "history test"
    assert restored_job.params.seed == 99
    assert len(restored_job.results) == 2

    # Check image content is intact
    restored_a = restored_job.results[0]
    restored_b = restored_job.results[1]
    assert restored_a.extent == Extent(512, 512)
    assert restored_b.extent == Extent(512, 512)
    # Pixel values should be close to the originals (allow for WebP compression rounding)
    assert Image.compare(restored_a, img_a) < 0.02
    assert Image.compare(restored_b, img_b) < 0.02


@qtapp
async def test_baron_archive_keeps_evicted_results(workflows_dir: Path):
    document = Krita.instance().openDocument("baron archive")
    document.setAnnotation(
        "ai_diffusion/ui.json",
        "Baron history",
        QByteArray(json.dumps({"version": 1, "baron_history_unlimited": True}).encode()),
    )
    model = _make_model(document, workflows_dir)
    sync = ModelSync(model)
    previous = settings.history_size, settings.history_storage
    settings.history_size = settings.history_storage = 0
    try:
        for index in range(2):
            params = JobParams(Bounds(12, 35, 32, 24), "character", seed=index)
            job = model.jobs.add_job(Job(f"baron-job-{index}", JobKind.diffusion, params))
            model.jobs.set_results(job, ImageCollection([Image.create(Extent(32, 24))]))
            model.jobs.notify_finished(job)
            if sync._image_task is not None:
                await sync._image_task
        assert len(sync._history) == 2
        assert model.jobs.find("baron-job-0") is None
        sync._save()
        state = json.loads(bytes(document.annotation("ai_diffusion/ui.json")))
        assert state["baron_history_unlimited"] is True
        assert len(state["history"]) == 2
        assert document.annotation("ai_diffusion/result0.webp").size() > 0
        model.jobs.clear()
        assert sync._history == []
        assert document.annotation("ai_diffusion/result0.webp").size() == 0
    finally:
        settings.history_size, settings.history_storage = previous
