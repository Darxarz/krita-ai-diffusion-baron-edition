import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from ai_diffusion.model import baron_updates
from ai_diffusion.model.updates import AutoUpdate, UpdateState


def archive(version="1.53.0-baron.6", extra=None):
    data = io.BytesIO()
    with ZipFile(data, "w") as z:
        z.writestr("ai_diffusion.desktop", "desktop entry")
        z.writestr("ai_diffusion/__init__.py", f'__version__ = "{version}"')
        z.writestr("ai_diffusion/updated.py", "new code")
        if extra:
            member = ZipInfo("placeholder")
            member.filename = extra[0]
            z.writestr(member, extra[1])
    return data.getvalue()


def manifest(data):
    return {
        "schema": 1,
        "edition": "baron",
        "channel": "stable",
        "windows": {
            "version": "1.53.0-baron.6",
            "url": "https://orchestrion.ink/baron-updates/releases/plugin.zip",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    }


class Requests:
    def __init__(self, data):
        self.data = data
        self.manifest = manifest(data)

    async def get(self, url, timeout):
        return self.manifest

    async def download(self, url, timeout, max_bytes):
        assert max_bytes == baron_updates.max_package_bytes
        return self.data


@pytest.mark.asyncio
async def test_update_keeps_settings_and_rolls_out_only_baron_files(tmp_path):
    root = tmp_path / "install"
    (root / "ai_diffusion/.appdata").mkdir(parents=True)
    (root / "ai_diffusion/__init__.py").write_text("old code")
    custom = root / "ai_diffusion/.appdata/settings.json"
    custom.write_text(json.dumps({"credential": "private", "theme": "dark"}))
    updater = AutoUpdate(plugin_dir=root, current_version="1.53.0-baron.5")
    updater._request_manager = Requests(archive())  # type: ignore
    await updater._check()
    assert updater.state is UpdateState.available
    await updater._run()
    assert updater.state is UpdateState.restart_required
    assert (root / "ai_diffusion/updated.py").read_text() == "new code"
    assert json.loads(custom.read_text())["credential"] == "private"
    assert next((root / ".baron-update-backups").rglob("__init__.py")).read_text() == "old code"


@pytest.mark.asyncio
async def test_corrupt_download_never_replaces_plugin(tmp_path):
    root = tmp_path / "install"
    root.mkdir()
    updater = AutoUpdate(plugin_dir=root, current_version="1.53.0-baron.5")
    requests = Requests(archive())
    updater._request_manager = requests  # type: ignore
    await updater._check()
    requests.data = b"corrupt"
    with pytest.raises(RuntimeError, match="corrupted"):
        await updater._run()
    assert not (root / "ai_diffusion").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "https://evil.example/baron-updates/releases/plugin.zip"),
        ("url", "http://orchestrion.ink/baron-updates/releases/plugin.zip"),
        ("version", "1.53.0"),
        ("bytes", -1),
        ("sha256", "broken"),
    ],
)
def test_update_metadata_rejects_wrong_channel_and_package(field, value):
    m = manifest(archive())
    m["windows"][field] = value
    with pytest.raises(ValueError):
        baron_updates.package_info(m, "1.53.0-baron.5", baron_updates.feed_url)


def test_update_does_not_downgrade():
    assert (
        baron_updates.package_info(manifest(archive()), "1.53.0-baron.8", baron_updates.feed_url)
        is None
    )


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "ai_diffusion/../../outside",
        "ai_diffusion/.appdata/settings.json",
        "ai_diffusion/NUL.py",
        "ai_diffusion\\outside.py",
    ],
)
def test_malicious_archive_is_rejected_before_install(path, tmp_path):
    p = tmp_path / "package.zip"
    p.write_bytes(archive(extra=(path, "untrusted")))
    with pytest.raises(ValueError):
        baron_updates.install_package(p, tmp_path / "install", "1.53.0-baron.6")
    assert not (tmp_path / "install").exists()


def test_write_failure_restores_previous_files(monkeypatch, tmp_path):
    p = tmp_path / "package.zip"
    p.write_bytes(archive())
    root = tmp_path / "install"
    (root / "ai_diffusion").mkdir(parents=True)
    previous = root / "ai_diffusion/__init__.py"
    previous.write_text("previous")
    real_replace = baron_updates.os.replace

    def replace(source, target):
        if Path(target).name == "updated.py":
            raise OSError("Simulated write failure")
        return real_replace(source, target)

    monkeypatch.setattr(baron_updates.os, "replace", replace)
    with pytest.raises(OSError):
        baron_updates.install_package(p, root, "1.53.0-baron.6")
    assert previous.read_text() == "previous"
    assert not (root / "ai_diffusion/updated.py").exists()


def test_identical_locked_font_is_not_replaced(monkeypatch, tmp_path):
    font_bytes = b"unchanged registered font"
    p = tmp_path / "package.zip"
    p.write_bytes(archive(extra=("ai_diffusion/fonts/AlexBrush-Regular.ttf", font_bytes)))
    root = tmp_path / "install"
    target = root / "ai_diffusion/fonts/AlexBrush-Regular.ttf"
    target.parent.mkdir(parents=True)
    target.write_bytes(font_bytes)
    real_replace = baron_updates.os.replace

    def replace(source, destination):
        if Path(destination) == target:
            raise PermissionError("Windows has registered this font")
        return real_replace(source, destination)

    monkeypatch.setattr(baron_updates.os, "replace", replace)
    baron_updates.install_package(p, root, "1.53.0-baron.6")
    assert target.read_bytes() == font_bytes
    assert (root / "ai_diffusion/updated.py").read_text() == "new code"
    assert not list(root.rglob("*.baron-pending-*"))
