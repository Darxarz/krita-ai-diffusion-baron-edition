import json
import shutil
from pathlib import Path

from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QFontDatabase

from .util import client_logger as log

_font_ids: dict[Path, int] = {}


def _bundle_directory():
    directory = Path(__file__).parent / "fonts"
    bundled = directory / "bundled"
    return bundled if (bundled / "manifest.json").exists() else directory


def install_krita_fonts(resource_directory: Path):
    bundle = _bundle_directory()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("public_bundle") or manifest.get("personal_bundle"):
            raise ValueError("Expected the reviewed public font bundle")
        destination = resource_directory / "fonts" / "BaronEdition"
        destination.mkdir(parents=True, exist_ok=True)
        for row in manifest["files"]:
            name = row["file"]
            if Path(name).name != name:
                raise ValueError("Invalid font filename")
            target = destination / name
            if not target.exists():
                shutil.copyfile(bundle / name, target)
        shutil.copytree(bundle / "licenses", destination / "licenses", dirs_exist_ok=True)
        shutil.copyfile(manifest_path, destination / "manifest.json")
    except (OSError, ValueError, KeyError) as error:
        log.warning(f"Could not install bundled fonts in the Krita resource folder: {error}")


def load_fonts(directory: Path | None = None) -> tuple[str, ...]:
    directory = (directory or _bundle_directory()).resolve()
    families: set[str] = set()
    for path in sorted(directory.glob("*.ttf")):
        if path not in _font_ids:
            font_id = QFontDatabase.addApplicationFontFromData(QByteArray(path.read_bytes()))
            if font_id < 0:
                log.warning(f"Could not load bundled font: {path.name}")
                continue
            _font_ids[path] = font_id
        families.update(QFontDatabase.applicationFontFamilies(_font_ids[path]))
    return tuple(sorted(families))
