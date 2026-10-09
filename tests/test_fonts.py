import json
import subprocess
import sys
from pathlib import Path

from ai_diffusion.fonts import install_krita_fonts


def test_krita_resource_fonts_preserve_existing_files(tmp_path: Path):
    destination = tmp_path / "fonts" / "BaronEdition"
    destination.mkdir(parents=True)
    custom = destination / "Creepster-Regular.ttf"
    custom.write_bytes(b"existing local font")
    install_krita_fonts(tmp_path)
    assert custom.read_bytes() == b"existing local font"
    assert len(list(destination.glob("*.ttf"))) == 96
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert all((destination / row["license_file"]).exists() for row in manifest["files"])
    install_krita_fonts(tmp_path)
    assert custom.read_bytes() == b"existing local font"


def test_bundled_fonts_register_once_and_have_reviewed_licenses():
    root = Path(__file__).parents[1]
    script = """
import hashlib
import json
from pathlib import Path
from PyQt5.QtGui import QFont, QFontDatabase, QGuiApplication, QRawFont
from ai_diffusion.fonts import load_fonts, _font_ids

app = QGuiApplication([])
folder = Path('ai_diffusion/fonts')
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
assert manifest['public_bundle'] and not manifest['personal_bundle']
families = load_fonts()
assert len(families) >= 96, families
ids = dict(_font_ids)
assert load_fonts() == families and _font_ids == ids
assert load_fonts(folder / 'missing') == ()
for row in manifest['files']:
    path = folder / row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    assert row['license'] in ('OFL-1.1', 'Apache-2.0')
    assert hashlib.sha256((folder / row['license_file']).read_bytes()).hexdigest() == row['license_sha256']
    names = QFontDatabase.applicationFontFamilies(ids[path.resolve()])
    assert names, row['family']
    if row['cyrillic']:
        font = QRawFont.fromFont(QFont(names[0]))
        assert all(font.supportsCharacter(ord(c)) for c in 'АЯаяЁё'), names
print(json.dumps({'registered': len(ids), 'families': len(families)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["registered"] == 96


def test_registered_font_does_not_lock_its_file(tmp_path: Path):
    root = Path(__file__).parents[1]
    script = """
import shutil
import sys
from pathlib import Path
from PyQt5.QtGui import QGuiApplication
from ai_diffusion.fonts import load_fonts
app = QGuiApplication([])
folder = Path(sys.argv[1])
font = folder / 'AlexBrush-Regular.ttf'
shutil.copyfile('ai_diffusion/fonts/AlexBrush-Regular.ttf', font)
assert load_fonts(folder)
font.replace(folder / 'moved.ttf')
assert not font.exists()
print('Registered font can be replaced while Qt remains active')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
