import ast
import json
from pathlib import Path
from string import Formatter

import pytest

from ai_diffusion.backend.orchestrion_client import quote_text
from ai_diffusion.localization import Localization

plugin = Path(__file__).parents[1] / "ai_diffusion"
languages = sorted((plugin / "language").glob("*.json"))


def source_keys(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def format_fields(text):
    return {field for _, field, _, _ in Formatter().parse(text) if field}


@pytest.mark.parametrize("language_file", languages, ids=lambda p: p.stem)
def test_new_workspaces_have_translations_and_preserve_parameters(language_file):
    keys = set()
    for path in [
        "backend/orchestrion_client.py",
        "backend/background.py",
        "model/background.py",
        "ui/background.py",
        "ui/orchestrion.py",
    ]:
        keys |= source_keys(plugin / path)
    keys |= {k for k in source_keys(plugin / "backend/workflow.py") if "Krea 2" in k}
    keys |= {k for k in source_keys(plugin / "style.py") if "Krea 2" in k}
    keys |= {
        "Identity",
        "Style",
        "768 is a balanced starting point. Try 1024 for face detail, or 512 for stronger scene changes.",
        "Higher values preserve appearance, pose and background. Start with 2–6 for edits. Generation in a new scene limits this to 1 to let the composition change.",
        "Browser login, balance, prices and your website model gallery.",
        "Connect to Orchestrion",
    }
    data = json.loads(language_file.read_text(encoding="utf-8"))
    if data["id"] == "en":
        return
    for key in keys:
        translated = data["translations"].get(key)
        assert translated, f"{data['id']}: {key}"
        assert format_fields(key) == format_fields(translated), f"{data['id']}: {key}"


def test_bleatbucks_name_follows_plugin_language_and_ignores_fiat(monkeypatch):
    russian = Localization.load("ru", plugin / "language/ru.json")
    monkeypatch.setattr(Localization, "current", russian)
    assert quote_text({"coins": 7, "fiat": {"amount": 999, "currency": "USD"}}, 3) == (
        "≈ 21 беелетиков"
    )
    monkeypatch.setattr(Localization, "current", Localization())
    assert quote_text({"coins": 7}, 3) == "≈ 21 bleatbucks"
