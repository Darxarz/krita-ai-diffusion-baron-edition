from scripts.translation import extract_source_strings


def test_translation_extraction_preserves_escaped_and_multiline_strings(tmp_path):
    source = tmp_path / "widget.py"
    source.write_text(
        'label = _("\\nTrigger words: ")\n'
        'hint = _("First sentence. "\n"Second sentence.")\n'
        'other = render("Untranslated text")\n',
        encoding="utf-8",
    )
    assert extract_source_strings(source) == {
        "\nTrigger words: ",
        "First sentence. Second sentence.",
    }
