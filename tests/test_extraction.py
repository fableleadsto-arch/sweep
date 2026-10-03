from app.scraping import markdown


def test_short_pages_fall_back_when_main_extractor_returns_nothing(monkeypatch):
    class EmptyExtractor:
        def extract(self, *args, **kwargs):
            return None

    monkeypatch.setattr(markdown, "HAS_TRAFILATURA", True)
    monkeypatch.setattr(markdown, "trafilatura", EmptyExtractor(), raising=False)
    doc = markdown.html_to_markdown(
        "<h1>Short page</h1><p>Useful content.</p><script>secret_script</script>",
        "https://example.com",
    )
    assert "Useful content." in doc.text
    assert "secret_script" not in doc.text


def test_plain_div_text_is_preserved_without_model_extractor(monkeypatch):
    monkeypatch.setattr(markdown, "HAS_TRAFILATURA", False)
    doc = markdown.html_to_markdown("<div>Just some text</div>", "https://example.com")
    assert doc.text == "Just some text"


def test_markdown_character_limit_is_exact(monkeypatch):
    monkeypatch.setattr(markdown, "HAS_TRAFILATURA", False)
    doc = markdown.html_to_markdown("<p>" + "word " * 100 + "</p>", "https://example.com", max_chars=50)
    assert doc.truncated
    assert len(doc.markdown) <= 50 and len(doc.text) <= 50
