"""Tests for the NLP preprocessing layer."""

from src.preprocessing import PREPROCESSOR_VERSION, clean_text, tokenize


def test_strips_urls_and_emails():
    text = "Read more at https://example.com/x or mail a@b.com today"
    cleaned = clean_text(text)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "a@b.com" not in cleaned
    assert "url" in cleaned
    assert "email" in cleaned


def test_strips_html_and_scripts():
    text = "<p>Hello <b>world</b></p><script>alert(1)</script>"
    cleaned = clean_text(text)
    assert "<p>" not in cleaned
    assert "alert" not in cleaned
    assert "hello world" in cleaned


def test_lowercases_and_collapses_whitespace():
    assert clean_text("  HUGE   News  STORY  ") == "huge news story"


def test_handles_none_and_non_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == "12345"


def test_entities_are_unescaped():
    # HTML entities are decoded first, then punctuation is stripped.
    cleaned = clean_text("rock &amp; roll <b>tonight</b>")
    assert "amp" not in cleaned
    assert "rock roll tonight" in cleaned


def test_tokenize_removes_stopwords():
    tokens = tokenize("The cat is sitting on the mat")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "cat" in tokens
    assert "mat" in tokens


def test_tokenize_can_keep_stopwords():
    tokens = tokenize("the cat is here", remove_stopwords=False)
    assert "the" in tokens


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_preprocessor_version_is_exposed():
    assert isinstance(PREPROCESSOR_VERSION, str) and PREPROCESSOR_VERSION
