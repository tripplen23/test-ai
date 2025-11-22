"""Tests for data cleaning utilities."""

from __future__ import annotations

from agentic_rag.data.types import RawRecord


def test_remove_html_tags() -> None:
    """Test remove HTML tags từ text."""
    from agentic_rag.data.cleaning import remove_html_tags

    text = "<p>This is <b>bold</b> text</p>"
    result = remove_html_tags(text)
    assert "<" not in result
    assert ">" not in result
    assert "bold" in result
    assert "This is" in result
    assert "This is bold text" == result


def test_normalize_whitespace() -> None:
    """Test smart normalize whitespace (multiple spaces → single)."""
    from agentic_rag.data.cleaning import smart_normalize_whitespace

    text = "This   has    multiple     spaces"
    result = smart_normalize_whitespace(text)
    assert "  " not in result  # No double spaces
    assert "This has multiple spaces" == result


def test_handle_encoding_issues() -> None:
    """Test handle encoding errors."""
    from agentic_rag.data.cleaning import clean_text

    # Test with text that might have encoding issues
    text = "<html>Test text with   special   chars:   àáâãäå</html>"
    result = clean_text(text)
    assert len(result) > 0
    assert "Test text with special chars: àáâãäå" == result


def test_clean_empty_text() -> None:
    """Test handle empty text after cleaning."""
    from agentic_rag.data.cleaning import clean_text

    # Empty string
    assert clean_text("") == ""
    # Only whitespace
    assert clean_text("   \n\t  ") == ""
    # Only HTML tags
    assert clean_text("<p></p><div></div>") == ""


def test_clean_preserves_structure() -> None:
    """Test preserve text structure."""
    from agentic_rag.data.cleaning import clean_text

    text = "First paragraph.\n\nSecond paragraph."
    result = clean_text(text)
    # Should preserve paragraph structure
    assert "First paragraph" in result
    assert "Second paragraph" in result


def test_validate_record() -> None:
    """Test validate RawRecord (skip invalid)."""
    from agentic_rag.data.cleaning import validate_record

    # Valid record
    valid_record = RawRecord(
        identifier="doc1",
        title="Test Title",
        body="Test content",
    )
    assert validate_record(valid_record) is True

    # Invalid: empty identifier
    invalid_record1 = RawRecord(identifier="", title="Test", body="Test")
    assert validate_record(invalid_record1) is False

    # Invalid: empty body
    invalid_record2 = RawRecord(identifier="doc1", title="Test", body="")
    assert validate_record(invalid_record2) is False

    # Invalid: body becomes empty after cleaning
    invalid_record3 = RawRecord(identifier="doc1", title="Test", body="   ")
    assert validate_record(invalid_record3) is False

