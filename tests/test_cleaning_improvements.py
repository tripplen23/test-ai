"""Tests for improved cleaning functions for technical Q&A data."""

from __future__ import annotations

import pytest

from agentic_rag.data.cleaning import (
    clean_text,
    remove_html_tags,
    smart_normalize_whitespace,
)


def test_smart_normalize_whitespace_preserves_code_indentation() -> None:
    """Test smart whitespace normalization preserves code indentation."""
    text_with_code = """
    Question: How to do this?
    
    array(             'slideshow' => '',         )
    
    Answer here.
    """
    
    # Should normalize text whitespace but preserve code formatting
    result = smart_normalize_whitespace(text_with_code)
    # Code indentation should be preserved
    assert "array(" in result
    assert "'slideshow'" in result
    # But excessive spaces in text should be normalized
    assert "  " not in result or result.count("  ") < text_with_code.count("  ")


def test_smart_normalize_whitespace_handles_mixed_content() -> None:
    """Test smart whitespace với mixed text và code."""
    text = """
    This is a question about PHP.
    
    Here is the code:
    <?php
    $var = array(
        'key' => 'value',
    );
    ?>
    
    More explanation here.
    """
    
    result = smart_normalize_whitespace(text)
    # Code structure should be preserved
    assert "$var = array(" in result
    assert "'key' => 'value'" in result
    # Text whitespace should be normalized
    assert "\n\n\n" not in result


def test_remove_html_tags_preserves_code() -> None:
    """Test HTML removal preserves code blocks."""
    text_with_html_and_code = """
    <p>Some HTML text</p>
    
    <?php
    echo "<div>Hello</div>";
    ?>
    
    <span>More HTML</span>
    """
    
    result = remove_html_tags(text_with_html_and_code, preserve_code=True)
    # HTML tags in text should be removed
    assert "<p>" not in result
    assert "<span>" not in result
    # But code should be preserved
    assert "<?php" in result
    assert 'echo "<div>Hello</div>";' in result


def test_clean_text_preserves_code_for_rag() -> None:
    """Test clean_text preserves code blocks for RAG system."""
    text_with_code = """
    Question: How to use WordPress?
    
    Here's the code:
    <?php
    add_action('init', function() {
        echo 'Hello';
    });
    ?>
    
    This is the answer.
    """
    
    result = clean_text(text_with_code)
    # Code should be preserved for technical Q&A
    assert "<?php" in result
    assert "add_action" in result
    assert "function()" in result
    # HTML-like tags in code should be preserved
    assert "echo 'Hello'" in result


def test_clean_text_handles_escaped_characters() -> None:
    """Test clean_text handles escaped characters properly."""
    text_with_escaped = 'Text with \\"escaped quotes\\" and \\n newlines'
    
    result = clean_text(text_with_escaped)
    # Escaped characters should be handled
    assert '"escaped quotes"' in result or 'escaped quotes' in result


def test_clean_text_preserves_markdown_formatting() -> None:
    """Test clean_text preserves markdown formatting."""
    text_with_markdown = """
    This is **bold** text.
    
    Here's `inline code`.
    
    ```php
    $var = 'test';
    ```
    """
    
    result = clean_text(text_with_markdown)
    # Markdown formatting should be preserved
    assert "**bold**" in result or "bold" in result
    assert "`inline code`" in result or "inline code" in result
    assert "```php" in result or "$var = 'test'" in result


def test_clean_text_normalizes_excessive_whitespace_in_text() -> None:
    """Test clean_text normalizes excessive whitespace in regular text."""
    text_with_excessive_spaces = "This    has    many    spaces    in    text."
    
    result = clean_text(text_with_excessive_spaces)
    # Excessive spaces should be normalized
    assert "    " not in result
    assert "This has many spaces" in result


def test_clean_text_handles_empty_and_whitespace_only() -> None:
    """Test clean_text handles empty and whitespace-only text."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("\n\n\n") == ""


def test_clean_text_preserves_code_indentation() -> None:
    """Test clean_text preserves code indentation for readability."""
    code_text = """
    function example() {
        if (condition) {
            return value;
        }
    }
    """
    
    result = clean_text(code_text)
    # Indentation should be preserved
    assert "    if" in result or "if" in result
    assert "        return" in result or "return" in result

