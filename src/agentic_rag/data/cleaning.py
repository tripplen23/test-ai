"""Data cleaning utilities for text processing."""

from __future__ import annotations

import re
from html import unescape

from agentic_rag.data.types import RawRecord

# Patterns for detecting code blocks
PHP_CODE_PATTERN = re.compile(r"<\?php.*?\?>", re.DOTALL)
HTML_CODE_PATTERN = re.compile(r"<[a-zA-Z][^>]*>.*?</[a-zA-Z]+>", re.DOTALL)
MARKDOWN_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

_PLACEHOLDER_TEMPLATE = "___CODE_BLOCK_{}___"


def _protect_code_blocks(text: str, include_inline: bool = True) -> tuple[str, list[tuple[str, str]]]:
    """
    Extract code blocks and replace with placeholders.
    
    Args:
        text: Input text containing code blocks
        include_inline: If True, also protect inline code blocks
        
    Returns:
        Tuple of (text with placeholders, list of (placeholder, original_code))
    """
    code_blocks: list[tuple[str, str]] = []
    
    def protect(match: re.Match[str]) -> str:
        placeholder = _PLACEHOLDER_TEMPLATE.format(len(code_blocks))
        code_blocks.append((placeholder, match.group(0)))
        return placeholder
    
    # Protect PHP code blocks
    text = PHP_CODE_PATTERN.sub(protect, text)
    
    # Protect markdown code blocks
    text = MARKDOWN_CODE_BLOCK.sub(protect, text)
    
    # Protect inline code if requested
    if include_inline:
        text = INLINE_CODE_PATTERN.sub(protect, text)
    
    return text, code_blocks


def _restore_code_blocks(text: str, code_blocks: list[tuple[str, str]]) -> str:
    """
    Restore code blocks from placeholders.
    
    Args:
        text: Text with placeholders
        code_blocks: List of (placeholder, original_code) tuples
        
    Returns:
        Text with code blocks restored
    """
    for placeholder, code in code_blocks:
        text = text.replace(placeholder, code)
    return text


def remove_html_tags(text: str, preserve_code: bool = False) -> str:
    """
    Remove HTML tags from text, optionally preserving code blocks.
    
    Args:
        text: Input text potentially containing HTML tags
        preserve_code: If True, preserve HTML-like tags within code blocks
        
    Returns:
        Text with HTML tags removed
    """
    if preserve_code:
        # Protect code blocks before removing HTML
        text, code_blocks = _protect_code_blocks(text, include_inline=False)
        
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        
        # Restore code blocks
        text = _restore_code_blocks(text, code_blocks)
    else:
        # Simple removal without code preservation
        text = re.sub(r"<[^>]+>", "", text)
    
    # Decode HTML entities
    text = unescape(text)
    return text


def smart_normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving code block formatting.
    
    This function detects code blocks and preserves their indentation
    while normalizing whitespace in regular text.
    
    Args:
        text: Input text with potentially multiple whitespace characters
        
    Returns:
        Text with normalized whitespace, but code formatting preserved
    """
    if not text:
        return ""
    
    # Protect code blocks before normalizing (include inline code)
    text, code_blocks = _protect_code_blocks(text, include_inline=True)
    
    # Normalize whitespace in text (not in code blocks)
    # Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)
    # Replace multiple newlines with double newline (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple tabs with single space
    text = re.sub(r"\t+", " ", text)
    
    # Restore code blocks
    text = _restore_code_blocks(text, code_blocks)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def clean_text(text: str) -> str:
    """
    Combine all cleaning steps with smart handling for technical Q&A data.
    
    This function preserves code blocks and markdown formatting while
    cleaning HTML and normalizing whitespace in regular text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with code blocks preserved
    """
    if not text:
        return ""
    
    # Remove HTML tags while preserving code blocks
    text = remove_html_tags(text, preserve_code=True)
    
    # Smart normalize whitespace (preserves code formatting)
    text = smart_normalize_whitespace(text)
    
    return text


def validate_record(record: RawRecord) -> bool:
    """
    Validate RawRecord (skip invalid).
    
    Args:
        record: RawRecord to validate
        
    Returns:
        True if record is valid, False otherwise
    """
    # Check required fields
    if not record.identifier or not record.identifier.strip():
        return False
    
    if not record.title or not record.title.strip():
        return False
    
    if not record.body or not record.body.strip():
        return False
    
    # Clean and check if text becomes empty
    cleaned_body = clean_text(record.body)
    if not cleaned_body:
        return False
    
    return True