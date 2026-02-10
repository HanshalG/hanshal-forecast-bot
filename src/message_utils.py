from __future__ import annotations

from typing import Any


def _extract_text_from_block(block: Any) -> str:
    """Extract plain text from a single content block."""
    if isinstance(block, str):
        return block

    if isinstance(block, dict):
        block_type = str(block.get("type", "")).lower()
        if block_type in {"text", "output_text", "input_text"}:
            text = block.get("text")
            return text if isinstance(text, str) else ""

        text = block.get("text")
        if isinstance(text, str):
            return text

        content = block.get("content")
        if isinstance(content, str):
            return content
        if content is not None:
            return content_to_text(content)
        return ""

    text_attr = getattr(block, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(block, "content", None)
    if content_attr is not None:
        return content_to_text(content_attr)

    return ""


def content_to_text(content: Any) -> str:
    """Normalize model content into plain text, ignoring reasoning-only blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        parts = [_extract_text_from_block(part).strip() for part in content]
        parts = [part for part in parts if part]
        return "\n\n".join(parts)

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr

    return _extract_text_from_block(content).strip() or str(content)


def message_to_text(message: Any) -> str:
    """Extract plain text from a LangChain/OpenAI message object."""
    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr
    return content_to_text(getattr(message, "content", message))
