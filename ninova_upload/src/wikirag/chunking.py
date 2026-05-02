from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into fixed-size overlapping chunks.

    The implementation avoids loading windows recursively, so it can handle
    large documents with predictable memory use.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        candidate = cleaned[start:end]

        if end < text_length:
            split_at = max(candidate.rfind("\n\n"), candidate.rfind(". "), candidate.rfind("; "))
            if split_at > chunk_size * 0.55:
                end = start + split_at + 1
                candidate = cleaned[start:end]

        chunk = candidate.strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(0, end - overlap)

    return chunks

