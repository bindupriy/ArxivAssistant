"""Extract contiguous evidence windows for the QA prompt, keeping originals intact."""

from __future__ import annotations

import re

_TERMS = re.compile(r"[\w-]{3,}")
_STOP = {"about", "after", "among", "and", "are", "does", "for", "from", "how", "its",
         "paper", "papers", "that", "the", "their", "these", "this", "what", "which", "with"}


def _excerpt(text: str, query: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    terms = {word.lower() for word in _TERMS.findall(query)} - _STOP
    starts = {0, len(text) - limit}
    for term in terms:
        for match in re.finditer(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE):
            starts.add(min(max(0, match.start() - limit // 3), len(text) - limit))
    start = max(
        starts,
        key=lambda offset: (
            len(terms & set(word.lower() for word in _TERMS.findall(text[offset:offset + limit]))),
            -offset,
        ),
    )
    return text[start:start + limit]


def compress_context(
    chunks: list[dict], query: str, *, max_chars_per_chunk: int, max_total_chars: int
) -> list[dict]:
    """Add a prompt-only `context` substring; never replace the full `text`."""
    remaining = max_total_chars
    result: list[dict] = []
    for index, chunk in enumerate(chunks):
        budget = min(max_chars_per_chunk, remaining // (len(chunks) - index))
        snippet = _excerpt(str(chunk.get("text", "")), query, budget) if budget else ""
        result.append({**chunk, "context": snippet})
        remaining -= len(snippet)
    return result