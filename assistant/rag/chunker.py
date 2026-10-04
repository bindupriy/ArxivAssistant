"""Page- and section-aware chunker.

Chunks never cross physical PDF pages when page ranges are available. Long
section portions on one page are split with overlap within that page only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from assistant.rag.parser import ParsedPaper, Section


@dataclass
class Chunk:
    id: str
    paper_id: str
    section_type: str
    section_title: str
    order: int
    text: str
    char_start: int
    char_end: int
    page_number: int | None = None


def chunk_paper(
    paper_id: str,
    parsed: ParsedPaper,
    *,
    max_chars: int = 900,
    overlap_chars: int = 135,
) -> list[Chunk]:
    if max_chars <= 0 or not 0 <= overlap_chars < max_chars:
        raise ValueError("max_chars must be positive and overlap_chars must be smaller")
    out: list[Chunk] = []
    order = 0
    for section in parsed.sections:
        if section.section_type in {"references", "acknowledgments"}:
            continue
        ranges = (
            [
                (max(section.char_start, start), min(section.char_end, end), number)
                for number, (start, end) in enumerate(parsed.page_ranges, start=1)
                if start < section.char_end and end > section.char_start
            ]
            if parsed.page_ranges else [(section.char_start, section.char_end, None)]
        )
        for start, end, page_number in ranges:
            raw = parsed.full_text[start:end] if page_number is not None else section.text
            text = raw.strip()
            if not text:
                continue
            start += len(raw) - len(raw.lstrip())
            part = Section(section.title, section.section_type, text, start, start + len(text))
            for piece, (cs, ce) in _split(part, max_chars, overlap_chars):
                out.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        paper_id=paper_id,
                        section_type=section.section_type,
                        section_title=section.title,
                        order=order,
                        text=piece,
                        char_start=cs,
                        char_end=ce,
                        page_number=page_number,
                    )
                )
                order += 1
    return out


def _split(section: Section, max_chars: int, overlap: int) -> list[tuple[str, tuple[int, int]]]:
    text = section.text
    if len(text) <= max_chars:
        return [(text, (section.char_start, section.char_start + len(text)))]
    pieces: list[tuple[str, tuple[int, int]]] = []
    step = max_chars - overlap
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        piece = text[i:end]
        pieces.append((piece, (section.char_start + i, section.char_start + end)))
        if end == len(text):
            break
        i += step
    return pieces
