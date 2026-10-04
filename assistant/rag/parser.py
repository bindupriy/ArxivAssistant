"""PDF -> structured sections.

MVP uses PyMuPDF + a heuristic section detector. Marker can be plugged in
later as a higher-quality parser by replacing `parse_pdf()` — the rest of the
pipeline only cares about the returned `ParsedPaper` shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Common section headers we want to recognize in ML/CS papers.
_SECTION_KEYWORDS = {
    "abstract": "abstract",
    "introduction": "intro",
    "background": "background",
    "related work": "related",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "experiment": "experiments",
    "experiments": "experiments",
    "evaluation": "experiments",
    "results": "results",
    "discussion": "discussion",
    "analysis": "analysis",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "limitations",
    "references": "references",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "appendix": "appendix",
}

_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?([A-Za-z][A-Za-z \-/&]{2,60})\s*$"
)


@dataclass
class Section:
    title: str
    section_type: str  # one of _SECTION_KEYWORDS' values, or 'body'
    text: str
    char_start: int
    char_end: int


@dataclass
class ParsedPaper:
    full_text: str
    sections: list[Section]
    abstract: str | None
    page_ranges: list[tuple[int, int]] = field(default_factory=list)  # 0-based text offsets, physical page order


def parse_pdf(pdf_path: Path) -> ParsedPaper:
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]
    ranges: list[tuple[int, int]] = []
    offset = 0
    for text in pages:
        ranges.append((offset, offset + len(text)))
        offset += len(text) + 1  # separator used by join
    return _structure("\n".join(pages), ranges)


def _classify(line: str) -> str | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    key = m.group(1).strip().lower()
    return _SECTION_KEYWORDS.get(key)


def _structure(full: str, page_ranges: list[tuple[int, int]] | None = None) -> ParsedPaper:
    lines = full.splitlines(keepends=True)
    sections: list[Section] = []
    cur_title = "Header"
    cur_type = "header"
    cur_start = 0
    cur_lines: list[str] = []
    cursor = 0

    def flush() -> None:
        if not cur_lines:
            return
        raw_text = "".join(cur_lines)
        text = raw_text.strip()
        if text:
            start = cur_start + len(raw_text) - len(raw_text.lstrip())
            sections.append(
                Section(
                    title=cur_title,
                    section_type=cur_type,
                    text=text,
                    char_start=start,
                    char_end=start + len(text),
                )
            )

    for raw in lines:
        sec_type = _classify(raw.strip()) if raw.strip() else None
        if sec_type is not None:
            flush()
            cur_title = raw.strip()
            cur_type = sec_type
            cur_start = cursor + len(raw)
            cur_lines = []
        else:
            cur_lines.append(raw)
        cursor += len(raw)
    flush()

    abstract = None
    for s in sections:
        if s.section_type == "abstract":
            abstract = s.text
            break
    return ParsedPaper(full_text=full, sections=sections, abstract=abstract, page_ranges=page_ranges or [])
