"""arxiv metadata + PDF fetch.

Thin wrapper over the `arxiv` package. Returns a dataclass we feed into the
ingestion pipeline. Phase 6 will route arxiv access through MCP; this module
remains the direct-API fallback (and is what Phase 3 tests against).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# arxiv IDs: legacy (cs/0501001) or new (2401.12345 / 2401.12345v1)
ARXIV_ID_RE = re.compile(r"^(?:[a-z\-]+/\d{7}|\d{4}\.\d{4,6})(?:v\d+)?$", re.IGNORECASE)


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_path: Path
    venue: str | None = None
    year: int | None = None
    categories: list[str] = field(default_factory=list)


def is_arxiv_id(s: str) -> bool:
    return bool(ARXIV_ID_RE.match(s.strip()))


def search_papers(query: str, limit: int = 8) -> list[dict]:
    """Preview matching arXiv papers without downloading their PDFs."""
    import arxiv

    query = query.strip()
    if is_arxiv_id(query):
        search = arxiv.Search(id_list=[query])
    else:
        tokens = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9-]{2,}", query)))[:8]
        if not tokens:
            return []
        search = arxiv.Search(
            query=" OR ".join(f"all:{term}" for term in tokens),
            max_results=30,
            sort_by=arxiv.SortCriterion.Relevance,
        )
    matches = []
    for result in arxiv.Client(page_size=30, num_retries=2).results(search):
        matches.append({
            "arxiv_id": result.get_short_id(),
            "title": result.title.strip(),
            "authors": [author.name for author in result.authors],
            "abstract": result.summary.strip(),
            "year": result.published.year if result.published else None,
        })
    if not is_arxiv_id(query):
        terms = {
            token.lower()[:-1] if len(token) > 4 and token.lower().endswith("s")
            and not token.lower().endswith("is") else token.lower()
            for token in tokens
        }
        matches.sort(
            key=lambda paper: (
                -sum(term in paper["title"].lower() for term in terms) * 2
                - sum(term in paper["abstract"].lower() for term in terms),
                paper["arxiv_id"],
            )
        )
    return matches[:limit]


def _download_pdf(url: str, target: Path) -> None:
    import httpx

    partial = target.with_suffix(f"{target.suffix}.part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                first = True
                for chunk in response.iter_bytes():
                    if first and not chunk.startswith(b"%PDF-"):
                        raise ValueError(f"arxiv returned non-PDF content for {url}")
                    first = False
                    output.write(chunk)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def fetch(arxiv_id: str, out_dir: Path) -> ArxivPaper:
    """Download a paper's PDF + metadata. Idempotent — re-uses existing file."""
    import arxiv  # lazy import — keeps CLI startup fast

    out_dir.mkdir(parents=True, exist_ok=True)
    search = arxiv.Search(id_list=[arxiv_id])
    client = arxiv.Client()
    result = next(client.results(search), None)
    if result is None:
        raise ValueError(f"arxiv ID not found: {arxiv_id}")

    safe_id = result.get_short_id().replace("/", "_")
    pdf_path = out_dir / f"{safe_id}.pdf"
    if not pdf_path.exists():
        _download_pdf(result.pdf_url, pdf_path)

    return ArxivPaper(
        arxiv_id=result.get_short_id(),
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip(),
        pdf_path=pdf_path,
        year=result.published.year if result.published else None,
        categories=list(result.categories or []),
    )
