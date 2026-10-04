"""One-hop citation-graph expansion of the existing BM25 + embedding retriever.

Only explicit arXiv references create edges. The graph never overrides the
paper/topic scope or the QA node's quote-and-verifier citation policy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from sqlalchemy import or_, select

from assistant.config import get_config
from assistant.rag.reranker import rerank
from assistant.rag.retriever import RetrievedChunk, hybrid_retrieve
from assistant.storage import Paper, PaperCitation, session_scope

_REFERENCE = re.compile(
    r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)"
    r"(?P<id>(?:[a-z-]+/\d{7}|\d{4}\.\d{4,6})(?:v\d+)?)(?!\d)",
    re.IGNORECASE,
)
_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def _base_id(identifier: str) -> str:
    return _VERSION.sub("", identifier.strip()).lower()


def extract_arxiv_references(text: str, source_id: str) -> set[str]:
    """Only link papers with an explicit arXiv identifier in the PDF text."""
    source = _base_id(source_id)
    return {
        identifier
        for match in _REFERENCE.finditer(text)
        if (identifier := _base_id(match.group("id"))) != source
    }


def related_papers(
    seeds: list[str], paper_ids: list[str] | None, max_neighbors: int
) -> list[str]:
    if not seeds or paper_ids == []:
        return []
    with session_scope() as session:
        rows = session.execute(select(Paper.id, Paper.arxiv_id)).all()
        allowed = set(paper_ids) if paper_ids is not None else {paper_id for paper_id, _ in rows}
        by_arxiv = {
            _base_id(arxiv_id): paper_id for paper_id, arxiv_id in rows
            if paper_id in allowed and arxiv_id
        }
        seed_arxiv = {
            _base_id(arxiv_id): paper_id for paper_id, arxiv_id in rows
            if paper_id in seeds and arxiv_id
        }
        links = session.execute(
            select(PaperCitation.source_id, PaperCitation.target_arxiv_id).where(
                or_(
                    PaperCitation.source_id.in_(seeds),
                    PaperCitation.target_arxiv_id.in_(seed_arxiv),
                )
            )
        ).all()

    seed_rank = {paper_id: 1 / (rank + 1) for rank, paper_id in enumerate(seeds)}
    scores: dict[str, float] = defaultdict(float)
    for source, target_arxiv in links:
        target = by_arxiv.get(target_arxiv)
        if source in seed_rank and target in allowed and target not in seed_rank:
            scores[target] += seed_rank[source]
        if target_arxiv in seed_arxiv and source in allowed and source not in seed_rank:
            scores[source] += seed_rank[seed_arxiv[target_arxiv]]
    return sorted(scores, key=lambda paper_id: (-scores[paper_id], paper_id))[:max_neighbors]


def graph_retrieve(
    query: str, *, top_k: int, paper_ids: list[str] | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Reserve a few answer-context slots for passages from linked papers."""
    cfg = get_config().rag.graph
    baseline = hybrid_retrieve(query, top_k=max(20, top_k * 3), paper_ids=paper_ids, metadata_filter=metadata_filter)
    if not baseline or cfg.graph_slots == 0:
        return baseline[:top_k]
    seeds = list(dict.fromkeys(chunk.paper_id for chunk in baseline[:max(top_k, 8)]))
    neighbors = related_papers(seeds, paper_ids, cfg.max_neighbors)
    if not neighbors:
        return baseline[:top_k]
    from_graph = hybrid_retrieve(query, top_k=max(top_k * 2, 8), paper_ids=neighbors, metadata_filter=metadata_filter)
    top_ids = {chunk.id for chunk in baseline[:top_k]}
    novel = [chunk for chunk in from_graph if chunk.id not in top_ids]
    slots = min(cfg.graph_slots, len(novel), max(0, top_k - 1))
    if not slots:
        return baseline[:top_k]
    # On systems without an optional reranker, its fallback retains this order.
    promoted_ids = {chunk.id for chunk in novel[:slots]}
    candidates = baseline[:top_k - slots] + novel[:slots]
    candidates.extend(chunk for chunk in baseline[top_k - slots:] if chunk.id not in promoted_ids)
    seen_ids = {chunk.id for chunk in candidates}
    for chunk in novel[slots:]:
        if chunk.id not in seen_ids:
            candidates.append(chunk)
            seen_ids.add(chunk.id)
    return rerank(query, candidates, top_k=top_k)