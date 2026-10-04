"""Retrieval Agent — vanilla (Phase 4) and agentic (Phase 7).

Vanilla path: hybrid retrieve -> rerank -> populate state.retrieved_chunks.
Agentic path: see Phase 7; selected by config.rag.mode.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import func, select

from assistant.config import get_config
from assistant.llm import get_router
from assistant.rag.agentic import agentic_retrieve
from assistant.rag.filters import MetadataFilters
from assistant.rag.graph_retriever import graph_retrieve
from assistant.rag.reranker import rerank
from assistant.rag.retriever import hybrid_retrieve
from assistant.state import GraphState
from assistant.storage import Domain, Paper, session_scope

log = logging.getLogger(__name__)
_ARXIV_ID = re.compile(r"\b(?:[a-z-]+/\d{7}|\d{4}\.\d{4,6})(?:v\d+)?\b", re.IGNORECASE)
_QUOTED = re.compile(r'"([^"\n]+)"')
_REWRITE_SYSTEM = """Rewrite the user's question into one self-contained search query for a research-paper index.
If conversation history is provided, resolve references such as "it", "their method", and "that paper".
For a standalone question, clarify its search terms without changing its meaning.
Preserve exact paper names, quoted phrases, technical terms, dates, numbers, and arXiv IDs.
Do not answer the question, invent new paper claims, or add a metadata restriction.
Return only one plain-text search query on a single line."""


def _send_progress(stage: str) -> None:
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "progress", "stage": stage})
    except (LookupError, RuntimeError):
        pass


def _topic_paper_ids(domain_name: str) -> list[str]:
    with session_scope() as session:
        return list(
            session.execute(
                select(Paper.id)
                .join(Domain, Paper.domain_id == Domain.id)
                .where(Domain.name == domain_name)
            ).scalars()
        )


def _standalone_question(question: str, history: list[dict]) -> str:
    cfg = get_config().rag.query_rewrite
    if not cfg.enabled or (not history and not cfg.first_turn):
        return question
    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')}")
    conversation = "\n".join(lines)
    try:
        message = get_router().chat("retrieval").invoke([
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=f"Conversation:\n{conversation or '(first turn)'}\n\nCurrent question:\n{question}"),
        ])
        rewritten = getattr(message, "content", "")
    except Exception:
        log.warning("Query rewrite unavailable; using original question.", exc_info=True)
        return question
    if not isinstance(rewritten, str):
        return question
    rewritten = rewritten.strip().strip('"').strip()
    if not rewritten or "\n" in rewritten or len(rewritten) > cfg.max_chars:
        return question
    if not {value.lower() for value in _ARXIV_ID.findall(question)}.issubset(
        {value.lower() for value in _ARXIV_ID.findall(rewritten)}
    ) or any(value not in rewritten for value in _QUOTED.findall(question)):
        return question
    return rewritten


def _eligible_papers(paper_ids: list[str] | None, filters: MetadataFilters) -> list[str] | None:
    if filters.min_year is None and filters.max_year is None and filters.venue is None:
        return paper_ids
    if paper_ids == []:
        return []
    with session_scope() as session:
        stmt = select(Paper.id)
        if paper_ids is not None:
            stmt = stmt.where(Paper.id.in_(paper_ids))
        if filters.min_year is not None:
            stmt = stmt.where(Paper.year >= filters.min_year)
        if filters.max_year is not None:
            stmt = stmt.where(Paper.year <= filters.max_year)
        if filters.venue is not None:
            stmt = stmt.where(func.lower(Paper.venue) == filters.venue.lower())
        return list(session.execute(stmt).scalars())


def retrieval_node(state: GraphState) -> GraphState:
    cfg = get_config()
    question = state.get("question", "").strip()
    if not question:
        return {"retrieved_chunks": []}

    _send_progress("retrieving")
    filters = MetadataFilters.model_validate((state.get("scratch") or {}).get("metadata_filters") or {})
    paper_ids = state.get("paper_ids")
    if paper_ids is None and state.get("domain"):
        paper_ids = _topic_paper_ids(str(state["domain"]))
    paper_ids = _eligible_papers(paper_ids, filters)
    if paper_ids == []:
        return {"retrieved_chunks": []}

    history = list((state.get("scratch") or {}).get("history") or [])
    search_question = _standalone_question(question, history[-cfg.rag.history_turns:])
    metadata_filter = {"section_type": filters.section_types} if filters.section_types else None

    if cfg.rag.mode == "agentic":
        top = agentic_retrieve(search_question, top_k=cfg.rag.top_k, paper_ids=paper_ids, metadata_filter=metadata_filter)
    elif cfg.rag.graph.enabled:
        top = graph_retrieve(search_question, top_k=cfg.rag.top_k, paper_ids=paper_ids, metadata_filter=metadata_filter)
    else:
        candidates = hybrid_retrieve(
            search_question,
            top_k=max(cfg.rag.top_k * 3, 20),
            paper_ids=paper_ids,
            metadata_filter=metadata_filter,
        )
        top = rerank(search_question, candidates, top_k=cfg.rag.top_k)
    return {
        "retrieved_chunks": [
            {
                "id": c.id,
                "paper_id": c.paper_id,
                "text": c.text,
                "section_type": c.section_type,
                "section_title": c.section_title,
                "score": c.score,
                "page_number": c.page_number,
            }
            for c in top
        ]
    }
