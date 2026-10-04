"""Agentic RAG — decompose the question, retrieve iteratively, critique gaps.

Enabled via config.rag.mode = "agentic". The loop is bounded by
config.rag.agentic.max_iters. Returns reranked chunks against the *original*
question (not the sub-queries) so the answer prompt sees the most relevant
material first.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.config import get_config
from assistant.llm import get_router
from assistant.rag.graph_retriever import graph_retrieve
from assistant.rag.reranker import rerank
from assistant.rag.retriever import RetrievedChunk, hybrid_retrieve

log = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = """Split the user's research question into 2-4 focused sub-queries that
collectively cover the original question. Each sub-query should be answerable from a chunk
of a research paper. Return JSON: {"sub_queries": [...]}"""

_CRITIQUE_SYSTEM = """You are reviewing whether the retrieved chunks are sufficient to answer
the user's question. If gaps remain, list 1-3 additional sub-queries that would close them.
If coverage is good, return an empty list.

Return JSON: {"gaps": [...]}"""


def _parse_json_list(raw: str, key: str) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    val = data.get(key) or []
    return [str(x).strip() for x in val if str(x).strip()]


def _decompose(question: str) -> list[str]:
    llm = get_router().chat("retrieval")
    msg = llm.invoke(
        [SystemMessage(content=_DECOMPOSE_SYSTEM), HumanMessage(content=question)]
    )
    subs = _parse_json_list(getattr(msg, "content", str(msg)), "sub_queries")
    return subs or [question]


def _critique(question: str, chunks: list[RetrievedChunk]) -> list[str]:
    if not chunks:
        return []
    snippets = "\n\n".join(f"[{c.id}] {c.text[:400]}" for c in chunks[:12])
    llm = get_router().chat("retrieval")
    msg = llm.invoke(
        [
            SystemMessage(content=_CRITIQUE_SYSTEM),
            HumanMessage(content=f"Question:\n{question}\n\nChunks:\n{snippets}"),
        ]
    )
    return _parse_json_list(getattr(msg, "content", str(msg)), "gaps")


def agentic_retrieve(
    question: str,
    *,
    top_k: int | None = None,
    paper_ids: list[str] | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    cfg = get_config()
    top_k = top_k or cfg.rag.top_k
    max_iters = cfg.rag.agentic.max_iters

    sub_queries = _decompose(question)
    seen: dict[str, RetrievedChunk] = {}
    iteration = 0
    while iteration < max_iters:
        for sq in sub_queries:
            retrieve = graph_retrieve if cfg.rag.graph.enabled else hybrid_retrieve
            for c in retrieve(sq, top_k=max(top_k, 8), paper_ids=paper_ids, metadata_filter=metadata_filter):
                seen.setdefault(c.id, c)
        gaps = _critique(question, list(seen.values()))
        if not gaps:
            break
        sub_queries = gaps
        iteration += 1

    return rerank(question, list(seen.values()), top_k=top_k)
