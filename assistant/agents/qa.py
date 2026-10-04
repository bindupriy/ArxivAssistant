"""Generate claim-level answers and verify paper evidence before rendering citations."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from assistant.agents.qa_prompts import ANSWER_SYSTEM_PROMPT as _SYSTEM
from assistant.agents.qa_prompts import VERIFY_SYSTEM_PROMPT as _VERIFY_SYSTEM
from assistant.agents.qa_schemas import _Claim, _Draft, _Evidence, _Verdict, _Verification
from assistant.agents.retrieval import retrieval_node
from assistant.config import get_config
from assistant.llm import get_router
from assistant.rag.compress import compress_context
from assistant.state import GraphState

log = logging.getLogger(__name__)

def _verified_evidence(claims: list[_Claim], chunks: list[dict]) -> dict[int, list[int]]:
    candidates = []
    locations: dict[int, tuple[int, int]] = {}
    for claim_index, claim in enumerate(claims):
        for evidence in claim.evidence:
            if evidence.source > len(chunks):
                continue
            passage = str(chunks[evidence.source - 1].get("text", ""))
            displayed = str(chunks[evidence.source - 1].get("context", passage))
            quote = " ".join(evidence.quote.split())
            if (not quote or quote not in " ".join(displayed.split())
                    or quote not in " ".join(passage.split())):
                continue
            candidate_id = len(candidates) + 1
            locations[candidate_id] = (claim_index, evidence.source)
            candidates.append({
                "candidate": candidate_id,
                "claim": claim.text,
                "quote": evidence.quote,
                "passage": passage,
            })
    if not candidates:
        return {}
    try:
        roles = get_config().llm.roles
        verifier = get_router().chat("citation_verifier" if "citation_verifier" in roles else "qa")
        message = verifier.invoke([
            SystemMessage(content=_VERIFY_SYSTEM),
            HumanMessage(content=json.dumps({"candidates": candidates}, ensure_ascii=True)),
        ])
        verified = _Verification.model_validate(_parse(str(message.content)))
        verdict_ids = [verdict.candidate for verdict in verified.verdicts]
        if len(verdict_ids) != len(locations) or set(verdict_ids) != set(locations):
            return {}
    except Exception:
        log.warning("Citation verification unavailable; withholding paper citations.", exc_info=True)
        return {}
    accepted: dict[int, list[int]] = {}
    for verdict in verified.verdicts:
        if verdict.supported:
            claim_index, source = locations[verdict.candidate]
            sources = accepted.setdefault(claim_index, [])
            if source not in sources:
                sources.append(source)
    return accepted


def _render_claims(
    claims: list[_Claim], chunks: list[dict], accepted: dict[int, list[int]]
) -> tuple[str, list[str]]:
    citations: list[str] = []
    blocks = []
    for claim_index, claim in enumerate(claims):
        text = re.sub(r"\[\d+\]", "", claim.text).strip()
        if not text:
            continue
        markers = []
        for source in accepted.get(claim_index, []):
            chunk_id = str(chunks[source - 1]["id"])
            if chunk_id not in citations:
                citations.append(chunk_id)
            markers.append(f"[{citations.index(chunk_id) + 1}]")
        attribution = " ".join(markers) if markers else "*(LLM knowledge)*"
        blocks.append(f"{text} {attribution}")
    return "\n\n".join(blocks), citations


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for index, c in enumerate(chunks, start=1):
        title = c.get("section_title") or c.get("section_type", "body")
        page = f", page={c['page_number']}" if c.get("page_number") else ""
        parts.append(f"[{index}] (paper={c['paper_id']}, section={title}{page})\n{c.get('context', c['text'])}")
    return "\n\n".join(parts) if parts else "(no chunks retrieved)"


def _format_history(history: list[dict], limit: int) -> str:
    turns = history[-limit:] if limit > 0 else []
    return "\n\n".join(
        f"User: {turn.get('question', '')}\nAssistant: {turn.get('answer', '')}"
        for turn in turns
    )


def _format_mcp(results: list[dict]) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        if "summary" in r:
            parts.append(f"[external] {r['summary']}")
        else:
            parts.append(f"[tool={r.get('tool')}] {r.get('args')}")
    return "\n\n".join(parts)


def _parse(raw: str) -> dict:
    # Try strict JSON first, then fall back to the first {...} block.
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _send_progress(stage: str) -> None:
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "progress", "stage": stage})
    except (LookupError, RuntimeError):
        pass


def qa_node(state: GraphState) -> GraphState:
    question = state.get("question", "").strip()
    if not question:
        return {"answer": "(empty question)", "citations": [], "confidence": 0.0}

    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        chunks = retrieval_node(state).get("retrieved_chunks", []) or []
    compression = getattr(get_config().rag, "compression", None)
    if compression is not None and compression.enabled and chunks:
        chunks = compress_context(
            chunks, question,
            max_chars_per_chunk=compression.max_chars_per_chunk,
            max_total_chars=compression.max_total_chars,
        )

    _send_progress("answering")
    llm = get_router().chat("qa")
    mcp_block = _format_mcp(state.get("mcp_results") or [])
    user_content = f"Question:\n{question}\n\nRetrieved chunks:\n{_format_context(chunks)}"
    history = list((state.get("scratch") or {}).get("history") or [])
    history_block = _format_history(history, get_config().rag.history_turns)
    if history_block:
        user_content = f"Conversation so far:\n{history_block}\n\n{user_content}"
    if mcp_block:
        user_content += f"\n\nAdditional context from external tools:\n{mcp_block}"
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]
    msg = llm.invoke(messages)
    try:
        draft = _Draft.model_validate(_parse(str(msg.content)))
    except ValidationError:
        try:
            # Smaller local models sometimes add extra fields; retry with schema-constrained output.
            draft = _Draft.model_validate(llm.with_structured_output(_Draft).invoke(messages))
        except Exception:
            return {
                "answer": "I couldn't reliably separate this answer's claims and evidence. Please try again.",
                "citations": [],
                "confidence": 0.0,
                "retrieved_chunks": chunks,
            }
    accepted = _verified_evidence(draft.claims, chunks)
    answer, citations = _render_claims(draft.claims, chunks, accepted)
    confidence = draft.confidence
    if any(claim.evidence and index not in accepted for index, claim in enumerate(draft.claims)):
        confidence = min(confidence, get_config().rag.low_confidence_threshold / 2)
    return {
        "answer": answer or "I couldn't produce a usable answer. Please try again.",
        "citations": citations,
        "confidence": confidence,
        "retrieved_chunks": chunks,
    }
