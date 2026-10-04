"""FastAPI routes for the local research-assistant UI."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Annotated, Iterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from starlette.middleware.trustedhost import TrustedHostMiddleware

from assistant.config import get_config
from assistant.memory.conversations import (
    conversation_history,
    create_conversation,
    delete_conversation,
    generate_conversation_title,
    get_conversation,
    list_conversations,
)
from assistant.memory.criteria_store import get_active
from assistant.memory.curation_store import (
    base_arxiv_id,
    get_decision,
    list_decisions,
    set_decision_status,
)
from assistant.memory.episodic import set_feedback
from assistant.rag.ingest import ingest_source
from assistant.rag.arxiv_fetch import search_papers as search_arxiv_papers
from assistant.rag.filters import MetadataFilters
from assistant.rag.retriever import invalidate_bm25_cache
from assistant.storage import Chunk, Domain, Paper, get_engine, session_scope
from assistant.storage.domains import sync_domains_from_config
from assistant.storage.qdrant_store import delete_paper_vectors
from assistant.storage.index_lock import index_lock, paper_lock, resource_lock
from assistant.web.runner import invoke_graph, jobs, stream_graph

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
log = logging.getLogger(__name__)
ReadingStatus = Literal["new", "reviewing", "read"]


class ConversationCreate(BaseModel):
    title: str = "New conversation"
    scope_type: str = "library"
    scope_target: str | None = None


class MessageCreate(BaseModel):
    message: str
    filters: MetadataFilters = Field(default_factory=MetadataFilters)


class FeedbackUpdate(BaseModel):
    score: int


class PaperAdd(BaseModel):
    arxiv_id: str
    domain: str | None = None


class PaperUpdate(BaseModel):
    domain: str | None = None
    reading_status: ReadingStatus = "new"
    is_favorite: bool = False


class CommentCreate(BaseModel):
    comment: str


def _citation_details(chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    with session_scope() as session:
        rows = session.execute(
            select(Chunk, Paper)
            .join(Paper, Chunk.paper_id == Paper.id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
    by_id = {
        chunk.id: {
            "chunk_id": chunk.id,
            "paper_id": paper.id,
            "title": paper.title,
            "arxiv_id": paper.arxiv_id,
            "section": chunk.section_title or chunk.section_type,
            "text": chunk.text,
            "page_number": chunk.page_number,
            "arxiv_url": f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else None,
            "pdf_url": (
                f"/api/papers/{paper.id}/pdf"
                + (f"#page={chunk.page_number}" if chunk.page_number else "")
            ) if paper.pdf_path else None,
        }
        for chunk, paper in rows
    }
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def _paper_detail(paper_id: str) -> dict:
    with session_scope() as session:
        result = session.execute(
            select(Paper, Domain.name)
            .outerjoin(Domain, Paper.domain_id == Domain.id)
            .where(Paper.id == paper_id)
        ).first()
        if result is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        paper, domain_name = result
        return {
            "id": paper.id,
            "title": paper.title,
            "authors": list(paper.authors or []),
            "abstract": paper.abstract,
            "venue": paper.venue,
            "year": paper.year,
            "arxiv_id": paper.arxiv_id,
            "domain": domain_name,
            "accepted_score": paper.accepted_score,
            "status": paper.status,
            "reading_status": paper.reading_status,
            "is_favorite": paper.is_favorite,
            "has_pdf": bool(paper.pdf_path),
        }


def _scope_state(conversation: dict) -> dict:
    if conversation["scope_type"] == "topic":
        return {"domain": conversation["scope_target"]}
    if conversation["scope_type"] == "paper":
        return {"paper_ids": [conversation["scope_target"]]}
    return {}


def create_app() -> FastAPI:
    # Finish schema setup before concurrent requests can race on first access.
    get_engine()
    app = FastAPI(title="AI Research Assistant", docs_url="/api/docs")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def require_requested_with(request: Request, call_next):
        if request.url.path.startswith("/api") and request.method in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }:
            if request.headers.get("X-Requested-With") != "assistant-ui":
                return JSONResponse(status_code=403, content={"detail": "Forbidden"})
        return await call_next(request)

    @app.get("/api/status")
    def api_status() -> dict:
        cfg = get_config()
        with session_scope() as session:
            paper_count = session.scalar(select(func.count()).select_from(Paper)) or 0
        return {
            "paper_count": paper_count,
            "rag_mode": cfg.rag.mode,
            "roles": {
                role: {"provider": spec.provider, "model": spec.model}
                for role, spec in cfg.llm.roles.items()
            },
        }

    @app.get("/api/jobs")
    def api_jobs() -> list[dict]:
        return jobs.list()

    @app.get("/api/arxiv/search")
    def api_search_arxiv(q: Annotated[str, Query(min_length=2, max_length=160)]) -> list[dict]:
        try:
            matches = search_arxiv_papers(q)
        except Exception as exc:
            log.warning("arXiv search failed", exc_info=True)
            raise HTTPException(status_code=502, detail="arXiv search unavailable; try again shortly") from exc
        if not matches:
            return []
        with session_scope() as session:
            indexed = {
                base_arxiv_id(arxiv_id): paper_id
                for paper_id, arxiv_id in session.execute(select(Paper.id, Paper.arxiv_id))
                if arxiv_id
            }
        return [{**match, "paper_id": indexed.get(base_arxiv_id(match["arxiv_id"]))}
                for match in matches]

    @app.get("/api/papers")
    def api_papers(
        q: str | None = None,
        domain: str | None = None,
        reading_status: ReadingStatus | None = None,
        is_favorite: bool | None = None,
    ) -> list[dict]:
        with session_scope() as session:
            query = select(Paper, Domain.name).outerjoin(Domain, Paper.domain_id == Domain.id)
            if q:
                pattern = f"%{q.lower()}%"
                query = query.where(
                    or_(func.lower(Paper.title).like(pattern), func.lower(Paper.id).like(pattern))
                )
            if domain:
                query = query.where(Domain.name == domain)
            if reading_status:
                query = query.where(Paper.reading_status == reading_status)
            if is_favorite is not None:
                query = query.where(Paper.is_favorite == is_favorite)
            rows = session.execute(query.order_by(Paper.created_at.desc())).all()
            return [
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": list(paper.authors or []),
                    "year": paper.year,
                    "arxiv_id": paper.arxiv_id,
                    "domain": domain_name,
                    "accepted_score": paper.accepted_score,
                    "reading_status": paper.reading_status,
                    "is_favorite": paper.is_favorite,
                    "has_pdf": bool(paper.pdf_path),
                }
                for paper, domain_name in rows
            ]

    @app.get("/api/papers/{paper_id}")
    def api_paper(paper_id: str) -> dict:
        return _paper_detail(paper_id)

    @app.patch("/api/papers/{paper_id}")
    def api_update_paper(paper_id: str, body: PaperUpdate) -> dict:
        with session_scope() as session:
            paper = session.get(Paper, paper_id)
            if paper is None:
                raise HTTPException(status_code=404, detail="Paper not found")
            if "domain" in body.model_fields_set:
                if body.domain:
                    domain = session.execute(
                        select(Domain).where(Domain.name == body.domain)
                    ).scalar_one_or_none()
                    if domain is None:
                        raise HTTPException(status_code=404, detail="Topic not found")
                    paper.domain_id = domain.id
                else:
                    paper.domain_id = None
            if "reading_status" in body.model_fields_set:
                paper.reading_status = body.reading_status
            if "is_favorite" in body.model_fields_set:
                paper.is_favorite = body.is_favorite
        return _paper_detail(paper_id)

    @app.delete("/api/papers/{paper_id}", status_code=204)
    def api_delete_paper(paper_id: str) -> Response:
        with paper_lock(paper_id), index_lock.write():
            with session_scope() as session:
                paper = session.get(Paper, paper_id)
                if paper is None:
                    raise HTTPException(status_code=404, detail="Paper not found")
                delete_paper_vectors(paper_id)
                session.delete(paper)
            invalidate_bm25_cache()
        return Response(status_code=204)

    @app.get("/api/papers/{paper_id}/pdf")
    def api_paper_pdf(paper_id: str):
        with session_scope() as session:
            paper = session.get(Paper, paper_id)
            pdf_path = Path(paper.pdf_path) if paper and paper.pdf_path else None
        if pdf_path is None or not pdf_path.is_file():
            raise HTTPException(status_code=404, detail="PDF not found")
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)

    @app.post("/api/papers/arxiv", status_code=202)
    def api_add_arxiv(body: PaperAdd) -> dict:
        sync_domains_from_config()
        job = jobs.submit(
            "ingest",
            body.arxiv_id,
            lambda: invoke_graph(
                {
                    "intent": "ingest",
                    "scratch": {"source": body.arxiv_id, "domain": body.domain},
                }
            ),
        )
        return job

    @app.post("/api/papers/upload", status_code=202)
    async def api_upload_pdf(
        file: Annotated[UploadFile, File()],
        domain: Annotated[str | None, Form()] = None,
    ) -> dict:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
        upload_dir = get_config().storage.pdf_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / f"{uuid.uuid4()}.pdf"
        size = 0
        first = True
        try:
            with target.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    if first and not chunk.startswith(b"%PDF-"):
                        raise HTTPException(status_code=400, detail="File is not a valid PDF")
                    first = False
                    size += len(chunk)
                    if size > _MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=400, detail="PDF exceeds 50 MB")
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="PDF is empty")
        sync_domains_from_config()
        return jobs.submit(
            "ingest",
            file.filename or target.name,
            lambda: invoke_graph(
                {"intent": "ingest", "scratch": {"source": str(target), "domain": domain}}
            ),
        )

    @app.get("/api/topics")
    def api_topics() -> list[dict]:
        cfg = get_config()
        with session_scope() as session:
            db_domains = {
                row.name: row
                for row in session.execute(select(Domain).order_by(Domain.name)).scalars()
            }
            counts = dict(
                session.execute(
                    select(Domain.name, func.count(Paper.id))
                    .outerjoin(Paper, Paper.domain_id == Domain.id)
                    .group_by(Domain.name)
                ).all()
            )
        config_domains = {item.name: item for item in cfg.domains}
        names = sorted(set(config_domains) | set(db_domains))
        result = []
        for name in names:
            configured = config_domains.get(name)
            stored = db_domains.get(name)
            criteria = get_active(name)
            result.append(
                {
                    "name": name,
                    "arxiv_categories": list(
                        configured.arxiv_categories if configured else stored.arxiv_categories
                    ),
                    "venues": list(configured.venues if configured else stored.venues),
                    "paper_count": counts.get(name, 0),
                    "monitored": configured is not None,
                    "synced": stored is not None,
                    "criteria": {
                        "version": criteria.version,
                        "structured_rules": criteria.structured_rules,
                        "nl_addendum": criteria.nl_addendum,
                    }
                    if criteria
                    else None,
                }
            )
        return result

    @app.post("/api/topics/sync")
    def api_sync_topics() -> dict:
        added, updated = sync_domains_from_config()
        return {"added": added, "updated": updated}

    @app.post("/api/topics/{domain}/criteria")
    def api_refine_criteria(domain: str, body: CommentCreate) -> dict:
        sync_domains_from_config()
        result = invoke_graph(
            {"intent": "criteria_update", "domain": domain, "user_comment": body.comment}
        )
        return {"version": result.get("new_criteria_version")}

    @app.post("/api/topics/{domain}/monitor", status_code=202)
    def api_monitor(domain: str) -> dict:
        sync_domains_from_config()

        def monitor() -> dict:
            with resource_lock(f"monitor:{domain}"):
                return invoke_graph({"intent": "monitor_tick", "domain": domain})

        return jobs.submit(
            "monitor",
            domain,
            monitor,
        )

    @app.get("/api/inbox")
    def api_inbox(domain: str | None = None, status: str | None = None) -> list[dict]:
        return list_decisions(domain=domain, status=status)

    @app.post("/api/inbox/{decision_id}/ingest", status_code=202)
    def api_ingest_decision(decision_id: int) -> dict:
        decision = get_decision(decision_id)
        if decision is None:
            raise HTTPException(status_code=404, detail="Decision not found")

        def ingest() -> dict:
            try:
                paper_id = ingest_source(
                    decision["arxiv_id"],
                    domain=decision["domain"],
                    accepted_score=decision["score"],
                )
                set_decision_status(decision_id, "ingested")
                return {"ingested": [paper_id]}
            except Exception as exc:
                set_decision_status(decision_id, "failed", str(exc))
                raise

        return jobs.submit("ingest", decision["title"], ingest)

    @app.post("/api/inbox/{decision_id}/dismiss")
    def api_dismiss_decision(decision_id: int) -> dict:
        try:
            return set_decision_status(decision_id, "dismissed")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/inbox/{decision_id}/disagree")
    def api_disagree_decision(decision_id: int, body: CommentCreate) -> dict:
        decision = get_decision(decision_id)
        if decision is None:
            raise HTTPException(status_code=404, detail="Decision not found")
        result = invoke_graph(
            {
                "intent": "criteria_update",
                "domain": decision["domain"],
                "user_comment": body.comment,
            }
        )
        set_decision_status(decision_id, "dismissed")
        return {"version": result.get("new_criteria_version")}

    @app.get("/api/conversations")
    def api_conversations() -> list[dict]:
        return list_conversations()

    @app.post("/api/conversations", status_code=201)
    def api_create_conversation(body: ConversationCreate) -> dict:
        try:
            return create_conversation(body.title, body.scope_type, body.scope_target)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/conversations/{conversation_id}")
    def api_conversation(conversation_id: str) -> dict:
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = conversation_history(conversation_id)
        for turn in history:
            turn["sources"] = _citation_details(turn["citations"])
        return {**conversation, "turns": history}

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def api_delete_conversation(conversation_id: str):
        if not delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return None

    @app.post("/api/conversations/{conversation_id}/messages")
    def api_send_message(conversation_id: str, body: MessageCreate) -> StreamingResponse:
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        question = body.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        def output() -> Iterator[str]:
            # A conversation's next turn must load history after its prior turn completes.
            # Different conversations still stream concurrently.
            with resource_lock(f"conversation:{conversation_id}"):
                current = get_conversation(conversation_id)
                if current is None:
                    yield json.dumps({"type": "error", "message": "Conversation not found"}) + "\n"
                    return
                history = conversation_history(conversation_id, get_config().rag.history_turns)
                state = {
                    "intent": "ask",
                    "question": question,
                    "conversation_id": conversation_id,
                    "scratch": {"history": history, "metadata_filters": body.filters.model_dump(exclude_defaults=True)},
                    **_scope_state(current),
                }
                yield json.dumps({"type": "conversation", "conversation": current}) + "\n"
                for line in stream_graph(state):
                    event = json.loads(line)
                    if event.get("type") == "final":
                        result = event.get("result") or {}
                        result["sources"] = _citation_details(list(result.get("citations") or []))
                        result["interaction_id"] = (result.get("scratch") or {}).get(
                            "interaction_id"
                        )
                        event["result"] = result
                    yield json.dumps(event, ensure_ascii=True) + "\n"
                    if event.get("type") == "final":
                        renamed = generate_conversation_title(conversation_id, question)
                        if renamed is not None:
                            yield json.dumps({"type": "conversation", "conversation": renamed}) + "\n"

        return StreamingResponse(output(), media_type="application/x-ndjson")

    @app.post("/api/interactions/{interaction_id}/feedback")
    def api_feedback(interaction_id: int, body: FeedbackUpdate) -> dict:
        try:
            set_feedback(interaction_id, body.score)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"interaction_id": interaction_id, "score": body.score}

    static_dir = Path(__file__).parent / "static"
    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        index = static_dir / "index.html"
        requested = (static_dir / path).resolve()
        if static_dir.resolve() in requested.parents and requested.is_file():
            return FileResponse(requested)
        if index.is_file():
            return FileResponse(index)
        return JSONResponse(
            status_code=503,
            content={"detail": "Web UI is not built. Run `assistant serve --build`."},
        )

    return app