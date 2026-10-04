"""End-to-end ingestion: source -> parsed -> chunks -> embeddings -> stores.

Handles both arxiv IDs and local PDF paths.
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import select

from assistant.config import get_config
from assistant.rag.arxiv_fetch import ArxivPaper, fetch, is_arxiv_id
from assistant.rag.chunker import Chunk as PaperChunk
from assistant.rag.chunker import chunk_paper
from assistant.rag.embedder import Embedder
from assistant.rag.graph_retriever import extract_arxiv_references
from assistant.rag.parser import parse_pdf
from assistant.storage.index_lock import index_lock, paper_lock
from assistant.storage import (
    CHUNKS_COLLECTION,
    PAPERS_COLLECTION,
    Chunk as ChunkRow,
    Domain,
    Paper as PaperRow,
    PaperCitation,
    VectorRecord,
    delete_by_paper,
    session_scope,
    upsert,
)


def ingest_source(
    source: str,
    *,
    domain: str | None = None,
    accepted_score: float | None = None,
) -> str:
    """source = arxiv ID or path to a PDF. Returns the paper id stored."""
    key = source.strip() if is_arxiv_id(source) else Path(source).stem
    with paper_lock(key):
        return _ingest_source(source, domain=domain, accepted_score=accepted_score)


def _ingest_source(source: str, *, domain: str | None, accepted_score: float | None) -> str:
    if is_arxiv_id(source):
        meta = fetch(source.strip(), get_config().storage.pdf_dir)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Not an arxiv ID and not a file: {source}")
        meta = ArxivPaper(
            arxiv_id=path.stem,
            title=path.stem,
            authors=[],
            abstract="",
            pdf_path=path,
        )

    parsed = parse_pdf(meta.pdf_path)
    paper_id = meta.arxiv_id
    chunks = chunk_paper(paper_id, parsed)

    abstract_text = meta.abstract or parsed.abstract or (chunks[0].text if chunks else "")
    embedder = Embedder()

    chunk_vectors = embedder.embed_many([c.text for c in chunks]) if chunks else []
    paper_vector = embedder.embed_query(abstract_text or meta.title)

    _persist(
        meta,
        parsed,
        chunks,
        chunk_vectors,
        paper_vector,
        abstract_text,
        domain=domain,
        accepted_score=accepted_score,
    )
    return paper_id


def _persist(
    meta: ArxivPaper,
    parsed,
    chunks: list[PaperChunk],
    chunk_vectors: list[list[float]],
    paper_vector: list[float],
    abstract: str,
    *,
    domain: str | None,
    accepted_score: float | None,
) -> None:
    with index_lock.write():
        try:
            _persist_unlocked(
                meta, parsed, chunks, chunk_vectors, paper_vector, abstract,
                domain=domain, accepted_score=accepted_score,
            )
        finally:
            # SQLite may have committed before a Qdrant write fails.
            from assistant.rag.retriever import invalidate_bm25_cache

            invalidate_bm25_cache()


def _persist_unlocked(
    meta: ArxivPaper,
    parsed,
    chunks: list[PaperChunk],
    chunk_vectors: list[list[float]],
    paper_vector: list[float],
    abstract: str,
    *,
    domain: str | None,
    accepted_score: float | None,
) -> None:
    paper_id = meta.arxiv_id

    with session_scope() as s:
        domain_id = None
        if domain:
            domain_id = s.execute(
                select(Domain.id).where(Domain.name == domain)
            ).scalar_one_or_none()
        existing = s.get(PaperRow, paper_id)
        if existing is None:
            s.add(
                PaperRow(
                    id=paper_id,
                    title=meta.title,
                    authors=list(meta.authors),
                    abstract=abstract,
                    venue=meta.venue,
                    year=meta.year,
                    arxiv_id=meta.arxiv_id,
                    pdf_path=str(meta.pdf_path),
                    domain_id=domain_id,
                    accepted_score=accepted_score,
                    status="ingested",
                    extra={"categories": list(meta.categories)},
                )
            )
        else:
            existing.title = meta.title
            existing.authors = list(meta.authors)
            existing.abstract = abstract
            existing.pdf_path = str(meta.pdf_path)
            if domain:
                existing.domain_id = domain_id
            if accepted_score is not None:
                existing.accepted_score = accepted_score
            existing.status = "ingested"

        # Clear and rewrite chunks so re-ingestion is idempotent.
        for old in list(s.query(ChunkRow).filter(ChunkRow.paper_id == paper_id)):
            s.delete(old)
        s.flush()

        for c in chunks:
            s.add(
                ChunkRow(
                    id=c.id,
                    paper_id=paper_id,
                    section_type=c.section_type,
                    section_title=c.section_title,
                    order=c.order,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    page_number=c.page_number,
                )
            )

        for old in list(s.query(PaperCitation).filter(PaperCitation.source_id == paper_id)):
            s.delete(old)
        s.flush()
        for target in sorted(extract_arxiv_references(parsed.full_text, paper_id) if parsed else []):
            s.add(PaperCitation(source_id=paper_id, target_arxiv_id=target))

    # Vectors: chunk-level and paper-level.
    delete_by_paper(paper_id)
    if chunks:
        upsert(
            CHUNKS_COLLECTION,
            (
                VectorRecord(
                    id=c.id,
                    vector=v,
                    payload={
                        "paper_id": paper_id,
                        "section_type": c.section_type,
                        "section_title": c.section_title,
                        "order": c.order,
                        "text": c.text,
                        "page_number": c.page_number,
                    },
                )
                for c, v in zip(chunks, chunk_vectors)
            ),
        )
    upsert(
        PAPERS_COLLECTION,
        [
            VectorRecord(
                id=paper_id,
                vector=paper_vector,
                payload={
                    "title": meta.title,
                    "authors": list(meta.authors),
                    "year": meta.year,
                    "arxiv_id": meta.arxiv_id,
                },
            )
        ],
    )
