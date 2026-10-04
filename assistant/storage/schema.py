"""SQLAlchemy schema for metadata + memory.

One SQLite file backs:
- `domains`: tracked research domains (name, arxiv cats, venues)
- `papers`: ingested papers (one row per paper)
- `chunks`: per-paper chunks (text + structural metadata; vector lives in Qdrant)
- `criteria_versions`: versioned curation criteria per domain
- `interactions`: episodic memory — every Q&A turn with citations
- `profile_facts`: user-profile key/value facts (manual or consolidated)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.utcnow()


class Domain(Base):
    __tablename__ = "domains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    arxiv_categories: Mapped[list[str]] = mapped_column(JSON, default=list)
    venues: Mapped[list[str]] = mapped_column(JSON, default=list)
    seed_papers: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    papers: Mapped[list["Paper"]] = relationship(back_populates="domain")
    criteria: Mapped[list["CriteriaVersion"]] = relationship(back_populates="domain")


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # arxiv_id or hash
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(Text, default="")
    venue: Mapped[str | None] = mapped_column(String(64), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("domains.id"), nullable=True)
    accepted_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ingested")
    reading_status: Mapped[str] = mapped_column(String(16), default="new", server_default="new")
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    domain: Mapped[Domain | None] = relationship(back_populates="papers")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    citations: Mapped[list["PaperCitation"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # uuid; matches Qdrant point id
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False)
    section_type: Mapped[str] = mapped_column(String(32), default="body")
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    order: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class PaperCitation(Base):
    """An explicit arXiv reference from one paper to another (which may not be ingested yet)."""

    __tablename__ = "paper_citations"

    source_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), primary_key=True)
    target_arxiv_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    paper: Mapped[Paper] = relationship(back_populates="citations")


class CriteriaVersion(Base):
    __tablename__ = "criteria_versions"
    __table_args__ = (UniqueConstraint("domain_id", "version", name="uq_criteria_domain_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("domains.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    structured_rules: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    nl_addendum: Mapped[str] = mapped_column(Text, default="")
    source_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    domain: Mapped[Domain] = relationship(back_populates="criteria")


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, default="")
    cited_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    rag_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # -1/0/+1
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    scope_type: Mapped[str] = mapped_column(String(16), default="library")
    scope_target: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class CurationDecision(Base):
    __tablename__ = "curation_decisions"
    __table_args__ = (
        UniqueConstraint("domain", "arxiv_id", name="uq_curation_domain_arxiv"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(Text, default="")
    categories: Mapped[list[str]] = mapped_column(JSON, default=list)
    domain: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    judge_reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="rejected")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class ProfileFact(Base):
    __tablename__ = "profile_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="manual")  # manual | consolidated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
