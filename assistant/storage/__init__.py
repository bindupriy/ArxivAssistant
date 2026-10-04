"""Persistence layer: Qdrant (vectors) + SQLite (metadata, memory, criteria)."""

from assistant.storage.qdrant_store import (
    CHUNKS_COLLECTION,
    PAPERS_COLLECTION,
    VectorRecord,
    delete_by_paper,
    ensure_collection,
    get_client as get_qdrant_client,
    search,
    upsert,
)
from assistant.storage.schema import (
    Base,
    Chunk,
    Conversation,
    CriteriaVersion,
    CurationDecision,
    Domain,
    Interaction,
    Paper,
    PaperCitation,
    ProfileFact,
)
from assistant.storage.sqlite_store import get_engine, session_scope

__all__ = [
    "Base",
    "Chunk",
    "Conversation",
    "CriteriaVersion",
    "CurationDecision",
    "Domain",
    "Interaction",
    "Paper",
    "PaperCitation",
    "ProfileFact",
    "VectorRecord",
    "CHUNKS_COLLECTION",
    "PAPERS_COLLECTION",
    "delete_by_paper",
    "ensure_collection",
    "get_engine",
    "get_qdrant_client",
    "search",
    "session_scope",
    "upsert",
]
