from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from qdrant_client import QdrantClient

from assistant.agents.retrieval import _eligible_papers, _standalone_question, retrieval_node
from assistant.rag.filters import MetadataFilters
from assistant.rag.agentic import agentic_retrieve
from assistant.rag.retriever import hybrid_retrieve, invalidate_bm25_cache
from assistant.storage import CHUNKS_COLLECTION, Chunk, Domain, Paper, VectorRecord, session_scope, upsert
from assistant.storage.sqlite_store import _session_factory, get_engine


class RewriteTests(unittest.TestCase):
    def test_rewrite_first_turn_and_followup_with_safe_fallback(self):
        rewrite = SimpleNamespace(enabled=True, first_turn=True, max_chars=300)
        cfg = SimpleNamespace(rag=SimpleNamespace(query_rewrite=rewrite))
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content="How does RoPE change attention in arXiv:2401.12345?")
        with patch("assistant.agents.retrieval.get_config", return_value=cfg), patch(
            "assistant.agents.retrieval.get_router"
        ) as router:
            router.return_value.chat.return_value = model
            question = "RoPE arXiv:2401.12345?"
            self.assertEqual(_standalone_question(question, []), model.invoke.return_value.content)
            self.assertIn("Conversation:", model.invoke.call_args.args[0][1].content)
            self.assertEqual(_standalone_question(question, [{"question": "RoPE", "answer": "Rotary positions"}]), model.invoke.return_value.content)
            model.invoke.return_value = SimpleNamespace(content="How does RoPE work?")
            self.assertEqual(_standalone_question(question, []), question)  # missing original ID
            model.invoke.return_value = SimpleNamespace(content="bad\nsecond query")
            self.assertEqual(_standalone_question(question, []), question)
            model.invoke.return_value = SimpleNamespace(content="How does attention work?")
            self.assertEqual(_standalone_question('What does "RoPE" do?', []), 'What does "RoPE" do?')
            model.invoke.side_effect = RuntimeError("unavailable")
            with self.assertLogs("assistant.agents.retrieval", level="WARNING"):
                self.assertEqual(_standalone_question(question, []), question)
            rewrite.first_turn = False
            model.invoke.reset_mock()
            self.assertEqual(_standalone_question(question, []), question)
            model.invoke.assert_not_called()

    def test_invalid_metadata_is_rejected(self):
        for filters in (
            {"min_year": 2025, "max_year": 2020},
            {"section_types": ["unsupported"]},
            {"surprise": "filter"},
        ):
            with self.subTest(filters=filters), self.assertRaises(ValidationError):
                MetadataFilters.model_validate(filters)
        self.assertIsNone(MetadataFilters(venue="  ").venue)


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        directory = Path(self.stack.enter_context(TemporaryDirectory()))
        storage = SimpleNamespace(sqlite_path=directory / "filtered.db", ensure_dirs=Mock())
        self.stack.enter_context(patch("assistant.storage.sqlite_store.get_config", return_value=SimpleNamespace(storage=storage)))
        get_engine.cache_clear()
        _session_factory.cache_clear()
        invalidate_bm25_cache()
        self.vectors = QdrantClient(":memory:")
        self.stack.enter_context(patch("assistant.storage.qdrant_store.get_client", return_value=self.vectors))
        with session_scope() as session:
            domain = Domain(name="topic")
            session.add(domain)
            session.flush()
            session.add_all([
                Paper(id="old", title="Old", year=2020, venue="ACL", domain_id=domain.id),
                Paper(id="new", title="New", year=2024, venue="NeurIPS", domain_id=domain.id),
                Paper(id="other", title="Other", year=2024, venue="NeurIPS"),
            ])
            session.flush()
            session.add_all([
                Chunk(id="old-method", paper_id="old", section_type="method", text="adaptive retrieval method"),
                Chunk(id="new-method", paper_id="new", section_type="method", text="adaptive retrieval method"),
                Chunk(id="new-results", paper_id="new", section_type="results", text="adaptive retrieval results"),
                Chunk(id="other-method", paper_id="other", section_type="method", text="adaptive retrieval method"),
            ])
        for chunk_id, paper_id, section in (
            ("old-method", "old", "method"), ("new-method", "new", "method"),
            ("new-results", "new", "results"), ("other-method", "other", "method"),
        ):
            upsert(CHUNKS_COLLECTION, [VectorRecord(
                id=chunk_id, vector=[1.0, 0.0], payload={"paper_id": paper_id,
                    "section_type": section, "section_title": "", "text": "adaptive retrieval method"},
            )])
        invalidate_bm25_cache()

    def tearDown(self):
        get_engine().dispose()
        get_engine.cache_clear()
        _session_factory.cache_clear()
        invalidate_bm25_cache()
        self.vectors.close()
        self.stack.close()

    def test_year_venue_topic_and_sections_filter_both_indexes(self):
        filters = MetadataFilters(min_year=2023, max_year=2024, venue="neurips", section_types=["method"])
        eligible = _eligible_papers(["old", "new"], filters)
        self.assertEqual(eligible, ["new"])
        with patch("assistant.rag.retriever.Embedder") as embedder:
            embedder.return_value.embed_query.return_value = [1.0, 0.0]
            chunks = hybrid_retrieve("adaptive retrieval", paper_ids=eligible,
                                     metadata_filter={"section_type": filters.section_types})
        self.assertEqual([chunk.id for chunk in chunks], ["new-method"])
        self.assertEqual(chunks[0].source, "fused")
        self.assertEqual(_eligible_papers(["old"], filters), [])
        self.assertEqual(_eligible_papers(None, MetadataFilters(min_year=2024)), ["new", "other"])

    def test_agent_forwards_filters_without_widening_scope(self):
        state = {"question": "Explain retrieval", "domain": "topic", "scratch": {
            "metadata_filters": {"min_year": 2023, "section_types": ["results"]},
        }}
        cfg = SimpleNamespace(rag=SimpleNamespace(top_k=4, mode="vanilla", graph=SimpleNamespace(enabled=False), history_turns=2))
        with patch("assistant.agents.retrieval.get_config", return_value=cfg), patch(
            "assistant.agents.retrieval._standalone_question", side_effect=lambda question, history: question,
        ), patch("assistant.agents.retrieval.hybrid_retrieve", return_value=[]) as retrieve, patch(
            "assistant.agents.retrieval.rerank", return_value=[],
        ):
            self.assertEqual(retrieval_node(state), {"retrieved_chunks": []})
        self.assertEqual(retrieve.call_args.kwargs["paper_ids"], ["new"])
        self.assertEqual(retrieve.call_args.kwargs["metadata_filter"], {"section_type": ["results"]})

    def test_agentic_graph_subqueries_keep_metadata_filters(self):
        cfg = SimpleNamespace(rag=SimpleNamespace(top_k=2, agentic=SimpleNamespace(max_iters=1), graph=SimpleNamespace(enabled=True)))
        with patch("assistant.rag.agentic.get_config", return_value=cfg), patch(
            "assistant.rag.agentic._decompose", return_value=["retrieval", "results"],
        ), patch("assistant.rag.agentic._critique", return_value=[]), patch(
            "assistant.rag.agentic.graph_retrieve", return_value=[],
        ) as retrieve, patch("assistant.rag.agentic.rerank", return_value=[]):
            agentic_retrieve("question", paper_ids=["new"], metadata_filter={"section_type": ["results"]})
        self.assertEqual(retrieve.call_count, 2)
        for call in retrieve.call_args_list:
            self.assertEqual(call.kwargs["paper_ids"], ["new"])
            self.assertEqual(call.kwargs["metadata_filter"], {"section_type": ["results"]})


if __name__ == "__main__":
    unittest.main()