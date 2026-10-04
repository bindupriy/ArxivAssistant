from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient
from sqlalchemy import select

from assistant.rag.arxiv_fetch import ArxivPaper
from assistant.rag.chunker import chunk_paper
from assistant.rag.graph_retriever import extract_arxiv_references, graph_retrieve, related_papers
from assistant.rag.ingest import _persist
from assistant.rag.retriever import RetrievedChunk, _rrf_fuse, hybrid_retrieve, invalidate_bm25_cache
from assistant.rag.parser import ParsedPaper, _structure
from assistant.storage import Paper, PaperCitation, session_scope
from assistant.storage.sqlite_store import _session_factory, get_engine


class GraphRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        directory = Path(self.stack.enter_context(TemporaryDirectory()))
        storage = SimpleNamespace(sqlite_path=directory / "papers.db", ensure_dirs=Mock())
        self.stack.enter_context(patch("assistant.storage.sqlite_store.get_config", return_value=SimpleNamespace(storage=storage)))
        get_engine.cache_clear()
        _session_factory.cache_clear()
        invalidate_bm25_cache()
        self.vectors = QdrantClient(":memory:")
        self.stack.enter_context(patch("assistant.storage.qdrant_store.get_client", return_value=self.vectors))
        self.directory = directory

    def tearDown(self):
        get_engine().dispose()
        get_engine.cache_clear()
        _session_factory.cache_clear()
        invalidate_bm25_cache()
        self.vectors.close()
        self.stack.close()

    def ingest(self, paper_id: str, text: str = "") -> None:
        meta = ArxivPaper(paper_id, paper_id, [], "", self.directory / f"{paper_id}.pdf")
        parsed = ParsedPaper(full_text=text, sections=[], abstract=None)
        _persist(meta, parsed, [], [], [1.0, 0.0], "", domain=None, accepted_score=None)

    def test_references_are_explicit_normalized_and_replaced_on_reingest(self):
        self.assertEqual(
            extract_arxiv_references("arXiv:2401.22222v3 https://arxiv.org/abs/2401.22222", "2401.11111"),
            {"2401.22222"},
        )
        self.assertEqual(extract_arxiv_references("arXiv:2401.1234567", "2401.11111"), set())
        self.ingest("2401.11111", "References\narXiv:2401.22222v3")
        self.ingest("2401.22222")
        self.assertEqual(related_papers(["2401.11111"], None, 6), ["2401.22222"])
        self.assertEqual(related_papers(["2401.22222"], None, 6), ["2401.11111"])
        self.assertEqual(related_papers(["2401.11111"], ["2401.11111"], 6), [])
        self.ingest("2401.11111", "No explicit references now")
        self.assertEqual(related_papers(["2401.11111"], None, 6), [])
        with session_scope() as session:
            self.assertEqual(session.execute(select(PaperCitation)).scalars().all(), [])

    def test_deleting_a_paper_clears_its_graph_edges(self):
        self.ingest("2401.11111", "arXiv:2401.22222")
        self.ingest("2401.22222")
        with session_scope() as session:
            session.delete(session.get(Paper, "2401.11111"))
        with session_scope() as session:
            self.assertEqual(session.execute(select(PaperCitation)).scalars().all(), [])

    def test_hybrid_dense_and_bm25_keep_page_provenance(self):
        full = "Method\nGraph attention retrieves evidence."
        parsed = _structure(full, [(0, len(full))])
        chunks = chunk_paper("2401.11111", parsed)
        meta = ArxivPaper("2401.11111", "Graph paper", [], "", self.directory / "source.pdf")
        _persist(meta, parsed, chunks, [[1.0, 0.0]], [1.0, 0.0], "", domain=None, accepted_score=None)
        invalidate_bm25_cache()
        with patch("assistant.rag.retriever.Embedder") as embedder:
            embedder.return_value.embed_query.return_value = [1.0, 0.0]
            result = hybrid_retrieve("graph attention", paper_ids=["2401.11111"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, chunks[0].id)
        self.assertEqual(result[0].source, "fused")
        self.assertEqual(result[0].page_number, 1)

    def test_hybrid_weights_choose_dense_or_bm25_priority(self):
        dense = RetrievedChunk("dense", "paper", "dense", "body", "", 0.0, "dense")
        sparse = RetrievedChunk("sparse", "paper", "sparse", "body", "", 0.0, "sparse")
        self.assertEqual(
            _rrf_fuse([dense, sparse], [sparse, dense], dense_weight=0.9, sparse_weight=0.1)[0].id,
            "dense",
        )
        self.assertEqual(
            _rrf_fuse([dense, sparse], [sparse, dense], dense_weight=0.1, sparse_weight=0.9)[0].id,
            "sparse",
        )

    def test_graph_expansion_reserves_slots_and_respects_scope(self):
        self.ingest("2401.11111", "arXiv:2401.22222")
        self.ingest("2401.22222")

        def chunk(cid, paper):
            return RetrievedChunk(cid, paper, f"passage {cid}", "body", "Methods", 1.0, "fused")

        base = [chunk("first", "2401.11111"), chunk("second", "2401.11111")]
        linked = [chunk("linked", "2401.22222")]
        config = SimpleNamespace(rag=SimpleNamespace(graph=SimpleNamespace(graph_slots=1, max_neighbors=6)))
        with patch("assistant.rag.graph_retriever.get_config", return_value=config), patch(
            "assistant.rag.graph_retriever.hybrid_retrieve",
            side_effect=lambda query, top_k, paper_ids, metadata_filter=None: linked if paper_ids == ["2401.22222"] else base,
        ) as retrieve, patch("assistant.rag.graph_retriever.rerank", side_effect=lambda query, chunks, top_k: chunks[:top_k]):
            result = graph_retrieve("question", top_k=2, metadata_filter={"section_type": ["method"]})
            self.assertEqual([c.id for c in result], ["first", "linked"])
            self.assertEqual(retrieve.call_count, 2)
            self.assertEqual(
                [call.kwargs["metadata_filter"] for call in retrieve.call_args_list],
                [{"section_type": ["method"]}] * 2,
            )
            scoped = graph_retrieve("question", top_k=2, paper_ids=["2401.11111"])
            self.assertEqual([c.id for c in scoped], ["first", "second"])
            self.assertEqual(retrieve.call_count, 3)  # no graph query outside scope


if __name__ == "__main__":
    unittest.main()