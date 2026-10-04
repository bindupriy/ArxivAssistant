from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from assistant.memory.conversations import create_conversation
from assistant.rag.arxiv_fetch import ArxivPaper
from assistant.rag.ingest import _persist
from assistant.rag.retriever import _bm25, invalidate_bm25_cache
from assistant.storage import Chunk, Domain, Paper, session_scope
from assistant.storage.qdrant_store import (
    CHUNKS_COLLECTION,
    PAPERS_COLLECTION,
    VectorRecord,
    upsert,
)
from assistant.storage.sqlite_store import _session_factory, get_engine
from assistant.web.app import _citation_details, create_app


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.directory = Path(self.stack.enter_context(TemporaryDirectory()))
        storage = SimpleNamespace(
            sqlite_path=self.directory / "library.db", ensure_dirs=Mock()
        )
        self.stack.enter_context(
            patch("assistant.storage.sqlite_store.get_config", return_value=SimpleNamespace(storage=storage))
        )
        get_engine.cache_clear()
        _session_factory.cache_clear()
        invalidate_bm25_cache()
        self.vectors = QdrantClient(":memory:")
        self.stack.enter_context(
            patch("assistant.storage.qdrant_store.get_client", return_value=self.vectors)
        )
        self.client = self.stack.enter_context(TestClient(create_app()))
        self.headers = {"X-Requested-With": "assistant-ui"}
        with session_scope() as session:
            domain = Domain(name="Research")
            session.add(domain)
            session.flush()
            session.add_all([
                Paper(id="first", title="First paper", domain_id=domain.id),
                Paper(id="second", title="Second paper"),
            ])
            session.flush()
            session.add_all([
                Chunk(id="chunk-first", paper_id="first", text="First research result"),
                Chunk(id="chunk-second", paper_id="second", text="Second research result"),
            ])

    def tearDown(self):
        get_engine().dispose()
        get_engine.cache_clear()
        _session_factory.cache_clear()
        invalidate_bm25_cache()
        self.vectors.close()
        self.stack.close()

    def update(self, paper_id="first", **body):
        return self.client.patch(f"/api/papers/{paper_id}", json=body, headers=self.headers)

    def test_defaults_and_partial_updates(self):
        paper = self.client.get("/api/papers/first").json()
        self.assertEqual(paper["reading_status"], "new")
        self.assertFalse(paper["is_favorite"])
        self.assertNotIn("sections", paper)
        for status in ("reviewing", "read", "new"):
            response = self.update(reading_status=status)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["reading_status"], status)
            self.assertEqual(response.json()["domain"], "Research")
        self.update(reading_status="read")
        paper = self.update(is_favorite=True).json()
        self.assertEqual(paper["reading_status"], "read")
        self.assertEqual(paper["domain"], "Research")
        self.assertTrue(paper["is_favorite"])
        paper = self.update(domain=None).json()
        self.assertIsNone(paper["domain"])
        self.assertTrue(paper["is_favorite"])
        self.assertEqual(paper["reading_status"], "read")
        self.assertFalse(self.update(is_favorite=False).json()["is_favorite"])
        self.assertEqual(self.update().json()["reading_status"], "read")

    def test_filters_combine(self):
        self.update(reading_status="reviewing", is_favorite=True)
        response = self.client.get(
            "/api/papers",
            params={"q": "first", "domain": "Research", "reading_status": "reviewing", "is_favorite": "true"},
        )
        self.assertEqual([paper["id"] for paper in response.json()], ["first"])
        self.assertEqual(len(self.client.get("/api/papers").json()), 2)
        self.assertEqual(len(self.client.get("/api/papers?reading_status=new").json()), 1)
        self.assertEqual(self.client.get("/api/papers?reading_status=read&is_favorite=true").json(), [])
        self.update(is_favorite=False)
        self.assertEqual(self.client.get("/api/papers?is_favorite=true").json(), [])

    def test_citation_source_points_to_physical_pdf_page(self):
        with session_scope() as session:
            session.get(Chunk, "chunk-first").page_number = 3
            session.get(Paper, "first").pdf_path = str(self.directory / "original.pdf")
        source = _citation_details(["chunk-first"])[0]
        self.assertEqual(source["page_number"], 3)
        self.assertEqual(source["pdf_url"], "/api/papers/first/pdf#page=3")

    def test_search_arxiv_previews_and_marks_indexed_papers(self):
        with session_scope() as session:
            session.add(Paper(id="2104.09864v5", title="RoFormer", arxiv_id="2104.09864v5"))
        preview = [{"arxiv_id": "2104.09864v5", "title": "RoFormer", "abstract": "RoPE",
                    "authors": ["Researcher"], "year": 2021}]
        with patch("assistant.web.app.search_arxiv_papers", return_value=preview) as search:
            response = self.client.get("/api/arxiv/search", params={"q": "RoPE"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["paper_id"], "2104.09864v5")
            search.assert_called_once_with("RoPE")
            self.assertEqual(self.client.get("/api/arxiv/search", params={"q": "x"}).status_code, 422)
        with patch("assistant.web.app.search_arxiv_papers", side_effect=RuntimeError("upstream")):
            self.assertEqual(self.client.get("/api/arxiv/search", params={"q": "RoPE"}).status_code, 502)

    def test_validation_and_write_protection(self):
        self.assertEqual(self.update(reading_status="invalid").status_code, 422)
        self.assertEqual(self.update(reading_status=None).status_code, 422)
        self.assertEqual(self.update(is_favorite=None).status_code, 422)
        self.assertEqual(self.client.get("/api/papers?reading_status=invalid").status_code, 422)
        self.assertEqual(self.update("missing", reading_status="read").status_code, 404)
        self.assertEqual(self.update(domain="missing").status_code, 404)
        self.assertEqual(self.client.patch("/api/papers/first", json={"is_favorite": True}).status_code, 403)
        self.assertEqual(self.client.delete("/api/papers/first").status_code, 403)
        self.assertEqual(self.client.get("/api/papers/first").json()["domain"], "Research")

    def test_delete_cleans_indexes_and_preserves_other_papers(self):
        for paper_id in ("first", "second"):
            upsert(CHUNKS_COLLECTION, [VectorRecord(
                id=f"chunk-{paper_id}", vector=[1.0, 0.0], payload={"paper_id": paper_id}
            )])
            upsert(PAPERS_COLLECTION, [VectorRecord(
                id=paper_id, vector=[1.0, 0.0], payload={"title": paper_id}
            )])
        self.assertEqual(len(_bm25().chunk_ids), 2)
        response = self.client.delete("/api/papers/first", headers=self.headers)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/api/papers/first").status_code, 404)
        self.assertEqual(self.client.delete("/api/papers/first", headers=self.headers).status_code, 404)
        with session_scope() as session:
            self.assertIsNone(session.get(Chunk, "chunk-first"))
            self.assertIsNotNone(session.get(Paper, "second"))
        self.assertEqual(_bm25().chunk_ids, ["chunk-second"])
        for collection in (CHUNKS_COLLECTION, PAPERS_COLLECTION):
            points, _ = self.vectors.scroll(collection)
            self.assertEqual(len(points), 1)
            self.assertIn("second", points[0].payload["_logical_id"])

    def test_delete_without_vector_collections(self):
        self.assertEqual(self.client.delete("/api/papers/first", headers=self.headers).status_code, 204)
        self.assertEqual(self.client.delete("/api/papers/second", headers=self.headers).status_code, 204)
        self.assertIsNone(_bm25())

    def test_vector_failure_keeps_paper(self):
        with patch("assistant.web.app.delete_paper_vectors", side_effect=RuntimeError("Unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Unavailable"):
                self.client.delete("/api/papers/first", headers=self.headers)
        self.assertEqual(self.client.get("/api/papers/first").status_code, 200)

    def test_reingest_preserves_user_preferences(self):
        self.update(reading_status="read", is_favorite=True)
        meta = ArxivPaper(
            arxiv_id="first", title="Reingested", authors=[], abstract="",
            pdf_path=self.directory / "source.pdf",
        )
        _persist(meta, None, [], [], [1.0, 0.0], "", domain=None, accepted_score=None)
        paper = self.client.get("/api/papers/first").json()
        self.assertEqual(paper["title"], "Reingested")
        self.assertEqual(paper["reading_status"], "read")
        self.assertTrue(paper["is_favorite"])

    def test_delete_keeps_pdf_and_saved_chat(self):
        pdf_path = self.directory / "original.pdf"
        pdf_path.touch()
        with session_scope() as session:
            session.get(Paper, "first").pdf_path = str(pdf_path)
        conversation = create_conversation("Paper chat", "paper", "first")
        self.assertEqual(self.client.delete("/api/papers/first", headers=self.headers).status_code, 204)
        self.assertTrue(pdf_path.exists())
        self.assertEqual(self.client.get(f"/api/conversations/{conversation['id']}").status_code, 200)


class MigrationTests(unittest.TestCase):
    def test_existing_database_upgrade_is_repeatable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            engine = create_engine(f"sqlite:///{path.as_posix()}")
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE papers (id VARCHAR(64) PRIMARY KEY, title TEXT NOT NULL)")
                connection.exec_driver_sql("INSERT INTO papers (id, title) VALUES ('old', 'Existing paper')")
                connection.exec_driver_sql("CREATE TABLE chunks (id VARCHAR(64) PRIMARY KEY, paper_id VARCHAR(64), text TEXT NOT NULL)")
                connection.exec_driver_sql("INSERT INTO chunks (id, paper_id, text) VALUES ('old-chunk', 'old', 'Old text')")
            engine.dispose()
            storage = SimpleNamespace(sqlite_path=path, ensure_dirs=Mock())
            with patch("assistant.storage.sqlite_store.get_config", return_value=SimpleNamespace(storage=storage)):
                for attempt in range(2):
                    get_engine.cache_clear()
                    engine = get_engine()
                    with engine.connect() as connection:
                        self.assertEqual(
                            connection.execute(text("SELECT reading_status, is_favorite FROM papers")).one(),
                            ("new", 0),
                        )
                        self.assertEqual(
                            connection.execute(text("SELECT id, page_number FROM chunks")).one(),
                            ("old-chunk", None),
                        )
                    engine.dispose()
                get_engine.cache_clear()


if __name__ == "__main__":
    unittest.main()