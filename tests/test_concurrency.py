from contextlib import ExitStack
import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from qdrant_client import QdrantClient
from sqlalchemy import select

from assistant.rag.arxiv_fetch import ArxivPaper
from assistant.rag.chunker import Chunk as PaperChunk
from assistant.rag.ingest import _persist
from assistant.rag.ingest import ingest_source
from assistant.rag.parser import _structure
from assistant.rag.retriever import _bm25, invalidate_bm25_cache
from assistant.agents.criteria import criteria_node
from assistant.memory.criteria_store import get_active
from assistant.storage import Domain, Paper, session_scope
from assistant.storage.index_lock import index_lock
from assistant.storage.sqlite_store import _session_factory, get_engine
from assistant.web.runner import JobManager, stream_graph


class ConcurrencyTests(unittest.TestCase):
    def test_background_jobs_run_in_parallel_and_extra_jobs_queue(self):
        manager = JobManager(max_workers=2)
        entered = threading.Barrier(3)
        release = threading.Event()

        def operation(label):
            entered.wait(timeout=10)
            self.assertTrue(release.wait(timeout=10))
            return {"label": label}

        try:
            first = manager.submit("ingest", "first", lambda: operation("first"))
            second = manager.submit("ingest", "second", lambda: operation("second"))
            third = manager.submit("ingest", "third", lambda: {"label": "third"})
            entered.wait(timeout=10)
            statuses = {item["id"]: item["status"] for item in manager.list()}
            self.assertEqual(statuses[first["id"]], "running")
            self.assertEqual(statuses[second["id"]], "running")
            self.assertEqual(statuses[third["id"]], "queued")
        finally:
            release.set()
            manager.shutdown()
        self.assertEqual({item["status"] for item in manager.list()}, {"completed"})

    def test_readers_overlap_but_writer_waits_for_both(self):
        readers = threading.Barrier(3)
        release = threading.Event()
        writing = threading.Event()
        attempted = threading.Event()

        def read():
            with index_lock.read():
                readers.wait(timeout=10)
                self.assertTrue(release.wait(timeout=10))

        def write():
            attempted.set()
            with index_lock.write():
                writing.set()
                with index_lock.write():  # invalidation during ingest is reentrant
                    pass

        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(read)
            second = executor.submit(read)
            try:
                readers.wait(timeout=10)
                writer = executor.submit(write)
                self.assertTrue(attempted.wait(timeout=10))
                self.assertFalse(writing.wait(timeout=0.1))
            finally:
                release.set()
            first.result(timeout=10)
            second.result(timeout=10)
            writer.result(timeout=10)
        self.assertTrue(writing.is_set())

    def test_same_paper_ingest_waits_while_other_paper_proceeds(self):
        first_started = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        other_started = threading.Event()

        def fake_ingest(source, *, domain, accepted_score):
            if source == "2401.11111":
                if not first_started.is_set():
                    first_started.set()
                    self.assertTrue(release.wait(timeout=10))
                else:
                    second_started.set()
            else:
                other_started.set()
            return source

        with patch("assistant.rag.ingest._ingest_source", side_effect=fake_ingest):
            with ThreadPoolExecutor(max_workers=3) as executor:
                first = executor.submit(ingest_source, "2401.11111")
                self.assertTrue(first_started.wait(timeout=10))
                second = executor.submit(ingest_source, "2401.11111")
                other = executor.submit(ingest_source, "2401.22222")
                try:
                    self.assertEqual(other.result(timeout=10), "2401.22222")
                    self.assertTrue(other_started.is_set())
                    self.assertFalse(second_started.is_set())
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=10), "2401.11111")
                self.assertEqual(second.result(timeout=10), "2401.11111")
        self.assertTrue(second_started.is_set())

    def test_parallel_persists_keep_both_sqlite_and_qdrant_indexes(self):
        with ExitStack() as stack:
            directory = Path(stack.enter_context(TemporaryDirectory()))
            storage = SimpleNamespace(sqlite_path=directory / "parallel.db", ensure_dirs=Mock())
            stack.enter_context(patch("assistant.storage.sqlite_store.get_config", return_value=SimpleNamespace(storage=storage)))
            get_engine.cache_clear()
            _session_factory.cache_clear()
            invalidate_bm25_cache()
            vectors = QdrantClient(":memory:")
            stack.enter_context(patch("assistant.storage.qdrant_store.get_client", return_value=vectors))

            def save(paper_id):
                text = f"Results for paper {paper_id}"
                meta = ArxivPaper(paper_id, paper_id, [], "", directory / f"{paper_id}.pdf")
                chunk = PaperChunk(str(uuid4()), paper_id, "results", "Results", 0, text, 0, len(text), 1)
                _persist(meta, _structure(text), [chunk], [[1.0, 0.0]], [1.0, 0.0], "", domain=None, accepted_score=None)
                return chunk.id

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    ids = [future.result(timeout=20) for future in (
                        executor.submit(save, "first"), executor.submit(save, "second")
                    )]
                with session_scope() as session:
                    self.assertEqual(set(session.execute(select(Paper.id)).scalars()), {"first", "second"})
                self.assertEqual(set(_bm25().chunk_ids), set(ids))
                points, _ = vectors.scroll("chunks")
                self.assertEqual({point.payload["_logical_id"] for point in points}, set(ids))
                with patch("assistant.rag.ingest.upsert", side_effect=RuntimeError("vector failure")):
                    with self.assertRaisesRegex(RuntimeError, "vector failure"):
                        save("third")
                # A vector error occurs after SQLite commits; BM25 still must refresh.
                self.assertEqual(len(_bm25().chunk_ids), 3)
            finally:
                get_engine().dispose()
                get_engine.cache_clear()
                _session_factory.cache_clear()
                invalidate_bm25_cache()
                vectors.close()

    def test_same_topic_criteria_edits_keep_both_comments(self):
        with ExitStack() as stack:
            directory = Path(stack.enter_context(TemporaryDirectory()))
            storage = SimpleNamespace(sqlite_path=directory / "criteria.db", ensure_dirs=Mock())
            stack.enter_context(patch("assistant.storage.sqlite_store.get_config", return_value=SimpleNamespace(storage=storage)))
            get_engine.cache_clear()
            _session_factory.cache_clear()
            with session_scope() as session:
                session.add(Domain(name="research"))

            started = threading.Event()
            release = threading.Event()

            def respond(messages):
                prompt = messages[1].content
                if "User comment:\nfirst" in prompt:
                    started.set()
                    self.assertTrue(release.wait(timeout=10))
                    words = ["first"]
                else:
                    words = ["first", "second"] if '"first"' in prompt else ["second"]
                return SimpleNamespace(content=json.dumps({
                    "structured_rules": {"include_keywords": words}, "nl_addendum": "",
                }))

            model = Mock()
            model.invoke.side_effect = respond
            router = stack.enter_context(patch("assistant.agents.criteria.get_router"))
            router.return_value.chat.return_value = model
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(criteria_node, {"domain": "research", "user_comment": "first"})
                    self.assertTrue(started.wait(timeout=10))
                    second = executor.submit(criteria_node, {"domain": "research", "user_comment": "second"})
                    release.set()
                    self.assertEqual(first.result(timeout=10)["new_criteria_version"], 1)
                    self.assertEqual(second.result(timeout=10)["new_criteria_version"], 2)
                self.assertEqual(get_active("research").structured_rules["include_keywords"], ["first", "second"])
            finally:
                release.set()
            get_engine().dispose()
            get_engine.cache_clear()
            _session_factory.cache_clear()

    def test_chat_streams_can_run_at_the_same_time(self):
        entered = threading.Barrier(3)
        release = threading.Event()

        def fake_stream(state, *, config, stream_mode):
            entered.wait(timeout=10)
            self.assertTrue(release.wait(timeout=10))
            yield "values", {"answer": state["question"]}

        with patch("assistant.web.runner.get_graph") as graph:
            graph.return_value.stream.side_effect = fake_stream
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(lambda: list(stream_graph({"question": "first"})))
                second = executor.submit(lambda: list(stream_graph({"question": "second"})))
                try:
                    entered.wait(timeout=10)
                finally:
                    release.set()
                self.assertEqual(json.loads(first.result(timeout=10)[0])["result"]["answer"], "first")
                self.assertEqual(json.loads(second.result(timeout=10)[0])["result"]["answer"], "second")


if __name__ == "__main__":
    unittest.main()