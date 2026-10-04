from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from assistant.memory.conversations import (
    create_conversation,
    delete_conversation,
    generate_conversation_title,
    get_conversation,
)
from assistant.storage import Conversation, Interaction, session_scope
from assistant.storage.sqlite_store import _session_factory, get_engine
from assistant.web.app import create_app


class ConversationTitleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        directory = Path(self.stack.enter_context(TemporaryDirectory()))
        storage = SimpleNamespace(sqlite_path=directory / "chats.db", ensure_dirs=Mock())
        self.stack.enter_context(patch(
            "assistant.storage.sqlite_store.get_config",
            return_value=SimpleNamespace(storage=storage),
        ))
        get_engine.cache_clear()
        _session_factory.cache_clear()
        self.config = SimpleNamespace(llm=SimpleNamespace(roles={"conversation_title": object()}))
        self.stack.enter_context(patch(
            "assistant.memory.conversations.get_config", return_value=self.config,
        ))
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(content="Hybrid Retrieval Tradeoffs")
        self.router = Mock()
        self.router.chat.return_value = self.model
        self.stack.enter_context(patch(
            "assistant.memory.conversations.get_router", return_value=self.router,
        ))
        self.conversation = create_conversation("New conversation", "library", None)
        self.conversation_id = self.conversation["id"]

    def tearDown(self):
        get_engine().dispose()
        get_engine.cache_clear()
        _session_factory.cache_clear()
        self.stack.close()

    def test_title_is_generated_and_persisted_once(self):
        result = generate_conversation_title(self.conversation_id, "Compare dense and sparse retrieval")
        self.assertEqual(result["title"], "Hybrid Retrieval Tradeoffs")
        self.assertEqual(get_conversation(self.conversation_id)["title"], result["title"])
        self.assertEqual(result["scope_type"], "library")
        self.assertIsNone(generate_conversation_title(self.conversation_id, "Follow up"))
        self.model.invoke.assert_called_once()
        self.router.chat.assert_called_once_with("conversation_title")

    def test_descriptive_titles_and_missing_chats_are_untouched(self):
        named = create_conversation("Paper title", "paper", "paper-a")
        self.assertIsNone(generate_conversation_title(named["id"], "Question"))
        self.assertIsNone(generate_conversation_title("missing", "Question"))
        self.model.invoke.assert_not_called()
        self.assertEqual(get_conversation(named["id"])["title"], "Paper title")

    def test_existing_unnamed_chat_uses_first_saved_question(self):
        with session_scope() as session:
            session.add_all([
                Interaction(question="Original research question", extra={"conversation_id": self.conversation_id}),
                Interaction(question="Later follow-up", extra={"conversation_id": self.conversation_id}),
            ])
        generate_conversation_title(self.conversation_id, "Newest follow-up")
        self.assertEqual(self.model.invoke.call_args.args[0][1].content, "Original research question")

    def test_model_failure_does_not_break_chat_or_prevent_retry(self):
        self.model.invoke.side_effect = RuntimeError("Provider unavailable")
        with self.assertLogs("assistant.memory.conversations", level="WARNING"):
            self.assertIsNone(generate_conversation_title(self.conversation_id, "Question"))
        self.assertEqual(get_conversation(self.conversation_id)["title"], "New conversation")
        self.model.invoke.side_effect = None
        self.assertIsNotNone(generate_conversation_title(self.conversation_id, "Question"))

    def test_empty_title_and_empty_question_keep_placeholder(self):
        self.assertIsNone(generate_conversation_title(self.conversation_id, "  "))
        self.model.invoke.assert_not_called()
        for title in (" ", "New conversation", []):
            self.model.invoke.return_value = SimpleNamespace(content=title)
            self.assertIsNone(generate_conversation_title(self.conversation_id, "Question"))
        self.assertEqual(get_conversation(self.conversation_id)["title"], "New conversation")

    def test_title_is_single_line_and_bounded(self):
        self.model.invoke.return_value = SimpleNamespace(content='"  Research\n title  "')
        result = generate_conversation_title(self.conversation_id, "Question")
        self.assertEqual(result["title"], "Research title")
        another = create_conversation("New conversation", "topic", "retrieval")
        self.model.invoke.return_value = SimpleNamespace(content="Long title " * 30)
        result = generate_conversation_title(another["id"], "Question " * 1000)
        self.assertLessEqual(len(result["title"]), 80)
        self.assertLessEqual(len(self.model.invoke.call_args.args[0][1].content), 2000)
        self.assertEqual(result["scope_target"], "retrieval")

    def test_old_configs_fall_back_to_qa(self):
        self.config.llm.roles = {"qa": object()}
        generate_conversation_title(self.conversation_id, "Question")
        self.router.chat.assert_called_once_with("qa")

    def test_deleted_chat_is_not_recreated(self):
        def delete_during_generation(messages):
            delete_conversation(self.conversation_id)
            return SimpleNamespace(content="Generated title")

        self.model.invoke.side_effect = delete_during_generation
        self.assertIsNone(generate_conversation_title(self.conversation_id, "Question"))
        self.assertIsNone(get_conversation(self.conversation_id))

    def test_concurrent_title_change_is_preserved(self):
        def rename_during_generation(messages):
            with session_scope() as session:
                session.get(Conversation, self.conversation_id).title = "Chosen title"
            return SimpleNamespace(content="Generated title")

        self.model.invoke.side_effect = rename_during_generation
        self.assertIsNone(generate_conversation_title(self.conversation_id, "Question"))
        self.assertEqual(get_conversation(self.conversation_id)["title"], "Chosen title")

    def stream_message(self, events, question="Explain hybrid retrieval"):
        config = SimpleNamespace(rag=SimpleNamespace(history_turns=4))
        lines = [json.dumps(event) + "\n" for event in events]
        with TestClient(create_app()) as client, patch(
            "assistant.web.app.get_config", return_value=config,
        ), patch("assistant.web.app.stream_graph", return_value=iter(lines)):
            return client.post(
                f"/api/conversations/{self.conversation_id}/messages",
                json={"message": question},
                headers={"X-Requested-With": "assistant-ui"},
            )

    def test_stream_delivers_answer_then_persisted_title(self):
        response = self.stream_message([
            {"type": "progress", "stage": "answering"},
            {"type": "final", "result": {"answer": "An answer", "citations": []}},
        ])
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["type"] for event in events], ["conversation", "progress", "final", "conversation"])
        self.assertEqual(events[2]["result"]["answer"], "An answer")
        self.assertEqual(events[3]["conversation"]["title"], "Hybrid Retrieval Tradeoffs")
        self.assertEqual(get_conversation(self.conversation_id)["title"], "Hybrid Retrieval Tradeoffs")

    def test_stream_keeps_answer_if_title_model_fails(self):
        self.model.invoke.side_effect = RuntimeError("Provider unavailable")
        with self.assertLogs("assistant.memory.conversations", level="WARNING"):
            response = self.stream_message([
                {"type": "final", "result": {"answer": "An answer", "citations": []}},
            ])
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["type"] for event in events], ["conversation", "final"])
        self.assertEqual(events[1]["result"]["answer"], "An answer")

    def test_stream_does_not_name_failed_or_empty_turns(self):
        response = self.stream_message([{"type": "error", "message": "Answer failed"}])
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["type"] for event in events], ["conversation", "error"])
        self.assertEqual(self.stream_message([], question="  ").status_code, 400)
        self.model.invoke.assert_not_called()

    def test_message_metadata_filters_are_validated_and_forwarded(self):
        config = SimpleNamespace(rag=SimpleNamespace(history_turns=4))
        lines = [json.dumps({"type": "final", "result": {"answer": "Answer", "citations": []}}) + "\n"]
        with TestClient(create_app()) as client, patch("assistant.web.app.get_config", return_value=config), patch(
            "assistant.web.app.stream_graph", return_value=iter(lines),
        ) as run:
            url = f"/api/conversations/{self.conversation_id}/messages"
            headers = {"X-Requested-With": "assistant-ui"}
            response = client.post(url, json={"message": "Methods?", "filters": {
                "min_year": 2020, "section_types": ["method"],
            }}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(run.call_args.args[0]["scratch"]["metadata_filters"], {
                "min_year": 2020, "section_types": ["method"],
            })
            invalid = client.post(url, json={"message": "Methods?", "filters": {
                "min_year": 2025, "max_year": 2020,
            }}, headers=headers)
            self.assertEqual(invalid.status_code, 422)
            run.assert_called_once()

    def test_same_conversation_turns_wait_and_reload_history(self):
        first_started = threading.Event()
        finish_first = threading.Event()
        first_saved = threading.Event()
        second_started = threading.Event()
        histories = {}

        def stream(state):
            question = state["question"]
            histories[question] = state["scratch"]["history"]
            if question == "first":
                first_started.set()
                self.assertTrue(finish_first.wait(timeout=10))
                first_saved.set()
            else:
                second_started.set()
            yield json.dumps({"type": "final", "result": {"answer": question, "citations": []}}) + "\n"

        def history(*args):
            return [{"question": "first", "answer": "first"}] if first_saved.is_set() else []

        config = SimpleNamespace(rag=SimpleNamespace(history_turns=4))
        with TestClient(create_app()) as client, patch(
            "assistant.web.app.get_config", return_value=config
        ), patch("assistant.web.app.stream_graph", side_effect=stream), patch(
            "assistant.web.app.conversation_history", side_effect=history
        ), patch("assistant.web.app.generate_conversation_title", return_value=None):
            def send(message):
                return client.post(
                    f"/api/conversations/{self.conversation_id}/messages",
                    json={"message": message}, headers={"X-Requested-With": "assistant-ui"},
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(send, "first")
                self.assertTrue(first_started.wait(timeout=10))
                second = executor.submit(send, "second")
                try:
                    self.assertFalse(second_started.wait(timeout=0.1))
                finally:
                    finish_first.set()
                self.assertEqual(first.result(timeout=10).status_code, 200)
                self.assertEqual(second.result(timeout=10).status_code, 200)
        self.assertEqual(histories["first"], [])
        self.assertEqual(histories["second"], [{"question": "first", "answer": "first"}])


if __name__ == "__main__":
    unittest.main()