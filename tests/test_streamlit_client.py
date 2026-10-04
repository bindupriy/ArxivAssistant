import json
import unittest

import httpx

from assistant.web.api_client import AssistantAPI, AssistantAPIError


class StreamlitClientTests(unittest.TestCase):
    def test_loopback_only_and_api_write_header(self):
        for url in ("https://127.0.0.1:8000", "http://example.org:8000", "http://127.0.0.1:8000/api"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                AssistantAPI(url)

        requests = []

        def respond(request):
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"paper_count": 0})
            if request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(201, json={"id": "conversation-1"})

        api = AssistantAPI(transport=httpx.MockTransport(respond))
        try:
            self.assertEqual(api.get("/status"), {"paper_count": 0})
            self.assertEqual(api.post("/conversations", {"scope_type": "library"})["id"], "conversation-1")
            api.delete("/conversations/conversation-1")
            self.assertEqual([request.url.path for request in requests], [
                "/api/status", "/api/conversations", "/api/conversations/conversation-1",
            ])
            self.assertNotIn("X-Requested-With", requests[0].headers)
            for request in requests[1:]:
                self.assertEqual(request.headers["X-Requested-With"], "assistant-ui")
        finally:
            api.close()

    def test_upload_is_multipart_and_errors_are_reported(self):
        def respond(request):
            self.assertEqual(request.headers["X-Requested-With"], "assistant-ui")
            self.assertTrue(request.headers["content-type"].startswith("multipart/form-data"))
            self.assertIn(b"%PDF-1.7", request.content)
            self.assertIn(b"topic-one", request.content)
            return httpx.Response(202, json={"status": "queued"})

        api = AssistantAPI(transport=httpx.MockTransport(respond))
        try:
            self.assertEqual(api.upload_pdf("paper.pdf", b"%PDF-1.7", "topic-one")["status"], "queued")
        finally:
            api.close()

        failure = AssistantAPI(transport=httpx.MockTransport(
            lambda request: httpx.Response(422, json={"detail": "Invalid section"}),
        ))
        try:
            with self.assertRaisesRegex(AssistantAPIError, "422: Invalid section"):
                failure.post("/conversations", {"scope_type": "invalid"})
        finally:
            failure.close()

    def test_stream_keeps_reading_for_title_after_final(self):
        seen = []

        def respond(request):
            self.assertEqual(request.headers["X-Requested-With"], "assistant-ui")
            seen.append(json.loads(request.content))
            lines = [
                {"type": "progress", "stage": "retrieving"},
                {"type": "final", "result": {"answer": "Evidence [1]", "citations": ["chunk-1"]}},
                {"type": "conversation", "conversation": {"title": "Named chat"}},
            ]
            return httpx.Response(200, content="\n".join(json.dumps(item) for item in lines) + "\n")

        api = AssistantAPI(transport=httpx.MockTransport(respond))
        try:
            events = list(api.stream_message("conversation-1", "How?", {"min_year": 2020}))
            self.assertEqual([event["type"] for event in events], ["progress", "final", "conversation"])
            self.assertEqual(seen, [{"message": "How?", "filters": {"min_year": 2020}}])
        finally:
            api.close()

    def test_missing_final_or_error_event_is_not_silently_accepted(self):
        api = AssistantAPI(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='{"type":"progress","stage":"retrieving"}\n'),
        ))
        try:
            with self.assertRaisesRegex(AssistantAPIError, "without an answer"):
                list(api.stream_message("conversation-1", "How?"))
        finally:
            api.close()


if __name__ == "__main__":
    unittest.main()