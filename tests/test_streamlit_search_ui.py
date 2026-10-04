from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # The Streamlit UI is an optional installation extra.
    AppTest = None


@unittest.skipUnless(AppTest is not None, "install the optional streamlit extra")
class StreamlitSearchTests(unittest.TestCase):
    def test_chat_can_find_and_queue_arxiv_paper_without_a_terminal(self):
        posted = []

        def get(api, path, **params):
            if path == "/status":
                return {"paper_count": 0, "rag_mode": "vanilla"}
            if path == "/arxiv/search":
                self.assertEqual(params["q"], "RoPE")
                return [{"arxiv_id": "2104.09864v5", "title": "RoFormer",
                         "abstract": "Rotary position embeddings", "authors": [],
                         "year": 2021, "paper_id": None}]
            return []

        def post(api, path, body=None):
            posted.append((path, body))
            return {"id": "queued-test-job", "status": "queued"}

        file_path = Path(__file__).resolve().parents[1] / "assistant" / "streamlit_app.py"
        with patch("assistant.web.api_client.AssistantAPI.get", autospec=True, side_effect=get), patch(
            "assistant.web.api_client.AssistantAPI.post", autospec=True, side_effect=post,
        ):
            app = AppTest.from_file(str(file_path)).run(timeout=30)
            self.assertFalse(app.exception)
            next(item for item in app.text_input if item.key == "arxiv-query-chat").input("RoPE")
            next(item for item in app.button if item.label == "Search arXiv").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any(item.label == "Download PDF and index paper" for item in app.button))
            next(item for item in app.button if item.label == "Download PDF and index paper").click().run(timeout=30)
            self.assertFalse(app.exception)
        self.assertEqual(posted, [
            ("/papers/arxiv", {"arxiv_id": "2104.09864v5", "domain": None}),
        ])


if __name__ == "__main__":
    unittest.main()