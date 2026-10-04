from datetime import datetime
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from assistant.rag.arxiv_fetch import search_papers


class ArxivSearchTests(unittest.TestCase):
    def test_topic_search_ranks_title_matches_and_bounds_results(self):
        class FakeSearch:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def paper(arxiv_id, title, abstract):
            return SimpleNamespace(
                get_short_id=lambda: arxiv_id, title=title,
                summary=abstract, authors=[SimpleNamespace(name="Researcher")],
                published=datetime(2024, 1, 1),
            )

        raw = [
            paper("2401.00001v1", "A generic model", "A RoPE embeddings comparison"),
            paper("2104.09864v5", "RoFormer Rotary Position Embedding", "Uses RoPE for attention"),
        ]

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def results(self, search):
                if "id_list" in search.kwargs:
                    self_outer.assertEqual(search.kwargs["id_list"], ["2104.09864"])
                    return iter([raw[1]])
                self_outer.assertEqual(search.kwargs["max_results"], 30)
                return iter(raw)

        self_outer = self
        fake = SimpleNamespace(
            Search=FakeSearch,
            SortCriterion=SimpleNamespace(Relevance="relevance"),
            Client=FakeClient,
        )
        with patch.dict(sys.modules, {"arxiv": fake}):
            matches = search_papers("RoPE embeddings", limit=1)
            self.assertEqual(matches[0]["arxiv_id"], "2104.09864v5")
            self.assertEqual(matches[0]["year"], 2024)
            self.assertEqual(len(matches), 1)
            self.assertEqual(search_papers("2104.09864")[0]["title"], raw[1].title)
            self.assertEqual(search_papers("  !  "), [])


if __name__ == "__main__":
    unittest.main()