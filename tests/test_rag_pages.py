from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import fitz

from assistant.rag.chunker import chunk_paper
from assistant.rag.parser import _structure, parse_pdf


class PageChunkTests(unittest.TestCase):
    def test_pdf_page_chunks_preserve_offsets_and_heading(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pages.pdf"
            with fitz.open() as doc:
                for text in ("Introduction\n" + "first page result " * 20,
                             "second page result " * 20):
                    page = doc.new_page()
                    page.insert_text((72, 72), text, fontsize=10)
                doc.save(path)
            parsed = parse_pdf(path)
            chunks = chunk_paper("paper", parsed, max_chars=90, overlap_chars=15)
            self.assertEqual({chunk.page_number for chunk in chunks}, {1, 2})
            self.assertTrue(all(chunk.section_type == "intro" for chunk in chunks))
            for chunk in chunks:
                self.assertEqual(parsed.full_text[chunk.char_start:chunk.char_end], chunk.text)
                page_start, page_end = parsed.page_ranges[chunk.page_number - 1]
                self.assertGreaterEqual(chunk.char_start, page_start)
                self.assertLessEqual(chunk.char_end, page_end)

    def test_short_pages_remain_separate_and_missing_pages_fall_back(self):
        full = "Abstract\nFirst page summary.\nSecond page summary."
        first_end = len("Abstract\nFirst page summary.")
        parsed = _structure(full, [(0, first_end), (first_end + 1, len(full))])
        chunks = chunk_paper("paper", parsed)
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 2])
        self.assertEqual([chunk.text for chunk in chunks], ["First page summary.", "Second page summary."])
        legacy = chunk_paper("paper", _structure(full))
        self.assertEqual(len(legacy), 1)
        self.assertIsNone(legacy[0].page_number)

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_paper("paper", _structure("Body"), max_chars=10, overlap_chars=10)


if __name__ == "__main__":
    unittest.main()