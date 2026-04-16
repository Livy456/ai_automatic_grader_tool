"""PDF verticalized text reflow + journal Q/A chunk pairing."""

from __future__ import annotations

import unittest

from app.grading.submission_chunks import build_submission_chunks
from app.grading.tools import normalize_verticalized_pdf_text


def _verticalize(sentence: str) -> str:
    """Mimic pypdf one-token-per-line exports (blank lines between tokens)."""
    return "\n\n".join(sentence.split())


class PdfVerticalReflowTests(unittest.TestCase):
    def test_reflows_word_per_line_layout(self) -> None:
        raw = _verticalize(
            "Did you select a specific color palette different from the one provided? "
            "If so why? Which type of chart do you prefer for your data, line plots or scatter plots?"
        )
        fixed = normalize_verticalized_pdf_text(raw)
        self.assertNotIn("\n\nDid", fixed)
        self.assertIn("Did you select", fixed)
        self.assertIn("scatter plots?", fixed)

    def test_does_not_corrupt_normal_pdf_lines(self) -> None:
        normal = (
            "This is a normal paragraph with enough content on one line.\n\n"
            "Second paragraph here also long enough to stay as layout text."
        )
        self.assertEqual(normalize_verticalized_pdf_text(normal), normal)


class JournalPdfChunkPairingTests(unittest.TestCase):
    def test_free_response_modality_enables_journal_prompt_boundaries(self) -> None:
        """Same journal logic when modality is free_response (not only journal_entry)."""
        intro = _verticalize(
            "Homework 2 After completing the second Climate Data colab please answer the questions below."
        )
        q1 = _verticalize(
            "What questions do your visualizations bring up about the climate?"
        )
        a1 = _verticalize("I think the charts show warming in the northern regions.")
        raw = "\n\n".join([intro, q1, a1])
        full = f"=== PDF TEXT ===\n{raw}"

        chunks = build_submission_chunks(
            full,
            assignment_title="test",
            modality_subtype="free_response",
            max_chunk_chars=None,
        )
        questions = [c["text"] for c in chunks if c.get("role") == "question"]
        self.assertTrue(
            any("What questions do your visualizations" in t for t in questions),
            questions,
        )

    def test_pdf_section_reflows_without_prior_extract_normalize(self) -> None:
        """Defense in depth: chunker reflows vertical PDF text even if not pre-normalized."""
        q = _verticalize(
            "Did you finish the reading assigned for this week and take notes on the main themes?"
        )
        a = _verticalize(
            "Yes I completed all of the assigned chapters on time and wrote a short summary."
        )
        full = f"=== PDF TEXT ===\n{q}\n\n{a}"
        chunks = build_submission_chunks(
            full,
            assignment_title="test",
            modality_subtype="journal_entry",
            max_chunk_chars=None,
        )
        qchunks = [c for c in chunks if c.get("role") == "question"]
        rchunks = [c for c in chunks if c.get("role") == "response"]
        self.assertTrue(
            any("Did you finish the reading" in c["text"] for c in qchunks),
            [c["text"] for c in qchunks],
        )
        self.assertTrue(
            any("Yes I completed" in c["text"] for c in rchunks),
            [c["text"] for c in rchunks],
        )

    def test_pairs_reflowed_prompt_with_student_answer(self) -> None:
        intro = _verticalize(
            "Homework 2 After completing the second Climate Data colab please answer the questions below."
        )
        q1 = _verticalize(
            "What questions do your visualizations bring up about the climate?"
        )
        a1 = _verticalize(
            "I really wonder the population growth my visualizations showed that there was "
            "a large increase in population but then it got dipped and its now decreasing and "
            "I would want to investigate what is causing this."
        )
        q2 = _verticalize(
            "Did you select a specific color palette different from the one provided? "
            "If so why? Which type of chart do you prefer for your data, line plots or scatter plots?"
        )
        a2 = _verticalize(
            "I kept the default ones because I felt they were easy to read the blue ones on white and red."
        )
        raw_pdf_body = "\n\n".join([intro, q1, a1, q2, a2])
        body = normalize_verticalized_pdf_text(raw_pdf_body)
        full = f"=== PDF TEXT ===\n{body}"

        chunks = build_submission_chunks(
            full,
            assignment_title="[Student 1] Journal Entry - Week 4",
            modality_subtype="journal_entry",
            max_chunk_chars=None,
        )

        questions = [c["text"] for c in chunks if c.get("role") == "question"]
        self.assertTrue(
            any("What questions do your visualizations" in t for t in questions),
            f"Expected homework question in question chunks; got {questions!r}",
        )
        self.assertTrue(
            any("Did you select" in t and "scatter plots?" in t for t in questions),
            f"Expected palette / chart prompt in question chunks; got {questions!r}",
        )
        self.assertFalse(
            any(t.strip().lower() == "questions" for t in questions),
            "Bare word 'questions' must not be a question header.",
        )

        by_pair: dict[int, list[dict]] = {}
        for c in chunks:
            pid = c.get("pair_id")
            if pid is None:
                continue
            by_pair.setdefault(int(pid), []).append(c)

        for pid, group in by_pair.items():
            qtxt = " ".join(
                x["text"] for x in group if x.get("role") == "question"
            )
            rtxt = " ".join(
                x["text"] for x in group if x.get("role") == "response"
            )
            if "Did you select" in qtxt:
                self.assertIn("I kept the default", rtxt, f"pair {pid}: {qtxt=!r} {rtxt=!r}")


if __name__ == "__main__":
    unittest.main()
