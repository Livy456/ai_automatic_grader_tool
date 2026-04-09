"""Tests for tolerant JSON extraction from LLM chat content."""

import json
import unittest

from app.grading.llm_router import parse_llm_json_content


class ParseLlmJsonTests(unittest.TestCase):
    def test_raw_decode_stops_after_first_object(self) -> None:
        s = '{"overall": {"score": 1}, "criteria": []}\nHere is extra explanation.'
        out = parse_llm_json_content(s)
        self.assertEqual(out["overall"]["score"], 1)

    def test_markdown_fence(self) -> None:
        s = '```json\n{"plan":[{"step":"x","tool":"none","notes":""}]}\n```'
        out = parse_llm_json_content(s)
        self.assertIn("plan", out)

    def test_leading_garbage_before_brace(self) -> None:
        s = 'Sure! Here is JSON:\n{"flags":[]}'
        out = parse_llm_json_content(s)
        self.assertEqual(out["flags"], [])

    def test_empty_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_llm_json_content("   ")


if __name__ == "__main__":
    unittest.main()
