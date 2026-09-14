# -*- coding: utf-8 -*-
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from format_math_sft_answers import canonicalize_response, extract_final_answer


class FormatMathSftAnswersTest(unittest.TestCase):
    def test_last_explicit_marker_wins(self):
        response = r"Intermediate \boxed{2}.\n#### 3\nThe answer is: 4"
        answer, marker = extract_final_answer(response.replace(r"\n", "\n"))
        self.assertEqual((answer, marker), ("4", "answer_is"))

    def test_balanced_boxed_is_preserved(self):
        formatted, answer, marker = canonicalize_response(
            r"Work. The answer is: \frac{3}{4}.")
        self.assertEqual(answer, r"\frac{3}{4}")
        self.assertEqual(marker, "answer_is")
        self.assertTrue(formatted.endswith(r"Final answer: \boxed{\frac{3}{4}}"))

    def test_unbraced_official_math_boxed_is_supported(self):
        formatted, answer, marker = canonicalize_response(
            r"It follows that $x^2 + y^2 = \boxed 9$.")
        self.assertEqual(answer, "9")
        self.assertEqual(marker, "boxed")
        self.assertTrue(formatted.endswith(r"Final answer: \boxed{9}"))

    def test_previous_boxed_and_trailing_marker_are_deduplicated(self):
        formatted, answer, marker = canonicalize_response(
            r"Work gives $\boxed{(2,8)}$. The answer is: (2,8)")
        self.assertEqual(answer, "(2,8)")
        self.assertEqual(marker, "answer_is")
        self.assertEqual(formatted.count(r"\boxed"), 1)
        self.assertNotIn("The answer is", formatted)
        self.assertTrue(formatted.endswith(r"Final answer: \boxed{(2,8)}"))

    def test_manual_override_repairs_nested_empty_box(self):
        formatted, answer, marker = canonicalize_response(
            r"All terms are composite, so the answer is $\boxed{\boxed{}}$.",
            answer_override="0")
        self.assertEqual(answer, "0")
        self.assertEqual(marker, "manual_correction")
        self.assertEqual(formatted.count(r"\boxed"), 1)
        self.assertTrue(formatted.endswith(r"Final answer: \boxed{0}"))

    def test_canonicalization_is_idempotent(self):
        original = r"Work. Final answer: \boxed{7}"
        formatted, answer, marker = canonicalize_response(original)
        self.assertEqual(formatted, original)
        self.assertEqual(answer, "7")
        self.assertEqual(marker, "already_canonical")

    def test_missing_marker_fails(self):
        with self.assertRaises(ValueError):
            canonicalize_response("Only reasoning, no explicit final answer.")


if __name__ == "__main__":
    unittest.main()
