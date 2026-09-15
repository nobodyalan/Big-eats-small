import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval_answer_nll import encode_answer_example, final_answer_span


class CharacterTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        result = SimpleNamespace(input_ids=[ord(char) for char in text])
        if return_offsets_mapping:
            result.offset_mapping = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class FinalAnswerSpanTests(unittest.TestCase):
    def test_nested_math_box(self):
        text = r"Reasoning. Therefore \\boxed{\\frac{1}{2}}."
        start, end = final_answer_span(text, "math")
        self.assertEqual(text[start:end], r"\\frac{1}{2}")

    def test_last_box_wins(self):
        text = r"First \\boxed{1}, corrected to \\boxed{2}."
        start, end = final_answer_span(text, "math")
        self.assertEqual(text[start:end], "2")

    def test_gsm8k_marker(self):
        text = "We calculate carefully.\n#### 1,024\n"
        start, end = final_answer_span(text, "gsm8k")
        self.assertEqual(text[start:end], "1,024")

    def test_missing_marker(self):
        self.assertIsNone(final_answer_span("answer 3", "math"))

    def test_only_answer_content_is_labeled(self):
        tokenizer = CharacterTokenizer()
        response = r"Reasoning. Therefore \boxed{\frac{1}{2}}."
        encoded, reason = encode_answer_example(
            tokenizer, "Question", response, "math", 512)
        self.assertIsNone(reason)
        labels = encoded["labels"][0].tolist()
        labeled_text = "".join(chr(token) for token in labels if token != -100)
        self.assertEqual(labeled_text, r"\frac{1}{2}")
        self.assertNotIn("boxed", labeled_text)

    def test_complete_answer_must_fit(self):
        encoded, reason = encode_answer_example(
            CharacterTokenizer(), "long prompt", r"work \boxed{2}",
            "math", max_len=5)
        self.assertIsNone(encoded)
        self.assertEqual(reason, "answer_sequence_truncated")


if __name__ == "__main__":
    unittest.main()
