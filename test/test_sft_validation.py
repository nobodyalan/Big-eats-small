# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval_math import load_sft_validation


class SftValidationTest(unittest.TestCase):
    def _dataset(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=False)
        prefix = ("Solve the following math problem step by step, and put your final "
                  "answer in \\boxed{...}:\n\n")
        with handle:
            for level in (1, 2):
                for index in range(3):
                    handle.write(json.dumps({
                        "prompt": prefix + f"MATH {level}-{index}",
                        "response": f"work \\boxed{{{level + index}}}",
                        "original_question": f"MATH {level}-{index}",
                        "source": "MATH_train",
                        "level": level,
                    }) + "\n")
            for index in range(3):
                handle.write(json.dumps({
                    "prompt": prefix + f"GSM {index}",
                    "response": f"work. The answer is: {index}",
                    "original_question": f"GSM {index}",
                    "source": "MetaMathQA_GSM8K",
                }) + "\n")
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_balanced_stratified_selection_is_reproducible(self):
        path = self._dataset()
        first = load_sft_validation(path, seed=42, limit=0, limit_per_stratum=2)
        second = load_sft_validation(path, seed=42, limit=0, limit_per_stratum=2)
        self.assertEqual([row["benchmark_id"] for row in first],
                         [row["benchmark_id"] for row in second])
        self.assertEqual(len(first), 6)
        counts = {name: sum(row["stratum"] == name for row in first)
                  for name in ("math_L1", "math_L2", "gsm8k")}
        self.assertEqual(counts, {"math_L1": 2, "math_L2": 2, "gsm8k": 2})
        self.assertTrue(all(not row["problem"].startswith("Solve the")
                            for row in first))


if __name__ == "__main__":
    unittest.main()
