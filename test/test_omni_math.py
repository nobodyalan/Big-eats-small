# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from omni_math_utils import (load_omni_math_items, parse_level_set,
                             recommend_level_range, summarize_by_difficulty)


class OmniMathUtilsTest(unittest.TestCase):
    def _dataset(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=False)
        with handle:
            for level in (1, 2, 3):
                for index in range(3):
                    handle.write(json.dumps({
                        "benchmark_id": f"L{level}-{index}",
                        "problem": f"problem {level}-{index}",
                        "answer": str(level + index),
                        "difficulty": level,
                    }) + "\n")
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_parse_level_set(self):
        self.assertEqual(parse_level_set("1-3,5"), {1, 2, 3, 5})
        self.assertIsNone(parse_level_set("all"))
        with self.assertRaises(ValueError):
            parse_level_set("11")

    def test_balanced_total_limit_is_reproducible(self):
        path = self._dataset()
        first = load_omni_math_items(path, seed=42, limit=4, levels="1-3")
        second = load_omni_math_items(path, seed=42, limit=4, levels="1-3")
        self.assertEqual([x["benchmark_id"] for x in first],
                         [x["benchmark_id"] for x in second])
        counts = {level: sum(x["difficulty"] == level for x in first)
                  for level in (1, 2, 3)}
        self.assertEqual(sorted(counts.values()), [1, 1, 2])

    def test_limit_per_level(self):
        items = load_omni_math_items(
            self._dataset(), seed=7, levels="1-3", limit_per_level=2)
        self.assertEqual(len(items), 6)
        self.assertEqual({x["difficulty"] for x in items}, {1, 2, 3})

    def test_fractional_difficulty_uses_integer_bands(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=False)
        with handle:
            for difficulty in (4.375, 7.5, 9.5):
                handle.write(json.dumps({
                    "problem": f"problem {difficulty}",
                    "answer": "1",
                    "difficulty": difficulty,
                }) + "\n")
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        items = load_omni_math_items(handle.name, levels="5-8")
        self.assertEqual([x["difficulty"] for x in items], [7.5])
        self.assertEqual(items[0]["difficulty_band"], 7)

    def test_summary_and_recommendation(self):
        rows = []
        # L5 偏简单，L6-L8 落入 10%-40%，L9 偏难。
        correct_by_level = {5: 6, 6: 4, 7: 3, 8: 2, 9: 0}
        for level, correct in correct_by_level.items():
            for index in range(10):
                rows.append({"difficulty": level,
                             "fusion_correct": index < correct})
        by_level = summarize_by_difficulty(rows)
        result = recommend_level_range(
            by_level, metric="fusion", target_min=0.10,
            target_max=0.40, min_samples=10)
        self.assertEqual(result["level_spec"], "6-8")


if __name__ == "__main__":
    unittest.main()
