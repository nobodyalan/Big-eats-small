# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_segments_parallel import BASE_TASKS, OMNI_TASK, build_summary


class ParallelEvalSuiteTest(unittest.TestCase):
    def _write_result(self, directory, filename, task, n=400, seed=42,
                      question_hash="a" * 64, errors=0,
                      suite_run_id="test-run"):
        summary = {"n": n, "lora_correct": 100, "lora_acc": 0.25}
        if task != "gsm8k":
            summary.update({
                "lora_math_verify_correct": 120,
                "lora_math_verify_acc": 0.30,
                "lora_math_verify_errors": errors,
            })
        payload = {
            "seed": seed,
            "suite_run_id": suite_run_id,
            "math_verify_version": "0.9.0",
            "question_set_sha256": question_hash,
            "summary": summary,
        }
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_four_tier_summary_accepts_exact_suite(self):
        tasks = [*BASE_TASKS, OMNI_TASK]
        with tempfile.TemporaryDirectory() as directory:
            for index, (task, filename, _) in enumerate(tasks):
                self._write_result(
                    directory, filename, task,
                    question_hash=f"{index + 1:064x}")
            report, errors = build_summary(
                Path(directory), tasks, expected_n=400,
                expected_seed=42, tag="test",
                expected_suite_run_id="test-run")
        self.assertEqual(errors, [])
        self.assertIn("omni_math", report)
        self.assertIn("validation: OK", report)

    def test_summary_rejects_wrong_count_and_judge_error(self):
        tasks = [*BASE_TASKS, OMNI_TASK]
        with tempfile.TemporaryDirectory() as directory:
            for task, filename, _ in tasks:
                self._write_result(directory, filename, task)
            self._write_result(
                directory, "04_omni_math.json", "omni_math",
                n=180, errors=1)
            _, errors = build_summary(
                Path(directory), tasks, expected_n=400,
                expected_seed=42, tag="test",
                expected_suite_run_id="test-run")
        self.assertTrue(any("实际 180 题" in error for error in errors))
        self.assertTrue(any("解析异常 1 题" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
