# -*- coding: utf-8 -*-
import json
import os
import random
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from prepare_math_majority import (group_fingerprint, main as prepare_main,
                                   normalize_problem, sample_grouped)


def row(query, original):
    return {
        "prompt": ("Solve the following math problem step by step, and put your "
                   "final answer in \\boxed{...}:\n\n" + query),
        "response": "answer \\boxed{1}",
        "original_question": original,
    }


class GroupDisjointDataTest(unittest.TestCase):
    def test_normalization_removes_only_cosmetic_differences(self):
        self.assertEqual(normalize_problem(r"$\\left( x + 1 \\right)$"),
                         normalize_problem(r"$\\left(x+1\\right)$"))
        self.assertNotEqual(normalize_problem("x+1"), normalize_problem("x-1"))

    def test_original_question_defines_group(self):
        first = row("Rewrite A", "Original problem")
        second = row("Rewrite B", " Original   problem ")
        self.assertEqual(group_fingerprint(first), group_fingerprint(second))

    def test_validation_group_is_blocked_from_training(self):
        rows = [
            row("validation rewrite", "same original"),
            row("training rewrite of same problem", "same original"),
            row("independent one", "original one"),
            row("independent two", "original two"),
        ]
        forbidden = {group_fingerprint(rows[0])}
        chosen, groups, _ = sample_grouped(
            rows, count=2, rng=random.Random(42),
            forbidden_groups=forbidden, seen_rows=set(),
            max_variants_per_group=1)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(groups.isdisjoint(forbidden))
        self.assertNotIn("same original", [r["original_question"] for r in chosen])

    def test_first_pass_maximizes_distinct_groups(self):
        rows = [row("a1", "A"), row("a2", "A"),
                row("b1", "B"), row("c1", "C")]
        chosen, groups, _ = sample_grouped(
            rows, count=3, rng=random.Random(1), forbidden_groups=set(),
            seen_rows=set(), max_variants_per_group=2)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len(groups), 3)

    def test_end_to_end_manifest_asserts_all_three_boundaries(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            math_zip = os.path.join(directory, "MATH.zip")
            with zipfile.ZipFile(math_zip, "w") as archive:
                for level in range(1, 6):
                    for index in range(2):
                        archive.writestr(
                            f"MATH/train/algebra/{level}-{index}.json",
                            json.dumps({"problem": f"official train L{level} Q{index}",
                                        "solution": "work \\boxed{1}",
                                        "level": f"Level {level}", "type": "Algebra"}))
                archive.writestr(
                    "MATH/test/algebra/test.json",
                    json.dumps({"problem": "blocked math test", "solution": "\\boxed{1}",
                                "level": "Level 1", "type": "Algebra"}))

            meta_math = os.path.join(directory, "meta_math.jsonl")
            meta_gsm = os.path.join(directory, "meta_gsm.jsonl")
            with open(meta_math, "w", encoding="utf-8") as stream:
                for index in range(4):
                    stream.write(json.dumps(row(
                        f"math rewrite {index}", f"independent math {index}")) + "\n")
                stream.write(json.dumps(row(
                    "test rewrite", "blocked math test")) + "\n")
            with open(meta_gsm, "w", encoding="utf-8") as stream:
                for index in range(5):
                    stream.write(json.dumps(row(
                        f"gsm rewrite {index}", f"independent gsm {index}")) + "\n")
                stream.write(json.dumps(row(
                    "gsm test rewrite", "blocked gsm test")) + "\n")

            gsm_path = os.path.join(directory, "gsm.parquet")
            pq.write_table(pa.table({"question": ["blocked gsm test"]}), gsm_path)
            omni_path = os.path.join(directory, "omni.jsonl")
            with open(omni_path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"problem": "blocked omni test", "answer": "1"}) + "\n")

            train_path = os.path.join(directory, "train.jsonl")
            val_path = os.path.join(directory, "val.jsonl")
            all_path = os.path.join(directory, "all.jsonl")
            manifest_path = os.path.join(directory, "manifest.json")
            argv = [
                "prepare_math_majority.py", "--math_zip", math_zip,
                "--meta_math", meta_math, "--meta_gsm8k", meta_gsm,
                "--gsm8k_test", gsm_path, "--omni_test", omni_path,
                "--meta_math_train", "2", "--meta_gsm_train", "2",
                "--meta_gsm_val", "1", "--max_variants_per_group", "1",
                "--out_train", train_path, "--out_val", val_path,
                "--out_combined", all_path, "--manifest", manifest_path,
            ]
            with mock.patch.object(sys, "argv", argv):
                prepare_main()

            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertTrue(manifest["isolation"]["asserted"])
            self.assertEqual(
                manifest["isolation"]["train_val_original_group_overlap"], 0)
            self.assertEqual(
                manifest["isolation"]["train_external_test_original_group_overlap"], 0)
            self.assertEqual(
                manifest["isolation"]["val_external_test_original_group_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
