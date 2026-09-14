import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_eval_null_answers import audit_file, prediction_labels


class AuditEvalNullAnswersTests(unittest.TestCase):
    def test_discovers_primary_prediction_labels_only(self):
        rows = [{
            "fusion_pred": None,
            "fusion_relaxed_pred": "2",
            "fusion_math_verify_pred": ["2"],
            "lora_pred": "3",
        }]
        self.assertEqual(prediction_labels(rows), ["fusion", "lora"])

    def test_counts_null_and_confirmed_length_stop(self):
        payload = {
            "summary": {"bench": "MATH_HIGH"},
            "results": [
                {
                    "id": 1,
                    "fusion_pred": None,
                    "fusion_text": "unfinished reasoning",
                    "fusion_relaxed_pred": None,
                    "fusion_math_verify_extracted": False,
                    "fusion_hit_max_new_tokens": True,
                },
                {
                    "id": 2,
                    "fusion_pred": None,
                    "fusion_text": "The answer is: 7",
                    "fusion_relaxed_pred": "7",
                    "fusion_math_verify_extracted": True,
                    "fusion_math_verify_correct": True,
                    "fusion_finish_reason": "eos",
                },
                {"id": 3, "fusion_pred": "9", "fusion_finish_reason": "eos"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = audit_file(path, ["fusion"])

        item = report["labels"][0]
        self.assertEqual(item["null"], 2)
        self.assertEqual(item["null_with_nonempty_raw_text"], 2)
        self.assertEqual(item["null_recovered_by_relaxed_parser"], 1)
        self.assertEqual(item["null_extracted_by_math_verify"], 1)
        self.assertEqual(item["null_correct_by_math_verify"], 1)
        self.assertEqual(item["null_stop_metadata_n"], 2)
        self.assertEqual(item["null_confirmed_length_stops"], 1)


if __name__ == "__main__":
    unittest.main()
