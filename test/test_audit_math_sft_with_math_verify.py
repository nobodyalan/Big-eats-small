# -*- coding: utf-8 -*-
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_math_sft_with_math_verify import inspect_response


class DummyConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_parse(text, extraction_config, raise_on_error, **kwargs):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return [matches[-1]] if matches else []


def fake_verify(gold, prediction, raise_on_error, **kwargs):
    return bool(gold and prediction and gold[-1] == prediction[-1])


RUNTIME = {
    "parse": fake_parse,
    "verify": fake_verify,
    "expr_config": DummyConfig,
    "latex_config": DummyConfig,
}


class AuditMathSftTest(unittest.TestCase):
    def test_canonical_full_response_is_equivalent(self):
        result = inspect_response(
            r"Reasoning with intermediate \boxed{2}. Final answer: \boxed{7}",
            RUNTIME)
        self.assertTrue(result["canonical_format"])
        self.assertTrue(result["gold_parseable"])
        self.assertTrue(result["full_parseable"])
        self.assertTrue(result["full_equivalent"])
        self.assertEqual(result["reasons"], [])

    def test_noncanonical_response_is_reported(self):
        result = inspect_response(r"Reasoning. The answer is: 7", RUNTIME)
        self.assertFalse(result["canonical_format"])
        self.assertIn("not_canonical", result["reasons"])


if __name__ == "__main__":
    unittest.main()
