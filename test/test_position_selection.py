# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core_training"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from validate_positions import inclusive_small_end, load_candidates
from compare_position_predictions import exact_mcnemar_p

try:
    import torch
    from main import GatedAdapter
    from select_positions import (build_task_candidates, rank_task_aware_results,
                                  ridge_affine)
    from train_fusion import JS_MARGIN, js_contrastive
except ModuleNotFoundError:
    torch = None


class PositionSelectionTests(unittest.TestCase):
    def test_exclusive_end_conversion(self):
        self.assertEqual(inclusive_small_end(4), 3)
        self.assertEqual(inclusive_small_end(1), 0)
        with self.assertRaises(ValueError):
            inclusive_small_end(0)

    def test_candidate_json_preserves_exclusive_end_and_l2(self):
        payload = {"task_aware": [
            {"L": 17, "l2": 29, "a": 3, "b": 4,
             "b_semantics": "exclusive"}
        ]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "candidates.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            self.assertEqual(load_candidates(path, 1), [(17, 29, 3, 4)])

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 5), 0.0625)



@unittest.skipUnless(torch is not None, "服务器模型环境才安装 torch/transformers")
class TorchPositionSelectionTests(unittest.TestCase):
    def test_deep_bridge_starts_as_zero_mapping(self):
        adapter = GatedAdapter(8, 4, 16, depth=2)
        self.assertEqual(len(adapter.blocks), 1)
        out = adapter(torch.randn(2, 3, 8))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_task_candidates_keep_default_and_length_diversity(self):
        entries = [{"L": 17, "a": 3}, {"L": 8, "a": 10},
                   {"L": 28, "a": 19}]
        full = []
        for e in entries:
            for length in (1, 2, 4):
                full.append({"L": e["L"], "a": e["a"],
                             "b": e["a"] + length,
                             "e_seg_amp": float(length)})
        out = build_task_candidates(entries, full, 36, 28,
                                    lengths=[1, 2, 4], spans=[6, 12], limit=8)
        self.assertEqual(out[0], (12, 24, 9, 19))
        self.assertGreater(len({b - a for _, _, a, b in out[1:]}), 1)
        self.assertEqual({(L, a) for L, _, a, _ in out[1:4]},
                         {(17, 3), (8, 10), (28, 19)})
        self.assertGreater(len({l2 - L for L, l2, _, _ in out[1:]}), 1)

    def test_task_ranking_requires_value_beyond_bridge_control(self):
        shallow = {"name": "shallow", "rank_delta_nll": -8.125,
                   "incremental_delta_nll": 4.09375, "Q": 0.0129}
        useful = {"name": "useful", "rank_delta_nll": -2.3125,
                  "incremental_delta_nll": -0.875, "Q": 0.0294}
        self.assertEqual(rank_task_aware_results([shallow, useful])[0]["name"],
                         "useful")

    def test_ridge_affine_recovers_affine_mapping(self):
        torch.manual_seed(0)
        x = torch.randn(128, 5)
        w = torch.randn(5, 3)
        bias = torch.randn(3)
        y = x @ w + bias
        got_w, got_bias = ridge_affine(x, y, 1e-8)
        pred = x @ got_w + got_bias
        self.assertLess(float((pred - y).pow(2).mean()), 1e-8)

    def test_js_contrastive_uses_forward_js_in_nats(self):
        labels = torch.tensor([[0, 1, 0]])
        same = torch.zeros(1, 3, 2)
        self.assertAlmostEqual(float(js_contrastive(same, same, labels)),
                               JS_MARGIN, places=5)
        p = torch.tensor([[[8.0, -8.0], [8.0, -8.0], [8.0, -8.0]]])
        q = -p
        self.assertLess(float(js_contrastive(p, q, labels)), 1e-3)


if __name__ == "__main__":
    unittest.main()
