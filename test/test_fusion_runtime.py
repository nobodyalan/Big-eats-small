# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import unittest

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core_training"))

from main import Config, attach_fusion, load_fusion, save_fusion
from train_fusion import (answer_start_char, causal_lm_loss, causal_lm_loss_parts,
                          causal_lm_loss_per_example, baseline_improvement_loss,
                          classify_branch_evaluation, diagnose_train_branch,
                          mismatched_hidden, plot_losses)


def tiny_qwen(layers=4):
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_sliding_window=False,
    )
    return Qwen3ForCausalLM(cfg).eval()


def tiny_fusion(large_end=2):
    small, large = tiny_qwen(), tiny_qwen()
    cfg = Config()
    cfg.fusion_large_start = 0
    cfg.fusion_large_end = large_end
    cfg.fusion_small_start = 1
    cfg.fusion_small_end = 3
    cfg.fusion_mlp_dim = 32
    fusion = attach_fusion(large, small, cfg)
    fusion.eval()
    return small, large, fusion


def randomize_up_projections(fusion):
    with torch.no_grad():
        for module in fusion.modules():
            if hasattr(module, "up") and isinstance(module.up, torch.nn.Linear):
                torch.nn.init.normal_(module.up.weight, std=0.02)
                torch.nn.init.zeros_(module.up.bias)


class FusionRuntimeTests(unittest.TestCase):
    def test_answer_weighting_uses_last_supported_marker(self):
        text = r"first \\boxed{2}, correction: #### 3. The answer is: 4"
        self.assertEqual(answer_start_char(text), text.index("The answer is:"))

    def test_rezero_is_exact_identity_with_nonzero_branch(self):
        _, _, fusion = tiny_fusion()
        randomize_up_projections(fusion)
        h = torch.randn(2, 5, 16)
        with torch.no_grad():
            fusion.branch_alpha.zero_()
            out = fusion(h)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_rezero_scale_learns_at_exact_zero(self):
        _, _, fusion = tiny_fusion()
        randomize_up_projections(fusion)
        h = torch.randn(1, 4, 16)
        target_direction = torch.randn(1, 4, 16)
        fusion.branch_alpha.data.zero_()
        loss = (fusion(h) * target_direction).sum()
        loss.backward()
        self.assertIsNotNone(fusion.branch_alpha.grad)
        self.assertGreater(abs(float(fusion.branch_alpha.grad)), 1e-8)

    def test_fusion_rmsnorm_gammas_are_frozen(self):
        _, _, fusion = tiny_fusion()
        norms = [m for m in fusion.modules() if isinstance(m, torch.nn.RMSNorm)]
        self.assertGreaterEqual(len(norms), 2)
        for norm in norms:
            self.assertFalse(norm.weight.requires_grad)
            self.assertTrue(torch.equal(norm.weight, torch.ones_like(norm.weight)))

    def test_cached_generation_matches_full_causal_segment(self):
        _, _, fusion = tiny_fusion()
        randomize_up_projections(fusion)
        h = torch.randn(1, 7, 16)
        with torch.no_grad():
            fusion.branch_alpha.fill_(0.25)
            full = fusion(h)
            fusion.begin_generation()
            incremental = torch.cat([fusion(h[:, i:i + 1])
                                     for i in range(h.size(1))], dim=1)
            fusion.end_generation()
        self.assertTrue(torch.allclose(full, incremental, atol=2e-5, rtol=2e-4),
                        f"max diff={float((full - incremental).abs().max())}")
        self.assertFalse(fusion._generation_mode)
        self.assertIsNone(fusion._small_cache)

    def test_checkpoint_rejects_wrong_large_positions(self):
        _, _, fusion = tiny_fusion(large_end=2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "fusion.pt")
            save_fusion(fusion, path)
            _, _, wrong = tiny_fusion(large_end=1)
            with self.assertRaises(ValueError):
                load_fusion(wrong, path)

    def test_legacy_sigmoid_checkpoint_still_loads(self):
        _, _, fusion = tiny_fusion()
        legacy = {k: v for k, v in fusion.state_dict().items()
                  if k != "branch_alpha"}
        legacy["gate_logit"] = torch.tensor(1.0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "legacy.pt")
            torch.save(legacy, path)
            _, _, restored = tiny_fusion()
            load_fusion(restored, path)
        self.assertEqual(restored.gate_mode, "sigmoid")
        self.assertAlmostEqual(float(restored.scale()),
                               float(torch.sigmoid(torch.tensor(1.0))), places=6)

    def test_loss_parts_support_token_weighted_cross_batch_aggregation(self):
        torch.manual_seed(0)
        logits1 = torch.randn(1, 4, 7)
        labels1 = torch.tensor([[-100, 1, 2, 3]])
        logits2 = torch.randn(1, 3, 7)
        labels2 = torch.tensor([[-100, -100, 4]])
        s1, n1 = causal_lm_loss_parts(logits1, labels1)
        s2, n2 = causal_lm_loss_parts(logits2, labels2)
        combined = (s1 + s2) / (n1 + n2)
        wrong_batch_mean = (causal_lm_loss(logits1, labels1)
                            + causal_lm_loss(logits2, labels2)) / 2
        self.assertNotAlmostEqual(float(combined), float(wrong_batch_mean), places=5)
        self.assertEqual(float(n1), 3.0)
        self.assertEqual(float(n2), 1.0)

    def test_baseline_improvement_is_paired_and_margin_based(self):
        logits = torch.zeros(2, 3, 4)
        labels = torch.tensor([[-100, 1, 2], [-100, 2, 3]])
        logits[0, 0, 1] = logits[0, 1, 2] = 8.0
        logits[1, 0, 2] = logits[1, 1, 3] = 8.0
        correct = causal_lm_loss_per_example(logits, labels)
        inactive, inactive_ratio = baseline_improvement_loss(
            logits, correct + 0.1, labels, margin=0.02)
        active, active_ratio = baseline_improvement_loss(
            logits, correct - 0.01, labels, margin=0.02)
        self.assertAlmostEqual(float(inactive), 0.0, places=6)
        self.assertAlmostEqual(float(inactive_ratio), 0.0, places=6)
        self.assertGreater(float(active), 0.0)
        self.assertAlmostEqual(float(active_ratio), 1.0, places=6)

    def test_branch_health_diagnostics_detect_drop_and_usefulness(self):
        warnings = diagnose_train_branch(
            1e-8, 0.0, 0.0, alpha_grad=0.0, alpha_trainable=True)
        self.assertTrue(any("幅度过低" in warning for warning in warnings))
        self.assertTrue(any("alpha 接近零" in warning for warning in warnings))
        self.assertTrue(any("bridge 梯度近零" in warning for warning in warnings))
        self.assertEqual(
            classify_branch_evaluation(0.8, 1.0, 1.1)[0], "USEFUL")
        self.assertEqual(
            classify_branch_evaluation(1.0, 1.0, 1.0)[0], "DROPPED")
        self.assertEqual(
            classify_branch_evaluation(1.1, 1.0, 1.2)[0], "HARMFUL")

    def test_mismatched_hidden_avoids_source_padding(self):
        h = torch.full((3, 5, 2), -99.0)
        mask = torch.tensor([[1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1]])
        for i, n in enumerate(mask.sum(dim=1).tolist()):
            h[i, :n] = float(i + 1)
        wrong = mismatched_hidden(h, mask)
        for i, n in enumerate(mask.sum(dim=1).tolist()):
            self.assertFalse(torch.any(wrong[i, :n] == -99))
            self.assertFalse(torch.any(wrong[i, :n] == float(i + 1)))
        self.assertFalse(wrong.requires_grad)

    def test_plot_losses_accepts_fusion_and_lora_histories(self):
        with tempfile.TemporaryDirectory() as td:
            fusion_path = os.path.join(td, "fusion.png")
            lora_path = os.path.join(td, "lora.png")
            plot_losses([(1, 1.0)], [(1, 0.9, 1.0, 1.1)], fusion_path)
            plot_losses([(1, 1.0)], [(1, 0.9, 1.0)], lora_path)
            self.assertTrue(os.path.isfile(fusion_path))
            self.assertTrue(os.path.isfile(lora_path))


if __name__ == "__main__":
    unittest.main()
