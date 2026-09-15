# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core_training"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from main import (Config, attach_fusion, attached_fusions, load_multi_fusion,
                  save_multi_fusion)
from train_multi_fusion import (bridge_parameters, randomize_bridge_outputs,
                                scale_parameter, set_bridge_trainable)
from eval_math import attach_multi_fusion_checkpoint, generate


def tiny_qwen(layers=6):
    config = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_sliding_window=False,
    )
    return Qwen3ForCausalLM(config).eval()


def fusion_config(l1, l2, s1, s2):
    config = Config()
    config.fusion_large_start = l1
    config.fusion_large_end = l2
    config.fusion_small_start = s1
    config.fusion_small_end = s2
    config.fusion_mlp_dim = 32
    return config


class MultiFusionRuntimeTests(unittest.TestCase):
    def make_pair(self):
        torch.manual_seed(7)
        small, large = tiny_qwen(), tiny_qwen()
        upstream = attach_fusion(
            large, small, fusion_config(0, 1, 0, 1), name="upstream")
        downstream = attach_fusion(
            large, small, fusion_config(2, 4, 2, 3), name="downstream")
        for fusion in (upstream, downstream):
            randomize_bridge_outputs(fusion)
            fusion.branch_alpha.data.fill_(0.1)
        return small, large, upstream, downstream

    def test_two_hooks_are_registered_without_overwriting_single_alias(self):
        _, large, upstream, downstream = self.make_pair()
        self.assertIs(large._bes_fusion, upstream)
        self.assertEqual(attached_fusions(large), [upstream, downstream])
        self.assertIsNot(upstream.branch_alpha, downstream.branch_alpha)
        self.assertNotEqual(upstream.branch_alpha.data_ptr(),
                            downstream.branch_alpha.data_ptr())

    def test_four_ablation_paths_execute_and_change_output(self):
        _, large, upstream, downstream = self.make_pair()
        ids = torch.randint(0, 64, (2, 6))
        outputs = {}
        with torch.no_grad():
            for key, up, down in (("off", False, False), ("up", True, False),
                                  ("down", False, True), ("both", True, True)):
                upstream.enabled, downstream.enabled = up, down
                outputs[key] = large.model(input_ids=ids, use_cache=False).last_hidden_state
        self.assertFalse(torch.equal(outputs["off"], outputs["up"]))
        self.assertFalse(torch.equal(outputs["off"], outputs["down"]))
        self.assertFalse(torch.equal(outputs["up"], outputs["both"]))

    def test_staged_freezes_all_upstream_parameters(self):
        _, large, upstream, downstream = self.make_pair()
        for parameter in large.parameters():
            parameter.requires_grad_(False)
        large.enable_input_require_grads()
        set_bridge_trainable(upstream, False, False)
        set_bridge_trainable(downstream, True, True)
        self.assertFalse(any(p.requires_grad for p in upstream.parameters()))
        self.assertTrue(all(p.requires_grad for p in bridge_parameters(downstream)))
        self.assertTrue(scale_parameter(downstream).requires_grad)
        norms = [m for m in downstream.modules() if isinstance(m, torch.nn.RMSNorm)]
        self.assertTrue(all(not module.weight.requires_grad for module in norms))
        ids = torch.randint(0, 64, (2, 6))
        loss = large.model(input_ids=ids, use_cache=False).last_hidden_state.square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in bridge_parameters(downstream)))
        self.assertTrue(all(p.grad is None for p in upstream.parameters()))

    def test_multi_checkpoint_round_trip(self):
        _, _, upstream, downstream = self.make_pair()
        before = {name: {key: value.detach().clone() for key, value in fusion.state_dict().items()}
                  for name, fusion in {"upstream": upstream,
                                       "downstream": downstream}.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "multi.pt")
            save_multi_fusion({"upstream": upstream, "downstream": downstream},
                              path, {"strategy": "staged"})
            for fusion in (upstream, downstream):
                for parameter in fusion.parameters():
                    parameter.data.zero_()
            meta = load_multi_fusion(
                {"upstream": upstream, "downstream": downstream}, path)
        self.assertEqual(meta["strategy"], "staged")
        for name, fusion in {"upstream": upstream, "downstream": downstream}.items():
            for key, value in fusion.state_dict().items():
                self.assertTrue(torch.equal(value, before[name][key]), key)

    def test_eval_loader_restores_two_alphas_and_forces_unit_rms_gamma(self):
        _, _, upstream, downstream = self.make_pair()
        upstream.branch_alpha.data.fill_(0.07)
        downstream.branch_alpha.data.fill_(0.19)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "multi.pt")
            save_multi_fusion({"upstream": upstream, "downstream": downstream},
                              path, {"pair": "tiny"})
            payload = torch.load(path, map_location="cpu")
        for saved in payload["fusions"].values():
            for key, value in saved["state_dict"].items():
                if key.endswith("norm.weight"):
                    value.fill_(3.0)  # 模拟旧版曾训练 gamma 的 checkpoint
        small, large = tiny_qwen(), tiny_qwen()
        restored, meta = attach_multi_fusion_checkpoint(large, small, payload)
        self.assertEqual(meta["pair"], "tiny")
        self.assertAlmostEqual(float(restored["upstream"].branch_alpha), 0.07)
        self.assertAlmostEqual(float(restored["downstream"].branch_alpha), 0.19)
        for fusion in restored.values():
            norms = [module for module in fusion.modules()
                     if isinstance(module, torch.nn.RMSNorm)]
            self.assertTrue(all(torch.equal(module.weight,
                                            torch.ones_like(module.weight))
                                for module in norms))
            self.assertTrue(all(not module.weight.requires_grad for module in norms))

    def test_eval_loader_can_attach_each_bridge_to_a_distinct_small_model(self):
        _, _, upstream, downstream = self.make_pair()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "multi.pt")
            save_multi_fusion(
                {"upstream": upstream, "downstream": downstream}, path,
                {"small_lora": {"mode": "independent"}})
            payload = torch.load(path, map_location="cpu")
        small_upstream, small_downstream, large = (
            tiny_qwen(), tiny_qwen(), tiny_qwen())
        restored, meta = attach_multi_fusion_checkpoint(
            large,
            {"upstream": small_upstream, "downstream": small_downstream},
            payload)
        self.assertEqual(meta["small_lora"]["mode"], "independent")
        self.assertIs(restored["upstream"].small_model_ref,
                      small_upstream.model)
        self.assertIs(restored["downstream"].small_model_ref,
                      small_downstream.model)
        self.assertIsNot(restored["upstream"].small_model_ref,
                         restored["downstream"].small_model_ref)

    def test_eval_generate_starts_and_ends_both_branch_caches(self):
        class CacheProbe:
            enabled = True

            def __init__(self):
                self.starts = 0
                self.ends = 0

            def begin_generation(self):
                self.starts += 1

            def end_generation(self):
                self.ends += 1

        class DummyTokenizer:
            chat_template = None
            eos_token_id = 2

            def __call__(self, text, return_tensors=None):
                return SimpleNamespace(input_ids=torch.tensor([[1, 3]]),
                                       attention_mask=torch.ones(1, 2, dtype=torch.long))

            def decode(self, ids, skip_special_tokens=True):
                return "ok"

        class DummyModel(torch.nn.Module):
            def __init__(self, branches):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                object.__setattr__(self, "_bes_fusions", branches)

            def generate(self, ids, **kwargs):
                return torch.cat([ids, torch.tensor([[4]], device=ids.device)], dim=1)

        branches = [CacheProbe(), CacheProbe()]
        result = generate(DummyModel(branches), DummyTokenizer(), "question", 1)
        self.assertEqual(result, "ok")
        self.assertEqual([(branch.starts, branch.ends) for branch in branches],
                         [(1, 1), (1, 1)])

        _, meta = generate(
            DummyModel(branches), DummyTokenizer(), "question", 1,
            return_meta=True)
        self.assertEqual(meta["generated_tokens"], 1)
        self.assertEqual(meta["finish_reason"], "length")
        self.assertTrue(meta["hit_max_new_tokens"])
        self.assertEqual(meta["max_new_tokens"], 1)

    def test_eval_generate_does_not_mark_final_eos_as_truncated(self):
        class DummyTokenizer:
            chat_template = None
            eos_token_id = 2

            def __call__(self, text, return_tensors=None):
                return SimpleNamespace(input_ids=torch.tensor([[1, 3]]),
                                       attention_mask=torch.ones(1, 2, dtype=torch.long))

            def decode(self, ids, skip_special_tokens=True):
                return "done"

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def generate(self, ids, **kwargs):
                return torch.cat(
                    [ids, torch.tensor([[4, 2]], device=ids.device)], dim=1)

        _, meta = generate(
            DummyModel(), DummyTokenizer(), "question", 2, return_meta=True)
        self.assertEqual(meta["generated_tokens"], 2)
        self.assertEqual(meta["finish_reason"], "eos")
        self.assertFalse(meta["hit_max_new_tokens"])


if __name__ == "__main__":
    unittest.main()
