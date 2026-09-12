# -*- coding: utf-8 -*-
"""
门控残差融合架构冒烟测试(不训练,只验证架构)
  ① 初始恒等性: ReZero branch_alpha=0 → 旁路输出必须精确为 0
  ② 挂载后解码: 4B_text 路径输出应与纯 4B 完全一致(旁路此时无效)

用法: python test_fusion.py
"""

import os
import sys
import torch

sys.stdout.reconfigure(encoding="utf-8")
# 项目根目录 = 上一级(BES); 把 core_training/(main) 与 eval/ 加入导入路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from main import (Config, attach_fusion, fused_inference, model_device,
                  resolve_dtype, resolve_model_path)

cfg = Config()
cfg.max_new_tokens = 64
dt = resolve_dtype(cfg.dtype)

print("加载模型 ...")
tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(cfg.model_large_id, cfg.model_large_local))
small = AutoModelForCausalLM.from_pretrained(
    resolve_model_path(cfg.model_small_id, cfg.model_small_local),
    dtype=dt, device_map=cfg.device_map_small or cfg.device_map).eval()
large = AutoModelForCausalLM.from_pretrained(
    resolve_model_path(cfg.model_large_id, cfg.model_large_local),
    dtype=dt, device_map=cfg.device_map).eval()

fusion = attach_fusion(large, small, cfg)

# ① 初始恒等验证
h = torch.randn(1, 8, large.config.hidden_size, device=model_device(large), dtype=dt)
with torch.no_grad():
    branch = fusion(h)
print(f"\n初始旁路输出范数: {float(branch.norm()):.6f} (应为 0.0)")
print(f"ReZero branch_alpha = {float(fusion.branch_alpha):.6f} (应为 0.0)")

# ② 挂载后正常解码
print("\n解码(4B 直接文本, 旁路初始无效, 应与纯 4B 输出一致) ...")
answer, stats = fused_inference(large, tokenizer, cfg, cfg.prompt)
print(f"解码结果: {answer[:150]}")
print(f"统计: {stats}")
