# -*- coding: utf-8 -*-
"""两条小模型 Bridge 的分阶段/联合训练。

实验变量只有一个：
  staged: 加载并冻结已经训练好的上游 Bridge，只训练新接入的下游 Bridge；
  joint:  使用完全相同的初始化，同时训练上游和下游 Bridge。

两条旁路共享同一份冻结 0.6B 权重（数学上等价于复制两份相同冻结模型，
但显存更省），各自拥有独立 adapter、branch alpha、hook 状态和生成 cache。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from main import (Config, attach_fusion, load_fusion, load_multi_fusion,
                  model_device, resolve_dtype, resolve_model_path,
                  save_multi_fusion)
from train_fusion import (TextDataset, causal_lm_loss, causal_lm_loss_parts,
                          collate, load_records)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# (4B capture, 4B inject, 0.6B first layer, 0.6B last layer), all 0-based
BRIDGE_PRESETS = {
    "early": (6, 11, 3, 8),       # small [3,9)
    "middle": (12, 24, 9, 18),   # small [9,19)
    "late": (25, 31, 18, 26),    # small [18,27)
}
PAIR_PRESETS = {
    "early-middle": ("early", "middle"),
    "middle-late": ("middle", "late"),
    "early-late": ("early", "late"),
}


def config_for_position(base: Config, position: str) -> Config:
    l1, l2, s1, s2 = BRIDGE_PRESETS[position]
    cfg = Config(**vars(base))
    cfg.fusion_large_start = l1
    cfg.fusion_large_end = l2
    cfg.fusion_small_start = s1
    cfg.fusion_small_end = s2
    return cfg


def randomize_bridge_outputs(fusion) -> None:
    """与单 Bridge 新训练保持相同初始化：解除全零 up 导致的梯度阻塞。"""
    with torch.no_grad():
        for module in fusion.modules():
            up = getattr(module, "up", None)
            if isinstance(up, torch.nn.Linear):
                torch.nn.init.normal_(up.weight, std=0.02)
                torch.nn.init.zeros_(up.bias)


def bridge_parameters(fusion):
    return [p for name, p in fusion.named_parameters()
            if name not in {"branch_alpha", "gate_logit"}
            and not name.endswith("norm.weight")]


def scale_parameter(fusion):
    return fusion.gate_logit if fusion.gate_mode == "sigmoid" else fusion.branch_alpha


def set_bridge_trainable(fusion, trainable: bool, scale_trainable: bool) -> None:
    for p in fusion.parameters():
        p.requires_grad_(False)
    if trainable:
        for p in bridge_parameters(fusion):
            p.requires_grad_(True)
    scale_parameter(fusion).requires_grad_(scale_trainable)
    # gamma 必须保持冻结，避免与 branch alpha 出现尺度退化。
    for module in fusion.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.weight.requires_grad_(False)


def grad_norm_and_clip(params, max_norm: float) -> float:
    params = [p for p in params if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    value = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
    return float(value)


def plot_losses(train_history, eval_history, path: str) -> None:
    if not path:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("    [提示] 未安装 matplotlib，跳过画图")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if train_history:
        steps = [x[0] for x in train_history]
        values = [x[1] for x in train_history]
        axes[0].plot(steps, values, alpha=0.3, lw=0.6, label="per-step CE")
        if len(values) >= 5:
            width = min(20, len(values))
            smooth = np.convolve(values, np.ones(width) / width, mode="valid")
            axes[0].plot(steps[width - 1:], smooth, lw=2,
                         label=f"smoothed (win {width})")
        axes[0].set_title("Training CE")
        axes[0].set_xlabel("optimizer step")
        axes[0].legend()
    if eval_history:
        steps = [x["step"] for x in eval_history]
        for key, label in (("both", "both"), ("upstream", "upstream only"),
                           ("downstream", "downstream only"), ("off", "both off")):
            axes[1].plot(steps, [x[key] for x in eval_history], marker="o",
                         ms=3, label=label)
        axes[1].set_title("Validation ablation CE")
        axes[1].set_xlabel("optimizer step")
        axes[1].legend()
    fig.tight_layout()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
        print(f"    loss 图已保存: {path}")
    finally:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="训练两条顺序接入的小模型 Bridge")
    parser.add_argument("--pair", choices=sorted(PAIR_PRESETS), required=True)
    parser.add_argument("--strategy", choices=("staged", "joint"), required=True,
                        help="staged=冻结上游只训下游；joint=两条一起训练")
    parser.add_argument("--upstream_ckpt", required=True,
                        help="对应上游位置的已训练单 Bridge checkpoint")
    parser.add_argument("--downstream_ckpt", default="",
                        help="可选；默认让新增下游 Bridge 从相同随机初始化开始")
    parser.add_argument("--resume", default="", help="恢复本脚本保存的双 Bridge checkpoint")
    parser.add_argument("--data", default="data/math_majority_all.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=1536)
    parser.add_argument("--bridge_depth", type=int, default=1)
    parser.add_argument("--bridge_mlp_dim", type=int, default=4096)
    parser.add_argument("--upstream_lr", type=float, default=2e-5,
                        help="joint 时预训练上游 Bridge 的较小学习率")
    parser.add_argument("--downstream_lr", type=float, default=1e-4,
                        help="新增下游 Bridge 学习率")
    parser.add_argument("--upstream_alpha_lr", type=float, default=1e-4)
    parser.add_argument("--downstream_alpha_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--branch_warmup_steps", type=int, default=400,
                        help="前 N 步固定新增下游 alpha；上游保留 checkpoint scale")
    parser.add_argument("--branch_warmup_alpha", type=float, default=0.05)
    parser.add_argument("--branch_alpha_max", type=float, default=0.25)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--alpha_grad_clip", type=float, default=0.1)
    parser.add_argument("--grad_checkpoint", type=int, default=1)
    parser.add_argument("--attn_impl", default="flash_attention_2")
    parser.add_argument("--answer_weight", type=float, default=2.0)
    parser.add_argument("--eval_samples", type=int, default=1999)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_max_samples", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot", default="")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "max_len", "bridge_depth", "bridge_mlp_dim",
                 "eval_batch_size", "eval_every", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} 必须 > 0")
    for name in ("warmup_steps", "branch_warmup_steps", "eval_samples",
                 "eval_max_samples", "patience"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} 必须 >= 0")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_cfg = Config()
    base_cfg.seed = args.seed
    base_cfg.fusion_bridge_depth = args.bridge_depth
    base_cfg.fusion_mlp_dim = args.bridge_mlp_dim
    dtype = resolve_dtype(base_cfg.dtype)
    upstream_name, downstream_name = PAIR_PRESETS[args.pair]
    print(f"实验: {args.pair} | 策略: {args.strategy}")
    print(f"精度: {dtype} | CUDA: {torch.cuda.is_available()}")

    small_path = resolve_model_path(base_cfg.model_small_id, base_cfg.model_small_local)
    large_path = resolve_model_path(base_cfg.model_large_id, base_cfg.model_large_local)
    tokenizer = AutoTokenizer.from_pretrained(large_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn_kwargs = {"attn_implementation": args.attn_impl} if args.attn_impl else {}
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dtype).cuda()
    large = AutoModelForCausalLM.from_pretrained(
        large_path, dtype=dtype, **attn_kwargs).cuda()
    for model in (small, large):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    large.train()
    small.eval()
    if args.grad_checkpoint:
        large.gradient_checkpointing_enable()
        large.enable_input_require_grads()
        print("    梯度检查点已开启")

    upstream = attach_fusion(
        large, small, config_for_position(base_cfg, upstream_name), name="upstream")
    downstream = attach_fusion(
        large, small, config_for_position(base_cfg, downstream_name), name="downstream")
    fusions = {"upstream": upstream, "downstream": downstream}
    if upstream.l2 >= downstream.l1:
        raise ValueError("本程序要求上游注入点严格早于下游捕获点")

    # AdamW 的主权重必须为 fp32；否则 bf16 精度会吞掉小更新。
    for fusion in fusions.values():
        for parameter in fusion.parameters():
            parameter.data = parameter.data.float()

    load_fusion(upstream, args.upstream_ckpt)
    if args.downstream_ckpt:
        load_fusion(downstream, args.downstream_ckpt)
        if (not args.resume and downstream.gate_mode == "rezero"
                and args.branch_warmup_steps > 0):
            with torch.no_grad():
                downstream.branch_alpha.fill_(args.branch_warmup_alpha)
    else:
        randomize_bridge_outputs(downstream)
        with torch.no_grad():
            downstream.branch_alpha.fill_(args.branch_warmup_alpha)
    if args.resume:
        load_multi_fusion(fusions, args.resume)

    downstream_warmup = (not args.resume and downstream.gate_mode == "rezero"
                         and args.branch_warmup_steps > 0)
    joint = args.strategy == "joint"
    set_bridge_trainable(upstream, joint, joint)
    set_bridge_trainable(downstream, True, not downstream_warmup)
    upstream.train(joint)
    downstream.train()

    up_bridge = bridge_parameters(upstream) if joint else []
    down_bridge = bridge_parameters(downstream)
    up_scale = [scale_parameter(upstream)] if joint else []
    down_scale = [scale_parameter(downstream)]
    groups = []
    if up_bridge:
        groups.append({"params": up_bridge, "lr": args.upstream_lr,
                       "weight_decay": args.weight_decay, "name": "upstream_bridge"})
        groups.append({"params": up_scale, "lr": args.upstream_alpha_lr,
                       "weight_decay": 0.0, "name": "upstream_alpha"})
    groups.append({"params": down_bridge, "lr": args.downstream_lr,
                   "weight_decay": args.weight_decay, "name": "downstream_bridge"})
    groups.append({"params": down_scale, "lr": args.downstream_alpha_lr,
                   "weight_decay": 0.0, "name": "downstream_alpha"})
    optimizer = torch.optim.AdamW(groups)

    n_up = sum(p.numel() for p in up_bridge)
    n_down = sum(p.numel() for p in down_bridge)
    print(f"小模型权重: 两条旁路共享且冻结；adapter/alpha 相互独立")
    print(f"可训练参数: upstream={n_up / 1e6:.2f}M | "
          f"downstream={n_down / 1e6:.2f}M | total={(n_up + n_down) / 1e6:.2f}M")
    print("优化器参数组: " + " | ".join(
        f"{g['name']} lr={g['lr']:.2e} wd={g['weight_decay']:g}" for g in groups))
    if args.strategy == "staged":
        print("训练策略: 上游 adapter 与 alpha 完全冻结；只更新新增下游")
    else:
        print("训练策略: 上游低学习率、下游高学习率联合更新")
    if downstream_warmup:
        print(f"下游旁路 warmup: 前 {args.branch_warmup_steps} step 固定 "
              f"alpha={args.branch_warmup_alpha:g}")
    print("监督目标: response-only causal CE；四路消融只用于验证，不进入训练 loss")

    records = load_records(args.data, 0)
    if len(records) < 2:
        raise SystemExit(f"[错误] 数据不足: {args.data}")
    n_eval = min(args.eval_samples, len(records) - 1) if args.eval_samples else 0
    eval_records = records[-n_eval:] if n_eval else []
    train_pool = records[:-n_eval] if n_eval else records
    train_records = train_pool[:args.max_samples] if args.max_samples > 0 else train_pool
    print(f"训练样本: {len(train_records)} | 评估样本: {len(eval_records)}")
    batch_collate = lambda batch: collate(
        batch, tokenizer, args.max_len, args.answer_weight)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TextDataset(train_records), batch_size=args.batch_size,
                              shuffle=True, collate_fn=batch_collate,
                              generator=generator)
    eval_loader = (DataLoader(TextDataset(eval_records), batch_size=args.eval_batch_size,
                              shuffle=False, collate_fn=batch_collate)
                   if eval_records else None)

    total_steps = len(train_loader) * args.epochs
    scheduler = None
    if args.warmup_steps > 0:
        def lr_lambda(step):
            if step < args.warmup_steps:
                return (step + 1) / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(
                1, total_steps - args.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    device = model_device(large)

    def forward_logits(ids, mask):
        context = (torch.autocast("cuda", dtype=dtype)
                   if dtype in (torch.bfloat16, torch.float16)
                   and torch.cuda.is_available() else nullcontext())
        with context:
            output = large.model(input_ids=ids, attention_mask=mask, use_cache=False)
            return large.lm_head(output.last_hidden_state)

    def set_enabled(up: bool, down: bool):
        upstream.enabled = up
        downstream.enabled = down
        upstream.branch_override = None
        downstream.branch_override = None

    @torch.no_grad()
    def evaluate():
        large.eval()
        upstream.eval()
        downstream.eval()
        totals = {key: [0.0, 0.0] for key in
                  ("both", "upstream", "downstream", "off")}
        seen = 0
        modes = (("off", False, False), ("upstream", True, False),
                 ("downstream", False, True), ("both", True, True))
        for ids, mask, labels, weights in eval_loader:
            ids, mask = ids.to(device), mask.to(device)
            labels, weights = labels.to(device), weights.to(device)
            for key, up, down in modes:
                set_enabled(up, down)
                loss_sum, weight_sum = causal_lm_loss_parts(
                    forward_logits(ids, mask), labels, weights)
                totals[key][0] += float(loss_sum)
                totals[key][1] += float(weight_sum)
            seen += ids.size(0)
            if args.eval_max_samples > 0 and seen >= args.eval_max_samples:
                break
        set_enabled(True, True)
        large.train()
        upstream.train(joint)
        downstream.train()
        return {key: value[0] / max(value[1], 1.0)
                for key, value in totals.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train_history = []
    eval_history = []
    best = float("inf")
    best_incremental = float("inf")
    best_step = best_incremental_step = 0
    patience_count = 0
    step = 0
    stop = False
    started = time.time()

    def checkpoint(path, evaluation=None):
        metadata = {
            "pair": args.pair,
            "strategy": args.strategy,
            "seed": args.seed,
            "step": step,
            "upstream_source": args.upstream_ckpt,
            "downstream_source": args.downstream_ckpt or "fresh",
            "evaluation": evaluation,
            "positions": {
                "upstream": BRIDGE_PRESETS[upstream_name],
                "downstream": BRIDGE_PRESETS[downstream_name],
            },
        }
        save_multi_fusion(fusions, path, metadata)

    for epoch in range(args.epochs):
        for ids, mask, labels, weights in train_loader:
            if downstream_warmup and step == args.branch_warmup_steps:
                scale_parameter(downstream).requires_grad_(True)
                print(f"    step {step}: 下游 alpha warmup 结束，开始优化 alpha")
            ids, mask = ids.to(device), mask.to(device)
            labels, weights = labels.to(device), weights.to(device)
            set_enabled(True, True)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(ids, mask)
            ce = causal_lm_loss(logits, labels, weights)
            if not torch.isfinite(ce):
                raise FloatingPointError(f"step {step + 1}: CE 非有限")
            ce.backward()
            up_grad = grad_norm_and_clip(up_bridge, args.grad_clip)
            down_grad = grad_norm_and_clip(down_bridge, args.grad_clip)
            up_alpha_grad = grad_norm_and_clip(up_scale, args.alpha_grad_clip)
            down_alpha_grad = grad_norm_and_clip(down_scale, args.alpha_grad_clip)
            optimizer.step()
            for fusion in fusions.values():
                if (fusion.gate_mode == "rezero" and args.branch_alpha_max > 0
                        and fusion.branch_alpha.requires_grad):
                    with torch.no_grad():
                        fusion.branch_alpha.clamp_(
                            -args.branch_alpha_max, args.branch_alpha_max)
            if scheduler is not None:
                scheduler.step()
            step += 1
            train_history.append((step, float(ce.detach())))
            if step % args.log_every == 0:
                ratios = (upstream.hook_state.get("branch_rms_ratio", float("nan")),
                          downstream.hook_state.get("branch_rms_ratio", float("nan")))
                lrs = ",".join(f"{g['name']}={g['lr']:.2e}"
                               for g in optimizer.param_groups)
                print(f"epoch {epoch + 1} step {step:>6} | CE {float(ce):.4f} "
                      f"| alpha[up={float(upstream.scale()):.5f},"
                      f"down={float(downstream.scale()):.5f}] "
                      f"| RMS[up={ratios[0]:.4f},down={ratios[1]:.4f}] "
                      f"| grad[up={up_grad:.2e},down={down_grad:.2e},"
                      f"up_a={up_alpha_grad:.2e},down_a={down_alpha_grad:.2e}] "
                      f"| lr[{lrs}] | {time.time() - started:.0f}s")

            if eval_loader is not None and step % args.eval_every == 0:
                losses = evaluate()
                row = {"step": step, **losses}
                eval_history.append(row)
                downstream_gain = losses["upstream"] - losses["both"]
                upstream_gain = losses["downstream"] - losses["both"]
                print(f"    [四路评估] both={losses['both']:.4f} | "
                      f"up-only={losses['upstream']:.4f} | "
                      f"down-only={losses['downstream']:.4f} | off={losses['off']:.4f}")
                print(f"    [边际价值] 新下游增益={downstream_gain:+.4f} | "
                      f"上游条件增益={upstream_gain:+.4f} (>0 才有独立贡献)")
                if losses["both"] < best - 1e-4:
                    best = losses["both"]
                    best_step = step
                    patience_count = 0
                    checkpoint(args.out + ".best", row)
                else:
                    patience_count += 1
                if (downstream_gain > 1e-4
                        and losses["both"] < best_incremental - 1e-4):
                    best_incremental = losses["both"]
                    best_incremental_step = step
                    checkpoint(args.out + ".best_incremental", row)
                if args.patience > 0 and patience_count >= args.patience:
                    print(f"    早停: 连续 {args.patience} 次评估未改善")
                    stop = True
            if stop:
                break
        if stop:
            break

    checkpoint(args.out, eval_history[-1] if eval_history else None)
    summary = {
        "pair": args.pair,
        "strategy": args.strategy,
        "seed": args.seed,
        "steps": step,
        "best_both_ce": None if math.isinf(best) else best,
        "best_step": best_step,
        "best_incremental_ce": None if math.isinf(best_incremental) else best_incremental,
        "best_incremental_step": best_incremental_step,
        "train_ce": train_history,
        "evaluations": eval_history,
    }
    with open(args.out + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    plot_losses(train_history, eval_history, args.plot)
    print(f"训练完成: {step} steps | final={args.out}")
    if best_step:
        print(f"最优双路 CE={best:.4f} @ step {best_step} | {args.out}.best")
    if best_incremental_step:
        print(f"最优且下游有增量价值 @ step {best_incremental_step} | "
              f"{args.out}.best_incremental")
    else:
        print("[警告] 尚未观察到 both 明确优于 up-only；新增下游未证明独立价值")


if __name__ == "__main__":
    main()
