# -*- coding: utf-8 -*-
"""
LoRA 对照组训练: 对纯 4B 做 LoRA 微调, 参数预算 ≈ 门控旁路的 44M, 用于公平对比。

与 train_fusion.py 的差异:
  - 不加载 0.6B, 无门控旁路
  - 对 4B 的 7 个线性层(q/k/v/o/gate/up/down)挂 LoRA
  - SFT 目标/数据/超参/调度与 train_fusion.py 完全一致(chat 模板 + prompt mask + 答案段 CE)

用法(在 BES 目录下执行):
  python core_training/train_lora.py --data data/mix_all.jsonl --eval_samples 4000 --epochs 3
依赖: pip install peft
"""

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

from main import Config, resolve_dtype, resolve_model_path, model_device
from train_fusion import (load_records, TextDataset, collate,
                          causal_lm_loss, plot_losses)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IGNORE = -100
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser(description="LoRA 对照组训练(纯 4B)")
    parser.add_argument("--data", default="data/mix_all.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_checkpoint", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=21,
                        help="LoRA rank(21≈43.4M 参数, 对齐旁路 44.1M)")
    parser.add_argument("--lora_alpha", type=int, default=0, help="LoRA alpha(0=自动 2×r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=4000)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_max_samples", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--out", default=None,
                        help="LoRA 权重目录(默认=自动 cache/lora_r<r>_<时间戳>)")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--plot", default=None,
                        help="loss 图路径(默认=自动带 rank+时间戳; 传空字符串=不画)")
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)
    dt = resolve_dtype(cfg.dtype)
    print(f"精度: {dt} | CUDA: {torch.cuda.is_available()}")
    # 输出文件自动命名: 带 rank + 时间戳, 避免不同实验互相覆盖
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or f"cache/lora_r{args.lora_r}_{ts}"
    plot_path = args.plot if args.plot is not None else f"cache/train_lora_r{args.lora_r}_{ts}.png"
    print(f"LoRA 输出目录: {out_dir} | loss 图: {plot_path}")

    # ── 加载 4B + 挂 LoRA ──
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    tokenizer = AutoTokenizer.from_pretrained(large_path)
    model = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt).cuda()

    from peft import LoraConfig, get_peft_model
    targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]
    alpha = args.lora_alpha or 2 * args.lora_r
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    # LoRA 参数用 fp32 主权重(与 train_fusion 对旁路的处理一致, 防 bf16 梯度冻结)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA 可训练参数: {n_train/1e6:.2f}M (rank={args.lora_r}, alpha={alpha})")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    dev = model_device(model)

    # ── 数据(与 train_fusion 完全一致) ──
    recs = load_records(args.data, 0)
    if not recs:
        raise SystemExit(f"[错误] 无训练数据: {args.data}")
    n_eval = min(args.eval_samples, len(recs) - 1) if args.eval_samples > 0 else 0
    eval_recs = recs[-n_eval:] if n_eval else []
    train_pool = recs[:-n_eval] if n_eval else recs
    train_recs = train_pool[:args.max_samples] if args.max_samples > 0 else train_pool
    print(f"训练样本: {len(train_recs)} 条 | 评估样本: {len(eval_recs)} 条")

    coll = lambda b: collate(b, tokenizer, args.max_len, 1.0)
    loader = DataLoader(TextDataset(train_recs), batch_size=args.batch_size,
                        shuffle=True, collate_fn=coll)
    eval_loader = DataLoader(TextDataset(eval_recs), batch_size=args.eval_batch_size,
                             shuffle=False, collate_fn=coll) if eval_recs else None

    # ── 调度(与 train_fusion 一致) ──
    scheduler = None
    if args.warmup_steps > 0:
        total_steps = len(loader) * args.epochs

        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            p = min(1.0, max(0.0, p))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def forward_logits(ids, mask):
        # bf16/fp16 权重用同精度 autocast(前向 bf16 + fp32 主权重, 梯度回传升回 fp32);
        # fp32 权重则不降精度直接算
        ctx = torch.autocast("cuda", dtype=dt) \
            if dt in (torch.bfloat16, torch.float16) and torch.cuda.is_available() \
            else nullcontext()
        with ctx:
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        return out.logits

    @torch.no_grad()
    def evaluate():
        """LoRA 开 vs 关(即微调后 vs 原 4B)在验证集答案段上的 CE"""
        model.eval()
        lora_sum = base_sum = 0.0
        n = 0
        seen = 0
        for ids, mask, labels, _w in eval_loader:
            ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
            model.enable_adapter_layers()
            lora_sum += causal_lm_loss(forward_logits(ids, mask), labels).item()
            model.disable_adapter_layers()
            base_sum += causal_lm_loss(forward_logits(ids, mask), labels).item()
            model.enable_adapter_layers()
            n += 1
            seen += ids.size(0)
            if args.eval_max_samples > 0 and seen >= args.eval_max_samples:
                break
        return lora_sum / n, base_sum / n

    # ── 训练循环 ──
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)
    model.train()
    train_ce_history = []
    eval_history = []
    best_lora = float("inf")
    best_step = 0
    patience_counter = 0
    stop = False
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for ids, mask, labels, _w in loader:
            ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
            model.train()          # 确保梯度检查点生效(也开启 LoRA dropout)
            opt.zero_grad()
            logits = forward_logits(ids, mask)
            ce = causal_lm_loss(logits, labels)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            step += 1
            train_ce_history.append((step, ce.item()))
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"epoch {ep + 1} step {step:>6} | CE {ce.item():.4f} "
                      f"| lr {opt.param_groups[0]['lr']:.2e} | {el:.0f}s")

            if eval_loader is not None and step % args.eval_every == 0:
                l_loss, b_loss = evaluate()
                eval_history.append((step, l_loss, b_loss))
                print(f"    [评估] step {step:>6} | LoRA {l_loss:.4f} | "
                      f"base {b_loss:.4f} | 增益 {b_loss - l_loss:+.4f} (>0=微调有效)")
                if l_loss < best_lora - 1e-4:
                    best_lora, best_step = l_loss, step
                    patience_counter = 0
                    model.save_pretrained(out_dir + ".best")
                else:
                    patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"    早停触发 @ step {best_step}")
                    stop = True
            if stop:
                break
        if stop:
            break

    model.save_pretrained(out_dir)
    if plot_path:
        plot_losses(train_ce_history, eval_history, plot_path)
    print(f"\n训练完成, 共 {step} 步, LoRA 已保存: {out_dir}")
    if best_lora < float("inf"):
        print(f"最优 LoRA loss {best_lora:.4f} @ step {best_step}, 已保存: {out_dir}.best")


if __name__ == "__main__":
    main()
