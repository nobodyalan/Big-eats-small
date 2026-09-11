# -*- coding: utf-8 -*-
"""
门控残差融合适配器训练脚本
==========================
冻结 4B(主脑)与 0.6B(旁路中段宿主),只训练门控残差旁路的可训练参数:
  adapter1 / adapter2 / gate_logit(共约 22M)。

训练目标: 因果语言建模(下一 token 预测)损失。
梯度路径: loss → 4B 第 24 层注入点 → adapter2 → 0.6B 中段层 → adapter1。

针对 H100 等大显存 GPU 的默认设置:
  bf16 权重 + 关闭梯度检查点(更快) + batch=8 + 最大序列 512 + 3 epoch。
小显存(如 12GB)请改用: --grad_checkpoint 1 --batch_size 1 --max_len 256。
(注意: 梯度检查点要求模型处于 .train() 状态才会生效; Qwen3 的
 attention_dropout=0.0, 所以 train 模式不会引入 dropout 噪声。)

用法:
  python train_fusion.py                                          # 默认全量
  python train_fusion.py --max_samples 2000 --epochs 1            # 小规模试跑
  python train_fusion.py --resume cache/fusion_adapter.pt         # 断点续训
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录 = 上一级(BES); 把 core_training/(main) 与 eval/(main 的依赖) 加入导入路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from main import (Config, attach_fusion, save_fusion, load_fusion,
                  model_device, resolve_dtype, resolve_model_path,
                  resolve_small_range, resolve_large_range)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IGNORE = -100


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ① 数据加载                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def load_records(path: str, max_samples: int):
    """读 jsonl, 每行返回 {"prompt":..., "response":...}。
    兼容旧格式 {"text": "题目\\n解题思路:答案"}(自动按 '解题思路:' 拆成 prompt/response)。"""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"text": line}
            if "prompt" in obj:
                recs.append({"prompt": obj.get("prompt", ""),
                             "response": obj.get("response", "")})
            else:
                t = obj.get("text", "") or line
                if "解题思路:" in t:
                    p, r = t.split("解题思路:", 1)
                    recs.append({"prompt": p + "解题思路:", "response": r})
                else:
                    recs.append({"prompt": t, "response": ""})
            if max_samples > 0 and len(recs) >= max_samples:
                break
    return [r for r in recs if r["response"].strip() or r["prompt"].strip()]


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def collate(batch, tokenizer, max_len: int, answer_weight: float = 1.0):
    """SFT 式编码: 只对 response 算 loss(prompt 位置 label=IGNORE),
    并给最终答案段(\\boxed / #### 之后)加权。返回 (input_ids, attention_mask, labels, weights)。"""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    ids_list, labels_list, weights_list = [], [], []
    for rec in batch:
        prompt = rec.get("prompt", "")
        resp = rec.get("response", "")
        # 与评测一致: prompt 套 chat 模板(add_generation_prompt=True), response 直接拼在后面
        if getattr(tokenizer, "chat_template", None):
            p_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            p_text = prompt
        # prompt 编码(给答案至少留 1 个位置)
        p_ids = tokenizer(p_text, add_special_tokens=False).input_ids
        p_ids = p_ids[:max_len - 1]
        # response 编码 + 字符偏移(用于定位答案段)
        enc = tokenizer(resp, add_special_tokens=False, return_offsets_mapping=True)
        r_ids, r_off = enc.input_ids, enc.offset_mapping
        r_ids = r_ids[:max_len - len(p_ids)]
        r_off = r_off[:len(r_ids)]
        if not r_ids or r_ids[-1] != eos:
            r_ids = r_ids + [eos]
            r_off = r_off + [(0, 0)]
        r_ids = r_ids[:max_len - len(p_ids)]
        r_off = r_off[:len(r_ids)]
        # 定位答案段起点(最后一个 \boxed 或 ####), 之后加权
        ans_char = -1
        for marker in ("\\boxed", "####"):
            i = resp.rfind(marker)
            if i != -1:
                ans_char = i
                break
        ans_tok = None
        if ans_char != -1:
            for k, (s, e) in enumerate(r_off):
                if s <= ans_char < e:
                    ans_tok = k
                    break
        w = [1.0] * len(r_ids)
        if ans_tok is not None:
            for k in range(ans_tok, len(r_ids)):
                w[k] = answer_weight
        ids_list.append(torch.tensor(p_ids + r_ids, dtype=torch.long))
        labels_list.append(torch.tensor([IGNORE] * len(p_ids) + r_ids, dtype=torch.long))
        weights_list.append(torch.tensor([0.0] * len(p_ids) + w, dtype=torch.float32))
    L = max(x.size(0) for x in ids_list)
    B = len(ids_list)
    input_ids = torch.full((B, L), pad, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    weights = torch.zeros((B, L), dtype=torch.float32)
    for i in range(B):
        n = ids_list[i].size(0)
        input_ids[i, :n] = ids_list[i]
        attention_mask[i, :n] = 1
        labels[i, :n] = labels_list[i]
        weights[i, :n] = weights_list[i]
    return input_ids, attention_mask, labels, weights


def causal_lm_loss(logits, labels, weights=None):
    """因果 LM: 每个位置预测下一个 token; padding/prompt 位置忽略(IGNORE)。
    weights 非空时对每个有效位置按权重加权(答案段加权用)。"""
    # 保持 bf16 传给 F.cross_entropy(内部按 fp32 算 log_softmax), 省掉整张 fp32 logits 拷贝
    # (batch×seq×151936 词表的 fp32 拷贝 ≈5GB, 是 4B 训练 OOM 的直接触发点)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if weights is None:
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1), ignore_index=IGNORE)
    ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                         shift_labels.view(-1), ignore_index=IGNORE, reduction="none")
    ce = ce.view(shift_labels.shape)
    w = weights[:, 1:].to(ce.device).float()
    mask = (shift_labels != IGNORE).float()
    return (ce * w * mask).sum() / (w * mask).sum().clamp(min=1)


JS_MARGIN = 0.69   # JS 散度上界 ln2≈0.69(InterLat 的铰链阈值)
EPS = 1e-8


def js_contrastive(logits_n, logits_r, labels, margin=JS_MARGIN):
    """
    InterLat 同款对比损失(JS 散度铰链):
      正确旁路 vs 错配旁路, 两边的输出分布应尽量不同(JS→ln2)。
    防止模型"无视这条旁路": 若旁路内容不影响预测, JS≈0 → 损失把 JS 往 ln2 推,
    逼模型去读旁路注入的隐状态。
    """
    # 与 CE 同对齐: logits[:, :-1] 预测 labels[:, 1:]
    shift_n = logits_n[:, :-1, :].contiguous()
    shift_r = logits_r[:, :-1, :].contiguous()
    shift_lab = labels[:, 1:].contiguous()
    valid = shift_lab != IGNORE
    # 只对有效 token(远小于 batch×seq)转 fp32 做 softmax, 避免整张 fp32 拷贝
    p = F.softmax(shift_n[valid].float(), dim=-1).clamp_min(EPS)
    q = F.softmax(shift_r[valid].float(), dim=-1).clamp_min(EPS)
    m = 0.5 * (p + q)
    js_nats = 0.5 * F.kl_div(p.log(), m, reduction="batchmean") \
              + 0.5 * F.kl_div(q.log(), m, reduction="batchmean")
    js_bits = js_nats / 1.4426950408889634   # nats → bits (1/ln2)
    return torch.clamp(margin - js_bits, min=0.0)


def plot_losses(train_ce_history, eval_history, path):
    """画训练 CE + 评估 fusion/baseline 两张子图并保存 PNG"""
    if not path:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("    [提示] 未安装 matplotlib, 跳过画图(可 pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if train_ce_history:
        steps = [s for s, _ in train_ce_history]
        ces = [c for _, c in train_ce_history]
        axes[0].plot(steps, ces, alpha=0.35, color="tab:blue", lw=0.6, label="per-step CE")
        if len(ces) >= 5:
            k = min(20, len(ces))
            sm = np.convolve(ces, np.ones(k) / k, mode="valid")
            axes[0].plot(steps[k - 1:], sm, color="tab:blue", lw=2, label=f"smoothed (win {k})")
        axes[0].set_title("Training CE loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("CE")
        axes[0].legend()
    if eval_history:
        steps = [s for s, _, _ in eval_history]
        f = [x for _, x, _ in eval_history]
        b = [x for _, _, x in eval_history]
        axes[1].plot(steps, b, marker="o", ms=3, label="baseline (branch off)")
        axes[1].plot(steps, f, marker="o", ms=3, label="fusion (branch on)")
        axes[1].set_title("Eval: fusion vs baseline")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("CE")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    loss 图已保存: {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ② 训练主流程                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    parser = argparse.ArgumentParser(description="训练门控残差融合适配器")
    parser.add_argument("--data", default="data/mix_all.jsonl")
    parser.add_argument("--max_samples", type=int, default=0, help="最多训练条数(0=全部; 评估集另算, 不被截断)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1024, help="单条最大 token 数(显存不足先调小)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="学习率线性 warmup 步数(0=关闭; 之后余弦衰减到 10%%)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gate_init", type=float, default=0.0,
                        help="训练起始 gate_logit(0→sigmoid=0.5);-10 会以 4.5e-5 阻塞梯度")
    parser.add_argument("--small_start", type=int, default=None,
                        help="0.6B 旁路起始层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--small_end", type=int, default=None,
                        help="0.6B 旁路结束层(含, 0-based); -1=最后一层; None=自动 2/3 位置")
    parser.add_argument("--large_start", type=int, default=None,
                        help="4B 取隐状态层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--large_end", type=int, default=None,
                        help="4B 加回残差层(含, 0-based); None=自动 2/3 位置")
    parser.add_argument("--grad_checkpoint", type=int, default=0, help="1=梯度检查点(省显存但更慢)")
    parser.add_argument("--batch_size", type=int, default=8, help="训练 batch(H100 可用 8~16)")
    parser.add_argument("--contrast_weight", type=float, default=0.5,
                        help="InterLat 式 JS 对比损失权重(0=关闭;>0 时防止旁路被无视/门控塌缩)")
    parser.add_argument("--answer_weight", type=float, default=1.0,
                        help="最终答案段(\\boxed/#### 之后)的 loss 权重(SFT 答案加权)")
    parser.add_argument("--eval_every", type=int, default=200, help="每隔 N 步对比一次 baseline/fusion loss")
    parser.add_argument("--eval_samples", type=int, default=64, help="从数据里留出多少条作评估集")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估时的 batch(越大评估越快)")
    parser.add_argument("--eval_max_samples", type=int, default=400,
                        help="每次评估最多用多少条(0=全部; 验证集大时应设小, 否则每步评估很慢)")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--out", default=None,
                        help="最终权重路径(默认=自动 cache/fusion_s<起>_<止>_<时间戳>.pt)")
    parser.add_argument("--resume", default="", help="从已保存的旁路参数继续")
    parser.add_argument("--patience", type=int, default=0,
                        help="早停耐心: 连续 N 次评估 fusion loss 无改善就停(0=关闭)")
    parser.add_argument("--plot", default=None,
                        help="loss 图路径(默认=自动带层范围+时间戳; 传空字符串=不画)")
    parser.add_argument("--attn_impl", default="",
                        help="注意力实现(sdpa/flash_attention_2/eager; 空=自动)。旁路训练建议保持 sdpa")
    args = parser.parse_args()

    cfg = Config()
    if args.small_start is not None:
        cfg.fusion_small_start = args.small_start
    if args.small_end is not None:
        cfg.fusion_small_end = args.small_end
    if args.large_start is not None:
        cfg.fusion_large_start = args.large_start
    if args.large_end is not None:
        cfg.fusion_large_end = args.large_end
    torch.manual_seed(cfg.seed)
    dt = resolve_dtype(cfg.dtype)
    print(f"精度: {dt} | CUDA: {torch.cuda.is_available()}")
    if dt not in (torch.bfloat16, torch.float16):
        print("    [提示] 建议用 bf16/fp16,4B 权重 fp32 会爆 12GB 显存")

    # ── 加载模型(全在 GPU,bf16;不用 device_map='auto' 的 offload) ──
    small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    tokenizer = AutoTokenizer.from_pretrained(small_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # flash 只给大模型: 小模型的层是手写循环调用(传 4D 掩码), flash 不支持 4D 掩码
    attn_kwargs = {"attn_implementation": args.attn_impl} if args.attn_impl else {}
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).cuda()
    large = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt, **attn_kwargs).cuda()
    # 双模型全冻结,只训旁路(22M)
    for m in (small, large):
        for p in m.parameters():
            p.requires_grad_(False)
    # 输出文件自动命名: 带大模型位置 + 旁路层范围 + 时间戳, 避免不同实验互相覆盖
    n_small = small.config.num_hidden_layers
    s1, s2 = resolve_small_range(cfg, n_small)
    l1, l2 = resolve_large_range(cfg, large.config.num_hidden_layers)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"cache/fusion_L{l1}-{l2}_s{s1}_{s2}_{ts}.pt"
    plot_path = args.plot if args.plot is not None else f"cache/train_fusion_L{l1}-{l2}_s{s1}_{s2}_{ts}.png"
    print(f"接入位置: 4B 第{l1}层取 → 第{l2}层加回 | 0.6B 第{s1}~{s2}层")
    print(f"输出权重: {out_path} | loss 图: {plot_path}")
    # 梯度检查点需要模型处于 train 模式才生效;Qwen3 attention_dropout=0 无噪声
    large.train()
    small.eval()
    if args.grad_checkpoint:
        large.gradient_checkpointing_enable()
        print("    梯度检查点已开启")

    # ── 挂载门控残差旁路 ──
    fusion = attach_fusion(large, small, cfg)
    # 关键修复: 适配器参数用 fp32 主权重。attach_fusion 把它们搬到 bf16, 若直接
    # 让 AdamW 更新 bf16 参数, 梯度小到 bf16 精度(约 3 位有效数字)就归零,
    # 训练几十步后彻底冻结。这里转回 fp32, 前向用 autocast 做 bf16 计算。
    # 必须先转 fp32 再 resume: 否则 fp32 检查点会被先压成 bf16 再升回 fp32, 丢尾数。
    for p in fusion.parameters():
        p.data = p.data.float()
    if args.resume:
        load_fusion(fusion, args.resume)
    else:
        # 关键: attach_fusion 里 gate_logit=-10(sigmoid≈4.5e-5)是为了推理时的
        # "初始恒等"。但训练时 branch = gate · adapter2(...),-10 会把所有适配器
        # 梯度整体缩小 4.5e-5 倍 → 训练卡死。这里把 gate 重置到可训练值;
        # adapter2 仍零初始化, 所以 step 0 时 branch 依然是 0, 恒等保持不破坏。
        with torch.no_grad():
            fusion.gate_logit.fill_(args.gate_init)
        # 关键修复2: 零初始化 up 投影把梯度链锁死——只有 adapter2.up 自己有梯度,
        # 其余所有参数(adapter1.*、adapter2.down/gate、两个 norm、gate_logit)的梯度
        # 都要穿过 adapter2.up 这个零点, 一开始全为 0, 只能等 adapter2.up 慢慢长起来
        # 才"解冻", adapter1 因此几乎学不动。训练时不需要恒等, 直接把两个 up 投影
        # 重初始化为小随机值, 让整条旁路第 0 步就有梯度; 输出尺度由 output_norm /
        # 0.6B 的 input_layernorm 归一化兜底, 不会爆炸。
        for m in (fusion.adapter1, fusion.adapter2):
            torch.nn.init.normal_(m.up.weight, std=0.02)
            torch.nn.init.zeros_(m.up.bias)
    params = [p for p in fusion.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"可训练旁路参数: {n_params / 1e6:.1f}M")

    opt = torch.optim.AdamW(params, lr=args.lr)
    dev = model_device(large)

    # ── 数据: 切出评估集(不参与训练, 用来对比 baseline/fusion loss) ──
    # 先全量加载, 从末尾切评估集, 再用 --max_samples 限制训练条数:
    # 这样 --max_samples 做快速测试时, 评估集仍来自文件末尾的真实验证集, 不会混进训练
    recs = load_records(args.data, 0)
    if not recs:
        raise SystemExit(f"[错误] 无训练数据: {args.data}")
    # 按 --eval_samples 从末尾切出评估集; 不再强行 10% 上限, 只保证至少留 1 条训练样本
    n_eval = min(args.eval_samples, len(recs) - 1) if args.eval_samples > 0 else 0
    eval_recs = recs[-n_eval:] if n_eval else []
    train_pool = recs[:-n_eval] if n_eval else recs
    train_recs = train_pool[:args.max_samples] if args.max_samples > 0 else train_pool
    print(f"训练样本: {len(train_recs)} 条 | 评估样本: {len(eval_recs)} 条")

    coll = lambda b: collate(b, tokenizer, args.max_len, args.answer_weight)
    loader = DataLoader(TextDataset(train_recs), batch_size=args.batch_size,
                        shuffle=True, collate_fn=coll)
    eval_loader = DataLoader(TextDataset(eval_recs), batch_size=args.eval_batch_size,
                             shuffle=False, collate_fn=coll) if eval_recs else None

    # 学习率调度: 线性 warmup → 余弦衰减到 10%×lr
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
            out = large.model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = large.lm_head(out.last_hidden_state)
        return logits

    @torch.no_grad()
    def evaluate():
        """在评估集上算 (fusion_loss, baseline_loss) 均值;baseline 关掉旁路"""
        large.eval()          # 关梯度检查点(只在 train 模式生效),no_grad 下干净前向
        f_sum = b_sum = 0.0
        n = 0
        seen = 0
        for ids, mask, labels, _w in eval_loader:
            ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
            fusion.enabled = True
            f_sum += causal_lm_loss(forward_logits(ids, mask), labels).item()
            fusion.enabled = False
            b_sum += causal_lm_loss(forward_logits(ids, mask), labels).item()
            fusion.enabled = True
            n += 1
            seen += ids.size(0)
            if args.eval_max_samples > 0 and seen >= args.eval_max_samples:
                break
        large.train()         # 恢复训练模式(重新启用梯度检查点)
        return f_sum / n, b_sum / n

    # ── 训练循环 ──
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fusion.train()
    train_ce_history = []   # (step, ce) 每一步训练 CE
    eval_history = []       # (step, fusion_loss, baseline_loss)
    best_fusion = float("inf")
    best_step = 0
    patience_counter = 0
    stop = False
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for ids, mask, labels, weights in loader:
            ids = ids.to(dev)
            mask = mask.to(dev)
            labels = labels.to(dev)
            weights = weights.to(dev)

            opt.zero_grad()
            fusion.enabled = True
            large.train()                     # 确保梯度检查点生效
            logits = forward_logits(ids, mask)
            ce = causal_lm_loss(logits, labels, weights)
            total = ce

            contrast = None
            if args.contrast_weight > 0:
                # 错配旁路: batch 内交换 h12(batch=1 时加噪声, 同 InterLat 的退化处理)
                h12 = fusion.hook_state.get("h")
                if h12 is not None:
                    h12_swap = h12.detach()
                    B = h12_swap.size(0)
                    h12_swap = h12_swap[torch.arange(B - 1, -1, -1)] if B >= 2 \
                        else h12_swap + torch.randn_like(h12_swap) * 0.01
                    fusion.branch_override = h12_swap
                    large.eval()
                    with torch.no_grad():
                        logits_r = forward_logits(ids, mask)
                    large.train()
                    fusion.branch_override = None
                    # 关键: rand 前向的 capture 钩子覆盖了 state["h"],必须恢复成
                    # 正常前向的 h12,否则 backward 重算时 branch 图结构不一致 → checkpoint 报错
                    fusion.hook_state["h"] = h12
                    contrast = js_contrastive(logits, logits_r, labels)
                    total = total + args.contrast_weight * contrast

            total.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            step += 1
            train_ce_history.append((step, ce.item()))
            if step % args.log_every == 0:
                el = time.time() - t0
                gate = float(torch.sigmoid(fusion.gate_logit).detach())
                extra = f" | js对比 {contrast.item():.4f}" if contrast is not None else ""
                print(f"epoch {ep + 1} step {step:>6} | CE {ce.item():.4f}{extra} "
                      f"| gate {gate:.5f} | lr {opt.param_groups[0]['lr']:.2e} | {el:.0f}s")

            if eval_loader is not None and step % args.eval_every == 0:
                f_loss, b_loss = evaluate()
                eval_history.append((step, f_loss, b_loss))
                gate = float(torch.sigmoid(fusion.gate_logit).detach())
                print(f"    [评估] step {step:>6} | fusion {f_loss:.4f} | "
                      f"baseline {b_loss:.4f} | 增益 {b_loss - f_loss:+.4f} "
                      f"(>0=旁路有用) | gate {gate:.5f}")
                # 早停: 评估 fusion loss 连续 patience 次无改善(改善阈值 1e-4)就停
                if f_loss < best_fusion - 1e-4:
                    best_fusion, best_step = f_loss, step
                    patience_counter = 0
                    save_fusion(fusion, out_path + ".best")
                else:
                    patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"    早停触发: {args.patience} 次评估无改善 "
                          f"(最优 fusion {best_fusion:.4f} @ step {best_step})")
                    stop = True
            if stop:
                break
        if stop:
            break

    save_fusion(fusion, out_path)
    if plot_path:
        plot_losses(train_ce_history, eval_history, plot_path)
    print(f"\n训练完成, 共 {step} 步, 旁路参数已保存: {out_path}")
    if best_fusion < float("inf"):
        print(f"最优 fusion loss {best_fusion:.4f} @ step {best_step}, 已保存: {out_path}.best")
    print("验证: 在 test_fusion.py 里 load_fusion 后解码, 或直接跑 python main.py 看效果")


if __name__ == "__main__":
    main()
