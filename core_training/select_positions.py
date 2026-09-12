# -*- coding: utf-8 -*-
"""
Approach B: 数据驱动选择 bridge 接入位置(依据 big-small-approaches.pdf)

三步筛选:
  ① 入口: CKA + 仿射对齐误差 E_in，只作可连接性诊断
  ② 片段: E_seg / 误差放大率，只淘汰明显不稳定片段
  ③ 任务: 梯度 Q + 等 RMS 真实 ΔNLL + identity control，作为最终排序依据
片段统一使用右开区间 [a,b)；训练脚本仍接收 inclusive small_end，调用时传 b-1。

用法(服务器, 在 BES 目录下):
  python3 core_training/select_positions.py \
      --data data/mix_all.jsonl --num_samples 128 --max_len 128 --topk 5
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))

from main import (Config, resolve_dtype, resolve_model_path, model_device)
from train_fusion import load_records

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EPS = 1e-8


def centered(X):
    return X - X.mean(dim=0, keepdim=True)


def finalize_hidden_buckets(buckets):
    """逐层拼接并转 fp32，及时释放碎片，避免一次性转换造成显存峰值。"""
    out = []
    for i in range(len(buckets)):
        parts = buckets[i]
        value = torch.cat(parts, dim=0).float() if parts else None
        buckets[i] = None
        out.append(value)
    return out


def rel_err(pred, target, ref):
    """相对误差: ||pred-target||² / (||target-ref||² + ε)"""
    return float((pred - target).pow(2).sum() / ((target - ref).pow(2).sum() + EPS))


def mean_ci95(values):
    """返回均值与正态近似 95% CI；筛选阶段只作不确定性诊断。"""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, [mean, mean]
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance / len(values))
    return mean, [mean - half, mean + half]


def run_small_segment_scan(sm, x, a):
    """逐层运行 0.6B 的 layers[a:], 返回每层输出的列表(保留完整序列/位置/因果 mask, 不加最终 norm)。
    outs[k] = S_{a:a+k+1}(x)  (经过 k+1 层)。"""
    from transformers.models.qwen3.modeling_qwen3 import (
        create_causal_mask, create_sliding_window_causal_mask)
    dev = x.device
    B, L, _ = x.shape
    pos_ids = torch.arange(L, device=dev).unsqueeze(0)
    cache_position = torch.arange(L, device=dev)
    pos_emb = sm.rotary_emb(x, pos_ids)
    mask_kwargs = dict(config=sm.config, inputs_embeds=x, attention_mask=None,
                       cache_position=cache_position, past_key_values=None,
                       position_ids=pos_ids)
    mask_map = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(sm, "has_sliding_layers", False):
        mask_map["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    layer_types = getattr(sm.config, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * sm.config.num_hidden_layers
    h = x
    outs = []
    for j in range(a, sm.config.num_hidden_layers):
        mask = mask_map[layer_types[j]]
        h = sm.layers[j](h, attention_mask=mask, position_embeddings=pos_emb,
                         position_ids=pos_ids, past_key_values=None, use_cache=False)
        outs.append(h)
    return outs


def build_sft_batch(tokenizer, recs, max_len, dev):
    """把 (prompt,response) 编码成 SFT 的 (ids, mask, labels), prompt 位置 label=-100"""
    ids_list, lab_list = [], []
    for rec in recs:
        prompt, resp = rec.get("prompt", ""), rec.get("response", "")
        if getattr(tokenizer, "chat_template", None):
            p_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        else:
            p_text = prompt
        p_ids = tokenizer(p_text, add_special_tokens=False).input_ids[:max_len - 1]
        r_ids = tokenizer(resp, add_special_tokens=False).input_ids
        r_ids = r_ids[:max_len - len(p_ids)]
        if not r_ids:
            r_ids = [tokenizer.eos_token_id]
        ids_list.append(p_ids + r_ids)
        lab_list.append([-100] * len(p_ids) + r_ids)
    L = max(len(x) for x in ids_list)
    B = len(ids_list)
    ids_t = torch.full((B, L), tokenizer.eos_token_id, dtype=torch.long, device=dev)
    mask_t = torch.zeros((B, L), dtype=torch.long, device=dev)
    lab_t = torch.full((B, L), -100, dtype=torch.long, device=dev)
    for i in range(B):
        n = len(ids_list[i])
        ids_t[i, :n] = torch.tensor(ids_list[i], dtype=torch.long, device=dev)
        mask_t[i, :n] = 1
        lab_t[i, :n] = torch.tensor(lab_list[i], dtype=torch.long, device=dev)
    return ids_t, mask_t, lab_t


def ridge_affine(X, Y, strength):
    """尺度不敏感的 ridge 仿射拟合；strength 相对于 Gram 的平均特征方差。"""
    Xm, Ym = X.mean(dim=0), Y.mean(dim=0)
    Xc, Yc = X - Xm, Y - Ym
    gram = Xc.t() @ Xc
    scale = gram.diagonal().mean().clamp_min(EPS)
    reg = strength * scale
    W = torch.linalg.solve(
        gram + reg * torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype),
        Xc.t() @ Yc)
    bias = Ym - Xm @ W
    return W, bias


def parse_int_list(spec):
    vals = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return sorted(set(v for v in vals if v > 0))


def select_diverse_entries(entries, limit, n_large, n_small):
    """从 3x3 深度网格中取各格最优项，避免 E_in top-k 全挤在相邻层。"""
    if limit <= 0:
        return []
    winners = {}
    for e in entries:
        lb = min(2, 3 * int(e["L"]) // max(1, n_large))
        ab = min(2, 3 * int(e["a"]) // max(1, n_small))
        winners.setdefault((lb, ab), e)
    ordered = []

    def add(e):
        key = (int(e["L"]), int(e["a"]))
        if all((int(x["L"]), int(x["a"])) != key for x in ordered):
            ordered.append(e)

    if entries:
        add(entries[0])
    default = min(entries, key=lambda e: abs(e["L"] - n_large // 3)
                  + abs(e["a"] - n_small // 3)) if entries else None
    if default is not None:
        add(default)
    for e in sorted(winners.values(), key=lambda x: x["e_in"]):
        add(e)
    for e in entries:
        add(e)
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def build_task_candidates(entry_candidates, top_full, n_large, n_small,
                          lengths, spans, limit):
    """均衡覆盖入口、片段长度和大模型跨度；b 始终为 exclusive。

    旧实现先在每个“长度×跨度”桶内按 E_seg 取最优入口。在候选预算较小
    时，每个桶往往反复选中同一个浅层入口，名义上的分层最后仍退化成少数
    L/a 的重复组合。这里用循环错位设计，让第一轮先覆盖不同入口，同时让
    length/span 随入口轮换；后续轮次再补充同一入口的其他尺度。

    top_full 参数保留用于兼容调用方；E_seg 只作为诊断，不再提前支配任务候选。
    """
    del top_full
    lengths = sorted(set(int(x) for x in lengths if int(x) > 0))
    spans = sorted(set(int(x) for x in spans if int(x) > 0))
    configs = []
    if lengths and spans:
        # 互质式轮换的前几个配置就能同时覆盖长/短片段和不同注入跨度。
        total = len(lengths) * len(spans)
        for i in range(total):
            cfg = (lengths[i % len(lengths)], spans[i % len(spans)])
            if cfg not in configs:
                configs.append(cfg)
        # 长度数和跨度数不互质时，上面的循环可能未覆盖完整笛卡尔积。
        for length in lengths:
            for span in spans:
                if (length, span) not in configs:
                    configs.append((length, span))

    pool = []
    seen = set()
    for round_idx in range(len(configs)):
        for entry_idx, e in enumerate(entry_candidates):
            length, span = configs[(entry_idx + round_idx) % len(configs)]
            L, a = int(e["L"]), int(e["a"])
            cand = (L, min(n_large - 1, L + span),
                    a, min(n_small, a + length))
            if cand[3] <= cand[2] or cand[1] <= cand[0] or cand in seen:
                continue
            seen.add(cand)
            pool.append(cand)

    # 始终保留当前默认 1/3→2/3，避免筛选器把强基线提前丢掉。
    default = (n_large // 3, min(n_large - 1, 2 * n_large // 3),
               n_small // 3, min(n_small, 2 * n_small // 3 + 1))
    out = [default]
    for cand in pool:
        if cand not in out:
            out.append(cand)
        if len(out) >= limit:
            break
    return out[:limit]


def collect_gradients(model, ids, mask, labels):
    """前反一次, 返回 grads、各层状态和基线 loss。"""
    stored = {}

    def make_hook(i):
        def h(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out.retain_grad()
            stored[i] = out
            return output
        return h
    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.model.layers)]
    embeds = model.model.embed_tokens(ids)
    embeds.requires_grad_(True)
    out = model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
    logits = model.lm_head(out.last_hidden_state)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                           labels[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    for h in handles:
        h.remove()
    grads = {i: stored[i].grad.detach().float() for i in stored if stored[i].grad is not None}
    X_full = {i + 1: stored[i].detach().float() for i in stored}
    return grads, X_full, float(loss.detach())


def _segment_feature(small, mapped, a, b_exclusive):
    """运行 [a,b) 并应用实际 fusion 使用的小模型末端 norm。"""
    if b_exclusive <= a:
        h = mapped
    else:
        h = run_small_segment_scan(small.model, mapped, a)[b_exclusive - a - 1]
    return small.model.norm(h)


@torch.no_grad()
def intervention_loss(model, row, layer_idx, update, rms_ratio):
    """在指定层输出注入等 RMS 更新，返回单样本 response NLL。"""
    valid = row["mask"].bool().unsqueeze(-1)
    base = row["X"][layer_idx + 1]
    u_rms = update.float()[valid.expand_as(update)].pow(2).mean().sqrt()
    r_rms = base.float()[valid.expand_as(base)].pow(2).mean().sqrt()
    scale = float(rms_ratio) * r_rms / u_rms.clamp_min(EPS)
    delta = update * scale

    def inject(_module, _args, output):
        if isinstance(output, tuple):
            return (output[0] + delta.to(output[0].dtype),) + output[1:]
        return output + delta.to(output.dtype)

    handle = model.model.layers[layer_idx].register_forward_hook(inject)
    try:
        out = model.model(input_ids=row["ids"], attention_mask=row["mask"], use_cache=False)
        logits = model.lm_head(out.last_hidden_state)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                               row["labels"][:, 1:].reshape(-1), ignore_index=-100)
        return float(loss)
    finally:
        handle.remove()


def exit_screening(small, large, tokenizer, recs, cands, args, Xf, Yf, dev):
    """任务感知筛选。

    cands 使用 (L,l2,a,b_exclusive)。除了梯度余弦 Q，还在独立题目上执行
    等 RMS 的真实局部注入并测量配对 ΔNLL；同时与“不运行小模型层”的
    identity control 比较，避免把 bridge 自身能力误当作小模型片段价值。
    """
    sample = recs[:args.exit_samples]
    if len(sample) < 2:
        return []
    n_fit = max(1, min(len(sample) - 1, int(len(sample) * args.exit_fit_ratio)))
    fit_recs, val_recs = sample[:n_fit], sample[n_fit:]

    def collect(recs_sub):
        rows = []
        for rec in recs_sub:
            ids, mask, labels = build_sft_batch(tokenizer, [rec], args.exit_max_len, dev)
            grads, X_full, loss = collect_gradients(large, ids, mask, labels)
            rows.append({"ids": ids, "mask": mask, "labels": labels,
                         "D": grads, "X": X_full, "loss": loss})
        return rows

    rows_f = collect(fit_recs)
    rows_v = collect(val_recs)
    small_dt = next(small.parameters()).dtype
    ratios = [float(x) for x in args.intervention_ratios.split(",") if x.strip()]
    if not ratios:
        ratios = [args.rank_ratio]

    results = []
    for (L, l2, a, b) in cands:
        A, c = ridge_affine(Xf[L + 1], Yf[a], args.lam)

        def collect_tokens(rows):
            Zl, Z0l, Dl, full = [], [], [], []
            for row in rows:
                mapped = (row["X"][L + 1] @ A + c).to(small_dt)
                z = _segment_feature(small, mapped, a, b)
                z0 = small.model.norm(mapped)
                m = row["mask"].bool()[0]
                Zl.append(z[0][m].float())
                Z0l.append(z0[0][m].float())
                Dl.append(row["D"][l2][0][m].float())
                full.append((row, z.float(), z0.float()))
            return torch.cat(Zl), torch.cat(Z0l), torch.cat(Dl), full

        Z_f, Z0_f, D_f, _ = collect_tokens(rows_f)
        Z_v, Z0_v, D_v, full_v = collect_tokens(rows_v)
        W, wc = ridge_affine(Z_f, -D_f, args.exit_lam)
        W0, w0c = ridge_affine(Z0_f, -D_f, args.exit_lam)
        ZW_v = Z_v @ W + wc
        Z0W_v = Z0_v @ W0 + w0c
        num = -(D_v * ZW_v).sum()
        den = D_v.norm() * ZW_v.norm() + EPS
        q = float(num / den)
        q0 = float(-(D_v * Z0W_v).sum() / (D_v.norm() * Z0W_v.norm() + EPS))

        deltas, control_deltas = {}, {}
        delta_samples, control_delta_samples = {}, {}
        delta_ci95, control_delta_ci95 = {}, {}
        for ratio in ratios:
            key = f"{ratio:g}"
            ds, ds0 = [], []
            for row, z, z0 in full_v:
                ds.append(intervention_loss(large, row, l2, z @ W + wc, ratio) - row["loss"])
                ds0.append(intervention_loss(large, row, l2, z0 @ W0 + w0c, ratio) - row["loss"])
            deltas[key], delta_ci95[key] = mean_ci95(ds)
            control_deltas[key], control_delta_ci95[key] = mean_ci95(ds0)
            delta_samples[key] = ds
            control_delta_samples[key] = ds0
        rank_key = f"{args.rank_ratio:g}"
        if rank_key not in deltas:
            nearest = min(ratios, key=lambda x: abs(x - args.rank_ratio))
            rank_key = f"{nearest:g}"
        incremental_samples = [x - y for x, y in zip(
            delta_samples[rank_key], control_delta_samples[rank_key])]
        incremental_mean, incremental_ci = mean_ci95(incremental_samples)
        results.append({
            "L": L, "l2": l2, "a": a, "b": b, "b_semantics": "exclusive",
            "length": b - a, "Q": q, "Q_control": q0,
            "delta_nll": deltas, "control_delta_nll": control_deltas,
            "delta_nll_ci95": delta_ci95,
            "control_delta_nll_ci95": control_delta_ci95,
            "delta_nll_samples": delta_samples,
            "control_delta_nll_samples": control_delta_samples,
            "rank_ratio": float(rank_key),
            "rank_delta_nll": deltas[rank_key],
            "rank_delta_nll_ci95": delta_ci95[rank_key],
            "incremental_delta_nll": incremental_mean,
            "incremental_delta_nll_ci95": incremental_ci,
            "task_fit_samples": len(rows_f), "task_val_samples": len(rows_v),
        })
    return rank_task_aware_results(results)


def rank_task_aware_results(results):
    """优先保留既改善任务 NLL、又优于同容量 identity control 的候选。"""
    return sorted(results, key=lambda x: (
        x["rank_delta_nll"] >= 0,
        x["incremental_delta_nll"] >= 0,
        x["incremental_delta_nll"],
        x["rank_delta_nll"],
        -x["Q"],
    ))


def main():
    ap = argparse.ArgumentParser(description="数据驱动选择 bridge 接入位置(Approach B)")
    ap.add_argument("--data", default="data/mix_all.jsonl")
    ap.add_argument("--num_samples", type=int, default=128,
                    help="表示映射校准样本数（任务筛选另取 exit_samples 条独立题目）")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--fit_ratio", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=9,
                    help="做片段诊断的入口候选数；按大/小模型深度网格分层抽取")
    ap.add_argument("--lam", type=float, default=0.1,
                    help="ridge 正则(相对 gram 对角尺度, 实际 λ = lam × fit token 数)")
    ap.add_argument("--out", default="cache/position_selection.json")
    # 出口筛选(任务梯度对齐 Q)
    ap.add_argument("--inject_span", type=int, default=12,
                    help="兼容旧命令；未显式给 --inject_spans 时使用")
    ap.add_argument("--inject_spans", default="6,12,18",
                    help="候选的大模型捕获→注入跨度，逗号分隔")
    ap.add_argument("--segment_lengths", default="1,2,4,8,12",
                    help="候选小模型片段长度，逗号分隔")
    ap.add_argument("--exit_topk", type=int, default=12,
                    help="执行任务感知 ΔNLL 筛选的分层候选数")
    ap.add_argument("--exit_samples", type=int, default=64,
                    help="任务感知筛选样本数；从入口映射未见过的题目中抽取")
    ap.add_argument("--exit_fit_ratio", type=float, default=0.5,
                    help="任务 probe 拟合比例；其余样本用于候选间配对比较")
    ap.add_argument("--exit_max_len", type=int, default=96)
    ap.add_argument("--exit_lam", type=float, default=0.1,
                    help="输出 bridge 的 ridge(相对 gram 尺度)")
    ap.add_argument("--intervention_ratios", default="0.01,0.03,0.10",
                    help="局部注入更新/原残差的 RMS 比例")
    ap.add_argument("--rank_ratio", type=float, default=0.03,
                    help="用于候选主排序的 RMS 比例")
    args = ap.parse_args()

    cfg = Config()
    dt = resolve_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed)
    print(f"精度 {dt} | CUDA {torch.cuda.is_available()}")

    small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    # 任务训练与最终解码都由 4B Instruct 定义格式，筛选必须使用同一 chat template。
    tokenizer = AutoTokenizer.from_pretrained(large_path)
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).cuda().eval()
    large = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt).cuda().eval()
    for m in (small, large):
        for p in m.parameters():
            p.requires_grad_(False)

    n_large = large.config.num_hidden_layers
    n_small = small.config.num_hidden_layers
    print(f"4B: {n_large} 层 | 0.6B: {n_small} 层")

    # ── 采样校准文本 ──
    all_recs = load_records(args.data, 0)
    if not all_recs:
        raise SystemExit(f"[错误] 无数据: {args.data}")
    random.Random(cfg.seed).shuffle(all_recs)
    # num_samples 与 exit_samples 是两个互斥数据池，避免增大任务样本数时反而
    # 挤占表示拟合样本；数据不足时优先保留至少两个表示样本和两个任务样本。
    n_task = min(args.exit_samples, max(0, len(all_recs) - 2))
    n_repr = min(args.num_samples, len(all_recs) - n_task)
    if n_repr < 2 or n_task < 2:
        raise SystemExit("[错误] 数据太少，表示校准和任务筛选都至少需要 2 条")
    repr_recs = all_recs[:n_repr]
    task_recs = all_recs[n_repr:n_repr + n_task]
    def calibration_text(r):
        prompt, response = r.get("prompt", ""), r.get("response", "")
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        return (prompt + response)[:2000]

    texts = [calibration_text(r) for r in repr_recs]
    # 三路按题目切分：表示映射 fit / 表示诊断 val / 最终 task screen。
    # task screen 使用 task_recs，不参与 E_in/E_seg 候选构造。
    n_fit = max(1, min(n_repr - 1, int(n_repr * args.fit_ratio)))
    print(f"校准样本 {n_repr + n_task} (repr fit {n_fit} / repr val {n_repr - n_fit} "
          f"/ task {n_task})")

    dev = model_device(large)

    # ── 收集隐状态: 按样本归属 fit/val, 展平有效 token ──
    # X[L+1]: 捕获层 L 的输出(即 hidden_states[L+1]); Y[a]: 小模型第 a 层输入(hidden_states[a])
    Xf = [[] for _ in range(n_large + 1)]
    Xv = [[] for _ in range(n_large + 1)]
    Yf = [[] for _ in range(n_small + 1)]
    Yv = [[] for _ in range(n_small + 1)]

    t0 = time.time()
    for s in range(0, n_repr, args.batch_size):
        idxs = list(range(s, min(s + args.batch_size, n_repr)))
        batch = [texts[i] for i in idxs]
        toks = tokenizer(batch, max_length=args.max_len, truncation=True,
                         padding=True, return_tensors="pt")
        ids = toks.input_ids.to(dev)
        mask = toks.attention_mask.to(dev)
        with torch.no_grad():
            bh = large.model(input_ids=ids, attention_mask=mask,
                             output_hidden_states=True, use_cache=False).hidden_states
            sh = small.model(input_ids=ids, attention_mask=mask,
                             output_hidden_states=True, use_cache=False).hidden_states
        for k, idx in enumerate(idxs):
            m = mask[k].bool()
            Xd = Xf if idx < n_fit else Xv
            Yd = Yf if idx < n_fit else Yv
            for L in range(n_large):
                Xd[L + 1].append(bh[L + 1][k][m])
            # 需要包含 hidden_states[n_small]，供 [a,b) 在 b=n_small 时计算
            # 原生片段输出的 fit 均值。
            for a in range(n_small + 1):
                Yd[a].append(sh[a][k][m])

    # 逐层拼接并转 fp32，避免先同时保留全部 bf16 cat、再整体复制 fp32 的峰值。
    Xf = finalize_hidden_buckets(Xf)
    Xv = finalize_hidden_buckets(Xv)
    Yf = finalize_hidden_buckets(Yf)
    Yv = finalize_hidden_buckets(Yv)
    print(f"隐状态收集完成 ({time.time() - t0:.0f}s)")

    # ── ① 入口筛选: 所有 (L, a) 的 CKA 和 E_in ──
    entries = []
    for L in range(n_large):          # 捕获层 L, 用 Xf[L+1]
        XfL = centered(Xf[L + 1])
        Gxx = XfL.t() @ XfL                     # 复用: CKA 分母 + ridge 的 Gram
        xx_frob = Gxx.pow(2).sum().sqrt()
        for a in range(n_small):      # 入口层 a
            Yfa = centered(Yf[a])
            Xy = XfL.t() @ Yfa                  # 复用: CKA 分子 + ridge 的 RHS
            A, c = ridge_affine(Xf[L + 1], Yf[a], args.lam)
            # 关键: 预测用未中心化的 Xv @ A + c(中心化只在 fit 集做, 不能用 val 自己的均值)
            pred = Xv[L + 1] @ A + c
            e_in = rel_err(pred, Yv[a], Yf[a].mean(dim=0))
            yy_frob = (Yfa.t() @ Yfa).pow(2).sum().sqrt()
            cka = float(Xy.pow(2).sum() / (xx_frob * yy_frob + EPS))
            entries.append({"L": L, "a": a, "cka": round(cka, 4), "e_in": e_in})

    entries.sort(key=lambda e: e["e_in"])
    print("\n===== 入口候选(按 E_in 升序, 越小越易对齐) top 12 =====")
    print(f"{'L(4B取)':>8} {'a(0.6B入)':>10} {'CKA':>8} {'E_in':>10}")
    for e in entries[:12]:
        print(f"{e['L']:>8} {e['a']:>10} {e['cka']:>8.4f} {e['e_in']:>10.4f}")

    # ── ② 片段筛选: 对 top-K 入口, 用验证集算 E_seg ──
    entry_candidates = select_diverse_entries(entries, args.topk, n_large, n_small)
    print(f"\n===== 片段诊断(分层抽取 {len(entry_candidates)} 个入口, b 为 exclusive) =====")
    top_full = []
    for e in entry_candidates:
        L, a = e["L"], e["a"]
        A, c = ridge_affine(Xf[L + 1], Yf[a], args.lam)

        # 文档定义的分母使用 fit 集逐特征均值。原生完整前向已收集了
        # 每层 hidden state，无需为每个候选重复运行小模型。
        native_fit_mean = [Yf[a + k + 1].mean(dim=0)
                           for k in range(n_small - a)]

        # 逐条验证样本跑片段, 累计每段终点 b 的误差
        num = [0.0] * (n_small - a)
        den = [0.0] * (n_small - a)
        for vi in range(n_fit, n_repr):
            toks = tokenizer(texts[vi], max_length=args.max_len, truncation=True,
                             return_tensors="pt")
            ids = toks.input_ids.to(dev)
            mask = toks.attention_mask.to(dev)
            with torch.no_grad():
                bh = large.model(input_ids=ids, attention_mask=mask,
                                 output_hidden_states=True, use_cache=False).hidden_states
                sh = small.model(input_ids=ids, attention_mask=mask,
                                 output_hidden_states=True, use_cache=False).hidden_states
            x_seq = bh[L + 1]                       # (1, Lseq, d_large), bf16
            y_native = sh[a]                        # (1, Lseq, d_small), bf16
            # A 是 fp32, 先转 x_seq 再乘, 最后对齐小模型 dtype
            mapped = (x_seq.float() @ A + c).to(y_native.dtype)
            m = mask.bool()[0]                      # (Lseq,) 一维有效位置掩码
            with torch.no_grad():
                mapped_outs = run_small_segment_scan(small.model, mapped, a)
            for k in range(len(mapped_outs)):
                # 原生完整前向的 hidden_states[a+k+1] 就是 S_[a:a+k+1) 的输出。
                nv = sh[a + k + 1][0][m].float()
                mv = mapped_outs[k][0][m].float()
                num[k] += float((mv - nv).pow(2).sum())
                den[k] += float((nv - native_fit_mean[k]).pow(2).sum())
        e_seg = [num[k] / (den[k] + EPS) for k in range(len(num))]
        best_b = int(a + 1 + min(range(len(e_seg)), key=lambda k: e_seg[k]))
        print(f"L={L:>2} a={a:>2} | E_seg[前3个b]={['%.3f' % v for v in e_seg[:3]]} "
              f"... 最优 b={best_b} (E_seg={e_seg[best_b - a - 1]:.3f})")
        for k, v in enumerate(e_seg):
            top_full.append({"L": L, "a": a, "b": a + 1 + k,
                             "b_semantics": "exclusive", "length": k + 1,
                             "e_seg": v, "e_seg_amp": v / max(e["e_in"], EPS),
                             "cka": e["cka"], "e_in": e["e_in"]})

    top_full.sort(key=lambda x: x["e_seg"])
    print("\n===== 片段诊断(按 E_seg 展示；不作为最终推荐) top 10 =====")
    print(f"{'L(4B取)':>8} {'a(入)':>6} {'b(止)':>6} {'E_seg':>9} {'E_in':>9} {'CKA':>8}")
    for x in top_full[:10]:
        print(f"{x['L']:>8} {x['a']:>6} {x['b']:>6} {x['e_seg']:>9.4f} "
              f"{x['e_in']:>9.4f} {x['cka']:>8.4f}")

    # ── ③ 任务感知筛选: 分层候选，不允许 E_seg 把深片段提前淘汰 ──
    exit_results = []
    lengths = parse_int_list(args.segment_lengths)
    spans = parse_int_list(args.inject_spans) or [args.inject_span]
    cands = build_task_candidates(entry_candidates, top_full, n_large, n_small,
                                  lengths, spans, args.exit_topk)
    if cands:
        print(f"\n===== 任务感知筛选(分层 {len(cands)} 候选, 独立样本最多 {args.exit_samples}) =====")
        exit_results = exit_screening(small, large, tokenizer, task_recs,
                                      cands, args, Xf, Yf, dev)
        print(f"{'L':>4} {'L2':>4} {'[a,b)':>9} {'Q':>8} {'ΔNLL':>10} {'相对control':>12}")
        for x in exit_results:
            segment = f"[{x['a']},{x['b']})"
            print(f"{x['L']:>4} {x['l2']:>4} {segment:>9} "
                  f"{x['Q']:>8.4f} {x['rank_delta_nll']:>+10.5f} "
                  f"{x['incremental_delta_nll']:>+12.5f}")

    payload = {"meta": {"data": args.data, "num_samples": args.num_samples,
                        "max_len": args.max_len, "lam": args.lam, "seed": cfg.seed,
                        "repr_fit_samples": n_fit,
                        "repr_val_samples": n_repr - n_fit,
                        "task_samples": n_task},
               "entry_candidates": entry_candidates,
               "task_candidates": [{"L": x[0], "l2": x[1], "a": x[2], "b": x[3],
                                     "b_semantics": "exclusive"} for x in cands],
               "entries": entries, "top_full": top_full,
               "task_aware": exit_results, "exit": exit_results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}")

    if exit_results or top_full:
        best = exit_results[0] if exit_results else top_full[0]
        src = "任务感知 ΔNLL" if exit_results else "E_seg"
        print(f"\n建议训练命令(按 {src}):")
        l2 = best.get("l2", min(best["L"] + args.inject_span, n_large - 1))
        # train_fusion 的 small_end 仍为 inclusive，因此 exclusive b 必须减 1。
        print(f"  python3 core_training/train_fusion.py --large_start {best['L']} "
              f"--small_start {best['a']} --small_end {best['b'] - 1} "
              f"--large_end {l2} --contrast_weight 0 ...")


if __name__ == "__main__":
    main()
