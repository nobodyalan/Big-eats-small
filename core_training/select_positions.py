# -*- coding: utf-8 -*-
"""
Approach B: 数据驱动选择 bridge 接入位置(依据 big-small-approaches.pdf)

三步筛选:
  ① 入口: CKA + 仿射对齐误差 E_in   → 筛 (大模型捕获层 L, 小模型入口层 a)
  ② 片段: 相对输出误差 E_seg        → 筛 片段终点 b
  ③ 出口: 任务梯度对齐度 Q          → 对 top-k 候选按 Q 重排(与下游任务最相关)
输出: 推荐 --large_start L --small_start a --small_end b, 保存 cache/position_selection.json

用法(服务器, 在 BES 目录下):
  python3 core_training/select_positions.py \
      --data data/mix_all.jsonl --num_samples 128 --max_len 128 --topk 5
"""

import argparse
import json
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


def rel_err(pred, target, ref):
    """相对误差: ||pred-target||² / (||target-ref||² + ε)"""
    return float((pred - target).pow(2).sum() / ((target - ref).pow(2).sum() + EPS))


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


def collect_gradients(model, ids, mask, labels):
    """前反一次, 返回 (grads, X_full)。grads[l]=∂L/∂(layer l 输出); X_full[l+1]=layer l 输出。"""
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
    return grads, X_full


def exit_screening(small, large, tokenizer, recs, cands, args, Xf, Yf, dev):
    """出口筛选: 对候选 (L,a,b) 算任务梯度对齐度 Q; 返回按 Q 降序的 [{L,a,b,Q}]
    Q = -⟨D_val, Z_val W*⟩ / (‖D_val‖·‖Z_val W*‖), W* 为线性输出 bridge。"""
    n_large = large.config.num_hidden_layers
    sample = recs[:args.exit_samples]
    n_fit = max(1, int(len(sample) * args.fit_ratio))
    fit_recs, val_recs = sample[:n_fit], sample[n_fit:]

    def collect(recs_sub):
        Xs, Ds, masks = [], [], []
        for rec in recs_sub:
            ids, mask, labels = build_sft_batch(tokenizer, [rec], args.exit_max_len, dev)
            grads, X_full = collect_gradients(large, ids, mask, labels)
            Xs.append(X_full)
            Ds.append(grads)
            masks.append(mask.bool()[0])
        return Xs, Ds, masks

    Xs_f, Ds_f, mk_f = collect(fit_recs)
    Xs_v, Ds_v, mk_v = collect(val_recs)
    small_dt = next(small.parameters()).dtype

    results = []
    for (L, a, b) in cands:
        l2 = min(L + args.inject_span, n_large - 1)
        XfL = centered(Xf[L + 1])
        lam_in = args.lam * Xf[L + 1].size(0)
        A = torch.linalg.solve(
            XfL.t() @ XfL + lam_in * torch.eye(XfL.size(1), device=dev, dtype=XfL.dtype),
            XfL.t() @ centered(Yf[a]))
        c = Yf[a].mean(dim=0) - Xf[L + 1].mean(dim=0) @ A

        def collect_tokens(Xs, Ds, ms):
            Zl, Dl = [], []
            for k in range(len(Xs)):
                mapped = (Xs[k][L + 1] @ A + c).to(small_dt)
                outs = run_small_segment_scan(small.model, mapped, a)
                Zl.append(outs[b - a - 1][0][ms[k]].float())
                Dl.append(Ds[k][l2][0][ms[k]])
            return torch.cat(Zl, dim=0), torch.cat(Dl, dim=0)

        Z_f, D_f = collect_tokens(Xs_f, Ds_f, mk_f)
        Z_v, D_v = collect_tokens(Xs_v, Ds_v, mk_v)
        lam_out = args.exit_lam * Z_f.size(0)
        W = torch.linalg.solve(
            Z_f.t() @ Z_f + lam_out * torch.eye(Z_f.size(1), device=dev, dtype=Z_f.dtype),
            Z_f.t() @ (-D_f))
        ZW_v = Z_v @ W
        num = -(D_v * ZW_v).sum()
        den = D_v.norm() * ZW_v.norm() + EPS
        results.append({"L": L, "a": a, "b": b, "Q": float(num / den)})
    results.sort(key=lambda x: -x["Q"])
    return results


def main():
    ap = argparse.ArgumentParser(description="数据驱动选择 bridge 接入位置(Approach B)")
    ap.add_argument("--data", default="data/mix_all.jsonl")
    ap.add_argument("--num_samples", type=int, default=128, help="校准样本数")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--fit_ratio", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=5, help="做片段(E_seg)筛选的入口候选数")
    ap.add_argument("--lam", type=float, default=0.1,
                    help="ridge 正则(相对 gram 对角尺度, 实际 λ = lam × fit token 数)")
    ap.add_argument("--out", default="cache/position_selection.json")
    # 出口筛选(任务梯度对齐 Q)
    ap.add_argument("--inject_span", type=int, default=12, help="加回位置 = 捕获层 + span")
    ap.add_argument("--exit_topk", type=int, default=5, help="出口筛选的候选数(取 E_seg 前 N)")
    ap.add_argument("--exit_samples", type=int, default=32, help="出口筛选用的校准样本数")
    ap.add_argument("--exit_max_len", type=int, default=96)
    ap.add_argument("--exit_lam", type=float, default=0.1,
                    help="输出 bridge 的 ridge(相对 gram 尺度)")
    args = ap.parse_args()

    cfg = Config()
    dt = resolve_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed)
    print(f"精度 {dt} | CUDA {torch.cuda.is_available()}")

    small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    tokenizer = AutoTokenizer.from_pretrained(small_path)
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).cuda().eval()
    large = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt).cuda().eval()
    for m in (small, large):
        for p in m.parameters():
            p.requires_grad_(False)

    n_large = large.config.num_hidden_layers
    n_small = small.config.num_hidden_layers
    print(f"4B: {n_large} 层 | 0.6B: {n_small} 层")

    # ── 采样校准文本 ──
    recs = load_records(args.data, 0)
    if not recs:
        raise SystemExit(f"[错误] 无数据: {args.data}")
    random.Random(cfg.seed).shuffle(recs)
    recs = recs[:args.num_samples]
    texts = [(r.get("prompt", "") + r.get("response", ""))[:2000] for r in recs]
    n_fit = max(1, int(len(texts) * args.fit_ratio))
    print(f"校准样本 {len(texts)} (fit {n_fit} / val {len(texts) - n_fit})")

    dev = model_device(large)

    # ── 收集隐状态: 按样本归属 fit/val, 展平有效 token ──
    # X[L+1]: 捕获层 L 的输出(即 hidden_states[L+1]); Y[a]: 小模型第 a 层输入(hidden_states[a])
    Xf = [[] for _ in range(n_large + 1)]
    Xv = [[] for _ in range(n_large + 1)]
    Yf = [[] for _ in range(n_small + 1)]
    Yv = [[] for _ in range(n_small + 1)]

    t0 = time.time()
    for s in range(0, len(texts), args.batch_size):
        idxs = list(range(s, min(s + args.batch_size, len(texts))))
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
            for a in range(n_small):
                Yd[a].append(sh[a][k][m])

    Xf = [torch.cat(x, dim=0) if x else None for x in Xf]
    Xv = [torch.cat(x, dim=0) if x else None for x in Xv]
    Yf = [torch.cat(y, dim=0) if y else None for y in Yf]
    Yv = [torch.cat(y, dim=0) if y else None for y in Yv]
    # 转 fp32 再算 Gram/岭回归, 避免 bf16 中心化/累加精度损失
    Xf = [x.float() if x is not None else None for x in Xf]
    Xv = [x.float() if x is not None else None for x in Xv]
    Yf = [y.float() if y is not None else None for y in Yf]
    Yv = [y.float() if y is not None else None for y in Yv]
    print(f"隐状态收集完成 ({time.time() - t0:.0f}s)")

    # ── ① 入口筛选: 所有 (L, a) 的 CKA 和 E_in ──
    # λ 随 fit token 数缩放(gram 对角 ≈ V, 使 --lam 表示"相对正则强度")
    lam = args.lam * Xf[1].size(0)
    entries = []
    for L in range(n_large):          # 捕获层 L, 用 Xf[L+1]
        XfL = centered(Xf[L + 1])
        Gxx = XfL.t() @ XfL                     # 复用: CKA 分母 + ridge 的 Gram
        xx_frob = Gxx.pow(2).sum().sqrt()
        Gx_reg = Gxx + lam * torch.eye(Gxx.size(0), device=Gxx.device, dtype=Gxx.dtype)
        for a in range(n_small):      # 入口层 a
            Yfa = centered(Yf[a])
            Xy = XfL.t() @ Yfa                  # 复用: CKA 分子 + ridge 的 RHS
            A = torch.linalg.solve(Gx_reg, Xy)
            c = Yf[a].mean(dim=0) - Xf[L + 1].mean(dim=0) @ A
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
    print(f"\n===== 片段筛选(对 top{args.topk} 入口, 完整序列跑小模型片段) =====")
    top_full = []
    for e in entries[:args.topk]:
        L, a = e["L"], e["a"]
        # 该 (L,a) 的仿射映射(在 fit 上重算, 用与入口筛选相同的缩放 λ)
        XfL = centered(Xf[L + 1])
        A = torch.linalg.solve(
            XfL.t() @ XfL + lam * torch.eye(XfL.size(1), device=XfL.device, dtype=XfL.dtype),
            XfL.t() @ centered(Yf[a]))
        c = Yf[a].mean(dim=0) - Xf[L + 1].mean(dim=0) @ A

        # 逐条验证样本跑片段, 累计每段终点 b 的误差
        num = [0.0] * (n_small - a)
        den = [0.0] * (n_small - a)
        for vi in range(n_fit, len(texts)):
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
                native_outs = run_small_segment_scan(small.model, y_native, a)
                mapped_outs = run_small_segment_scan(small.model, mapped, a)
            for k in range(len(native_outs)):
                nv = native_outs[k][0][m]
                mv = mapped_outs[k][0][m]
                num[k] += float((mv - nv).pow(2).sum())
                den[k] += float((nv - nv.mean()).pow(2).sum())
        e_seg = [num[k] / (den[k] + EPS) for k in range(len(num))]
        best_b = int(a + 1 + min(range(len(e_seg)), key=lambda k: e_seg[k]))
        print(f"L={L:>2} a={a:>2} | E_seg[前3个b]={['%.3f' % v for v in e_seg[:3]]} "
              f"... 最优 b={best_b} (E_seg={e_seg[best_b - a - 1]:.3f})")
        for k, v in enumerate(e_seg):
            top_full.append({"L": L, "a": a, "b": a + 1 + k, "e_seg": v,
                             "cka": e["cka"], "e_in": e["e_in"]})

    top_full.sort(key=lambda x: x["e_seg"])
    print("\n===== 综合推荐(按 E_seg 升序) top 10 =====")
    print(f"{'L(4B取)':>8} {'a(入)':>6} {'b(止)':>6} {'E_seg':>9} {'E_in':>9} {'CKA':>8}")
    for x in top_full[:10]:
        print(f"{x['L']:>8} {x['a']:>6} {x['b']:>6} {x['e_seg']:>9.4f} "
              f"{x['e_in']:>9.4f} {x['cka']:>8.4f}")

    # ── ③ 出口筛选: 任务梯度对齐 Q(对 E_seg 前 exit_topk 个去重候选) ──
    exit_results = []
    seen_c, cands = set(), []
    for x in top_full:
        key = (x["L"], x["a"], x["b"])
        if key not in seen_c:
            seen_c.add(key)
            cands.append(key)
        if len(cands) >= args.exit_topk:
            break
    if cands:
        print(f"\n===== 出口筛选(Q 梯度对齐, top{len(cands)} 候选, {args.exit_samples} 样本) =====")
        exit_results = exit_screening(small, large, tokenizer, recs, cands, args, Xf, Yf, dev)
        print(f"{'L':>6} {'a':>4} {'b':>4} {'Q':>9}")
        for x in exit_results:
            print(f"{x['L']:>6} {x['a']:>4} {x['b']:>4} {x['Q']:>9.4f}")

    payload = {"meta": {"data": args.data, "num_samples": args.num_samples,
                        "max_len": args.max_len, "lam": args.lam, "seed": cfg.seed},
               "entries": entries, "top_full": top_full, "exit": exit_results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}")

    if exit_results or top_full:
        best = exit_results[0] if exit_results else top_full[0]
        src = "Q 梯度对齐" if exit_results else "E_seg"
        print(f"\n建议训练命令(按 {src}):")
        print(f"  python3 core_training/train_fusion.py --large_start {best['L']} "
              f"--small_start {best['a']} --small_end {best['b']} "
              f"--large_end {min(best['L'] + args.inject_span, 35)} ...")


if __name__ == "__main__":
    main()
