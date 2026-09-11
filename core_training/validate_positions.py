# -*- coding: utf-8 -*-
"""
短训验证候选接入位置(Approach B 的"短训筛选"步骤)

从 select_positions.py 的结果里取 top-N 个 (L, a, b) 候选, 加上"默认 1/3~2/3"作对照,
用相同预算各短训一次, 比谁在验证集上的 eval 增益最高(增益 = baseline - fusion)。

用法(服务器, BES 目录下):
  CUDA_VISIBLE_DEVICES=1 python3 core_training/validate_positions.py \
      --candidates_json cache/position_selection.json --topn 3 \
      --max_samples 1000 --epochs 1
"""

import argparse
import json
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(_ROOT, "core_training", "train_fusion.py")

EVAL_PAT = re.compile(
    r"\[评估\].*?fusion\s+([\d.e+\-]+).*?baseline\s+([\d.e+\-]+).*?增益\s+([+\-][\d.e+\-]+)")


def parse_best(log_path):
    """从训练日志里找 fusion loss 最低的那次评估, 返回 (fusion, baseline, gain) 或 None"""
    best = None
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = EVAL_PAT.search(line)
            if m:
                fl, bl, gl = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if best is None or fl < best[0]:
                    best = (fl, bl, gl)
    return best


def load_candidates(json_path, topn):
    """从 select_positions.py 的结果里取 top-N 个去重的 (L, a, b)"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    seen, cands = set(), []
    for x in data.get("top_full", []):
        key = (int(x["L"]), int(x["a"]), int(x["b"]))
        if key in seen:
            continue
        seen.add(key)
        cands.append(key)
        if len(cands) >= topn:
            break
    return cands


def main():
    ap = argparse.ArgumentParser(description="短训验证候选接入位置")
    ap.add_argument("--candidates_json", default="cache/position_selection.json")
    ap.add_argument("--topn", type=int, default=3, help="从结果里取几个候选")
    ap.add_argument("--inject_span", type=int, default=12, help="加回位置 = 捕获层 + span")
    # 短训预算(所有候选用完全相同的一套)
    ap.add_argument("--data", default="data/mix_all.jsonl")
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--grad_checkpoint", type=int, default=1)
    ap.add_argument("--attn_impl", default="flash_attention_2")
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--eval_samples", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--out_dir", default="cache/validate_positions")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 配置列表: (文件tag, 显示描述, 位置元组或 None=默认)
    configs = [("baseline_default", "默认 1/3~2/3", None)]
    if os.path.exists(args.candidates_json):
        for i, (L, a, b) in enumerate(load_candidates(args.candidates_json, args.topn)):
            l2 = min(L + args.inject_span, 35)
            tag = f"cand{i}_L{L}_{l2}_s{a}_{b}"
            configs.append((tag, f"L={L}→{l2}, a={a}~{b}", (L, l2, a, b)))
    else:
        print(f"[提示] 未找到 {args.candidates_json}, 只跑 baseline 对照")

    results = []
    for tag, desc, pos in configs:
        out_path = os.path.join(args.out_dir, f"{tag}.pt")
        log_path = os.path.join(args.out_dir, f"{tag}.log")
        cmd = [sys.executable, TRAIN_PY, "--data", args.data,
               "--max_samples", str(args.max_samples), "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size), "--max_len", str(args.max_len),
               "--grad_checkpoint", str(args.grad_checkpoint),
               "--attn_impl", args.attn_impl, "--warmup_steps", str(args.warmup_steps),
               "--eval_samples", str(args.eval_samples), "--eval_every", str(args.eval_every),
               "--out", out_path, "--plot", ""]
        if pos is not None:
            L, l2, a, b = pos
            cmd += ["--large_start", str(L), "--large_end", str(l2),
                    "--small_start", str(a), "--small_end", str(b)]
        print(f"\n===== 短训 [{desc}] =====")
        with open(log_path, "w", encoding="utf-8") as logf:
            rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=_ROOT)
        best = parse_best(log_path)
        if best is None:
            print(f"[{desc}] 无 eval 结果 (退出码 {rc}), 详见 {log_path}")
            results.append((desc, pos, None))
        else:
            fl, bl, gl = best
            print(f"[{desc}] 最优 fusion {fl:.4f} | baseline {bl:.4f} | 增益 {gl:+.4f}")
            results.append((desc, pos, (fl, bl, gl)))

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("短训验证汇总 (增益 = baseline - fusion, 越大越好)")
    print("=" * 80)
    print(f"{'配置':<28} {'fusion':>9} {'增益':>10}")
    for desc, pos, r in results:
        if r is None:
            print(f"{desc:<28} {'N/A':>9} {'N/A':>10}")
        else:
            print(f"{desc:<28} {r[0]:>9.4f} {r[2]:>+10.4f}")

    valid = [(d, p, r) for d, p, r in results if r is not None]
    if valid:
        best_desc, best_pos, best_r = min(valid, key=lambda x: x[2][0])
        print(f"\n最优: {best_desc} (fusion {best_r[0]:.4f}, 增益 {best_r[2]:+.4f})")
        if best_pos is not None:
            L, l2, a, b = best_pos
            print(f"完整训练命令: python3 core_training/train_fusion.py "
                  f"--large_start {L} --large_end {l2} --small_start {a} --small_end {b} ...")

    summary = {"results": [
        {"name": d, "pos": p,
         "best_fusion": (r[0] if r else None),
         "gain": (r[2] if r else None)} for d, p, r in results]}
    out_json = os.path.join(args.out_dir, "validation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {out_json}")


if __name__ == "__main__":
    main()
