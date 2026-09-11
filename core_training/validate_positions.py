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
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(_ROOT, "core_training", "train_fusion.py")
EVAL_PY = os.path.join(_ROOT, "eval", "eval_math.py")

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
    """从 select_positions.py 的结果里取 top-N 个去重的 (L, a, b)。
    优先用出口筛选 Q 的排序(exit), 否则退回 top_full(E_seg)。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    src = data.get("exit") or data.get("top_full") or []
    seen, cands = set(), []
    for x in src:
        key = (int(x["L"]), int(x["a"]), int(x["b"]))
        if key in seen:
            continue
        seen.add(key)
        cands.append(key)
        if len(cands) >= topn:
            break
    return cands


def latest(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def read_accuracy(out_dir: str, seg_name: str, tag: str):
    """读某段(如 GSM8K / MATH_lo1-3 / MATH_hi4-5)的正确率, 返回 fusion_acc 或 None"""
    jf = latest(os.path.join(out_dir, f"eval_{seg_name.lower()}_{tag}_*.json"))
    if not jf:
        return None
    with open(jf, encoding="utf-8") as f:
        s = json.load(f).get("summary", {})
    return s.get("fusion_acc", s.get("lora_acc"))


def run_one(tag, desc, pos, gpu, args):
    """跑一个候选: 短训 + 准确率评测(三难度各 N 题), 返回结果 dict"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    out_path = os.path.join(args.out_dir, f"{tag}.pt")
    log_path = os.path.join(args.out_dir, f"{tag}.log")

    train_cmd = [sys.executable, TRAIN_PY, "--data", args.data,
                 "--max_samples", str(args.max_samples), "--epochs", str(args.epochs),
                 "--batch_size", str(args.batch_size), "--max_len", str(args.max_len),
                 "--grad_checkpoint", str(args.grad_checkpoint),
                 "--attn_impl", args.attn_impl, "--warmup_steps", str(args.warmup_steps),
                 "--eval_samples", str(args.eval_samples), "--eval_every", str(args.eval_every),
                 "--out", out_path, "--plot", ""]
    if pos is not None:
        L, l2, a, b = pos
        train_cmd += ["--large_start", str(L), "--large_end", str(l2),
                      "--small_start", str(a), "--small_end", str(b)]
    print(f"[GPU {gpu}] 短训启动: {desc}", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.call(train_cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=_ROOT, env=env)
    ce = parse_best(log_path)

    acc = None
    if not args.skip_accuracy and os.path.exists(out_path + ".best"):
        acc_cmd = [sys.executable, EVAL_PY, "--bench", "segments",
                   "--limit", str(args.acc_limit), "--seed", str(args.acc_seed),
                   "--math_lo", args.math_lo, "--math_hi", args.math_hi,
                   "--fusion_only", "--ckpt", out_path + ".best",
                   "--out_dir", args.out_dir, "--tag", tag]
        if pos is not None:
            L, l2, a, b = pos
            acc_cmd += ["--large_start", str(L), "--large_end", str(l2),
                        "--small_start", str(a), "--small_end", str(b)]
        with open(log_path + ".acc", "w", encoding="utf-8") as logf:
            subprocess.call(acc_cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=_ROOT, env=env)
        acc = {seg: read_accuracy(args.out_dir, seg, tag)
               for seg in ("GSM8K", f"MATH_lo{args.math_lo}", f"MATH_hi{args.math_hi}")}
        if not any(v is not None for v in acc.values()):
            acc = None
    return {"desc": desc, "pos": pos, "ce": ce, "acc": acc}


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
    # 短训后的准确率评测(三难度: GSM8K + MATH低级 + MATH高级, 各 N 题)
    ap.add_argument("--acc_limit", type=int, default=50, help="短训后准确率评测: 每段题数")
    ap.add_argument("--acc_seed", type=int, default=42)
    ap.add_argument("--math_lo", default="1-3", help="MATH 低级区间")
    ap.add_argument("--math_hi", default="4-5", help="MATH 高级区间")
    ap.add_argument("--skip_accuracy", action="store_true", help="跳过准确率评测, 只比 CE 增益")
    ap.add_argument("--gpus", default="0", help="逗号分隔的 GPU 列表(候选轮转分配, 如 0,1)")
    ap.add_argument("--parallel", type=int, default=1, help="同时跑几个候选(≤ GPU 数×每卡可容纳数)")
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
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"并行度 {args.parallel} | GPU 列表 {gpus} | 候选数 {len(configs)}")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = []
        for i, (tag, desc, pos) in enumerate(configs):
            gpu = gpus[i % len(gpus)]
            futs.append(ex.submit(run_one, tag, desc, pos, gpu, args))
        results = [f.result() for f in futs]

    for r in results:
        if r["ce"]:
            print(f"[{r['desc']}] 最优 fusion {r['ce'][0]:.4f} | 增益 {r['ce'][2]:+.4f}")
        if r["acc"]:
            print(f"[{r['desc']}] 正确率: " + "  ".join(
                f"{k}={v:.2%}" if v is not None else f"{k}=N/A"
                for k, v in r["acc"].items()))

    # ── 汇总 ──
    print("\n" + "=" * 100)
    print(f"短训验证汇总 (准确率 = 三难度各 {args.acc_limit} 题, 越高越好)")
    print("=" * 100)
    print(f"{'配置':<22} {'fusion':>8} {'增益':>8} "
          f"{'GSM8K':>8} {'MATH_lo':>9} {'MATH_hi':>9} {'平均':>8}")
    for r in results:
        desc, ce, acc = r["desc"], r["ce"], r["acc"]
        line = f"{desc:<22} "
        line += f"{ce[0]:>8.4f} " if ce else f"{'N/A':>8} "
        line += f"{ce[2]:>+8.4f} " if ce else f"{'N/A':>8} "
        if acc:
            for k in ("GSM8K", f"MATH_lo{args.math_lo}", f"MATH_hi{args.math_hi}"):
                v = acc.get(k)
                line += f"{v:.2%} " if v is not None else "N/A      "
            vals = [v for v in acc.values() if v is not None]
            line += f"{sum(vals) / len(vals):.2%}" if vals else "N/A"
        else:
            line += "N/A      N/A       N/A       N/A"
        print(line)

    def sort_key(r):
        if r["acc"]:
            vals = [v for v in r["acc"].values() if v is not None]
            if vals:
                return -sum(vals) / len(vals)   # 平均正确率越高越优(取负转最小)
        if r["ce"]:
            return r["ce"][0]                    # 否则 CE 越低越优
        return float("inf")

    best = min(results, key=sort_key)
    if best["ce"] is not None or best["acc"] is not None:
        print(f"\n最优: {best['desc']}")
        if best["pos"] is not None:
            L, l2, a, b = best["pos"]
            print(f"完整训练命令: python3 core_training/train_fusion.py "
                  f"--large_start {L} --large_end {l2} --small_start {a} --small_end {b} ...")

    summary = {"results": [
        {"name": r["desc"], "pos": r["pos"],
         "best_fusion": (r["ce"][0] if r["ce"] else None),
         "gain": (r["ce"][2] if r["ce"] else None),
         "accuracy": r["acc"]} for r in results]}
    out_json = os.path.join(args.out_dir, "validation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {out_json}")


if __name__ == "__main__":
    main()
