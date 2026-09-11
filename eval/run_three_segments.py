# -*- coding: utf-8 -*-
"""
并行三模型 × 三段(GSM8K / MATH 低级 / MATH 高级)评测 + 汇总成对比表

用法(服务器, BES 目录下):
  CUDA_VISIBLE_DEVICES=1 python3 eval/run_three_segments.py \
      --old_ckpt <1/3旁路权重> --full_ckpt cache/fusion_full_sft.pt.best \
      --lora_ckpt cache/lora_r21_sft.best --limit 400

原理: 三个模型各跑一次 `eval_math.py --bench segments`(同 seed → 各段题目一致),
每个模型输出 GSM8K / MATH_lo / MATH_hi 三个 JSON, 最后汇总成一张表。
"""

import argparse
import glob
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PY = os.path.join(_ROOT, "eval", "eval_math.py")


def latest(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def read_summary(out_dir: str, seg_name: str, tag: str):
    jf = latest(os.path.join(out_dir, f"eval_{seg_name.lower()}_{tag}_*.json"))
    if not jf:
        return None
    with open(jf, encoding="utf-8") as f:
        s = json.load(f).get("summary", {})
    return {"model_acc": s.get("fusion_acc", s.get("lora_acc")),
            "baseline_acc": s.get("baseline_acc"), "file": jf}


def main():
    ap = argparse.ArgumentParser(description="三模型 × 三段(GSM8K/MATH低级/MATH高级)并行评测")
    ap.add_argument("--old_ckpt", default="", help="1/3 旁路权重(空=跳过)")
    ap.add_argument("--full_ckpt", default="cache/fusion_full_sft.pt.best", help="整段旁路权重")
    ap.add_argument("--lora_ckpt", default="cache/lora_r21_sft.best", help="LoRA 权重目录")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--math_lo", default="1-3", help="MATH 低级区间")
    ap.add_argument("--math_hi", default="4-5", help="MATH 高级区间")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--tag", default="", help="汇总表文件名标签")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = [sys.executable, EVAL_PY, "--bench", "segments",
            "--limit", str(args.limit), "--seed", str(args.seed),
            "--max_new", str(args.max_new), "--math_lo", args.math_lo,
            "--math_hi", args.math_hi, "--out_dir", args.out_dir]

    jobs = []
    if args.old_ckpt:
        jobs.append(("onethird", ["--ckpt", args.old_ckpt]))
    else:
        print("[提示] 未提供 --old_ckpt, 跳过 1/3 旁路")
    jobs.append(("full", ["--ckpt", args.full_ckpt,
                          "--small_start", "0", "--small_end", "-1"]))
    jobs.append(("lora", ["--lora_ckpt", args.lora_ckpt]))

    procs = {}
    for name, extra in jobs:
        cmd = base + extra + ["--tag", name]
        suffix = f"_{args.tag}" if args.tag else ""
        log_path = os.path.join(args.out_dir, f"eval_seg_{name}{suffix}.log")
        logf = open(log_path, "w", encoding="utf-8")
        print(f"启动 [{name}]: {' '.join(cmd)}")
        procs[name] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                        cwd=_ROOT), logf, log_path)

    for name, (p, logf, log_path) in procs.items():
        rc = p.wait()
        logf.close()
        print(f"[{name}] 退出码 {rc} (日志: {log_path})")

    # ── 汇总 ──
    segs = [("GSM8K", "GSM8K"),
            ("MATH_lo", f"MATH_lo{args.math_lo}"),
            ("MATH_hi", f"MATH_hi{args.math_hi}")]
    table = {"meta": {"seed": args.seed, "limit": args.limit,
                      "math_lo": args.math_lo, "math_hi": args.math_hi,
                      "old_ckpt": args.old_ckpt, "full_ckpt": args.full_ckpt,
                      "lora_ckpt": args.lora_ckpt}}
    print("\n" + "=" * 92)
    print(f"三模型 × 三段对比 (seed={args.seed}, 每段 {args.limit} 题)")
    print("=" * 92)
    for label, seg_name in segs:
        baseline = None
        cols = {}
        for name, _ in jobs:
            r = read_summary(args.out_dir, seg_name, name)
            cols[name] = r["model_acc"] if r else None
            if r and r["baseline_acc"] is not None and baseline is None:
                baseline = r["baseline_acc"]
        table[seg_name] = {"baseline_acc": baseline,
                           **{f"{n}_acc": v for n, v in cols.items()}}
        parts = [f"[{label}] baseline {baseline:.2%}" if baseline is not None
                 else f"[{label}] baseline N/A"]
        for name, _ in jobs:
            v = cols[name]
            if v is None:
                parts.append(f"{name} N/A")
            else:
                d = "" if baseline is None else f" ({v - baseline:+.2%})"
                parts.append(f"{name} {v:.2%}{d}")
        print(" | ".join(parts))

    out_json = os.path.join(args.out_dir,
                            f"compare_segments{('_' + args.tag) if args.tag else ''}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 92)
    print(f"汇总已保存: {out_json}")


if __name__ == "__main__":
    main()
