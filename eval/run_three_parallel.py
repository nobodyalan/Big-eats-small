# -*- coding: utf-8 -*-
"""
并行评测三模型(1/3 旁路 / 整段旁路 / LoRA)在完全相同的一批题目上, 最后汇总成对比表。

用法(服务器, 在 BES 目录下执行):
  CUDA_VISIBLE_DEVICES=1 python3 eval/run_three_parallel.py \
      --old_ckpt <1/3旁路权重路径> \
      --full_ckpt cache/fusion_full_sft.pt.best \
      --lora_ckpt cache/lora_r21_sft.best \
      --limit 400

原理: 三个 eval_math.py 子进程用同一个 --seed(默认 42)并行启动, 因此
MATH / GSM8K 会 shuffle 成完全相同的题目顺序 → 保证"同样题目"。
每个子进程各写一个带 --tag 的 JSON, 最后本脚本汇总成一张对比表。
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


def load_summary(out_dir: str, bench: str, tag: str):
    """读取某模型的 summary: 返回 {model_acc, baseline_acc} 或 None"""
    jf = latest(os.path.join(out_dir, f"eval_{bench.lower()}_{tag}_*.json"))
    if not jf:
        return None
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    s = d.get("summary", {})
    model_acc = s.get("fusion_acc", s.get("lora_acc"))
    return {"model_acc": model_acc, "baseline_acc": s.get("baseline_acc"), "file": jf}


def main():
    parser = argparse.ArgumentParser(description="并行三模型评测 + 汇总")
    parser.add_argument("--old_ckpt", default="", help="1/3~2/3 旧旁路权重路径(空=跳过)")
    parser.add_argument("--full_ckpt", default="cache/fusion_full_sft.pt.best",
                        help="整段旁路权重")
    parser.add_argument("--lora_ckpt", default="cache/lora_r21_sft.best", help="LoRA 权重目录")
    parser.add_argument("--bench", default="both")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--math_level", type=int, default=0, help="只测 MATH 指定难度(0=全部)")
    parser.add_argument("--out_dir", default="eval_results")
    parser.add_argument("--tag", default="", help="汇总表文件名标签")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = [sys.executable, EVAL_PY,
            "--bench", args.bench, "--limit", str(args.limit),
            "--seed", str(args.seed), "--max_new", str(args.max_new),
            "--fusion_only",          # 不再跑裸 4B baseline, 只输出各模型正确率
            "--out_dir", args.out_dir]
    if args.math_level:
        base += ["--math_level", str(args.math_level)]

    jobs = []
    if args.old_ckpt:
        jobs.append(("onethird", ["--ckpt", args.old_ckpt]))
    else:
        print("[提示] 未提供 --old_ckpt, 跳过 1/3 旁路")
    jobs.append(("full", ["--ckpt", args.full_ckpt,
                          "--small_start", "0", "--small_end", "-1"]))
    jobs.append(("lora", ["--lora_ckpt", args.lora_ckpt]))

    # 并行启动所有子进程(cwd=BES, 保证 data/.cache 相对路径正确)
    procs = {}
    for name, extra in jobs:
        cmd = base + extra + ["--tag", name]
        suffix = f"_{args.tag}" if args.tag else ""
        log_path = os.path.join(args.out_dir, f"eval_{name}{suffix}.log")
        logf = open(log_path, "w", encoding="utf-8")
        print(f"启动 [{name}]: {' '.join(cmd)}")
        procs[name] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                        cwd=_ROOT), logf, log_path)

    # 等所有完成
    for name, (p, logf, log_path) in procs.items():
        rc = p.wait()
        logf.close()
        print(f"[{name}] 退出码 {rc} (日志: {log_path})")

    # ── 汇总 ──
    if args.bench == "aime":
        benches = ["AIME"]
    else:
        benches = []
        if args.bench in ("math", "both"):
            benches.append(f"MATH_L{args.math_level}" if args.math_level else "MATH")
        if args.bench in ("gsm8k", "both"):
            benches.append("GSM8K")
    table = {"meta": {"seed": args.seed, "limit": args.limit, "bench": args.bench,
                      "old_ckpt": args.old_ckpt, "full_ckpt": args.full_ckpt,
                      "lora_ckpt": args.lora_ckpt}}
    print("\n" + "=" * 78)
    print(f"三模型正确率对比 (seed={args.seed}, 每数据集 {args.limit} 题, 不含 baseline)")
    print("=" * 78)
    for bench in benches:
        cols = {}
        for name, _ in jobs:
            r = load_summary(args.out_dir, bench, name)
            cols[name] = r["model_acc"] if r else None
        table[bench] = {f"{name}_acc": v for name, v in cols.items()}
        parts = [f"[{bench}]"]
        for name, _ in jobs:
            v = cols[name]
            parts.append(f"{name} {'N/A' if v is None else f'{v:.2%}'}")
        print(" | ".join(parts))

    out_json = os.path.join(args.out_dir,
                            f"compare_three{('_' + args.tag) if args.tag else ''}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    print(f"汇总已保存: {out_json}")
    print("各模型明细 JSON: eval_math_<tag>_*.json / eval_gsm8k_<tag>_*.json")


if __name__ == "__main__":
    main()
