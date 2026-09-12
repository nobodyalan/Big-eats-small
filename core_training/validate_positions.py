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


def inclusive_small_end(b_exclusive):
    """把筛选器的右开边界转换为 train_fusion 的 inclusive 参数。"""
    b_exclusive = int(b_exclusive)
    if b_exclusive <= 0:
        raise ValueError("exclusive small end 必须大于 0")
    return b_exclusive - 1


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


def load_candidates(json_path, topn, fallback_span=12):
    """读取任务感知候选，返回 (L,l2,a,b_exclusive)。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    src = data.get("task_aware") or data.get("exit") or data.get("top_full") or []
    seen, cands = set(), []
    for x in src:
        L, a, b = int(x["L"]), int(x["a"]), int(x["b"])
        l2 = int(x.get("l2", min(L + fallback_span, 35)))
        # select_positions 的 b 一直表示 [a,b) 的右边界；旧 validator 曾把它
        # 错传成 inclusive small_end。这里统一保留 exclusive 语义。
        key = (L, l2, a, b)
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
    """读取未舍入的正确数/总数，避免 round 后的伪排序。"""
    jf = latest(os.path.join(out_dir, f"eval_{seg_name.lower()}_{tag}_*.json"))
    if not jf:
        return None
    with open(jf, encoding="utf-8") as f:
        s = json.load(f).get("summary", {})
    correct = s.get("fusion_correct", s.get("lora_correct"))
    n = s.get("n")
    if correct is None or not n:
        return None
    return {"correct": int(correct), "n": int(n),
            "accuracy": int(correct) / int(n)}


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
                  "--contrast_weight", str(args.contrast_weight), "--seed", str(args.seed),
                  "--out", out_path, "--plot", ""]
    if pos is not None:
        L, l2, a, b = pos
        train_cmd += ["--large_start", str(L), "--large_end", str(l2),
                      "--small_start", str(a), "--small_end",
                      str(inclusive_small_end(b))]
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
                        "--small_start", str(a), "--small_end",
                        str(inclusive_small_end(b))]
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
    ap.add_argument("--contrast_weight", type=float, default=0.0,
                    help="位置比较默认关闭对比项，避免与任务 CE 混杂")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_samples", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--out_dir", default="cache/validate_positions")
    # 短训后的准确率评测(三难度: GSM8K + MATH低级 + MATH高级, 各 N 题)
    ap.add_argument("--acc_limit", type=int, default=50, help="短训后准确率评测: 每段题数")
    ap.add_argument("--acc_seed", type=int, default=42)
    ap.add_argument("--math_lo", default="1-3", help="MATH 低级区间")
    ap.add_argument("--math_hi", default="4-5", help="MATH 高级区间")
    ap.add_argument("--skip_accuracy", action="store_true", help="跳过准确率评测, 只比 CE 增益")
    ap.add_argument("--primary", choices=["gain", "ce", "micro_accuracy"], default="gain",
                    help="候选主排序；短训默认最大化独立验证集 CE 增益")
    ap.add_argument("--gpus", default="0", help="逗号分隔的 GPU 列表(候选轮转分配, 如 0,1)")
    ap.add_argument("--parallel", type=int, default=1, help="同时跑几个候选(≤ GPU 数×每卡可容纳数)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 配置列表: (文件tag, 显示描述, 位置元组或 None=默认)
    configs = [("baseline_default", "默认 1/3~2/3", None)]
    if os.path.exists(args.candidates_json):
        for i, (L, l2, a, b) in enumerate(
                load_candidates(args.candidates_json, args.topn, args.inject_span)):
            tag = f"cand{i}_L{L}_{l2}_s{a}_{b}x"
            configs.append((tag, f"L={L}→{l2}, [{a},{b})", (L, l2, a, b)))
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
                f"{k}={v['accuracy']:.2%} ({v['correct']}/{v['n']})"
                if v is not None else f"{k}=N/A"
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
                line += f"{v['accuracy']:.2%} " if v is not None else "N/A      "
            counts = [(v["correct"], v["n"]) for v in acc.values() if v is not None]
            line += (f"{sum(c for c, _ in counts) / sum(n for _, n in counts):.2%}"
                     if counts else "N/A")
        else:
            line += "N/A      N/A       N/A       N/A"
        print(line)

    def sort_key(r):
        if args.primary == "gain" and r["ce"]:
            return -r["ce"][2]
        if args.primary == "ce" and r["ce"]:
            return r["ce"][0]
        if r["acc"]:
            counts = [(v["correct"], v["n"]) for v in r["acc"].values()
                      if v is not None]
            if counts:
                return -sum(c for c, _ in counts) / sum(n for _, n in counts)
        if r["ce"]:
            return r["ce"][0]
        return float("inf")

    best = min(results, key=sort_key)
    if best["ce"] is not None or best["acc"] is not None:
        print(f"\n最优: {best['desc']}")
        if best["pos"] is not None:
            L, l2, a, b = best["pos"]
            print(f"完整训练命令: python3 core_training/train_fusion.py "
                  f"--large_start {L} --large_end {l2} --small_start {a} "
                  f"--small_end {inclusive_small_end(b)} --contrast_weight 0 ...")

    summary = {"results": [
        {"name": r["desc"], "pos": r["pos"],
         "best_fusion": (r["ce"][0] if r["ce"] else None),
         "gain": (r["ce"][2] if r["ce"] else None),
         "accuracy": r["acc"]} for r in results],
         "primary_metric": args.primary, "seed": args.seed,
         "small_segment_semantics": "[a,b) in this summary; train_fusion receives b-1"}
    out_json = os.path.join(args.out_dir, "validation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {out_json}")


if __name__ == "__main__":
    main()
