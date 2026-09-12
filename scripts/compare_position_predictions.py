# -*- coding: utf-8 -*-
"""对两个候选的逐题结果做配对 bootstrap 与精确 McNemar 检验。"""

import argparse
import json
import math
import random


def load_rows(path, label):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f).get("results", [])
    out = {}
    for row in rows:
        key = (row.get("problem"), str(row.get("gold")))
        correct = row.get(f"{label}_correct")
        if correct is None:
            raise ValueError(f"{path} 缺少 {label}_correct")
        out[key] = bool(correct)
    return out


def exact_mcnemar_p(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main():
    ap = argparse.ArgumentParser(description="逐题配对比较两个位置候选")
    ap.add_argument("--a", required=True, help="候选 A 的 eval JSON")
    ap.add_argument("--b", required=True, help="候选 B 的 eval JSON")
    ap.add_argument("--label", default="fusion", choices=["fusion", "lora", "baseline"],
                    help="两边使用同一结果字段（兼容旧用法）")
    ap.add_argument("--label_a", choices=["fusion", "lora", "baseline"], default=None)
    ap.add_argument("--label_b", choices=["fusion", "lora", "baseline"], default=None,
                    help="允许跨文件比较 fusion/LoRA 与独立 baseline")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    label_a = args.label_a or args.label
    label_b = args.label_b or args.label
    a, b = load_rows(args.a, label_a), load_rows(args.b, label_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("两个文件没有可配对的共同题目")
    if len(keys) != len(a) or len(keys) != len(b):
        print(f"[警告] 只比较共同题目 {len(keys)}；A={len(a)} B={len(b)}")

    pairs = [(a[k], b[k]) for k in keys]
    acc_a = sum(x for x, _ in pairs) / len(pairs)
    acc_b = sum(y for _, y in pairs) / len(pairs)
    n01 = sum((not x) and y for x, y in pairs)  # A错、B对
    n10 = sum(x and (not y) for x, y in pairs)  # A对、B错

    rng = random.Random(args.seed)
    diffs = []
    for _ in range(args.bootstrap):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(sum(y - x for x, y in sample) / len(sample))
    lo, hi = percentile(diffs, 0.025), percentile(diffs, 0.975)

    print(f"共同题目: {len(pairs)}")
    print(f"A({label_a})={acc_a:.2%} | B({label_b})={acc_b:.2%} | "
          f"Δ(B-A)={acc_b - acc_a:+.2%}")
    print(f"配对翻转: A错B对={n01} | A对B错={n10}")
    print(f"paired bootstrap 95% CI: [{lo:+.2%}, {hi:+.2%}]")
    print(f"exact McNemar p={exact_mcnemar_p(n01, n10):.6f}")


if __name__ == "__main__":
    main()
