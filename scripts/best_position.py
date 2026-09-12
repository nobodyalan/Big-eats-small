# -*- coding: utf-8 -*-
"""Print the best screened position as shell-friendly integer fields."""

import json
import math
import sys


def _incremental_samples(row):
    ratio = f"{row.get('rank_ratio', 0.03):g}"
    candidate = row.get("delta_nll_samples", {}).get(ratio)
    control = row.get("control_delta_nll_samples", {}).get(ratio)
    if not candidate or not control or len(candidate) != len(control):
        return None
    return [x - y for x, y in zip(candidate, control)]


def _ci95(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, mean
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(var / len(values))
    return mean - half, mean + half


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: best_position.py POSITION_SELECTION.json")
    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)
    candidates = payload.get("task_aware") or payload.get("exit") or []
    if not candidates:
        raise SystemExit("position selection contains no task-aware candidates")
    best = candidates[0]
    best_values = _incremental_samples(best)
    tied = []
    if best_values:
        for row in candidates:
            values = _incremental_samples(row)
            if not values or len(values) != len(best_values):
                continue
            lo, hi = _ci95([x - y for x, y in zip(best_values, values)])
            if lo <= 0 <= hi:
                tied.append(row)
    if len(tied) > 1:
        print(f"[警告] 筛选没有唯一最优位置；{len(tied)} 个候选与第一名统计并列。"
              "后续实验暂取排序第一作为代表，不能把它解释为已证明的最优解。",
              file=sys.stderr)
    large_start = int(best["L"])
    large_end = int(best["l2"])
    small_start = int(best["a"])
    small_end_inclusive = int(best["b"]) - 1
    print(large_start, large_end, small_start, small_end_inclusive)


if __name__ == "__main__":
    main()
