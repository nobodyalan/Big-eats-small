# -*- coding: utf-8 -*-
"""Omni-MATH 数据加载、分层抽样与难度区间推荐（纯标准库）。"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict


def parse_level_set(spec):
    """把 ``1-3,5`` 解析成难度整数集合；空、0、all 表示全部。"""
    if spec is None or str(spec).strip().lower() in ("", "0", "all"):
        return None
    levels = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x.strip()) for x in part.split("-", 1))
            if lo > hi:
                lo, hi = hi, lo
            levels.update(range(lo, hi + 1))
        else:
            levels.add(int(part))
    bad = sorted(x for x in levels if x < 1 or x > 10)
    if bad:
        raise ValueError(f"Omni-MATH 难度必须在 1..10，收到: {bad}")
    return levels or None


def difficulty_value(value):
    """读取官方连续难度值（数据中包含 7.5、4.375 等）。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无效 Omni-MATH difficulty: {value!r}") from exc
    if not math.isfinite(number) or number < 1 or number > 10:
        raise ValueError(f"Omni-MATH difficulty 必须在 1..10，收到: {value!r}")
    return number


def difficulty_level(value):
    """把连续难度映射到整数带；L7 表示原始难度 ``[7, 8)``。"""
    number = difficulty_value(value)
    return min(10, int(math.floor(number)))


def load_omni_math_items(path, seed=42, limit=0, levels=None, limit_per_level=0):
    """读取官方/规范化 JSONL，并以难度分层方式确定固定评测题。

    ``limit_per_level`` 用于小规模 pilot；否则 ``limit`` 会尽量平均分配到
    选中的难度，避免简单等级数量多而支配总分。
    """
    wanted = parse_level_set(levels) if isinstance(levels, str) else levels
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = str(row.get("problem", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not problem or not answer:
                raise ValueError(f"{path}:{line_no} 缺 problem/answer")
            raw_difficulty = difficulty_value(row.get("difficulty"))
            level = difficulty_level(raw_difficulty)
            if wanted is not None and level not in wanted:
                continue
            item = dict(row)
            item["problem"] = problem
            item["answer"] = answer
            item["difficulty"] = raw_difficulty
            item["difficulty_band"] = level
            item.setdefault("benchmark_id", f"omni_math_line_{line_no}")
            groups[level].append(item)

    if not groups:
        raise ValueError(f"{path} 中没有符合难度条件的数据")

    # 每个等级使用独立 RNG；增加/移除其他等级不会改变本等级的题目顺序。
    for level, rows in groups.items():
        random.Random(f"omni-math:{seed}:level:{level}").shuffle(rows)

    selected = []
    ordered_levels = sorted(groups)
    if limit_per_level > 0:
        for level in ordered_levels:
            selected.extend(groups[level][:limit_per_level])
    elif limit > 0:
        # round-robin 形成尽量均衡的难度分布。
        offsets = {level: 0 for level in ordered_levels}
        while len(selected) < limit:
            added = False
            for level in ordered_levels:
                idx = offsets[level]
                if idx < len(groups[level]) and len(selected) < limit:
                    selected.append(groups[level][idx])
                    offsets[level] += 1
                    added = True
            if not added:
                break
    else:
        for level in ordered_levels:
            selected.extend(groups[level])

    random.Random(f"omni-math:{seed}:final").shuffle(selected)
    return selected


def summarize_by_difficulty(results):
    """从逐题结果产生每个难度下 baseline/fusion/lora 的正确率。"""
    groups = defaultdict(list)
    for row in results:
        if "difficulty" in row:
            groups[int(row.get("difficulty_band",
                               difficulty_level(row["difficulty"])))].append(row)

    summary = {}
    for level in sorted(groups):
        rows = groups[level]
        part = {"n": len(rows)}
        for label in ("baseline", "fusion", "lora"):
            key = f"{label}_correct"
            present = [row[key] for row in rows if key in row]
            if present:
                correct = sum(bool(x) for x in present)
                part[f"{label}_correct"] = correct
                part[f"{label}_acc"] = round(correct / len(present), 4)
        summary[str(level)] = part
    return summary


def recommend_level_range(by_difficulty, metric="auto", target_min=0.10,
                          target_max=0.40, min_samples=10):
    """推荐准确率落在目标区间内的最高、最长连续难度段。"""
    if target_min < 0 or target_max > 1 or target_min > target_max:
        raise ValueError("target_min/target_max 必须满足 0 <= min <= max <= 1")

    if metric == "auto":
        for candidate in ("fusion_acc", "lora_acc", "baseline_acc"):
            if any(candidate in part for part in by_difficulty.values()):
                metric = candidate
                break
        else:
            raise ValueError("结果中没有 fusion/lora/baseline 准确率")
    elif not metric.endswith("_acc"):
        metric += "_acc"

    points = []
    for raw_level, part in by_difficulty.items():
        level = int(raw_level)
        if int(part.get("n", 0)) >= min_samples and metric in part:
            points.append((level, float(part[metric]), int(part["n"])))
    points.sort()
    if not points:
        raise ValueError(f"没有每档至少 {min_samples} 题的 {metric} 结果")

    qualified = {level for level, acc, _ in points if target_min <= acc <= target_max}
    runs = []
    current = []
    for level, _, _ in points:
        if level in qualified and (not current or level == current[-1] + 1):
            current.append(level)
        else:
            if current:
                runs.append(current)
            current = [level] if level in qualified else []
    if current:
        runs.append(current)

    if runs:
        chosen = max(runs, key=lambda run: (len(run), max(run)))
        reason = (f"{metric} 在目标区间 "
                  f"[{target_min:.0%}, {target_max:.0%}] 内")
    else:
        accuracies = [acc for _, acc, _ in points]
        if min(accuracies) > target_max:
            chosen = [points[-1][0]]
            reason = "所有已测等级都偏简单；暂取最高等级"
        elif max(accuracies) < target_min:
            chosen = [points[0][0]]
            reason = "所有已测等级都偏难；暂取最低等级"
        else:
            midpoint = (target_min + target_max) / 2
            level, _, _ = min(points, key=lambda x: (abs(x[1] - midpoint), -x[0]))
            chosen = [level]
            reason = "准确率不单调；取最接近目标中点的等级"

    return {
        "metric": metric,
        "levels": chosen,
        "level_spec": (str(chosen[0]) if len(chosen) == 1
                       else f"{chosen[0]}-{chosen[-1]}"),
        "target_min": target_min,
        "target_max": target_max,
        "min_samples": min_samples,
        "reason": reason,
        "observations": [
            {"level": level, "accuracy": acc, "n": n}
            for level, acc, n in points
        ],
    }
