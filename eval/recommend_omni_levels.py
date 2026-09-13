# -*- coding: utf-8 -*-
"""根据 Omni-MATH pilot 的逐难度正确率推荐正式评测等级。"""

import argparse
import json

from omni_math_utils import recommend_level_range, summarize_by_difficulty


def main():
    parser = argparse.ArgumentParser(
        description="读取 eval_math.py 的 Omni-MATH JSON 并推荐难度区间")
    parser.add_argument("result", help="pilot 结果 JSON")
    parser.add_argument("--metric", default="auto",
                        help="auto/fusion/lora/baseline 或完整的 *_acc")
    parser.add_argument("--target_min", type=float, default=0.10)
    parser.add_argument("--target_max", type=float, default=0.40)
    parser.add_argument("--min_samples", type=int, default=10)
    args = parser.parse_args()

    with open(args.result, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    by_difficulty = summary.get("by_difficulty")
    if not by_difficulty:
        by_difficulty = summarize_by_difficulty(payload.get("results", []))
    recommendation = recommend_level_range(
        by_difficulty, metric=args.metric,
        target_min=args.target_min, target_max=args.target_max,
        min_samples=args.min_samples)

    metric = recommendation["metric"]
    print(f"指标: {metric}")
    print("等级  样本  准确率")
    for row in recommendation["observations"]:
        print(f"L{row['level']:<2}  {row['n']:>4}  {row['accuracy']:.2%}")
    print(f"\n推荐正式评测等级: {recommendation['level_spec']}")
    print(f"原因: {recommendation['reason']}")
    print("正式参数:")
    print(f"  --bench omni_math --omni_levels "
          f"{recommendation['level_spec']} --limit 400")


if __name__ == "__main__":
    main()
