# -*- coding: utf-8 -*-
"""打印任务感知筛选结果，并判断筛选信号是否足够强。"""

import argparse
import json
import math


def mean_ci95(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, (mean, mean)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(var / len(values))
    return mean, (mean - half, mean + half)


def incremental_samples(row):
    """返回 candidate ΔNLL - identity-control ΔNLL 的逐题配对值。"""
    ratio = f"{row.get('rank_ratio', 0.03):g}"
    candidate = row.get("delta_nll_samples", {}).get(ratio)
    control = row.get("control_delta_nll_samples", {}).get(ratio)
    if not candidate or not control or len(candidate) != len(control):
        return None
    return [x - y for x, y in zip(candidate, control)]


def main():
    ap = argparse.ArgumentParser(description="汇总 position_selection.json")
    ap.add_argument("json_path")
    ap.add_argument("--topn", type=int, default=12)
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("task_aware") or payload.get("exit") or []
    if not rows:
        raise SystemExit("结果中没有 task_aware 候选；无法判断任务价值")

    print(f"{'#':>2} {'L→L2':>8} {'[a,b)':>9} {'len':>4} {'Q':>8} "
          f"{'ΔNLL':>11} {'raw 95% CI':>23} {'vs control':>12} "
          f"{'incremental 95% CI':>23} {'判定':>12}")
    for i, row in enumerate(rows[:args.topn], 1):
        ci = row.get("rank_delta_nll_ci95", [float("nan"), float("nan")])
        ici = row.get("incremental_delta_nll_ci95", [float("nan"), float("nan")])
        useful = ci[1] < 0 and ici[1] < 0
        if useful:
            verdict = "明确增量价值"
        elif row["rank_delta_nll"] >= 0:
            verdict = "无收益"
        elif row.get("incremental_delta_nll", 0) >= 0:
            verdict = "不如control"
        else:
            verdict = "增量不确定"
        print(f"{i:>2} {row['L']:>2}→{row.get('l2', '?')!s:<3} "
              f"[{row['a']},{row['b']})".rjust(9) +
              f" {row.get('length', row['b'] - row['a']):>4} {row.get('Q', 0):>8.4f} "
              f"{row['rank_delta_nll']:>+11.5f} "
              f"[{ci[0]:+.5f},{ci[1]:+.5f}] "
              f"{row.get('incremental_delta_nll', float('nan')):>+12.5f} "
              f"[{ici[0]:+.5f},{ici[1]:+.5f}] {verdict:>12}")

    best = rows[0]
    best_ci = best.get("rank_delta_nll_ci95", [best["rank_delta_nll"]] * 2)
    inc_ci = best.get("incremental_delta_nll_ci95",
                      [best.get("incremental_delta_nll", 0)] * 2)
    print("\n筛选价值判断:")
    if best_ci[1] >= 0:
        print("- 最优候选的 ΔNLL 置信区间仍包含 0：当前筛选没有证明任何位置能改善任务。")
    elif inc_ci[1] >= 0:
        print("- 候选能降低 NLL，但没有显著超过 identity/bridge-only control；"
              "小模型层位置本身的筛选价值不足。")
    else:
        print("- 最优候选同时稳定降低 NLL，并超过 bridge-only control；"
              "说明小模型片段位置包含可测的增量任务价值。")

    if len(rows) >= 2:
        a = incremental_samples(best)
        b = incremental_samples(rows[1])
        if a and b and len(a) == len(b):
            # 排序依据是相对 control 的增量，因此候选间也必须比较增量，
            # 不能比较旧版使用的 raw ΔNLL。
            diff, ci = mean_ci95([x - y for x, y in zip(a, b)])
            if ci[1] < 0:
                print(f"- 第一名相对第二名的增量价值有清晰分离：paired Δ={diff:+.5f}, "
                      f"95% CI=[{ci[0]:+.5f},{ci[1]:+.5f}]。")
            else:
                print(f"- 第一名与第二名的增量价值不可区分：paired Δ={diff:+.5f}, "
                      f"95% CI=[{ci[0]:+.5f},{ci[1]:+.5f}]；应视为并列候选。")

        ties = []
        if a:
            for row in rows:
                values = incremental_samples(row)
                if not values or len(values) != len(a):
                    continue
                _, pair_ci = mean_ci95([x - y for x, y in zip(a, values)])
                if pair_ci[0] <= 0 <= pair_ci[1]:
                    ties.append(f"{row['L']}→{row.get('l2', '?')} [{row['a']},{row['b']})")
        if len(ties) > 1:
            print(f"- 与第一名统计并列的候选共 {len(ties)} 个：" + "；".join(ties))
            print("- 因此本轮只能给出候选集合，不能声称找到了唯一最优接入位置。")


if __name__ == "__main__":
    main()
