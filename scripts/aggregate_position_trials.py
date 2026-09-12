# -*- coding: utf-8 -*-
"""聚合多随机种子的位置短训结果，并生成下一阶段兼容的候选 JSON。"""

import argparse
import glob
import json
import math
import os
import statistics


def mean_se(values):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def micro_accuracy(acc):
    if not acc:
        return None
    rows = [v for v in acc.values() if v and v.get("n")]
    if not rows:
        return None
    return sum(v["correct"] for v in rows) / sum(v["n"] for v in rows)


def main():
    ap = argparse.ArgumentParser(description="聚合多 seed 的位置验证结果")
    ap.add_argument("--inputs", required=True,
                    help="validation_summary.json 的 glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"没有匹配结果: {args.inputs}")

    groups = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        seed = payload.get("seed")
        for row in payload.get("results", []):
            pos = row.get("pos")
            key = "default" if pos is None else ",".join(str(int(x)) for x in pos)
            g = groups.setdefault(key, {"name": row.get("name", key), "pos": pos,
                                        "seeds": [], "gain": [], "fusion": [],
                                        "micro_accuracy": []})
            g["seeds"].append(seed)
            if row.get("gain") is not None:
                g["gain"].append(float(row["gain"]))
            if row.get("best_fusion") is not None:
                g["fusion"].append(float(row["best_fusion"]))
            acc = micro_accuracy(row.get("accuracy"))
            if acc is not None:
                g["micro_accuracy"].append(acc)

    rows = []
    for g in groups.values():
        gain, gain_se = mean_se(g.pop("gain"))
        fusion, fusion_se = mean_se(g.pop("fusion"))
        acc, acc_se = mean_se(g.pop("micro_accuracy"))
        g.update({"mean_gain": gain, "gain_se": gain_se,
                  "mean_fusion": fusion, "fusion_se": fusion_se,
                  "mean_micro_accuracy": acc, "micro_accuracy_se": acc_se,
                  "runs": len(g["seeds"])})
        rows.append(g)

    # 有准确率时以准确率为主，否则以验证 CE gain 为主。SE 只报告，不伪装成精确显著性。
    have_acc = any(x["mean_micro_accuracy"] is not None for x in rows)
    if have_acc:
        rows.sort(key=lambda x: (-(x["mean_micro_accuracy"] or -1),
                                 -(x["mean_gain"] or -1e9)))
    else:
        rows.sort(key=lambda x: -(x["mean_gain"] or -1e9))

    print(f"聚合 {len(paths)} 个结果文件 | 配置 {len(rows)}")
    print(f"{'配置':<28} {'runs':>4} {'gain(mean±SE)':>20} {'acc(mean±SE)':>20}")
    for x in rows:
        gain = "N/A" if x["mean_gain"] is None else (
            f"{x['mean_gain']:+.5f}±{(x['gain_se'] or 0):.5f}")
        acc = "N/A" if x["mean_micro_accuracy"] is None else (
            f"{x['mean_micro_accuracy']:.2%}±{(x['micro_accuracy_se'] or 0):.2%}")
        print(f"{x['name']:<28} {x['runs']:>4} {gain:>20} {acc:>20}")

    task_aware = []
    for x in rows:
        if x["pos"] is None:
            continue
        L, l2, a, b = [int(v) for v in x["pos"]]
        task_aware.append({"L": L, "l2": l2, "a": a, "b": b,
                           "b_semantics": "exclusive",
                           "mean_gain": x["mean_gain"],
                           "mean_micro_accuracy": x["mean_micro_accuracy"]})
        if len(task_aware) >= args.topn:
            break

    payload = {"meta": {"source_files": paths, "ranking":
                         "micro_accuracy_then_gain" if have_acc else "gain"},
               "aggregate": rows, "task_aware": task_aware}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"下一阶段候选已保存: {args.out}")


if __name__ == "__main__":
    main()
