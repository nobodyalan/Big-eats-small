#!/usr/bin/env python3
"""统计评测 JSON 中未能提取答案（``*_pred: null``）的样本。

``*_pred`` 为 null 只表示对应答案提取器失败，不能单独证明模型输出被
``max_new_tokens`` 截断。本脚本会同时汇总原始输出、宽松提取、Math-Verify
和生成停止元数据，帮助区分格式失败与已确认的长度截断。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PRIMARY_PRED_RE = re.compile(r"^(.+)_pred$")
EXCLUDED_PREFIX_SUFFIXES = ("_relaxed", "_math_verify")


def discover_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*.json")
                      if candidate.is_file())
    raise FileNotFoundError(f"路径不存在: {path}")


def prediction_labels(results: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for row in results:
        for key in row:
            matched = PRIMARY_PRED_RE.match(key)
            if not matched:
                continue
            label = matched.group(1)
            if label.endswith(EXCLUDED_PREFIX_SUFFIXES):
                continue
            labels.add(label)
    return sorted(labels)


def _is_length_stop(row: dict[str, Any], label: str) -> bool | None:
    hit_key = f"{label}_hit_max_new_tokens"
    reason_key = f"{label}_finish_reason"
    if hit_key in row:
        return bool(row[hit_key])
    if reason_key in row:
        return row[reason_key] == "length"
    return None


def audit_label(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    pred_key = f"{label}_pred"
    relevant = [row for row in results if pred_key in row]
    null_rows = [row for row in relevant if row[pred_key] is None]
    non_null_rows = [row for row in relevant if row[pred_key] is not None]

    text_key = f"{label}_text"
    relaxed_key = f"{label}_relaxed_pred"
    mv_extract_key = f"{label}_math_verify_extracted"
    mv_correct_key = f"{label}_math_verify_correct"

    stop_values = [_is_length_stop(row, label) for row in relevant]
    null_stop_values = [_is_length_stop(row, label) for row in null_rows]
    stop_metadata_n = sum(value is not None for value in stop_values)
    null_stop_metadata_n = sum(value is not None for value in null_stop_values)

    return {
        "label": label,
        "n": len(relevant),
        "null": len(null_rows),
        "null_rate": len(null_rows) / len(relevant) if relevant else 0.0,
        "non_null": len(non_null_rows),
        "null_with_raw_text": sum(
            isinstance(row.get(text_key), str) for row in null_rows),
        "null_with_nonempty_raw_text": sum(
            isinstance(row.get(text_key), str) and bool(row[text_key].strip())
            for row in null_rows),
        "null_recovered_by_relaxed_parser": sum(
            row.get(relaxed_key) is not None for row in null_rows),
        "null_extracted_by_math_verify": sum(
            bool(row.get(mv_extract_key)) for row in null_rows),
        "null_correct_by_math_verify": sum(
            bool(row.get(mv_correct_key)) for row in null_rows),
        "stop_metadata_n": stop_metadata_n,
        "confirmed_length_stops": sum(value is True for value in stop_values),
        "null_stop_metadata_n": null_stop_metadata_n,
        "null_confirmed_length_stops": sum(
            value is True for value in null_stop_values),
        "null_ids": [row.get("id") for row in null_rows],
    }


def audit_file(path: Path, requested_labels: list[str] | None) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not all(isinstance(row, dict) for row in results):
        return None

    available = prediction_labels(results)
    labels = requested_labels or available
    missing = sorted(set(labels) - set(available))
    if missing:
        raise ValueError(
            f"{path}: 找不到预测字段: "
            + ", ".join(f"{label}_pred" for label in missing))
    if not labels:
        return None
    return {
        "file": str(path),
        "bench": payload.get("summary", {}).get("bench"),
        "rows": len(results),
        "labels": [audit_label(results, label) for label in labels],
    }


def print_report(report: dict[str, Any], show_examples: int) -> None:
    bench = f" | bench={report['bench']}" if report.get("bench") else ""
    print(f"\n{report['file']} | rows={report['rows']}{bench}")
    for item in report["labels"]:
        print(
            f"  {item['label']}: null={item['null']}/{item['n']} "
            f"({item['null_rate']:.2%}) | null中非空原文="
            f"{item['null_with_nonempty_raw_text']} | 宽松提取恢复="
            f"{item['null_recovered_by_relaxed_parser']} | Math-Verify提取="
            f"{item['null_extracted_by_math_verify']} | Math-Verify判对="
            f"{item['null_correct_by_math_verify']}"
        )
        if item["null_stop_metadata_n"]:
            print(
                f"    null中确认命中生成上限="
                f"{item['null_confirmed_length_stops']}/"
                f"{item['null_stop_metadata_n']}（有停止元数据的样本）")
        else:
            print("    null中确认命中生成上限=无法判断（结果未保存停止元数据）")
        if show_examples and item["null_ids"]:
            ids = item["null_ids"][:show_examples]
            suffix = " ..." if len(item["null_ids"]) > show_examples else ""
            print(f"    null样本 id: {ids}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计 eval JSON 的 *_pred=null，并区分格式失败与确认截断。")
    parser.add_argument("path", type=Path, help="单个评测 JSON 或包含它们的目录")
    parser.add_argument(
        "--label", action="append", dest="labels",
        help="只统计指定前缀，例如 fusion/lora/baseline；可重复。默认自动发现")
    parser.add_argument(
        "--show_examples", type=int, default=10,
        help="每个模型最多展示多少个 null 样本 id；0 表示不展示")
    parser.add_argument(
        "--json_out", type=Path,
        help="可选：把机器可读汇总另存为 JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = []
    skipped = 0
    for path in discover_json_files(args.path):
        report = audit_file(path, args.labels)
        if report is None:
            skipped += 1
            continue
        reports.append(report)
        print_report(report, max(0, args.show_examples))

    if not reports:
        raise SystemExit("没有找到包含 results 和 *_pred 字段的评测 JSON")

    print(f"\n完成: 统计 {len(reports)} 个评测文件，跳过 {skipped} 个非逐题 JSON。")
    print("注意: *_pred=null 仅表示答案提取失败；只有保存了 "
          "*_hit_max_new_tokens 或 *_finish_reason=length 才能确认长度截断。")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump({"reports": reports, "skipped": skipped}, handle,
                      ensure_ascii=False, indent=2)
        print(f"JSON 汇总已保存: {args.json_out}")


if __name__ == "__main__":
    main()
