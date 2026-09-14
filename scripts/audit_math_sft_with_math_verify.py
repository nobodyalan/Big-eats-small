# -*- coding: utf-8 -*-
"""用正式评测相同的 Math-Verify 配置审计数学 SFT response。

逐条检查：
1. response 是否以规范的 ``Final answer: \\boxed{...}`` 结束；
2. 规范最终答案能否被 Math-Verify 解析；
3. Math-Verify 直接读取完整 response 时能否提取答案；
4. 完整 response 的解析结果是否与规范最终答案等价。

脚本只读训练数据；失败详情写 JSONL，汇总写 JSON。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from format_math_sft_answers import canonicalize_response


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_math_verify_runtime() -> dict:
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        from importlib.metadata import version
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 缺少 math-verify；请使用 bes-math-eval-v2 venv") from exc
    try:
        installed_version = version("math-verify")
    except Exception:
        installed_version = "unknown"
    return {
        "parse": parse,
        "verify": verify,
        "expr_config": ExprExtractionConfig,
        "latex_config": LatexExtractionConfig,
        "version": installed_version,
    }


def inspect_response(response: str, runtime: dict) -> dict:
    result = {
        "canonical_format": False,
        "answer_extracted": False,
        "gold_parseable": False,
        "full_parseable": False,
        "full_equivalent": False,
        "answer": None,
        "gold_parsed": [],
        "full_parsed": [],
        "reasons": [],
        "error": None,
    }
    stripped = str(response).rstrip()
    try:
        formatted, answer, marker = canonicalize_response(stripped)
    except (TypeError, ValueError) as exc:
        result["reasons"].append("answer_extract_error")
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        return result

    result["answer_extracted"] = bool(answer)
    result["answer"] = answer
    result["canonical_format"] = marker == "already_canonical" and formatted == stripped
    if not result["canonical_format"]:
        result["reasons"].append("not_canonical")

    extraction_config = (
        runtime["latex_config"](boxed_match_priority=0),
        runtime["expr_config"](),
    )
    gold_text = rf"\boxed{{{answer}}}"
    try:
        gold = runtime["parse"](
            gold_text, extraction_config=extraction_config, raise_on_error=True,
            parsing_timeout=None)
        result["gold_parsed"] = [str(item) for item in gold]
        result["gold_parseable"] = bool(gold)
    except Exception as exc:
        result["reasons"].append("gold_parse_error")
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        return result
    if not result["gold_parseable"]:
        result["reasons"].append("gold_not_parseable")
        return result

    try:
        full = runtime["parse"](
            stripped, extraction_config=extraction_config, raise_on_error=False,
            parsing_timeout=None)
        result["full_parsed"] = [str(item) for item in full]
        result["full_parseable"] = bool(full)
        if not full:
            result["reasons"].append("full_response_not_parseable")
            return result
        result["full_equivalent"] = bool(runtime["verify"](
            gold, full, raise_on_error=False, timeout_seconds=None))
        if not result["full_equivalent"]:
            result["reasons"].append("full_response_not_equivalent")
    except Exception as exc:
        result["reasons"].append("full_response_parse_error")
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return result


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Math-Verify 审计数学 SFT 答案格式")
    parser.add_argument("--data", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--expected_records", type=int, default=0)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument(
        "--fail_if_not_all", action="store_true",
        help="任一记录未通过四项检查时返回非零退出码")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    summary_path = Path(args.summary).resolve()
    failures_path = Path(args.failures).resolve()
    if not data_path.is_file():
        parser.error(f"数据文件不存在: {data_path}")
    runtime = load_math_verify_runtime()

    metrics = Counter()
    reasons = Counter()
    failures_by_source = Counter()
    failures_by_level = Counter()
    examples = []
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_failures = failures_path.with_name(
        failures_path.name + f".tmp.{os.getpid()}")

    try:
        with data_path.open(encoding="utf-8") as reader, \
                temporary_failures.open("w", encoding="utf-8") as failure_writer:
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                inspected = inspect_response(str(row.get("response", "")), runtime)
                metrics["records"] += 1
                for name in (
                        "canonical_format", "answer_extracted", "gold_parseable",
                        "full_parseable", "full_equivalent"):
                    metrics[name] += int(bool(inspected[name]))

                if inspected["reasons"]:
                    metrics["failed_records"] += 1
                    source = str(row.get("source", "unknown"))
                    level = str(row.get("level", "unknown"))
                    failures_by_source[source] += 1
                    failures_by_level[level] += 1
                    reasons.update(inspected["reasons"])
                    failure = {
                        "line_number": line_number,
                        "source": source,
                        "level": row.get("level"),
                        "subject": row.get("subject"),
                        "prompt": row.get("prompt", ""),
                        "response": row.get("response", ""),
                        **inspected,
                    }
                    failure_writer.write(
                        json.dumps(failure, ensure_ascii=False) + "\n")
                    if len(examples) < 20:
                        examples.append({
                            "line_number": line_number,
                            "source": source,
                            "level": row.get("level"),
                            "answer": inspected["answer"],
                            "reasons": inspected["reasons"],
                            "error": inspected["error"],
                            "gold_parsed": inspected["gold_parsed"],
                            "full_parsed": inspected["full_parsed"],
                            "prompt_excerpt": str(row.get("prompt", ""))[:500],
                            "response_excerpt": str(row.get("response", ""))[-500:],
                        })
                if args.progress_every > 0 and metrics["records"] % args.progress_every == 0:
                    print(
                        f"已检查 {metrics['records']} 条 | 完整等价 "
                        f"{metrics['full_equivalent']} | 失败 {metrics['failed_records']}",
                        flush=True)
                if args.max_records > 0 and metrics["records"] >= args.max_records:
                    break

        if args.expected_records and metrics["records"] != args.expected_records:
            raise RuntimeError(
                f"记录数不符: 实际 {metrics['records']}, 预期 {args.expected_records}")
        os.replace(temporary_failures, failures_path)
    finally:
        temporary_failures.unlink(missing_ok=True)

    total = metrics["records"]
    coverage = {
        name: {
            "count": metrics[name],
            "rate": round(metrics[name] / total, 8) if total else 0.0,
        }
        for name in (
            "canonical_format", "answer_extracted", "gold_parseable",
            "full_parseable", "full_equivalent")
    }
    summary = {
        "data": str(data_path),
        "math_verify_version": runtime["version"],
        "records": total,
        "failed_records": metrics["failed_records"],
        "coverage": coverage,
        "reason_counts": dict(sorted(reasons.items())),
        "failures_by_source": dict(sorted(failures_by_source.items())),
        "failures_by_level": dict(sorted(failures_by_level.items())),
        "failure_examples": examples,
        "failures_jsonl": str(failures_path),
    }
    atomic_write_json(summary_path, summary)

    print("\n===== Math-Verify SFT 格式审计 =====")
    print(f"版本: {runtime['version']} | 记录: {total}")
    for name, value in coverage.items():
        print(f"{name:>20}: {value['count']}/{total} ({value['rate']:.2%})")
    print(f"失败记录: {metrics['failed_records']} | 原因: {dict(sorted(reasons.items()))}")
    print(f"汇总: {summary_path}")
    print(f"失败明细: {failures_path}")
    if args.fail_if_not_all and metrics["failed_records"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
