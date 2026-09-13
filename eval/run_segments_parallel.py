# -*- coding: utf-8 -*-
"""在同一张可见 GPU 上并行运行固定数学评测套件。

默认保持旧版三档兼容；加 ``--include_omni_math`` 后并行运行四档：
GSM8K、MATH L1-3、MATH L4-5、Omni-MATH rule L1-10。每档写入一个
固定文件名，全部结束后核验样本数、seed、题集哈希和 Math-Verify 状态，
再生成一个便于查看的 ``summary.txt``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
EVAL_PY = ROOT / "eval" / "eval_math.py"

BASE_TASKS = (
    ("gsm8k", "01_gsm8k.json", ("--bench", "segments", "--segment_only", "gsm8k")),
    ("math_low", "02_math_low.json", ("--bench", "segments", "--segment_only", "math_lo")),
    ("math_high", "03_math_high.json", ("--bench", "segments", "--segment_only", "math_hi")),
)
OMNI_TASK = (
    "omni_math", "04_omni_math.json", ("--bench", "omni_math"),
)


def option_value(arguments: list[str], option: str, default=None):
    """读取转发参数中最后一个 ``--option value``。"""
    value = default
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} 缺少值")
            value = arguments[index + 1]
    return value


def metric_from_payload(payload: dict, task_name: str) -> dict:
    summary = payload.get("summary") or {}
    n = int(summary.get("n", 0))
    labels = [
        label for label in ("lora", "fusion", "baseline")
        if f"{label}_correct" in summary
    ]
    if len(labels) != 1:
        raise ValueError(
            f"{task_name}: 应恰有一个被评模型标签，实际为 {labels or '无'}")
    label = labels[0]

    if task_name == "gsm8k":
        correct_key = f"{label}_correct"
        acc_key = f"{label}_acc"
        judge = "GSM8K numeric"
    else:
        correct_key = f"{label}_math_verify_correct"
        acc_key = f"{label}_math_verify_acc"
        error_key = f"{label}_math_verify_errors"
        if correct_key not in summary or acc_key not in summary:
            raise ValueError(f"{task_name}: 缺少 Math-Verify 指标")
        errors = int(summary.get(error_key, -1))
        judge = f"Math-Verify {payload.get('math_verify_version') or 'unknown'}"

    return {
        "label": label,
        "n": n,
        "correct": int(summary[correct_key]),
        "accuracy": float(summary[acc_key]),
        "judge": judge,
        "judge_errors": errors if task_name != "gsm8k" else 0,
    }


def build_summary(out_dir: Path, tasks, expected_n: int, expected_seed: int,
                  tag: str, expected_suite_run_id: str = "") -> tuple[str, list[str]]:
    """读取固定结果文件，返回人读摘要和所有一致性错误。"""
    errors: list[str] = []
    rows = []
    question_hashes = {}

    for task_name, filename, _ in tasks:
        path = out_dir / filename
        if not path.is_file():
            errors.append(f"{task_name}: 缺结果文件 {path}")
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            metric = metric_from_payload(payload, task_name)
            seed = int(payload.get("seed", -1))
            suite_run_id = str(payload.get("suite_run_id", ""))
            question_hash = str(payload.get("question_set_sha256", ""))
            if metric["n"] != expected_n:
                errors.append(
                    f"{task_name}: 实际 {metric['n']} 题，要求 {expected_n} 题")
            if metric["judge_errors"] != 0:
                errors.append(
                    f"{task_name}: Math-Verify 解析异常 "
                    f"{metric['judge_errors']} 题")
            if seed != expected_seed:
                errors.append(
                    f"{task_name}: 实际 seed={seed}，要求 seed={expected_seed}")
            if expected_suite_run_id and suite_run_id != expected_suite_run_id:
                errors.append(
                    f"{task_name}: suite_run_id 不属于本轮，可能读到了旧结果")
            if len(question_hash) != 64:
                errors.append(f"{task_name}: question_set_sha256 缺失或无效")
            rows.append((task_name, filename, metric))
            question_hashes[task_name] = question_hash
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{task_name}: 无法验证结果: {exc}")

    suite_material = "\n".join(
        f"{name}:{question_hashes.get(name, '')}" for name, _, _ in tasks)
    suite_hash = hashlib.sha256(suite_material.encode("utf-8")).hexdigest()
    lines = [
        f"experiment: {tag}",
        f"seed: {expected_seed}",
        f"samples_per_tier: {expected_n}",
        f"suite_run_id: {expected_suite_run_id or 'not-set'}",
        f"suite_question_set_sha256: {suite_hash}",
        "",
        "tier         correct       accuracy   judge",
        "------------ ------------- ---------- ------------------------",
    ]
    for task_name, _, metric in rows:
        lines.append(
            f"{task_name:<12} {metric['correct']:>4}/{metric['n']:<7} "
            f"{metric['accuracy']:>9.2%}   {metric['judge']}")
    lines.extend(["", "question_set_sha256:"])
    for task_name, _, _ in tasks:
        lines.append(f"  {task_name}: {question_hashes.get(task_name, 'MISSING')}")
    lines.extend(["", "result_files:"])
    for task_name, filename, _ in tasks:
        lines.append(f"  {task_name}: {filename}")
    lines.extend(["", "validation: " + ("OK" if not errors else "FAILED")])
    if errors:
        lines.append("errors:")
        lines.extend(f"  - {error}" for error in errors)
    return "\n".join(lines) + "\n", errors


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="同一模型的三/四档数学评测并行启动器")
    parser.add_argument("--out_dir", required=True, help="结果 JSON 的共同目录")
    parser.add_argument("--tag", required=True, help="摘要中的实验标签")
    parser.add_argument("--include_omni_math", action="store_true",
                        help="增加 Omni-MATH rule，形成四档评测")
    parser.add_argument("--summary_path", default="",
                        help="摘要文件；默认 <out_dir>/summary.txt")
    parser.add_argument("--quiet", action="store_true",
                        help="不显示子进程逐题输出，只显示命令和最终摘要")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令，不启动评测")
    parser.add_argument(
        "eval_args", nargs=argparse.REMAINDER,
        help="放在 -- 后、原样传递给 eval_math.py 的模型与评测参数",
    )
    args = parser.parse_args()

    forwarded = list(args.eval_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forbidden = {
        "--bench", "--segment_only", "--out_dir", "--tag", "--result_path",
        "--suite_run_id",
    }
    conflicts = sorted(forbidden.intersection(forwarded))
    if conflicts:
        parser.error("以下参数由并行启动器管理，不要重复传入: " + ", ".join(conflicts))

    try:
        limit = int(option_value(forwarded, "--limit", 200))
        seed = int(option_value(forwarded, "--seed", 42))
        omni_per_level = int(option_value(forwarded, "--omni_limit_per_level", 0))
    except ValueError as exc:
        parser.error(str(exc))
    if limit <= 0:
        parser.error("并行套件要求 --limit 为正数，以便校验每档题数")
    if args.include_omni_math:
        if option_value(forwarded, "--math_judge", "legacy") != "both":
            parser.error("四档正式评测必须传 --math_judge both")
        if omni_per_level != 0:
            parser.error(
                "四档正式评测禁止 --omni_limit_per_level；它会把总题数变成每级题数之和")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path) if args.summary_path else out_dir / "summary.txt"
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    tasks = list(BASE_TASKS)
    if args.include_omni_math:
        tasks.append(OMNI_TASK)
    suite_run_id = f"{args.tag}-{uuid.uuid4().hex}"

    commands = []
    for task_name, filename, selector in tasks:
        commands.append((task_name, [
            sys.executable, str(EVAL_PY), *selector,
            "--out_dir", str(out_dir),
            "--tag", args.tag,
            "--result_path", str(out_dir / filename),
            "--suite_run_id", suite_run_id,
            *forwarded,
        ]))

    for task_name, command in commands:
        print(f"[{task_name}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return

    processes = []
    for task_name, command in commands:
        log_path = out_dir / f".{task_name}.running.log"
        log_handle = (log_path.open("w", encoding="utf-8")
                      if args.quiet else None)
        process = subprocess.Popen(
            command, cwd=ROOT,
            stdout=log_handle if log_handle is not None else None,
            stderr=subprocess.STDOUT if log_handle is not None else None,
        )
        processes.append((task_name, process, log_handle, log_path))
    failed = []
    try:
        for task_name, process, log_handle, log_path in processes:
            return_code = process.wait()
            if log_handle is not None:
                log_handle.close()
            print(f"[{task_name}] 退出码 {return_code}", flush=True)
            if return_code != 0:
                failed.append((task_name, return_code, log_path))
            elif log_handle is not None:
                log_path.unlink(missing_ok=True)
    except KeyboardInterrupt:
        for _, process, log_handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            if log_handle is not None and not log_handle.closed:
                log_handle.close()
        raise

    if failed:
        details = ", ".join(
            f"{name}={code} (日志 {path})" for name, code, path in failed)
        raise SystemExit(f"部分档位评测失败: {details}")

    report, validation_errors = build_summary(
        out_dir, tasks, expected_n=limit, expected_seed=seed, tag=args.tag,
        expected_suite_run_id=suite_run_id)
    atomic_write_text(summary_path, report)
    print("\n" + report, end="", flush=True)
    print(f"摘要已保存: {summary_path}", flush=True)
    if validation_errors:
        raise SystemExit("结果一致性校验失败；详见 summary.txt")


if __name__ == "__main__":
    main()
