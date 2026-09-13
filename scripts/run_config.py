# -*- coding: utf-8 -*-
"""从版本化 JSON 配置启动训练或评测。

配置只描述一个进程；多 GPU 实验使用多个配置分别启动。默认将子进程输出以
追加模式写入配置中的 ``log_path``，适合在服务器上配合 nohup 使用。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("仅支持 schema_version=1")
    for key in ("name", "program"):
        if not config.get(key):
            raise ValueError(f"配置缺少必填字段: {key}")
    if not (config.get("experiment_key") or config.get("run_id")):
        raise ValueError("配置缺少 experiment_key（旧配置可继续使用 run_id）")
    return config


def experiment_key(config: dict) -> str:
    """返回语义化实验键；兼容尚未迁移的旧 run_id 配置。"""
    return config.get("experiment_key") or config["run_id"]


def resolve_value(value):
    if isinstance(value, dict) and "first_existing" in value:
        candidates = value["first_existing"]
        if not candidates:
            raise ValueError("first_existing 至少需要一个候选路径")
        for raw in candidates:
            if repo_path(str(raw)).exists():
                return raw
        # dry-run 时尚未训练完成是正常的；展示优先级最高的预期路径。
        return candidates[0]
    return value


def cli_args(values: dict) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        value = resolve_value(value)
        # 当前仓库 argparse 参数使用下划线；保留配置键的原样拼写。
        option = "--" + name
        if value is None or value is False:
            continue
        if value is True:
            result.append(option)
        elif isinstance(value, list):
            result.extend([option, *map(str, value)])
        else:
            result.extend([option, str(value)])
    return result


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_paths(config: dict) -> None:
    for raw in config.get("required_paths", []):
        if not repo_path(raw).exists():
            raise FileNotFoundError(f"缺少必需文件或目录: {raw}")
    for candidates in config.get("required_any_paths", []):
        if not any(repo_path(raw).exists() for raw in candidates):
            raise FileNotFoundError(
                "以下候选路径至少需要存在一个: " + ", ".join(candidates)
            )

    args = config.get("args", {})
    for key in ("out_dir",):
        if args.get(key):
            repo_path(str(args[key])).mkdir(parents=True, exist_ok=True)
    for key in ("out", "plot"):
        if args.get(key):
            repo_path(str(args[key])).parent.mkdir(parents=True, exist_ok=True)

    for key in ("log_path", "resolved_config_path"):
        if config.get(key):
            repo_path(config[key]).parent.mkdir(parents=True, exist_ok=True)


def missing_required_paths(config: dict) -> list[str]:
    missing = [raw for raw in config.get("required_paths", [])
               if not repo_path(raw).exists()]
    for candidates in config.get("required_any_paths", []):
        if not any(repo_path(raw).exists() for raw in candidates):
            missing.append("任一: " + " | ".join(candidates))
    return missing


def command_for(config: dict) -> list[str]:
    program = repo_path(config["program"])
    if not program.is_file():
        raise FileNotFoundError(f"程序不存在: {config['program']}")
    command = [sys.executable, str(program), *cli_args(config.get("args", {}))]
    passthrough = config.get("passthrough", [])
    if passthrough:
        command.extend(["--", *(str(resolve_value(value)) for value in passthrough)])
    return command


def write_resolved(config_path: Path, config: dict, command: list[str]) -> None:
    output = config.get("resolved_config_path")
    if not output:
        return
    resolved = dict(config)
    resolved["source_config"] = str(config_path.resolve())
    resolved["resolved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    resolved["git_commit"] = git_commit()
    resolved["command"] = command
    destination = repo_path(output)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 BES JSON 实验配置")
    parser.add_argument("config", help="configs/ 下的 JSON 配置")
    parser.add_argument("--dry-run", action="store_true", help="检查配置并打印命令，不执行")
    parser.add_argument("--no-log", action="store_true", help="忽略 log_path，输出到当前终端")
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config = load_config(config_path)
    command = command_for(config)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})
    if config.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    print(f"experiment: {experiment_key(config)} | task: {config['name']}")
    print("command:", shlex.join(command))
    if config.get("gpu") is not None:
        print("CUDA_VISIBLE_DEVICES:", config["gpu"])
    if config.get("log_path") and not args.no_log:
        print("log:", config["log_path"], "(append)")
    if args.dry_run:
        missing = missing_required_paths(config)
        if missing:
            print("尚未就绪（正式运行前必须存在）:")
            for path in missing:
                print(" -", path)
        return

    prepare_paths(config)
    write_resolved(config_path, config, command)
    log_path = config.get("log_path")
    if log_path and not args.no_log:
        destination = repo_path(log_path)
        with destination.open("a", encoding="utf-8", buffering=1) as log:
            stamp = dt.datetime.now().astimezone().isoformat()
            log.write(f"\n===== {stamp} | {config['name']} =====\n")
            log.write("command: " + shlex.join(command) + "\n")
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
