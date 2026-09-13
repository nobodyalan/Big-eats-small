# -*- coding: utf-8 -*-
"""在同一张可见 GPU 上并行运行 GSM8K / MATH-low / MATH-high 三档评测。

模型相关参数原样放在 ``--`` 后转交给 eval_math.py。例如：

  CUDA_VISIBLE_DEVICES=0 python3 eval/run_segments_parallel.py \
    --out_dir cache/capacity_lora_experiments/eval \
    --tag large_lora_r32_seed42_n200 --quiet -- \
    --limit 200 --seed 42 --math_lo 1-3 --math_hi 4-5 --max_new 1024 \
    --attn_impl flash_attention_2 --fusion_only \
    --lora_ckpt cache/capacity_lora_experiments/large_lora_r32_seed42.best

三个子进程继承当前 CUDA_VISIBLE_DEVICES。每个进程会各自加载一份模型，使用前请
确认单卡显存能够同时容纳三份模型。
"""

import argparse
import os
import shlex
import subprocess
import sys


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_PY = os.path.join(_ROOT, "eval", "eval_math.py")
_SEGMENTS = ("gsm8k", "math_lo", "math_hi")


def main():
    parser = argparse.ArgumentParser(
        description="同一模型的三档数学评测并行启动器",
    )
    parser.add_argument("--out_dir", required=True, help="三档结果 JSON 的共同目录")
    parser.add_argument("--tag", required=True, help="输出文件的共同实验标签")
    parser.add_argument("--quiet", action="store_true",
                        help="丢弃三个子进程的逐题输出，只保留结果 JSON")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印三个命令，不启动评测")
    parser.add_argument(
        "eval_args", nargs=argparse.REMAINDER,
        help="放在 -- 后、原样传递给 eval_math.py 的模型与评测参数",
    )
    args = parser.parse_args()

    forwarded = list(args.eval_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forbidden = {"--bench", "--segment_only", "--out_dir", "--tag"}
    conflicts = sorted(forbidden.intersection(forwarded))
    if conflicts:
        parser.error("以下参数由并行启动器管理，不要在 -- 后重复传入: "
                     + ", ".join(conflicts))

    os.makedirs(args.out_dir, exist_ok=True)
    commands = []
    for segment in _SEGMENTS:
        commands.append([
            sys.executable, _EVAL_PY,
            "--bench", "segments",
            "--segment_only", segment,
            "--out_dir", args.out_dir,
            "--tag", args.tag,
            *forwarded,
        ])

    for segment, command in zip(_SEGMENTS, commands):
        print(f"[{segment}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return

    output = subprocess.DEVNULL if args.quiet else None
    processes = [subprocess.Popen(command, cwd=_ROOT, stdout=output, stderr=output)
                 for command in commands]
    failed = []
    try:
        for segment, process in zip(_SEGMENTS, processes):
            return_code = process.wait()
            print(f"[{segment}] 退出码 {return_code}", flush=True)
            if return_code != 0:
                failed.append((segment, return_code))
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise

    if failed:
        details = ", ".join(f"{name}={code}" for name, code in failed)
        raise SystemExit(f"部分档位评测失败: {details}")
    print(f"三档评测完成，结果目录: {args.out_dir}")


if __name__ == "__main__":
    main()
