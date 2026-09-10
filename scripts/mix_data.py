# -*- coding: utf-8 -*-
"""
中英数学数据混合切分: 各按参数抽取训练集 / 验证集

从中文(z)与英文(en)两个 jsonl 里, 分别随机抽取:
  --zh_train / --en_train  条 → 混合成训练集(整体再打乱)
  --zh_val   / --en_val    条 → 混合成验证集(整体再打乱)
训练集与验证集不重叠(每种语言先打乱再按顺序切分)。

输出:
  --out_train  训练集 jsonl(喂给 train_fusion.py 的 --data)
  --out_val    验证集 jsonl
  --out_combined(可选) train + val 拼成一个文件(val 在末尾),
               这样 train_fusion.py 用 --eval_samples <val 条数> 就能
               在训练过程中自动用验证集做 fusion/baseline 损失监控。

用法(在 BES 目录下执行):
  # 小规模试跑: 中英各抽少量
  python scripts/mix_data.py --zh_train 1000 --en_train 500 --zh_val 100 --en_val 100

  # 正式: 中文 2 万 + 英文 6k 训练, 验证集=训练集 20%(中 4k + 英 1.2k)
  python scripts/mix_data.py --zh_train 20000 --en_train 6000 --zh_val 4000 --en_val 1200 \
                             --out_train data/mix_train.jsonl --out_val data/mix_val.jsonl \
                             --out_combined data/mix_all.jsonl
"""

import argparse
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def read_lines(path: str):
    """读 jsonl 的原始行(去空白), 不解析 JSON, 保持与源文件完全一致的格式"""
    with open(path, encoding="utf-8") as f:
        return [ln for ln in (l.strip() for l in f) if ln]


def write_lines(path: str, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def split_pool(lines, train_n: int, val_n: int, rng: random.Random):
    """打乱后按顺序切: 前 train_n 条训练, 随后 val_n 条验证(不重叠)"""
    pool = lines[:]
    rng.shuffle(pool)
    train = pool[:train_n]
    val = pool[train_n:train_n + val_n]
    return train, val


def main():
    parser = argparse.ArgumentParser(description="中英数学数据混合切分(训练集/验证集)")
    parser.add_argument("--zh", default="data/train_metamath.jsonl", help="中文 jsonl")
    parser.add_argument("--en", default="data/math_train.jsonl", help="英文 jsonl")
    parser.add_argument("--zh_train", type=int, default=20000, help="中文训练条数")
    parser.add_argument("--en_train", type=int, default=6000, help="英文训练条数(英文池仅 7500, 需留出验证)")
    parser.add_argument("--zh_val", type=int, default=4000, help="中文验证条数(默认=训练集的 20%%)")
    parser.add_argument("--en_val", type=int, default=1200, help="英文验证条数(默认=训练集的 20%%)")
    parser.add_argument("--out_train", default="data/mix_train.jsonl")
    parser.add_argument("--out_val", default="data/mix_val.jsonl")
    parser.add_argument("--out_combined", default="",
                        help="可选: 训练+验证拼成一个文件(验证集在末尾), 便于 --eval_samples 监控")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    zh = read_lines(args.zh)
    en = read_lines(args.en)
    print(f"源数据: 中文 {len(zh)} 条 | 英文 {len(en)} 条")

    zh_train, zh_val = split_pool(zh, args.zh_train, args.zh_val, rng)
    en_train, en_val = split_pool(en, args.en_train, args.en_val, rng)
    if args.zh_train > len(zh):
        print(f"  [提示] 中文训练请求 {args.zh_train} 超过可用 {len(zh)}, 实际取 {len(zh_train)}")
    if args.en_train > len(en):
        print(f"  [提示] 英文训练请求 {args.en_train} 超过可用 {len(en)}, 实际取 {len(en_train)}")
    if len(zh_val) < args.zh_val:
        print(f"  [提示] 中文验证请求 {args.zh_val} 超过剩余可用, 实际取 {len(zh_val)}")
    if len(en_val) < args.en_val:
        print(f"  [提示] 英文验证请求 {args.en_val} 超过剩余可用, 实际取 {len(en_val)}")

    train = zh_train + en_train
    val = zh_val + en_val
    rng.shuffle(train)
    rng.shuffle(val)

    write_lines(args.out_train, train)
    write_lines(args.out_val, val)
    print(f"训练集: {len(train)} 条(中文 {len(zh_train)} + 英文 {len(en_train)}) → {args.out_train}")
    print(f"验证集: {len(val)} 条(中文 {len(zh_val)} + 英文 {len(en_val)}) → {args.out_val}")

    if args.out_combined:
        write_lines(args.out_combined, train + val)
        print(f"合并文件: {len(train) + len(val)} 条(验证集在末尾) → {args.out_combined}")
        print(f"\n训练命令(带评估监控, 验证集大小 {len(val)}):")
        print(f"  python core_training/train_fusion.py --data {args.out_combined} "
              f"--eval_samples {len(val)} --epochs 3 --eval_every 100 --patience 3")


if __name__ == "__main__":
    main()
