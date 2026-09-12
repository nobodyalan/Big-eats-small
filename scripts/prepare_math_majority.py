# -*- coding: utf-8 -*-
"""构造 MATH 为主、且 MATH Level 1--5 全覆盖的训练/验证数据。"""

import argparse
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict


INSTRUCTION = ("Solve the following math problem step by step, "
               "and put your final answer in \\boxed{...}:\n\n")


def math_level(value):
    match = re.search(r"([1-5])", str(value))
    return int(match.group(1)) if match else None


def load_math(zip_path):
    pools = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.startswith("MATH/train/") and n.endswith(".json"))
        for name in names:
            with zf.open(name) as f:
                src = json.load(f)
            level = math_level(src.get("level"))
            problem = (src.get("problem") or "").strip()
            solution = (src.get("solution") or "").strip()
            if level is None or not problem or not solution:
                continue
            pools[level].append({
                "prompt": INSTRUCTION + problem,
                "response": solution,
                "source": "MATH",
                "level": level,
                "subject": src.get("type", ""),
            })
    return pools


def normalize_gsm(record):
    out = dict(record)
    out["source"] = "GSM8K"
    return out


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(normalize_gsm(json.loads(line)))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def counts(rows):
    result = Counter()
    for row in rows:
        source = row.get("source", "unknown")
        key = f"MATH_L{row['level']}" if source == "MATH" else source
        result[key] += 1
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(
        description="生成约 75% MATH + 25% GSM8K 的分层训练数据")
    parser.add_argument("--math_zip", default="data/.cache/math/MATH.zip")
    parser.add_argument("--gsm8k", default="data/train_metamath.jsonl")
    parser.add_argument("--math_val_ratio", type=float, default=0.2)
    parser.add_argument("--gsm_train", type=int, default=2000)
    parser.add_argument("--gsm_val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_train", default="data/math_majority_train.jsonl")
    parser.add_argument("--out_val", default="data/math_majority_val.jsonl")
    parser.add_argument("--out_combined", default="data/math_majority_all.jsonl")
    parser.add_argument("--manifest", default="data/math_majority_manifest.json")
    args = parser.parse_args()

    if not 0.0 < args.math_val_ratio < 1.0:
        parser.error("--math_val_ratio 必须在 0 和 1 之间")

    rng = random.Random(args.seed)
    math_pools = load_math(args.math_zip)
    missing = sorted(set(range(1, 6)) - set(math_pools))
    if missing:
        raise SystemExit(f"MATH 训练集缺少 level: {missing}")

    math_train, math_val = [], []
    for level in range(1, 6):
        pool = math_pools[level]
        rng.shuffle(pool)
        n_val = max(1, round(len(pool) * args.math_val_ratio))
        math_val.extend(pool[:n_val])
        math_train.extend(pool[n_val:])

    gsm = load_jsonl(args.gsm8k)
    rng.shuffle(gsm)
    requested = args.gsm_train + args.gsm_val
    if requested > len(gsm):
        raise SystemExit(f"GSM8K 请求 {requested} 条，但只有 {len(gsm)} 条")
    gsm_train = gsm[:args.gsm_train]
    gsm_val = gsm[args.gsm_train:requested]

    train = math_train + gsm_train
    val = math_val + gsm_val
    rng.shuffle(train)
    rng.shuffle(val)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_val, val)
    write_jsonl(args.out_combined, train + val)

    manifest = {
        "seed": args.seed,
        "math_val_ratio": args.math_val_ratio,
        "train_total": len(train),
        "val_total": len(val),
        "eval_samples": len(val),
        "train_counts": counts(train),
        "val_counts": counts(val),
        "official_test_in_training": False,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"训练集 {len(train)} 条: {manifest['train_counts']}")
    print(f"验证集 {len(val)} 条: {manifest['val_counts']}")
    print(f"合并文件: {args.out_combined} (末尾 {len(val)} 条为验证集)")
    print(f"清单: {args.manifest}")


if __name__ == "__main__":
    main()
