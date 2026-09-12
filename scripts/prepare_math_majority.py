# -*- coding: utf-8 -*-
"""构造 MATH 主导的 MetaMathQA 增强训练集与独立验证集。"""

import argparse
import json
import os
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict


RECIPE_VERSION = "metamath_math_majority_v2"
INSTRUCTION = ("Solve the following math problem step by step, "
               "and put your final answer in \\boxed{...}:\n\n")


def math_level(value):
    match = re.search(r"([1-5])", str(value))
    return int(match.group(1)) if match else None


def problem_text(record):
    prompt = str(record.get("prompt", record.get("text", "")))
    if "解题思路:" in prompt:
        prompt = prompt.split("解题思路:", 1)[0]
    if prompt.startswith(INSTRUCTION):
        prompt = prompt[len(INSTRUCTION):]
    return prompt.strip()


def fingerprint(record):
    """跨数据源精确去重：保留数学符号，只统一 Unicode、大小写和空白。"""
    text = unicodedata.normalize("NFKC", problem_text(record)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def load_official_math(zip_path, split):
    pools = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.startswith(f"MATH/{split}/") and n.endswith(".json"))
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
                "source": f"MATH_{split}",
                "level": level,
                "subject": src.get("type", ""),
            })
    return pools


def load_jsonl(path, source):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not problem_text(row):
                continue
            if "prompt" not in row and "text" in row and "解题思路:" in row["text"]:
                prompt, response = row["text"].split("解题思路:", 1)
                row = {"prompt": prompt + "解题思路:", "response": response}
            if not str(row.get("response", "")).strip():
                continue
            row = dict(row)
            row["source"] = source
            rows.append(row)
    return rows


def unique_sample(rows, count, rng, blocked):
    rows = rows[:]
    rng.shuffle(rows)
    selected = []
    duplicate = 0
    for row in rows:
        key = fingerprint(row)
        if not key or key in blocked:
            duplicate += 1
            continue
        blocked.add(key)
        selected.append(row)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"去重后只取得 {len(selected)}/{count} 条；源池 {len(rows)}，跳过 {duplicate}")
    return selected, duplicate


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def counts(rows):
    result = Counter()
    for row in rows:
        source = row.get("source", "unknown")
        level = row.get("level")
        key = f"{source}_L{level}" if level is not None else source
        result[key] += 1
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(
        description="生成约 75% MATH + 25% GSM8K 的 MetaMathQA 增强数据")
    parser.add_argument("--math_zip", default="data/.cache/math/MATH.zip")
    parser.add_argument("--meta_math", default="data/metamath_math.jsonl")
    parser.add_argument("--meta_gsm8k", default="data/metamath_gsm8k.jsonl")
    parser.add_argument("--math_val_ratio", type=float, default=0.2)
    parser.add_argument("--meta_math_train", type=int, default=18000)
    parser.add_argument("--meta_gsm_train", type=int, default=8000)
    parser.add_argument("--meta_gsm_val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_train", default="data/math_majority_train.jsonl")
    parser.add_argument("--out_val", default="data/math_majority_val.jsonl")
    parser.add_argument("--out_combined", default="data/math_majority_all.jsonl")
    parser.add_argument("--manifest", default="data/math_majority_manifest.json")
    args = parser.parse_args()

    if not 0.0 < args.math_val_ratio < 1.0:
        parser.error("--math_val_ratio 必须在 0 和 1 之间")

    rng = random.Random(args.seed)
    official_train = load_official_math(args.math_zip, "train")
    official_test = load_official_math(args.math_zip, "test")
    missing = sorted(set(range(1, 6)) - set(official_train))
    if missing:
        raise SystemExit(f"MATH 训练集缺少 level: {missing}")

    # 官方 test 的题面从所有训练候选中封锁；验证集保留官方 train 的高质量题。
    test_keys = {fingerprint(row) for pool in official_test.values() for row in pool}
    blocked = set(test_keys)
    official_math_train, official_math_val = [], []
    official_overlap = 0
    for level in range(1, 6):
        pool = []
        for row in official_train[level]:
            if fingerprint(row) in test_keys:
                official_overlap += 1
            else:
                pool.append(row)
        rng.shuffle(pool)
        n_val = max(1, round(len(pool) * args.math_val_ratio))
        val_rows = pool[:n_val]
        train_rows = pool[n_val:]
        for row in val_rows + train_rows:
            key = fingerprint(row)
            if key in blocked:
                raise RuntimeError("官方 MATH train 内部出现重复题面")
            blocked.add(key)
        official_math_val.extend(val_rows)
        official_math_train.extend(train_rows)

    meta_math_pool = load_jsonl(args.meta_math, "MetaMathQA_MATH")
    meta_gsm_pool = load_jsonl(args.meta_gsm8k, "MetaMathQA_GSM8K")

    # 验证集先抽取并封锁，再取训练集，保证题面不重叠。
    gsm_val, gsm_val_dups = unique_sample(
        meta_gsm_pool, args.meta_gsm_val, rng, blocked)
    gsm_train, gsm_train_dups = unique_sample(
        meta_gsm_pool, args.meta_gsm_train, rng, blocked)
    meta_math_train, meta_math_dups = unique_sample(
        meta_math_pool, args.meta_math_train, rng, blocked)

    train = official_math_train + meta_math_train + gsm_train
    val = official_math_val + gsm_val
    rng.shuffle(train)
    rng.shuffle(val)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_val, val)
    write_jsonl(args.out_combined, train + val)

    manifest = {
        "recipe_version": RECIPE_VERSION,
        "seed": args.seed,
        "math_val_ratio": args.math_val_ratio,
        "train_total": len(train),
        "val_total": len(val),
        "eval_samples": len(val),
        "train_counts": counts(train),
        "val_counts": counts(val),
        "source_pool_counts": {
            "meta_math": len(meta_math_pool),
            "meta_gsm8k": len(meta_gsm_pool),
        },
        "dedup_skipped": {
            "meta_math": meta_math_dups,
            "gsm_train": gsm_train_dups,
            "gsm_val": gsm_val_dups,
        },
        "official_math_test_fingerprints_blocked": len(test_keys),
        "official_train_test_overlap_skipped": official_overlap,
        "official_test_in_training": False,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"训练集 {len(train)} 条: {manifest['train_counts']}")
    print(f"验证集 {len(val)} 条: {manifest['val_counts']}")
    print(f"去重/测试集过滤: {manifest['dedup_skipped']}")
    print(f"合并文件: {args.out_combined} (末尾 {len(val)} 条为验证集)")
    print(f"清单: {args.manifest}")


if __name__ == "__main__":
    main()
