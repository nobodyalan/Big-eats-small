# -*- coding: utf-8 -*-
"""构造原题组严格隔离的 MATH 主导 SFT 训练集与验证集。

隔离单位不是 MetaMathQA 的改写 query，而是 ``original_question``。脚本还会把
官方 MATH test、GSM8K test 和 Omni-MATH rule test 的规范化原题全部从训练和
验证候选中封锁，并在写文件前执行硬断言。
"""

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict


RECIPE_VERSION = "metamath_math_majority_v3_group_disjoint"
INSTRUCTION = ("Solve the following math problem step by step, "
               "and put your final answer in \\boxed{...}:\n\n")


def math_level(value):
    match = re.search(r"([1-5])", str(value))
    return int(match.group(1)) if match else None


def strip_instruction(text):
    text = str(text or "").strip()
    if "解题思路:" in text:
        text = text.split("解题思路:", 1)[0]
    if text.startswith(INSTRUCTION):
        text = text[len(INSTRUCTION):]
    return text.strip()


def problem_text(record):
    for field in ("problem", "question", "query", "prompt", "text"):
        value = record.get(field)
        if value:
            return strip_instruction(value)
    return ""


def normalize_problem(text):
    """跨来源原题指纹：保留数学语义，只消除排版差异。"""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\(?:,|!|;|:)\s*", "", text)
    return re.sub(r"\s+", "", text)


def row_fingerprint(record):
    """改写 query 的去重键；用于避免同一行被重复选中。"""
    return normalize_problem(problem_text(record))


def group_fingerprint(record):
    """原题组键；MetaMathQA 必须优先使用 raw 中的 original_question。"""
    original = record.get("original_question")
    return normalize_problem(original if original else problem_text(record))


def load_official_math(zip_path, split):
    pools = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.startswith(f"MATH/{split}/") and n.endswith(".json"))
        for name in names:
            with zf.open(name) as stream:
                src = json.load(stream)
            level = math_level(src.get("level"))
            problem = (src.get("problem") or "").strip()
            solution = (src.get("solution") or "").strip()
            if level is None or not problem or not solution:
                continue
            pools[level].append({
                "prompt": INSTRUCTION + problem,
                "response": solution,
                "original_question": problem,
                "source": f"MATH_{split}",
                "level": level,
                "subject": src.get("type", ""),
            })
    return pools


def load_jsonl(path, source, require_original=False):
    rows = []
    missing_original = 0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not problem_text(row):
                continue
            if "prompt" not in row and "text" in row and "解题思路:" in row["text"]:
                prompt, response = row["text"].split("解题思路:", 1)
                row = {"prompt": prompt + "解题思路:", "response": response,
                       "original_question": row.get("original_question"),
                       "meta_type": row.get("meta_type")}
            if not str(row.get("response", "")).strip():
                continue
            if require_original and not str(row.get("original_question", "")).strip():
                missing_original += 1
                continue
            row = dict(row)
            row["source"] = source
            rows.append(row)
    if require_original and missing_original:
        raise RuntimeError(
            f"{path} 有 {missing_original} 条缺少 original_question；先运行 "
            "python3 scripts/prepare_data.py --skip_zh --skip_mix 由本地 raw 重建转换文件")
    return rows


def load_gsm8k_test_keys(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺少 GSM8K test: {path}；先运行一次 eval/eval_math.py 下载测试集")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("读取 GSM8K test 需要 pyarrow") from exc
    table = pq.read_table(path, columns=["question"])
    return {normalize_problem(value) for value in table.column("question").to_pylist()
            if str(value or "").strip()}


def load_jsonl_test_keys(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {label} test: {path}")
    keys = set()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = normalize_problem(problem_text(row))
            if key:
                keys.add(key)
    if not keys:
        raise RuntimeError(f"{label} test 未读到有效题面: {path}")
    return keys


def grouped_rows(rows, forbidden_groups, seen_rows):
    groups = defaultdict(list)
    skipped = Counter()
    for row in rows:
        group = group_fingerprint(row)
        query = row_fingerprint(row)
        if not group or not query:
            skipped["empty"] += 1
        elif group in forbidden_groups:
            skipped["forbidden_group"] += 1
        elif query in seen_rows:
            skipped["duplicate_query"] += 1
        else:
            groups[group].append(row)
    return groups, skipped


def sample_grouped(rows, count, rng, forbidden_groups, seen_rows,
                   max_variants_per_group):
    """先覆盖不同原题，再取同组其他变体，降低改写样本对数据量的虚增。"""
    groups, skipped = grouped_rows(rows, forbidden_groups, seen_rows)
    group_order = list(groups)
    rng.shuffle(group_order)
    for group in group_order:
        rng.shuffle(groups[group])
    selected = []
    selected_groups = set()
    for variant_index in range(max_variants_per_group):
        for group in group_order:
            variants = groups[group]
            if variant_index >= len(variants):
                continue
            row = variants[variant_index]
            query = row_fingerprint(row)
            if query in seen_rows:
                skipped["duplicate_query"] += 1
                continue
            selected.append(row)
            selected_groups.add(group)
            seen_rows.add(query)
            if len(selected) >= count:
                return selected, selected_groups, dict(skipped)
    raise RuntimeError(
        f"按原题分组且每组最多 {max_variants_per_group} 个变体后，仅取得 "
        f"{len(selected)}/{count} 条；可增加源数据或显式提高 --max_variants_per_group")


def deduplicate_official(rows, forbidden_groups, seen_rows):
    selected = []
    groups = set()
    skipped = Counter()
    for row in rows:
        group = group_fingerprint(row)
        query = row_fingerprint(row)
        if group in forbidden_groups:
            skipped["external_test_group"] += 1
        elif group in groups:
            skipped["duplicate_group"] += 1
        elif query in seen_rows:
            skipped["duplicate_query"] += 1
        else:
            selected.append(row)
            groups.add(group)
            seen_rows.add(query)
    return selected, groups, skipped


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def counts(rows):
    result = Counter()
    for row in rows:
        source = row.get("source", "unknown")
        level = row.get("level")
        result[f"{source}_L{level}" if level is not None else source] += 1
    return dict(sorted(result.items()))


def sha256_keys(keys):
    payload = "\n".join(sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="生成原题级 train/val/test 严格隔离的 MATH 主导数据")
    parser.add_argument("--math_zip", default="data/.cache/math/MATH.zip")
    parser.add_argument("--meta_math", default="data/metamath_math.jsonl")
    parser.add_argument("--meta_gsm8k", default="data/metamath_gsm8k.jsonl")
    parser.add_argument("--gsm8k_test", default="data/.cache/gsm8k_test.parquet")
    parser.add_argument("--omni_test", default="data/omni_math_rule_test.jsonl")
    parser.add_argument("--math_val_ratio", type=float, default=0.2)
    parser.add_argument("--meta_math_train", type=int, default=18000)
    parser.add_argument("--meta_gsm_train", type=int, default=8000)
    parser.add_argument("--meta_gsm_val", type=int, default=500)
    # MetaMathQA 的 MATH 部分由较少原题产生大量推理/改写变体。服务器实测在
    # 完整 test/val 原题封锁后，cap=3/cap=6 分别只能提供 10,529/15,291 条，
    # 均无法满足保持 v2 配方所需的 18,000 条 MetaMath-MATH。cap=10 给容量留出
    # 余量，同时仍显式限制单题权重并维持原题组级 split 隔离。
    parser.add_argument("--max_variants_per_group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # v3 使用新文件名，绝不静默覆盖历史实验依赖的 v2 数据。
    parser.add_argument("--out_train", default="data/math_majority_v3_train.jsonl")
    parser.add_argument("--out_val", default="data/math_majority_v3_val.jsonl")
    parser.add_argument("--out_combined", default="data/math_majority_v3_all.jsonl")
    parser.add_argument("--manifest", default="data/math_majority_v3_manifest.json")
    args = parser.parse_args()

    if not 0.0 < args.math_val_ratio < 1.0:
        parser.error("--math_val_ratio 必须在 0 和 1 之间")
    if args.max_variants_per_group < 1:
        parser.error("--max_variants_per_group 必须 >= 1")

    rng = random.Random(args.seed)
    official_train = load_official_math(args.math_zip, "train")
    official_test = load_official_math(args.math_zip, "test")
    missing = sorted(set(range(1, 6)) - set(official_train))
    if missing:
        raise SystemExit(f"MATH 训练集缺少 level: {missing}")

    external_sets = {
        "math_test": {group_fingerprint(row) for pool in official_test.values()
                      for row in pool},
        "gsm8k_test": load_gsm8k_test_keys(args.gsm8k_test),
        "omni_math_test": load_jsonl_test_keys(args.omni_test, "Omni-MATH"),
    }
    external_groups = set().union(*external_sets.values())

    seen_rows = set()
    official_math_train, official_math_val = [], []
    official_train_groups, val_groups = set(), set()
    official_skipped = Counter()
    for level in range(1, 6):
        pool, _, skipped = deduplicate_official(
            official_train[level], external_groups, seen_rows)
        official_skipped.update(skipped)
        rng.shuffle(pool)
        n_val = max(1, round(len(pool) * args.math_val_ratio))
        val_rows, train_rows = pool[:n_val], pool[n_val:]
        official_math_val.extend(val_rows)
        official_math_train.extend(train_rows)
        val_groups.update(group_fingerprint(row) for row in val_rows)
        official_train_groups.update(group_fingerprint(row) for row in train_rows)

    meta_math_pool = load_jsonl(
        args.meta_math, "MetaMathQA_MATH", require_original=True)
    meta_gsm_pool = load_jsonl(
        args.meta_gsm8k, "MetaMathQA_GSM8K", require_original=True)

    # GSM 验证原题不得与任何 official MATH 训练/验证原题相撞；每组只取一条。
    gsm_val_forbidden = external_groups | official_train_groups | val_groups
    gsm_val, gsm_val_groups, gsm_val_stats = sample_grouped(
        meta_gsm_pool, args.meta_gsm_val, rng, gsm_val_forbidden,
        seen_rows, max_variants_per_group=1)
    val_groups.update(gsm_val_groups)

    # 所有训练来源统一封锁完整验证原题组和三套外部测试原题组。
    train_forbidden = external_groups | val_groups
    meta_math_train, _, meta_math_stats = sample_grouped(
        meta_math_pool, args.meta_math_train, rng, train_forbidden,
        seen_rows, args.max_variants_per_group)
    gsm_train, _, gsm_train_stats = sample_grouped(
        meta_gsm_pool, args.meta_gsm_train, rng, train_forbidden,
        seen_rows, args.max_variants_per_group)

    train = official_math_train + meta_math_train + gsm_train
    val = official_math_val + gsm_val
    train_groups = {group_fingerprint(row) for row in train}
    val_groups_final = {group_fingerprint(row) for row in val}
    train_val_overlap = train_groups & val_groups_final
    train_test_overlap = train_groups & external_groups
    val_test_overlap = val_groups_final & external_groups
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "原题级隔离断言失败: "
            f"train-val={len(train_val_overlap)}, "
            f"train-test={len(train_test_overlap)}, val-test={len(val_test_overlap)}")

    rng.shuffle(train)
    rng.shuffle(val)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_val, val)
    write_jsonl(args.out_combined, train + val)

    manifest = {
        "recipe_version": RECIPE_VERSION,
        "seed": args.seed,
        "split_unit": "normalized original_question group",
        "normalization": "NFKC + casefold + cosmetic LaTeX removal + all whitespace removal",
        "math_val_ratio": args.math_val_ratio,
        "max_variants_per_group": args.max_variants_per_group,
        "train_total": len(train),
        "val_total": len(val),
        "eval_samples": len(val),
        "train_original_groups": len(train_groups),
        "val_original_groups": len(val_groups_final),
        "train_counts": counts(train),
        "val_counts": counts(val),
        "source_pool_counts": {
            "meta_math": len(meta_math_pool),
            "meta_gsm8k": len(meta_gsm_pool),
        },
        "selection_skipped": {
            "official": dict(official_skipped),
            "meta_math": meta_math_stats,
            "gsm_train": gsm_train_stats,
            "gsm_val": gsm_val_stats,
        },
        "external_test_groups": {
            name: {"count": len(keys), "sha256": sha256_keys(keys)}
            for name, keys in external_sets.items()
        },
        "isolation": {
            "train_val_original_group_overlap": 0,
            "train_external_test_original_group_overlap": 0,
            "val_external_test_original_group_overlap": 0,
            "asserted": True,
        },
    }
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print(f"训练集 {len(train)} 条 / {len(train_groups)} 原题组: {manifest['train_counts']}")
    print(f"验证集 {len(val)} 条 / {len(val_groups_final)} 原题组: {manifest['val_counts']}")
    print("隔离断言: train∩val=0, train∩external_test=0, val∩external_test=0")
    print(f"合并文件: {args.out_combined} (末尾 {len(val)} 条为验证集)")
    print(f"清单: {args.manifest}")


if __name__ == "__main__":
    main()
