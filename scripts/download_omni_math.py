# -*- coding: utf-8 -*-
"""下载并规范化 Omni-MATH 官方测试集。

默认使用官方 ``omni-math-rule`` 子集，便于本地规则判分。完整 4428 题版本
可用 ``--variant full`` 下载，但复杂答案建议用官方 judge 复核。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
import urllib.request
from collections import Counter


SOURCES = {
    "rule": {
        "url": ("https://raw.githubusercontent.com/KbsdJames/"
                "omni-math-rule/main/omni_math_rule.jsonl"),
        "cache": "data/.cache/math/omni_math_rule.jsonl",
        "out": "data/omni_math_rule_test.jsonl",
        "manifest": "data/omni_math_rule_manifest.json",
        "expected": 2821,
    },
    "full": {
        "url": ("https://raw.githubusercontent.com/KbsdJames/"
                "Omni-MATH/main/Omni-Math.jsonl"),
        "cache": "data/.cache/math/omni_math_full.jsonl",
        "out": "data/omni_math_full_test.jsonl",
        "manifest": "data/omni_math_full_manifest.json",
        "expected": 4428,
    },
}


def _download(url, destination):
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".",
        suffix=".part", dir=os.path.dirname(destination) or ".")
    os.close(fd)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "BES-Omni-MATH-downloader/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(temporary, "wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                print(f"\r下载 {total / 1024 / 1024:.1f} MiB", end="", flush=True)
        print()
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fingerprint(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(raw_path, out_path):
    rows = []
    seen = set()
    duplicates = 0
    difficulty_counts = Counter()
    band_counts = Counter()
    source_counts = Counter()
    with open(raw_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = str(row.get("problem", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not problem or not answer:
                raise RuntimeError(f"{raw_path}:{line_no} 缺 problem/answer")
            number = float(row.get("difficulty"))
            if not math.isfinite(number) or not 1 <= number <= 10:
                raise RuntimeError(
                    f"{raw_path}:{line_no} difficulty 不在 1..10: {number}")
            band = min(10, int(math.floor(number)))
            fingerprint = _fingerprint(problem)
            if fingerprint in seen:
                duplicates += 1
            seen.add(fingerprint)
            stable_id = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:12]
            normalized = {
                "benchmark_id": f"omni_math_{stable_id}",
                "problem": problem,
                "answer": answer,
                "difficulty": number,
                "difficulty_band": band,
                "domain": row.get("domain", []),
                "source": row.get("source", ""),
            }
            if row.get("solution") is not None:
                normalized["solution"] = row["solution"]
            rows.append(normalized)
            difficulty_counts[f"{number:g}"] += 1
            band_counts[band] += 1
            source_counts[str(row.get("source", "unknown"))] += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".",
        suffix=".part", dir=os.path.dirname(out_path) or ".", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return rows, duplicates, difficulty_counts, band_counts, source_counts


def main():
    parser = argparse.ArgumentParser(description="下载并规范化 Omni-MATH 测试集")
    parser.add_argument("--variant", choices=sorted(SOURCES), default="rule",
                        help="rule=官方规则判分子集（默认）；full=完整 4428 题")
    parser.add_argument("--url", default="", help="自定义下载地址/镜像")
    parser.add_argument("--cache", default="", help="原始 JSONL 缓存位置")
    parser.add_argument("--out", default="", help="规范化测试集输出位置")
    parser.add_argument("--manifest", default="", help="清单 JSON 输出位置")
    parser.add_argument("--force", action="store_true", help="重新下载原始文件")
    args = parser.parse_args()

    spec = SOURCES[args.variant]
    url = args.url or spec["url"]
    cache_path = args.cache or spec["cache"]
    out_path = args.out or spec["out"]
    manifest_path = args.manifest or spec["manifest"]

    if args.force or not os.path.isfile(cache_path) or os.path.getsize(cache_path) == 0:
        print(f"下载 Omni-MATH-{args.variant}: {url}")
        _download(url, cache_path)
    else:
        print(f"原始文件已存在，复用: {cache_path}")

    rows, duplicates, difficulties, bands, sources = _normalize(cache_path, out_path)
    if len(rows) != spec["expected"]:
        print(f"[警告] 官方预期约 {spec['expected']} 条，当前读取 {len(rows)} 条；"
              "请检查上游版本是否变化")

    manifest = {
        "benchmark": "Omni-MATH",
        "variant": args.variant,
        "source_url": url,
        "raw_path": cache_path,
        "normalized_path": out_path,
        "raw_sha256": _sha256(cache_path),
        "records": len(rows),
        "exact_problem_duplicates": duplicates,
        "difficulty_counts": {
            k: difficulties[k] for k in sorted(difficulties, key=float)
        },
        "difficulty_band_counts": {str(k): bands[k] for k in sorted(bands)},
        "source_counts": dict(sorted(sources.items())),
        "evaluation_note": (
            "rule 版本适合规则判分；正式论文结果仍应与 Omni-MATH 官方 evaluator 对齐"
        ),
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"完成: {len(rows)} 条 -> {out_path}")
    print("整数难度带分布:",
          " ".join(f"L{k}={bands[k]}" for k in sorted(bands)))
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
