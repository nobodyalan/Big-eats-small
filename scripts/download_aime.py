# -*- coding: utf-8 -*-
"""
下载 AIME(2024 + 2025, 比 MATH 更难、整数答案 0-999) → data/aime_test.jsonl

用法(服务器, 在 BES 目录下执行):
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/download_aime.py

依赖: pip install datasets (若未装)
输出: data/aime_test.jsonl, 每行 {"problem": 题目, "answer": 答案}
"""

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = "data/aime_test.jsonl"
DATASETS = ["opencompass/AIME_2024", "opencompass/AIME_2025"]


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("[错误] 未装 datasets, 先执行: pip install datasets")

    rows = []
    for ds_id in DATASETS:
        try:
            ds = load_dataset(ds_id)
        except Exception as e:
            print(f"[警告] {ds_id} 下载失败: {e}")
            continue
        split = list(ds.keys())[0] if ds else None
        if split is None:
            print(f"[警告] {ds_id} 无可用 split")
            continue
        d = ds[split]
        cols = d.column_names
        q_col = "question" if "question" in cols else cols[0]
        a_col = "answer" if "answer" in cols else cols[-1]
        before = len(rows)
        for ex in d:
            q = ex.get(q_col)
            a = ex.get(a_col)
            if q is None or a is None:
                continue
            rows.append({"problem": str(q).strip(), "answer": str(a).strip()})
        print(f"{ds_id}: 新增 {len(rows) - before} 条 (累计 {len(rows)})")

    if not rows:
        raise SystemExit("[错误] 什么都没下到, 检查网络/HF_ENDPOINT")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成: 共 {len(rows)} 条 → {OUT}")


if __name__ == "__main__":
    main()
