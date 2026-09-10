# -*- coding: utf-8 -*-
"""
MATH (Hendrycks competition_math) 下载并转成桥训练 jsonl

默认从阿里云 OSS 直下 MATH.zip(modelscope 官方镜像 AI-ModelScope/competition_math 同源,
国内直连快), 直接读取 zip 内每个 problem 的 JSON 转成一行 {"text": "题目\\n解题思路:答案"}。
输出格式与 core_training/train_fusion.py 的 load_texts 约定一致。

用法(在 BES 目录下执行):
  python scripts/download_math.py                 # train 全量 7500 条 → data/math_train.jsonl
  python scripts/download_math.py --split test    # test 5000 条(留作泛化评测, 别混进训练)
  python scripts/download_math.py --limit 200     # 只转前 200 条试跑
  python scripts/download_math.py --source hf     # 改用 HuggingFace datasets 下载(需已装 datasets)

数据源结构:
  MATH.zip 内为 MATH/{train,test}/<学科>/<编号>.json, 每个 json 含
  problem / solution / level / type 四个字段。train=7500 条, test=5000 条。
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OSS_URL = "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/math/MATH.zip"
HF_ID = "hendrycks/competition_math"
DEFAULT_CACHE = "data/.cache/math"


def download_file(url: str, dest: str) -> str:
    """urllib 下载, 带简单进度显示; 失败时若装了 requests 则用它重试一次"""
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r下载中 {done/1024/1024:.1f}/{total/1024/1024:.1f} MB "
                              f"({done * 100 // total}%)", end="", flush=True)
            print()
    except Exception as e:
        print(f"\nurllib 下载失败: {e}")
        try:
            import requests
            print("改用 requests 重试 ...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 256):
                        f.write(chunk)
        except Exception as e2:
            raise RuntimeError(f"下载失败: {e2}") from e2
    os.replace(tmp, dest)
    return dest


def iter_math_zip(zip_path: str, split: str):
    """直接从 MATH.zip 读取 MATH/{split}/*.json, 逐个产出记录 dict
    (不落盘解压, 避免 Windows 下文件句柄占用导致清理失败)"""
    prefix = f"MATH/{split}/"
    zf = zipfile.ZipFile(zip_path)
    names = [n for n in zf.namelist()
             if n.startswith(prefix) and n.endswith(".json")]
    if not names:
        raise FileNotFoundError(f"zip 内未找到 {prefix}*.json, 请检查 --split 名")
    for name in sorted(names):
        with zf.open(name) as f:
            rec = json.load(f)
        yield rec


def iter_hf(split: str):
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split=split, trust_remote_code=True)
    for ex in ds:
        yield {"problem": ex.get("problem", ""), "solution": ex.get("solution", "")}


def convert(recs, out_path: str, limit: int) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in recs:
            q = (rec.get("problem") or "").strip()
            a = (rec.get("solution") or "").strip()
            if not q or not a:
                continue
            f.write(json.dumps({"text": f"{q}\n解题思路:{a}"},
                               ensure_ascii=False) + "\n")
            written += 1
            if limit and written >= limit:
                break
    return written


def main():
    parser = argparse.ArgumentParser(description="下载 MATH 并转成桥训练 jsonl")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="train=训练集(7500), test=测试集(5000)")
    parser.add_argument("--out", default="data/math_train.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=0, help="最多转换条数(0=全部)")
    parser.add_argument("--source", default="oss", choices=["oss", "hf"],
                        help="oss=阿里云 OSS 直连(默认), hf=HuggingFace datasets")
    parser.add_argument("--url", default=OSS_URL, help="OSS 直连的 MATH.zip 地址")
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE,
                        help="MATH.zip 下载缓存目录")
    args = parser.parse_args()

    if args.source == "hf":
        recs = iter_hf(args.split)
    else:
        cache_root = Path(args.cache_dir)
        zip_path = cache_root / "MATH.zip"
        if not zip_path.exists():
            print(f"下载 MATH.zip ← {args.url}")
            download_file(args.url, str(zip_path))
        recs = iter_math_zip(str(zip_path), args.split)

    n = convert(recs, args.out, args.limit)
    print(f"完成: 写出 {n} 条 → {args.out}")


if __name__ == "__main__":
    main()
