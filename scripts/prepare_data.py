# -*- coding: utf-8 -*-
"""
一键数据准备: 建 data/ 目录 → 下载中英数学数据集 → 转 jsonl → 混合切分

步骤:
  1. 建 data/ 目录
  2. 下载中文 MetaMathQA_GSM8K_zh(231,685 条, 默认走 hf-mirror 国内镜像)
     → data/metamath_gsm8k_zh.json → 转 data/train_metamath.jsonl
  3. 下载英文 Hendrycks MATH(7,500 条, 阿里云 OSS 直连)
     → data/.cache/math/MATH.zip → 转 data/math_train.jsonl
  4. 按参数混合成训练集 / 验证集(同语言不重叠, 中英整体打乱)

已存在的文件默认跳过(幂等), 用 --force 强制重新下载/转换。

用法(在 BES 目录下执行):
  python scripts/prepare_data.py                       # 全量: 中文20k + 英文6k 训练, 验证集=20%(中4k+英1.2k)
  python scripts/prepare_data.py --zh_train 500 --en_train 500 --zh_val 100 --en_val 100  # 试跑混合
  python scripts/prepare_data.py --skip_mix            # 只下载+转换, 不混合
"""

import argparse
import json
import os
import random
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 下载源(可被 --zh_url / --en_url 覆盖; zh 支持逗号分隔多个备选地址)
ZH_URLS = [
    "https://hf-mirror.com/datasets/meta-math/MetaMathQA_GSM8K_zh/resolve/main/MetaMathQA_GSM8K_zh.json",
    "https://huggingface.co/datasets/meta-math/MetaMathQA_GSM8K_zh/resolve/main/MetaMathQA_GSM8K_zh.json",
]
EN_URL = "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/math/MATH.zip"

ZH_RAW = "data/metamath_gsm8k_zh.json"     # 中文原始 json
ZH_JSONL = "data/train_metamath.jsonl"     # 中文转出的 jsonl
EN_ZIP = "data/.cache/math/MATH.zip"       # 英文 zip 缓存
EN_JSONL = "data/math_train.jsonl"         # 英文转出的 jsonl


def download_file(url: str, dest: str) -> str:
    """urllib 下载(带进度); 失败时回退 requests"""
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
                        print(f"\r  {done/1048576:.1f}/{total/1048576:.1f} MB "
                              f"({done * 100 // total}%)", end="", flush=True)
            print()
    except Exception as e:
        print(f"\n  urllib 失败: {e}, 改用 requests 重试 ...")
        import requests
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    f.write(chunk)
    os.replace(tmp, dest)
    return dest


def fetch(urls, dest: str, force: bool) -> str:
    if os.path.exists(dest) and not force:
        print(f"  已存在, 跳过下载: {dest}")
        return dest
    last = None
    for u in urls:
        try:
            print(f"  下载: {u}")
            download_file(u, dest)
            print(f"  已保存: {dest}")
            return dest
        except Exception as e:
            last = e
            print(f"  失败: {e}")
    raise RuntimeError(f"所有地址均下载失败, 最后错误: {last}")


def convert_zh(src: str, out: str, force: bool):
    if os.path.exists(out) and not force:
        print(f"  已存在, 跳过转换: {out}")
        return
    print(f"  转换中文: {src} → {out}")
    data = json.load(open(src, encoding="utf-8"))
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in data:
            q = (r.get("query_zh") or "").strip()
            a = (r.get("response_zh") or "").strip()
            if not q or not a:
                continue
            f.write(json.dumps({"text": f"{q}\n解题思路:{a}"}, ensure_ascii=False) + "\n")
            written += 1
    print(f"  中文写出 {written} 条")


def convert_en(zip_path: str, out: str, force: bool):
    if os.path.exists(out) and not force:
        print(f"  已存在, 跳过转换: {out}")
        return
    print(f"  转换英文: {zip_path} → {out}")
    prefix = "MATH/train/"
    zf = zipfile.ZipFile(zip_path)
    names = [n for n in zf.namelist()
             if n.startswith(prefix) and n.endswith(".json")]
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for name in sorted(names):
            with zf.open(name) as zff:
                r = json.load(zff)
            q = (r.get("problem") or "").strip()
            a = (r.get("solution") or "").strip()
            if not q or not a:
                continue
            f.write(json.dumps({"text": f"{q}\n解题思路:{a}"}, ensure_ascii=False) + "\n")
            written += 1
    print(f"  英文写出 {written} 条")


def read_lines(path: str):
    with open(path, encoding="utf-8") as f:
        return [ln for ln in (l.strip() for l in f) if ln]


def write_lines(path: str, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def split_pool(lines, train_n: int, val_n: int, rng: random.Random):
    """打乱后顺序切: 前 train_n 训练, 随后 val_n 验证(同语言不重叠)"""
    pool = lines[:]
    rng.shuffle(pool)
    return pool[:train_n], pool[train_n:train_n + val_n]


def main():
    parser = argparse.ArgumentParser(description="一键准备中英数学训练数据")
    parser.add_argument("--zh_url", default=",".join(ZH_URLS),
                        help="中文数据集地址(逗号分隔多个备选)")
    parser.add_argument("--en_url", default=EN_URL, help="英文 MATH.zip 地址")
    parser.add_argument("--zh_train", type=int, default=20000, help="中文训练条数")
    parser.add_argument("--en_train", type=int, default=6000, help="英文训练条数(英文池仅 7500)")
    parser.add_argument("--zh_val", type=int, default=4000, help="中文验证条数(默认=训练集的 20%)")
    parser.add_argument("--en_val", type=int, default=1200, help="英文验证条数(默认=训练集的 20%)")
    parser.add_argument("--out_train", default="data/mix_train.jsonl")
    parser.add_argument("--out_val", default="data/mix_val.jsonl")
    parser.add_argument("--out_combined", default="data/mix_all.jsonl",
                        help="训练+验证拼成一个文件(验证集在末尾), 便于 --eval_samples 监控")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="强制重新下载与转换")
    parser.add_argument("--skip_zh", action="store_true", help="跳过中文")
    parser.add_argument("--skip_en", action="store_true", help="跳过英文")
    parser.add_argument("--skip_mix", action="store_true", help="只下载+转换, 不混合")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    print("① 建 data/ 目录 ...")

    # ── 中文 ──
    if not args.skip_zh:
        print("② 下载中文 MetaMathQA_GSM8K_zh ...")
        zh_urls = [u.strip() for u in args.zh_url.split(",") if u.strip()]
        fetch(zh_urls, ZH_RAW, args.force)
        convert_zh(ZH_RAW, ZH_JSONL, args.force)

    # ── 英文 ──
    if not args.skip_en:
        print("③ 下载英文 Hendrycks MATH ...")
        fetch([args.en_url], EN_ZIP, args.force)
        convert_en(EN_ZIP, EN_JSONL, args.force)

    # ── 混合 ──
    if args.skip_mix:
        print("完成(已跳过混合)")
        return

    print("④ 混合切分训练集 / 验证集 ...")
    rng = random.Random(args.seed)
    zh_train = zh_val = []
    en_train = en_val = []

    if not args.skip_zh:
        zh = read_lines(ZH_JSONL)
        zh_train, zh_val = split_pool(zh, args.zh_train, args.zh_val, rng)
    if not args.skip_en:
        en = read_lines(EN_JSONL)
        en_train, en_val = split_pool(en, args.en_train, args.en_val, rng)

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
