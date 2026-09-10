# -*- coding: utf-8 -*-
"""
一键数据准备: 建 data/ → 下载 MetaMathQA(中英) → 转 jsonl → 混合切分

数据来源:
  1. 中文: MetaMathQA_GSM8K_zh(231,685 条 GSM8K 中文, hf-mirror) → data/train_metamath.jsonl
  2. 英文: MetaMathQA-395K(hf-mirror) → 按 type 拆成两个池:
       MATH 子集 → data/metamath_math.jsonl, GSM8K 子集 → data/metamath_gsm8k.jsonl
混合(默认): 中文 GSM8K 10000 + 英文 MATH 5000 + 英文 GSM8K 5000, 验证集=各池 20%

已存在的文件默认跳过(幂等), 用 --force 强制重新下载/转换。

用法(在 BES 目录下执行):
  python scripts/prepare_data.py    # 默认: 中1万 + 英MATH5k + 英GSM8K5k
  python scripts/prepare_data.py --zh_train 500 --en_math_train 200 --en_gsm8k_train 200
  python scripts/prepare_data.py --skip_mix   # 只下载+转换, 不混合
"""

import argparse
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 下载源(可被 --zh_url / --meta_url 覆盖; 支持逗号分隔多个备选地址)
ZH_URLS = [
    "https://hf-mirror.com/datasets/meta-math/MetaMathQA_GSM8K_zh/resolve/main/MetaMathQA_GSM8K_zh.json",
    "https://huggingface.co/datasets/meta-math/MetaMathQA_GSM8K_zh/resolve/main/MetaMathQA_GSM8K_zh.json",
]
META_URLS = [
    "https://hf-mirror.com/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json",
    "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json",
]

ZH_RAW = "data/metamath_gsm8k_zh.json"       # 中文原始 json
ZH_JSONL = "data/train_metamath.jsonl"       # 中文 GSM8K 池
META_RAW = "data/.cache/math/MetaMathQA-395K.json"   # MetaMathQA 原始 json(~377MB)
META_MATH_JSONL = "data/metamath_math.jsonl"         # 英文 MATH 池
META_GSM8K_JSONL = "data/metamath_gsm8k.jsonl"       # 英文 GSM8K 池


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
            f.write(json.dumps({"prompt": f"{q}\n解题思路:", "response": a},
                               ensure_ascii=False) + "\n")
            written += 1
    print(f"  中文写出 {written} 条")


def convert_metamath(src: str, math_out: str, gsm8k_out: str, force: bool):
    """把 MetaMathQA-395K.json 按 type 拆成 MATH 池与 GSM8K 池, 转 prompt/response"""
    if os.path.exists(math_out) and os.path.exists(gsm8k_out) and not force:
        print(f"  已存在, 跳过转换: {math_out} / {gsm8k_out}")
        return
    print(f"  转换 MetaMathQA: {src} → MATH 池 + GSM8K 池")
    data = json.load(open(src, encoding="utf-8"))
    mw = gw = 0
    with open(math_out, "w", encoding="utf-8") as fm, \
         open(gsm8k_out, "w", encoding="utf-8") as fg:
        for r in data:
            t = r.get("type") or ""
            q = (r.get("query") or "").strip()
            a = (r.get("response") or "").strip()
            if not q or not a:
                continue
            line = json.dumps({"prompt": f"{q}\n解题思路:", "response": a},
                              ensure_ascii=False) + "\n"
            if t.startswith("MATH"):
                fm.write(line)
                mw += 1
            elif t.startswith("GSM"):
                fg.write(line)
                gw += 1
    print(f"  MetaMathQA 英文: MATH {mw} 条, GSM8K {gw} 条")


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
    parser = argparse.ArgumentParser(description="一键准备中英数学训练数据(MetaMathQA)")
    parser.add_argument("--zh_url", default=",".join(ZH_URLS),
                        help="中文数据集地址(逗号分隔多个备选)")
    parser.add_argument("--meta_url", default=",".join(META_URLS),
                        help="英文 MetaMathQA 地址(逗号分隔多个备选)")
    parser.add_argument("--zh_train", type=int, default=10000, help="中文 GSM8K 训练条数")
    parser.add_argument("--en_math_train", type=int, default=5000, help="英文 MATH 训练条数")
    parser.add_argument("--en_gsm8k_train", type=int, default=5000, help="英文 GSM8K 训练条数")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="每个池子留出的验证比例(默认 20%%)")
    parser.add_argument("--out_train", default="data/mix_train.jsonl")
    parser.add_argument("--out_val", default="data/mix_val.jsonl")
    parser.add_argument("--out_combined", default="data/mix_all.jsonl",
                        help="训练+验证拼成一个文件(验证集在末尾)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="强制重新下载与转换")
    parser.add_argument("--skip_zh", action="store_true", help="跳过中文")
    parser.add_argument("--skip_en", action="store_true", help="跳过英文")
    parser.add_argument("--skip_mix", action="store_true", help="只下载+转换, 不混合")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    print("① 建 data/ 目录 ...")

    if not args.skip_zh:
        print("② 下载中文 MetaMathQA_GSM8K_zh ...")
        zh_urls = [u.strip() for u in args.zh_url.split(",") if u.strip()]
        fetch(zh_urls, ZH_RAW, args.force)
        convert_zh(ZH_RAW, ZH_JSONL, args.force)

    if not args.skip_en:
        print("③ 下载英文 MetaMathQA ...")
        meta_urls = [u.strip() for u in args.meta_url.split(",") if u.strip()]
        fetch(meta_urls, META_RAW, args.force)
        convert_metamath(META_RAW, META_MATH_JSONL, META_GSM8K_JSONL, args.force)

    if args.skip_mix:
        print("完成(已跳过混合)")
        return

    print("④ 混合切分训练集 / 验证集 ...")
    rng = random.Random(args.seed)
    train, val = [], []

    def sample(path, train_n, name):
        pool = read_lines(path)
        n_val = int(round(train_n * args.val_ratio))
        t, v = split_pool(pool, train_n, n_val, rng)
        print(f"  {name}: 训练 {len(t)} + 验证 {len(v)}")
        return t, v

    if not args.skip_zh:
        t, v = sample(ZH_JSONL, args.zh_train, "中文GSM8K")
        train += t
        val += v
    if not args.skip_en:
        t, v = sample(META_MATH_JSONL, args.en_math_train, "英文MATH")
        train += t
        val += v
        t, v = sample(META_GSM8K_JSONL, args.en_gsm8k_train, "英文GSM8K")
        train += t
        val += v

    rng.shuffle(train)
    rng.shuffle(val)
    write_lines(args.out_train, train)
    write_lines(args.out_val, val)
    print(f"训练集: {len(train)} 条 → {args.out_train}")
    print(f"验证集: {len(val)} 条 → {args.out_val}")

    if args.out_combined:
        write_lines(args.out_combined, train + val)
        print(f"合并文件: {len(train) + len(val)} 条(验证集在末尾) → {args.out_combined}")
        print(f"\n训练命令(验证集大小 {len(val)}):")
        print(f"  python core_training/train_fusion.py --data {args.out_combined} "
              f"--eval_samples {len(val)} --epochs 3 --patience 3")


if __name__ == "__main__":
    main()
