# -*- coding: utf-8 -*-
"""
MetaMathQA_GSM8K_zh → 桥训练格式转换

输入: data/metamath_gsm8k_zh.json(365MB, 231,685 条)
      字段: query_zh(中文题) / response_zh(中文分步思路, 以 "#### 答案" 结尾) / type
输出: data/train_metamath.jsonl, 每行 {"text": "题目\\n解题思路:思路"}
      —— 与 bridge_train.py 的 split_sender_target 切分约定一致
      (发送者只见题目, 接收者生成思路)

用法:
  python convert_metamath.py                    # 全量 231685 条
  python convert_metamath.py --limit 20000 --type GSM_AnsAug
"""

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="MetaMathQA_GSM8K_zh → 训练 jsonl")
    parser.add_argument("--src", default="data/metamath_gsm8k_zh.json")
    parser.add_argument("--out", default="data/train_metamath.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="最多转换条数(0=全部)")
    parser.add_argument("--type", default="", help="只保留指定 type(如 GSM_AnsAug),空=全部")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)
    print(f"读取 {len(data)} 条")

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in data:
            if args.type and r.get("type") != args.type:
                continue
            q = (r.get("query_zh") or "").strip()
            a = (r.get("response_zh") or "").strip()
            if not q or not a:
                continue
            f.write(json.dumps({"text": f"{q}\n解题思路:{a}"}, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break
    print(f"完成: 写出 {written} 条 → {args.out}")


if __name__ == "__main__":
    main()
