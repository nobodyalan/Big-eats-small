# -*- coding: utf-8 -*-
"""
MATH test 划分评测: 融合 vs baseline 的最终答案准确率

数据: Hendrycks MATH test 划分(5000 题), 从 data/.cache/math/MATH.zip 直接读取
      (zip 缺失时自动从阿里云 OSS 下载)。
判分: 抽取生成文本里最后一个 \\boxed{...} 作为最终答案, 与标准答案做
      「归一化精确匹配 → 数值容差 → SymPy 代数等价」三级判定。
对比: 同一道题分别用 旁路开(fusion) 与 旁路关(baseline) 生成, 统计各自准确率。

用法(在 BES 目录下执行):
  python eval/eval_math.py --ckpt cache/fusion_adapter.pt --limit 200          # 随机 200 题
  python eval/eval_math.py --ckpt cache/fusion_adapter.pt --limit 0            # 全量 5000 题
  python eval/eval_math.py --ckpt cache/fusion_adapter.pt --limit 200 --fusion_only
"""

import argparse
import json
import os
import random
import re
import sys
import time
import zipfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from main import (Config, attach_fusion, load_fusion, resolve_dtype,
                  resolve_model_path, build_prompt_text)
from download_math import download_file

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ZIP_URL = "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/math/MATH.zip"
ZIP_PATH = "data/.cache/math/MATH.zip"


# ────────────────────────────── 判分工具 ──────────────────────────────
def extract_boxed(text: str):
    """取文本中最后一个 \\boxed{...} 的内容(平衡括号); 没有则返回 None"""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\dfrac|\\tfrac|\\frac", r"\\frac", s)
    s = re.sub(r"\\,|\\!|\\;|\\:|\\ ", "", s)
    s = re.sub(r"\\cdot|\\times", "*", s)
    s = re.sub(r"\s+", "", s)          # 去掉所有空白
    return s


def _frac_to_py(s: str) -> str:
    """把 \frac{a}{b} 转成 ((a)/(b)) (一层花括号, 不支持嵌套; 反复替换直到稳定)"""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    return s


def to_sympy(s: str):
    """把归一化后的答案转成 SymPy 表达式(不需要 antlr4)"""
    import sympy
    t = _frac_to_py(s)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t)
    t = t.replace(r"\pi", "pi")
    t = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", t)   # 隐式乘法: 2x / 2pi / 2(3)
    return sympy.sympify(t)


def answers_equal(pred, gold) -> bool:
    """分层判分: 归一化精确匹配 → SymPy 代数等价 → 数值容差"""
    if pred is None or gold is None:
        return False
    a = normalize_answer(pred)
    b = normalize_answer(gold)
    if a == b:
        return True
    try:
        import sympy
        ea = to_sympy(a)
        eb = to_sympy(b)
        if sympy.simplify(ea - eb) == 0:
            return True
        if ea.is_number and eb.is_number:
            return abs(float(ea) - float(eb)) < 1e-6
    except Exception:
        pass
    # 纯数值兜底(即使 sympy 解析不了, 也试试直接转浮点比较)
    try:
        return abs(float(_frac_to_py(a)) - float(_frac_to_py(b))) < 1e-6
    except Exception:
        pass
    return False


# ────────────────────────────── GSM8K 判分 ──────────────────────────────
def extract_gold_gsm8k(answer: str):
    """GSM8K 标准答案形如 '...#### 72', 取 #### 后面的内容"""
    if not answer:
        return None
    m = re.findall(r"####\s*(.+)", answer)
    return m[-1].strip() if m else None


def extract_pred_gsm8k(text: str):
    """优先取 \boxed{}, 否则取文本最后一个数字"""
    b = extract_boxed(text)
    if b is not None:
        return b
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1] if nums else None


def gsm8k_equal(pred, gold) -> bool:
    """GSM8K 判分: 数值匹配(容差 1e-6); 解析不了时退化为字符串匹配"""
    if pred is None or gold is None:
        return False
    def to_num(s):
        return float(str(s).replace(",", "").replace("$", "").replace("%", "").strip())
    try:
        return abs(to_num(pred) - to_num(gold)) < 1e-6
    except Exception:
        return str(pred).strip() == str(gold).strip()


# ────────────────────────────── 数据与生成 ──────────────────────────────
def load_test_problems(zip_path: str, seed: int, limit: int):
    zf = zipfile.ZipFile(zip_path)
    prefix = "MATH/test/"
    names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".json")]
    rng = random.Random(seed)
    rng.shuffle(names)
    if limit > 0:
        names = names[:limit]
    probs = []
    for name in names:      # 保持 shuffle 后的随机顺序(不要 sorted, 否则会按学科排序)
        with zf.open(name) as f:
            probs.append(json.load(f))
    return probs


def math_prompt(problem: str) -> str:
    return (f"Solve the following math problem step by step, and put your final "
            f"answer in \\boxed{{...}}:\n\n{problem}")


def generate(model, tokenizer, prompt: str, max_new: int) -> str:
    text = build_prompt_text(tokenizer, prompt)
    tok = tokenizer(text, return_tensors="pt")
    ids = tok.input_ids.to(next(model.parameters()).device)
    amask = tok.attention_mask.to(ids.device)
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=amask, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def ensure_zip(zip_path: str):
    if not os.path.exists(zip_path):
        print(f"MATH.zip 缺失, 开始下载: {ZIP_URL}")
        download_file(ZIP_URL, zip_path)


# ────────────────────────────── GSM8K 数据 ──────────────────────────────
GSM8K_TEST_URL = "https://modelscope.cn/datasets/AI-ModelScope/gsm8k/resolve/master/main/test-00000-of-00001.parquet"
GSM8K_TEST_PATH = "data/.cache/gsm8k_test.parquet"


def ensure_gsm8k():
    if not os.path.exists(GSM8K_TEST_PATH):
        print(f"GSM8K test 缺失, 开始下载: {GSM8K_TEST_URL}")
        download_file(GSM8K_TEST_URL, GSM8K_TEST_PATH)


def load_gsm8k_test(seed: int, limit: int):
    """读 GSM8K test(1319 题), 返回 [{"problem": question, "answer": answer}]"""
    ensure_gsm8k()
    import pyarrow.parquet as pq
    t = pq.read_table(GSM8K_TEST_PATH)
    qs = t.column("question").to_pylist()
    ans = t.column("answer").to_pylist()
    items = [{"problem": q, "answer": a} for q, a in zip(qs, ans)]
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit > 0:
        items = items[:limit]
    return items


# ────────────────────────────── 主流程 ──────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MATH / GSM8K 分层评测: fusion vs baseline")
    parser.add_argument("--ckpt", default="", help="训练好的 fusion 权重(空=用初始恒等旁路)")
    parser.add_argument("--bench", default="both", choices=["math", "gsm8k", "both"],
                        help="评测哪个数据集")
    parser.add_argument("--limit", type=int, default=200, help="每个数据集最多评测多少题(0=全部)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--zip", default=ZIP_PATH)
    parser.add_argument("--fusion_only", action="store_true", help="只测 fusion, 跳过 baseline")
    parser.add_argument("--out_dir", default="eval_results")
    args = parser.parse_args()

    # ── 选择 benchmark(数据 + 判分函数) ──
    benches = []
    if args.bench in ("math", "both"):
        ensure_zip(args.zip)
        benches.append(("MATH", load_test_problems(args.zip, args.seed, args.limit),
                        lambda r: extract_boxed(r.get("solution", "")),
                        extract_boxed, answers_equal))
    if args.bench in ("gsm8k", "both"):
        benches.append(("GSM8K", load_gsm8k_test(args.seed, args.limit),
                        lambda r: extract_gold_gsm8k(r.get("answer", "")),
                        extract_pred_gsm8k, gsm8k_equal))

    # ── 加载模型 + 旁路 ──
    cfg = Config()
    dt = resolve_dtype(cfg.dtype)
    small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    tokenizer = AutoTokenizer.from_pretrained(small_path)
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).cuda().eval()
    large = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt).cuda().eval()
    fusion = attach_fusion(large, small, cfg)
    if args.ckpt:
        load_fusion(fusion, args.ckpt)
    gate = float(torch.sigmoid(fusion.gate_logit).detach())
    print(f"旁路 gate = {gate:.5f} ({'训练后权重' if args.ckpt else '初始恒等'})")

    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries = {}
    for name, items, gold_fn, pred_fn, score_fn in benches:
        results = []
        t0 = time.time()
        n = len(items)
        print(f"\n===== 评测 {name} ({n} 题, seed={args.seed}) =====")
        for i, r in enumerate(items, 1):
            problem = r.get("problem", "")
            gold = gold_fn(r)
            prompt = math_prompt(problem)
            fusion.enabled = True
            pred_f = generate(large, tokenizer, prompt, args.max_new)
            fusion.enabled = False
            pred_b = "" if args.fusion_only else generate(large, tokenizer, prompt, args.max_new)
            fusion.enabled = True
            ans_f = pred_fn(pred_f)
            ok_f = score_fn(ans_f, gold)
            rec = {"id": i, "problem": problem, "gold": gold,
                   "fusion_pred": ans_f, "fusion_correct": ok_f}
            if not args.fusion_only:
                ans_b = pred_fn(pred_b)
                ok_b = score_fn(ans_b, gold)
                rec.update({"baseline_pred": ans_b, "baseline_correct": ok_b})
            results.append(rec)
            acc_f = sum(1 for x in results if x["fusion_correct"]) / i
            line = f"[{name} {i}/{n}] fusion_acc={acc_f:.3f}"
            if not args.fusion_only:
                acc_b = sum(1 for x in results if x["baseline_correct"]) / i
                line += f" | baseline_acc={acc_b:.3f}"
            print(line, flush=True)

        n_total = len(results)
        f_correct = sum(1 for x in results if x["fusion_correct"])
        summary = {"bench": name, "n": n_total, "fusion_correct": f_correct,
                   "fusion_acc": round(f_correct / n_total, 4) if n_total else 0}
        print("=" * 62)
        print(f"[{name}] fusion 准确率: {summary['fusion_acc']:.2%} ({f_correct}/{n_total})")
        if not args.fusion_only:
            b_correct = sum(1 for x in results if x["baseline_correct"])
            summary.update({"baseline_correct": b_correct,
                            "baseline_acc": round(b_correct / n_total, 4)})
            print(f"[{name}] baseline 准确率: {summary['baseline_acc']:.2%} ({b_correct}/{n_total})")
            n_diff = sum(1 for x in results if x.get("fusion_pred") != x.get("baseline_pred"))
            summary["n_answer_changed"] = n_diff
            print(f"[{name}] 答案不同题数: {n_diff}/{n_total}")
            print(f"[{name}] 差值(fusion - baseline): {summary['fusion_acc'] - summary['baseline_acc']:+.2%}")
        print(f"[{name}] 耗时: {time.time() - t0:.0f}s")
        all_summaries[name] = summary

        path = os.path.join(args.out_dir,
                            f"eval_{name.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "seed": args.seed, "summary": summary,
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {path}")

    if len(benches) > 1:
        print("\n" + "=" * 62)
        print("分层汇总:")
        for name, s in all_summaries.items():
            if args.fusion_only:
                print(f"  {name}: fusion {s['fusion_acc']:.2%}")
            else:
                print(f"  {name}: fusion {s['fusion_acc']:.2%} | baseline {s['baseline_acc']:.2%} "
                      f"| 差值 {s['fusion_acc'] - s['baseline_acc']:+.2%}")


if __name__ == "__main__":
    main()
