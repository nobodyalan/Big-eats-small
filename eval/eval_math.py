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
    parser.add_argument("--lora_ckpt", default="",
                        help="LoRA 权重目录(给定则评测 LoRA 模型; 与 --ckpt 同给=并行评测两者)")
    parser.add_argument("--small_start", type=int, default=None,
                        help="0.6B 旁路起始层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--small_end", type=int, default=None,
                        help="0.6B 旁路结束层(含, 0-based); -1=最后一层; None=自动 2/3 位置")
    parser.add_argument("--fusion_device", default="", help="旁路模型设备(空=自动 cuda:0)")
    parser.add_argument("--lora_device", default="", help="LoRA 模型设备(空=自动另一张卡)")
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

    # ── 加载模型(旁路 / LoRA / 两者并行) ──
    cfg = Config()
    if args.small_start is not None:
        cfg.fusion_small_start = args.small_start
    if args.small_end is not None:
        cfg.fusion_small_end = args.small_end
    dt = resolve_dtype(cfg.dtype)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)

    # 两个模型可同时挂载(旁路放 fusion 设备, LoRA 放另一张卡), 一次性并行验证两者正确率
    have_fusion = (not args.lora_ckpt) or bool(args.ckpt)   # 只给 --lora_ckpt 时不上旁路
    have_lora = bool(args.lora_ckpt)

    tokenizer = None
    large_f = large_l = None
    fusion = None

    def _resolve_dev(pref: str, fallback: str) -> str:
        d = pref or fallback
        if d.startswith("cuda") and torch.cuda.is_available():
            idx = int(d.split(":")[-1]) if ":" in d else 0
            if idx >= torch.cuda.device_count():
                print(f"    [警告] 设备 {d} 不存在, 退回 cuda:0")
                return "cuda:0"
        return d

    if have_fusion:
        small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
        tokenizer = AutoTokenizer.from_pretrained(small_path)
        fdev = _resolve_dev(args.fusion_device, "cuda:0")
        small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).to(fdev).eval()
        large_f = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt).to(fdev).eval()
        fusion = attach_fusion(large_f, small, cfg)
        if args.ckpt:
            load_fusion(fusion, args.ckpt)
        gate = float(torch.sigmoid(fusion.gate_logit).detach())
        print(f"旁路 gate = {gate:.5f} ({'训练后权重' if args.ckpt else '初始恒等'}) "
              f"(设备 {fdev})")

    if have_lora:
        from peft import PeftModel
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(large_path)
        ldev = _resolve_dev(args.lora_device,
                            "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
        large_l = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt).to(ldev).eval()
        large_l = PeftModel.from_pretrained(large_l, args.lora_ckpt)
        # 训练存的是 fp32 主权重(配合 autocast bf16 前向); 推理统一回 base 的 dt,
        # 否则 fp32 LoRA + bf16 base 混合 dtype, 且 PEFT 可能把部分 base 参数带到 fp32
        large_l = large_l.to(dt)
        large_l.eval()
        print(f"已加载 LoRA: {args.lora_ckpt} (设备 {ldev}, dtype {dt})")

    def set_fusion(on: bool):
        if fusion is not None:
            fusion.enabled = on

    def set_lora(on: bool):
        if large_l is not None:
            if on:
                large_l.enable_adapter_layers()
            else:
                large_l.disable_adapter_layers()

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
            rec = {"id": i, "problem": problem, "gold": gold}

            # ① 纯 4B baseline(旁路关 = LoRA 关, 二者等价, 只算一次)
            if not args.fusion_only:
                try:
                    if large_f is not None:
                        set_fusion(False)
                        pred_base = generate(large_f, tokenizer, prompt, args.max_new)
                    else:
                        set_lora(False)
                        pred_base = generate(large_l, tokenizer, prompt, args.max_new)
                finally:
                    # generate 抛异常也要恢复为开启, 防状态残留污染后续结果
                    set_fusion(True)
                    set_lora(True)
                ans_base = pred_fn(pred_base)
                rec["baseline_pred"] = ans_base
                rec["baseline_correct"] = score_fn(ans_base, gold)

            # ② 旁路开
            if large_f is not None:
                set_fusion(True)
                pred_f = generate(large_f, tokenizer, prompt, args.max_new)
                ans_f = pred_fn(pred_f)
                rec["fusion_pred"] = ans_f
                rec["fusion_correct"] = score_fn(ans_f, gold)

            # ③ LoRA 开
            if large_l is not None:
                set_lora(True)
                pred_l = generate(large_l, tokenizer, prompt, args.max_new)
                ans_l = pred_fn(pred_l)
                rec["lora_pred"] = ans_l
                rec["lora_correct"] = score_fn(ans_l, gold)

            results.append(rec)
            parts = [f"[{name} {i}/{n}]"]
            if large_f is not None:
                acc_f = sum(1 for x in results if x.get("fusion_correct")) / i
                parts.append(f"fusion={acc_f:.3f}")
            if large_l is not None:
                acc_l = sum(1 for x in results if x.get("lora_correct")) / i
                parts.append(f"lora={acc_l:.3f}")
            if not args.fusion_only:
                acc_b = sum(1 for x in results if x.get("baseline_correct")) / i
                parts.append(f"baseline={acc_b:.3f}")
            print(" | ".join(parts), flush=True)

        n_total = len(results)
        summary = {"bench": name, "n": n_total}
        print("=" * 62)
        for label in ("baseline", "fusion", "lora"):
            key = f"{label}_correct"
            if any(key in x for x in results):
                c = sum(1 for x in results if x[key])
                summary[key] = c
                summary[f"{label}_acc"] = round(c / n_total, 4) if n_total else 0
                print(f"[{name}] {label} 准确率: {summary[f'{label}_acc']:.2%} ({c}/{n_total})")
        if results and "fusion_pred" in results[0] and "baseline_pred" in results[0]:
            n_diff = sum(1 for x in results if x.get("fusion_pred") != x.get("baseline_pred"))
            summary["n_answer_changed"] = n_diff
            print(f"[{name}] 答案不同题数(fusion vs baseline): {n_diff}/{n_total}")
            print(f"[{name}] 差值(fusion - baseline): "
                  f"{summary['fusion_acc'] - summary['baseline_acc']:+.2%}")
        if "fusion_acc" in summary and "lora_acc" in summary:
            print(f"[{name}] 差值(fusion - lora): {summary['fusion_acc'] - summary['lora_acc']:+.2%}")
        print(f"[{name}] 耗时: {time.time() - t0:.0f}s")
        all_summaries[name] = summary

        path = os.path.join(args.out_dir,
                            f"eval_{name.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "lora_ckpt": args.lora_ckpt,
                       "small_start": args.small_start, "small_end": args.small_end,
                       "seed": args.seed, "summary": summary,
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {path}")

    if len(benches) > 1:
        print("\n" + "=" * 62)
        print("分层汇总:")
        for name, s in all_summaries.items():
            cols = []
            for label in ("baseline", "fusion", "lora"):
                if f"{label}_acc" in s:
                    cols.append(f"{label} {s[f'{label}_acc']:.2%}")
            line = f"  {name}: " + " | ".join(cols)
            if "fusion_acc" in s and "baseline_acc" in s:
                line += f"  (fusion-baseline {s['fusion_acc'] - s['baseline_acc']:+.2%})"
            if "fusion_acc" in s and "lora_acc" in s:
                line += f"  (fusion-lora {s['fusion_acc'] - s['lora_acc']:+.2%})"
            print(line)


if __name__ == "__main__":
    main()
