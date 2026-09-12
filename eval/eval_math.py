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
def parse_levels(spec):
    """把 '5' / '1,2,3' / '1-3' / '4-5' 解析成 int 集合; 空/0/None=全部(None)"""
    if spec in (None, "", "0"):
        return None
    s = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            s.update(range(int(lo), int(hi) + 1))
        else:
            s.add(int(part))
    return s or None


def load_test_problems(zip_path: str, seed: int, limit: int, levels=None):
    """读 MATH test; levels 为 int 集合(如 {1,2,3})或 None=全部"""
    zf = zipfile.ZipFile(zip_path)
    prefix = "MATH/test/"
    names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".json")]
    probs = []
    for name in names:
        with zf.open(name) as f:
            p = json.load(f)
            if levels is not None:
                lv = str(p.get("level", "")).lower()
                digits = "".join(ch for ch in lv if ch.isdigit())
                if not digits or int(digits) not in levels:
                    continue
            probs.append(p)
    rng = random.Random(seed)
    rng.shuffle(probs)       # 保持随机顺序(不要 sorted, 否则会按学科排序)
    if limit > 0:
        probs = probs[:limit]
    return probs


def math_prompt(problem: str) -> str:
    return (f"Solve the following math problem step by step, and put your final "
            f"answer in \\boxed{{...}}:\n\n{problem}")


def generate(model, tokenizer, prompt: str, max_new: int) -> str:
    text = build_prompt_text(tokenizer, prompt)
    tok = tokenizer(text, return_tensors="pt")
    ids = tok.input_ids.to(next(model.parameters()).device)
    amask = tok.attention_mask.to(ids.device)
    fusion = getattr(model, "_bes_fusion", None)
    cache_started = fusion is not None and fusion.enabled
    if cache_started:
        fusion.begin_generation()
    try:
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=amask, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
    finally:
        if cache_started:
            fusion.end_generation()
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


# ────────────────────────────── AIME 数据 ──────────────────────────────
AIME_PATH = "data/aime_test.jsonl"


def load_aime_test(seed: int, limit: int):
    """读 AIME(比 MATH 更难, 整数答案 0-999), 返回 [{"problem","answer"}]"""
    if not os.path.exists(AIME_PATH):
        raise SystemExit(f"[错误] 缺 AIME 数据 {AIME_PATH}, 先跑 scripts/download_aime.py")
    items = []
    with open(AIME_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit > 0:
        items = items[:limit]
    return items


# ────────────────────────────── 主流程 ──────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MATH / GSM8K 分层评测: fusion vs baseline")
    parser.add_argument("--ckpt", default="", help="训练好的 fusion 权重(空=用初始恒等旁路)")
    parser.add_argument("--small_lora_ckpt", default="",
                        help="与 fusion 配套的小模型 LoRA 目录")
    parser.add_argument("--bridge_depth", type=int, default=1,
                        help="fusion bridge 深度，必须与 checkpoint 一致")
    parser.add_argument("--bridge_mlp_dim", type=int, default=None,
                        help="fusion bridge 中间维度，必须与 checkpoint 一致")
    parser.add_argument("--lora_ckpt", default="",
                        help="LoRA 权重目录(给定则评测 LoRA 模型; 与 --ckpt 同给=并行评测两者)")
    parser.add_argument("--small_start", type=int, default=None,
                        help="0.6B 旁路起始层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--small_end", type=int, default=None,
                        help="0.6B 旁路结束层(含, 0-based); -1=最后一层; None=自动 2/3 位置")
    parser.add_argument("--large_start", type=int, default=None,
                        help="4B 取隐状态层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--large_end", type=int, default=None,
                        help="4B 加回残差层(含, 0-based); None=自动 2/3 位置")
    parser.add_argument("--bypass_small", action="store_true",
                        help="评测 bridge-only control checkpoint")
    parser.add_argument("--fusion_device", default="", help="旁路模型设备(空=自动 cuda:0)")
    parser.add_argument("--lora_device", default="", help="LoRA 模型设备(空=自动另一张卡)")
    parser.add_argument("--bench", default="both",
                        choices=["math", "gsm8k", "both", "aime", "segments"],
                        help="评测哪个数据集(math/gsm8k/both/aime/segments)")
    parser.add_argument("--limit", type=int, default=200, help="每个数据集最多评测多少题(0=全部)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--attn_impl", default="",
                        help="4B 注意力实现；正式评测应与训练一致")
    parser.add_argument("--zip", default=ZIP_PATH)
    parser.add_argument("--math_level", type=int, default=0,
                        help="只测 MATH 指定难度(1~5, 0=全部)")
    parser.add_argument("--math_lo", default="1-3",
                        help="segments 模式下 MATH 低级区间(如 1-3 / 1,2,3)")
    parser.add_argument("--math_hi", default="4-5",
                        help="segments 模式下 MATH 高级区间(如 4-5 / 5)")
    parser.add_argument("--fusion_only", action="store_true", help="只测 fusion, 跳过 baseline")
    parser.add_argument("--baseline_only", action="store_true",
                        help="只测纯 4B baseline；不加载小模型、bridge 或 LoRA")
    parser.add_argument("--out_dir", default="eval_results")
    parser.add_argument("--tag", default="", help="输出文件名标签(并行多模型评测时用于区分)")
    args = parser.parse_args()
    if args.fusion_only and args.baseline_only:
        parser.error("--fusion_only 与 --baseline_only 不能同时使用")
    if args.baseline_only and (args.ckpt or args.small_lora_ckpt or args.lora_ckpt):
        parser.error("--baseline_only 不应同时传入实验 checkpoint")

    # ── 选择 benchmark(数据 + 判分函数) ──
    benches = []
    if args.bench in ("math", "both"):
        ensure_zip(args.zip)
        lv = {args.math_level} if args.math_level else None
        lname = f"MATH_L{args.math_level}" if args.math_level else "MATH"
        benches.append((lname, load_test_problems(args.zip, args.seed, args.limit, levels=lv),
                        lambda r: extract_boxed(r.get("solution", "")),
                        extract_boxed, answers_equal))
    if args.bench in ("gsm8k", "both"):
        benches.append(("GSM8K", load_gsm8k_test(args.seed, args.limit),
                        lambda r: extract_gold_gsm8k(r.get("answer", "")),
                        extract_pred_gsm8k, gsm8k_equal))
    if args.bench == "aime":
        benches.append(("AIME", load_aime_test(args.seed, args.limit),
                        lambda r: str(r.get("answer", "")).strip(),
                        extract_pred_gsm8k, gsm8k_equal))
    if args.bench == "segments":
        # 三段: GSM8K + MATH 低级 + MATH 高级(同一 seed → 各段题目一致)
        ensure_zip(args.zip)
        lo, hi = parse_levels(args.math_lo), parse_levels(args.math_hi)
        benches.append(("GSM8K", load_gsm8k_test(args.seed, args.limit),
                        lambda r: extract_gold_gsm8k(r.get("answer", "")),
                        extract_pred_gsm8k, gsm8k_equal))
        benches.append((f"MATH_lo{args.math_lo}",
                        load_test_problems(args.zip, args.seed, args.limit, levels=lo),
                        lambda r: extract_boxed(r.get("solution", "")),
                        extract_boxed, answers_equal))
        benches.append((f"MATH_hi{args.math_hi}",
                        load_test_problems(args.zip, args.seed, args.limit, levels=hi),
                        lambda r: extract_boxed(r.get("solution", "")),
                        extract_boxed, answers_equal))

    # ── 加载模型(旁路 / LoRA / 两者并行) ──
    cfg = Config()
    cfg.fusion_bridge_depth = args.bridge_depth
    if args.bridge_mlp_dim is not None:
        cfg.fusion_mlp_dim = args.bridge_mlp_dim
    if args.small_start is not None:
        cfg.fusion_small_start = args.small_start
    if args.small_end is not None:
        cfg.fusion_small_end = args.small_end
    if args.large_start is not None:
        cfg.fusion_large_start = args.large_start
    if args.large_end is not None:
        cfg.fusion_large_end = args.large_end
    cfg.fusion_bypass_small = args.bypass_small
    dt = resolve_dtype(cfg.dtype)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    attn_kwargs = {"attn_implementation": args.attn_impl} if args.attn_impl else {}

    # 两个模型可同时挂载(旁路放 fusion 设备, LoRA 放另一张卡), 一次性并行验证两者正确率
    have_fusion = (not args.baseline_only) and (
        (not args.lora_ckpt) or bool(args.ckpt))  # 只给 --lora_ckpt 时不上旁路
    have_lora = (not args.baseline_only) and bool(args.lora_ckpt)

    tokenizer = None
    large_f = large_l = large_b = None
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
        tokenizer = AutoTokenizer.from_pretrained(large_path)
        fdev = _resolve_dev(args.fusion_device, "cuda:0")
        small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).to(fdev).eval()
        large_f = AutoModelForCausalLM.from_pretrained(
            large_path, dtype=dt, **attn_kwargs).to(fdev).eval()
        fusion = attach_fusion(large_f, small, cfg)
        if args.small_lora_ckpt:
            from peft import PeftModel
            # attach_fusion 已保存底层真实层引用；PEFT 原地替换这些层中的线性模块，
            # 因而手写的小模型片段 forward 会自动使用 LoRA。
            small_lora_model = PeftModel.from_pretrained(
                small, args.small_lora_ckpt, is_trainable=False).eval()
        if args.ckpt:
            load_fusion(fusion, args.ckpt)
        gate = float(fusion.scale().detach())
        print(f"旁路 scale = {gate:.5f} ({'训练后权重' if args.ckpt else '初始恒等'}) "
              f"(设备 {fdev})")

    if have_lora:
        from peft import PeftModel
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(large_path)
        ldev = _resolve_dev(args.lora_device,
                            "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
        large_l = AutoModelForCausalLM.from_pretrained(
            large_path, dtype=dt, **attn_kwargs).to(ldev).eval()
        large_l = PeftModel.from_pretrained(large_l, args.lora_ckpt)
        # 训练存的是 fp32 主权重(配合 autocast bf16 前向); 推理统一回 base 的 dt,
        # 否则 fp32 LoRA + bf16 base 混合 dtype, 且 PEFT 可能把部分 base 参数带到 fp32
        large_l = large_l.to(dt)
        large_l.eval()
        print(f"已加载 LoRA: {args.lora_ckpt} (设备 {ldev}, dtype {dt})")

    if args.baseline_only:
        tokenizer = AutoTokenizer.from_pretrained(large_path)
        bdev = _resolve_dev(args.fusion_device, "cuda:0")
        large_b = AutoModelForCausalLM.from_pretrained(
            large_path, dtype=dt, **attn_kwargs).to(bdev).eval()
        print(f"已加载纯 4B baseline (设备 {bdev}, dtype {dt})")

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
                    elif large_l is not None:
                        set_lora(False)
                        pred_base = generate(large_l, tokenizer, prompt, args.max_new)
                    else:
                        pred_base = generate(large_b, tokenizer, prompt, args.max_new)
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

        tag = f"_{args.tag}" if args.tag else ""
        path = os.path.join(args.out_dir,
                            f"eval_{name.lower()}{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "small_lora_ckpt": args.small_lora_ckpt,
                       "lora_ckpt": args.lora_ckpt,
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
