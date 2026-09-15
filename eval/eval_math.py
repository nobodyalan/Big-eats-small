# -*- coding: utf-8 -*-
"""
MATH test 划分评测: 融合 vs baseline 的最终答案准确率

数据: Hendrycks MATH test 划分(5000 题), 从 data/.cache/math/MATH.zip 直接读取
      (zip 缺失时自动从阿里云 OSS 下载)。
判分: 抽取生成文本里最后一个 \\boxed{...} 作为最终答案, 与标准答案做
      「归一化精确匹配 → 数值容差 → SymPy 代数等价」三级判定。
对比: 同一道题分别用 旁路开(fusion) 与 旁路关(baseline) 生成, 统计各自准确率。

用法(在 BES 目录下执行):
  python eval/eval_math.py --ckpt cache/fusion_adapter.pt                      # 默认 400 题
  python eval/eval_math.py --ckpt cache/fusion_adapter.pt --limit 0            # 全量 5000 题
  python eval/eval_math.py --ckpt cache/fusion_adapter.pt --limit 200 --fusion_only
  python eval/eval_math.py --bench segments --segment_only math_hi --limit 200 # 只跑高难档
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import zipfile
from importlib.metadata import PackageNotFoundError, version as package_version

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from main import (Config, attach_fusion, attached_fusions, load_fusion,
                  load_multi_fusion, resolve_dtype, resolve_model_path,
                  build_prompt_text)
from download_math import download_file
from omni_math_utils import (load_omni_math_items, parse_level_set,
                             summarize_by_difficulty)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ZIP_URL = "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/math/MATH.zip"
ZIP_PATH = "data/.cache/math/MATH.zip"


# ────────────────────────────── 判分工具 ──────────────────────────────
def extract_boxed(text: str):
    """取最后一个 ``\\boxed`` 的内容；兼容花括号与官方 MATH 无括号写法。"""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len(r"\boxed")
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return None
    if text[i] != "{":
        candidate = text[i:].splitlines()[0].strip()
        if candidate.startswith("$"):
            candidate = candidate[1:]
        math_end = candidate.find("$")
        if math_end >= 0:
            candidate = candidate[:math_end]
        return candidate.rstrip("。.;；").strip() or None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def extract_math_marker_answer(text: str):
    """MATH 格式诊断：boxed 缺失时，仅从明确的最终答案标记后提取。

    该函数不替代官方严格的 ``\\boxed{...}`` 判分，只用于区分“数学答案错误”与
    “答案正确但沿用了 MetaMathQA 的 ``The answer is:`` / ``####`` 格式”。
    为避免把推理过程中的最后一个数字误当答案，这里不做任意数字兜底。
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    if not text:
        return None

    matches = []
    patterns = (
        r"####\s*([^\r\n]+)",
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|:)\s*([^\r\n]+)",
    )
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None

    match = max(matches, key=lambda item: item.start())
    candidate = match.group(1).strip()
    # 常见输出是 ``The answer is: $...$.``；优先保留最后一个行内数学块。
    inline_math = re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", candidate)
    if inline_math:
        candidate = inline_math[-1].strip()
    if candidate.startswith(r"\(") and candidate.endswith(r"\)"):
        candidate = candidate[2:-2].strip()
    candidate = candidate.rstrip().rstrip(".。;,；")
    return candidate or None


def load_math_verify_runtime():
    """按需加载 Math-Verify；未启用时不增加基础评测依赖。"""
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as exc:
        raise RuntimeError(
            "启用了 --math_judge both，但当前 Python 缺少 math-verify；"
            "请使用安装了 math-verify 的评测 venv") from exc
    try:
        installed_version = package_version("math-verify")
    except PackageNotFoundError:
        installed_version = "unknown"
    return {
        "parse": parse,
        "verify": verify,
        "expr_config": ExprExtractionConfig,
        "latex_config": LatexExtractionConfig,
        "version": installed_version,
    }


def judge_with_math_verify(runtime, generated_text: str, gold_answer: str):
    """用 Math-Verify 对完整模型输出判分，并返回可 JSON 序列化的诊断。"""
    result = {
        "correct": False,
        "extracted": False,
        "prediction": [],
        "gold": [],
        "error": None,
    }
    if gold_answer is None or not str(gold_answer).strip():
        result["error"] = "missing gold answer"
        return result
    try:
        # MATH 的 gold 已由官方 solution 中的 boxed 内容提取；重新包进 boxed，
        # 能让分数、集合、区间等 LaTeX 结构按明确答案而非普通文本解析。
        gold_text = rf"\boxed{{{gold_answer}}}"
        latex_config = runtime["latex_config"](boxed_match_priority=0)
        extraction_config = (latex_config, runtime["expr_config"]())
        parsed_gold = runtime["parse"](
            gold_text, extraction_config=extraction_config, raise_on_error=True)
        # 对模型的无答案/畸形答案按官方默认语义返回空列表并计错，而不是把
        # “模型答错”误报成评测程序崩溃。gold 仍开启异常传播，防止标准答案损坏。
        parsed_prediction = runtime["parse"](
            generated_text, extraction_config=extraction_config, raise_on_error=False)
        result["gold"] = [str(item) for item in parsed_gold]
        result["prediction"] = [str(item) for item in parsed_prediction]
        result["extracted"] = bool(parsed_prediction)
        if parsed_gold and parsed_prediction:
            # Math-Verify 的 verify 非对称，官方要求 gold 在前、prediction 在后。
            result["correct"] = bool(runtime["verify"](
                parsed_gold, parsed_prediction, raise_on_error=False))
    except Exception as exc:  # 单题解析失败不应终止几小时的整轮生成评测
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


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
    names = sorted(
        n for n in zf.namelist()
        if n.startswith(prefix) and n.endswith(".json"))
    probs = []
    for name in names:
        with zf.open(name) as f:
            p = json.load(f)
            if levels is not None:
                lv = str(p.get("level", "")).lower()
                digits = "".join(ch for ch in lv if ch.isdigit())
                if not digits or int(digits) not in levels:
                    continue
            p = dict(p)
            p["benchmark_id"] = name
            probs.append(p)
    rng = random.Random(seed)
    rng.shuffle(probs)       # 保持随机顺序(不要 sorted, 否则会按学科排序)
    if limit > 0:
        probs = probs[:limit]
    return probs


def math_prompt(problem: str) -> str:
    return (f"Solve the following math problem step by step, and put your final "
            f"answer in \\boxed{{...}}:\n\n{problem}")


def attach_multi_fusion_checkpoint(model_large, model_small, payload):
    """依据 checkpoint 自带元数据挂载并加载 upstream/downstream 两条旁路。"""
    if (not isinstance(payload, dict)
            or payload.get("format") != "bes_multi_fusion_v1"):
        raise ValueError("不是 BES 双 Bridge checkpoint")
    saved_fusions = payload.get("fusions", {})
    required_names = ("upstream", "downstream")
    if any(name not in saved_fusions for name in required_names):
        raise ValueError("双 Bridge checkpoint 必须包含 upstream/downstream")
    fusion_map = {}
    for fusion_name in required_names:
        saved_meta = saved_fusions[fusion_name].get("meta", {})
        required_meta = ("large_l1", "large_l2", "small_s1", "small_s2",
                         "mlp_dim", "bridge_depth")
        missing_meta = [key for key in required_meta if key not in saved_meta]
        if missing_meta:
            raise ValueError(
                f"{fusion_name} checkpoint 缺少架构元数据: {missing_meta}")
        branch_cfg = Config()
        branch_cfg.fusion_large_start = int(saved_meta["large_l1"])
        branch_cfg.fusion_large_end = int(saved_meta["large_l2"])
        branch_cfg.fusion_injection_mode = str(
            saved_meta.get("injection_mode", "span"))
        branch_cfg.fusion_small_start = int(saved_meta["small_s1"])
        branch_cfg.fusion_small_end = int(saved_meta["small_s2"])
        branch_cfg.fusion_mlp_dim = int(saved_meta["mlp_dim"])
        branch_cfg.fusion_bridge_depth = int(saved_meta["bridge_depth"])
        branch_cfg.fusion_bypass_small = bool(saved_meta.get("bypass_small", False))
        fusion_map[fusion_name] = attach_fusion(
            model_large, model_small, branch_cfg, name=fusion_name)
    if (fusion_map["upstream"].injection_mode == "span"
            and fusion_map["downstream"].injection_mode == "span"
            and fusion_map["upstream"].l2 >= fusion_map["downstream"].l1):
        raise ValueError("双 Bridge checkpoint 的上游注入点必须早于下游捕获点")
    meta = load_multi_fusion(fusion_map, payload)
    return fusion_map, meta


def generate(model, tokenizer, prompt: str, max_new: int,
             return_meta: bool = False):
    text = build_prompt_text(tokenizer, prompt)
    tok = tokenizer(text, return_tensors="pt")
    ids = tok.input_ids.to(next(model.parameters()).device)
    amask = tok.attention_mask.to(ids.device)
    active_fusions = [fusion for fusion in attached_fusions(model)
                      if fusion.enabled]
    for fusion in active_fusions:
        fusion.begin_generation()
    try:
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=amask, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
    finally:
        for fusion in active_fusions:
            fusion.end_generation()
    generated_ids = out[0][ids.shape[1]:]
    generated_tokens = int(generated_ids.numel())
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    elif eos_ids is None:
        eos_ids = set()
    else:
        eos_ids = {int(token_id) for token_id in eos_ids}
    ended_with_eos = bool(
        generated_tokens and int(generated_ids[-1].item()) in eos_ids)
    hit_max_new_tokens = generated_tokens >= max_new and not ended_with_eos
    if ended_with_eos:
        finish_reason = "eos"
    elif hit_max_new_tokens:
        finish_reason = "length"
    else:
        finish_reason = "other"
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if not return_meta:
        return decoded
    return decoded, {
        "generated_tokens": generated_tokens,
        "finish_reason": finish_reason,
        "hit_max_new_tokens": hit_max_new_tokens,
        "max_new_tokens": int(max_new),
    }


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
    items = [{
        "benchmark_id": "gsm8k_" + hashlib.sha1(
            str(q).encode("utf-8")).hexdigest()[:12],
        "problem": q,
        "answer": a,
    } for q, a in zip(qs, ans)]
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


# ────────────────────────── Omni-MATH 数据 ──────────────────────────
OMNI_MATH_PATH = "data/omni_math_rule_test.jsonl"
SFT_VAL_PATH = "data/math_majority_v3_val.jsonl"


def _strip_sft_instruction(prompt: str) -> str:
    prefix = ("Solve the following math problem step by step, and put your final "
              "answer in \\boxed{...}:\n\n")
    prompt = str(prompt or "").strip()
    return prompt[len(prefix):].strip() if prompt.startswith(prefix) else prompt


def load_sft_validation(path: str, seed: int, limit: int,
                        limit_per_stratum: int = 0):
    """读取严格隔离的 SFT val，并作确定性分层抽样。

    v3 验证集由 official MATH L1--L5 与 MetaMathQA GSM8K 组成，因此默认
    strata 是 ``math_L1`` ... ``math_L5`` 和 ``gsm8k``。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 SFT 验证集: {path}")
    items = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem = _strip_sft_instruction(row.get("prompt", row.get("problem", "")))
            response = str(row.get("response", row.get("solution", "")))
            gold = extract_boxed(response) or extract_math_marker_answer(response)
            source = str(row.get("source", "unknown"))
            level = row.get("level")
            if "GSM" in source.upper():
                stratum = "gsm8k"
            elif level is not None:
                stratum = f"math_L{level}"
            else:
                stratum = source
            if not problem or gold is None:
                continue
            items.append({
                "benchmark_id": row.get(
                    "id", "sft_val_" + hashlib.sha1(
                        f"{line_number}:{problem}".encode("utf-8")).hexdigest()[:12]),
                "problem": problem,
                "answer": gold,
                "source": source,
                "level": level,
                "stratum": stratum,
                "original_question": row.get("original_question"),
            })
    rng = random.Random(seed)
    if limit_per_stratum > 0:
        grouped = {}
        for row in items:
            grouped.setdefault(row["stratum"], []).append(row)
        selected = []
        for stratum in sorted(grouped):
            pool = grouped[stratum]
            rng.shuffle(pool)
            selected.extend(pool[:limit_per_stratum])
        rng.shuffle(selected)
        return selected
    rng.shuffle(items)
    return items[:limit] if limit > 0 else items


# ────────────────────────────── 主流程 ──────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MATH / GSM8K 分层评测: fusion vs baseline")
    parser.add_argument("--ckpt", default="", help="训练好的 fusion 权重(空=用初始恒等旁路)")
    parser.add_argument("--multi_fusion_ckpt", default="",
                        help=("train_multi_fusion.py 保存的双 Bridge 权重；"
                              "位置/深度/MLP 维度从 checkpoint 自动恢复"))
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
    parser.add_argument("--injection_mode", choices=("span", "pre_block"),
                        default="span",
                        help="单 Bridge 注入模式；必须与 checkpoint 一致")
    parser.add_argument("--bypass_small", action="store_true",
                        help="评测 bridge-only control checkpoint")
    parser.add_argument("--fusion_device", default="", help="旁路模型设备(空=自动 cuda:0)")
    parser.add_argument("--lora_device", default="", help="LoRA 模型设备(空=自动另一张卡)")
    parser.add_argument("--bench", default="both",
                        choices=["math", "gsm8k", "both", "aime", "omni_math",
                                 "segments", "sft_val"],
                        help=("评测哪个数据集(math/gsm8k/both/aime/omni_math/"
                              "segments/sft_val)"))
    parser.add_argument("--limit", type=int, default=400, help="每个数据集最多评测多少题(0=全部)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new", type=int, default=1536)
    parser.add_argument("--attn_impl", default="flash_attention_2",
                        help="4B 注意力实现；默认固定为 flash_attention_2")
    parser.add_argument("--zip", default=ZIP_PATH)
    parser.add_argument("--math_level", type=int, default=0,
                        help="只测 MATH 指定难度(1~5, 0=全部)")
    parser.add_argument("--math_lo", default="1-3",
                        help="segments 模式下 MATH 低级区间(如 1-3 / 1,2,3)")
    parser.add_argument("--math_hi", default="4-5",
                        help="segments 模式下 MATH 高级区间(如 4-5 / 5)")
    parser.add_argument("--omni_path", default=OMNI_MATH_PATH,
                        help="规范化 Omni-MATH JSONL（默认官方 rule 子集）")
    parser.add_argument("--omni_levels", default="3-8",
                        help="Omni-MATH 难度，如 1-10 / 3-8 / 7,8；正式默认 3-8")
    parser.add_argument("--omni_limit_per_level", type=int, default=0,
                        help="每个难度抽取题数；>0 时覆盖 --limit，供等级 pilot 使用")
    parser.add_argument("--sft_val_path", default=SFT_VAL_PATH,
                        help="原题组严格隔离的 SFT 验证集 JSONL")
    parser.add_argument("--sft_val_limit_per_stratum", type=int, default=0,
                        help=("SFT val 每个层级抽取题数；>0 时覆盖 --limit。"
                              "v3 通常有 math_L1..L5、gsm8k 六层"))
    parser.add_argument(
        "--segment_only", default="all",
        choices=["all", "gsm8k", "math_lo", "math_hi"],
        help=("segments 模式下只运行指定档位；all=依次运行三档。"
              "可分别启动三个进程以并行评测同一模型"),
    )
    parser.add_argument("--fusion_only", action="store_true", help="只测 fusion, 跳过 baseline")
    parser.add_argument("--baseline_only", action="store_true",
                        help="只测纯 4B baseline；不加载小模型、bridge 或 LoRA")
    parser.add_argument("--diagnose_math_format", action="store_true",
                        help=("保留严格 boxed 指标，并额外报告 The answer is/#### "
                              "标记兜底的诊断指标"))
    parser.add_argument("--math_judge", choices=["legacy", "both"],
                        default="both",
                        help=("MATH 判分器：legacy=原严格 boxed；both=同时报告严格 "
                              "boxed 与 Math-Verify（推荐）"))
    parser.add_argument("--save_raw_output", action="store_true",
                        help="在逐题 JSON 中保存模型完整生成文本，供格式和截断排查")
    parser.add_argument("--out_dir", default="eval_results")
    parser.add_argument("--tag", default="", help="输出文件名标签(并行多模型评测时用于区分)")
    parser.add_argument("--result_path", default="",
                        help="单 benchmark 的确定输出 JSON；供并行套件避免拿错旧文件")
    parser.add_argument("--suite_run_id", default="",
                        help="并行套件本轮标识；写入结果以防误读上一次残留文件")
    args = parser.parse_args()
    if args.fusion_only and args.baseline_only:
        parser.error("--fusion_only 与 --baseline_only 不能同时使用")
    if args.ckpt and args.multi_fusion_ckpt:
        parser.error("--ckpt 与 --multi_fusion_ckpt 只能选择一个")
    if args.multi_fusion_ckpt and any(value is not None for value in (
            args.small_start, args.small_end, args.large_start, args.large_end)):
        parser.error("多 Bridge 的层位置来自 checkpoint，不要再传单 Bridge 层号")
    if args.baseline_only and (args.ckpt or args.multi_fusion_ckpt
                               or args.small_lora_ckpt or args.lora_ckpt):
        parser.error("--baseline_only 不应同时传入实验 checkpoint")
    if args.segment_only != "all" and args.bench != "segments":
        parser.error("--segment_only 仅可与 --bench segments 一起使用")
    if args.omni_limit_per_level < 0:
        parser.error("--omni_limit_per_level 必须 >= 0")
    if args.sft_val_limit_per_stratum < 0:
        parser.error("--sft_val_limit_per_stratum 必须 >= 0")
    if args.result_path and (
            args.bench == "both"
            or (args.bench == "segments" and args.segment_only == "all")):
        parser.error("--result_path 只支持恰好一个 benchmark")

    math_verify_runtime = None
    if args.math_judge == "both":
        try:
            math_verify_runtime = load_math_verify_runtime()
        except RuntimeError as exc:
            raise SystemExit(f"[错误] {exc}") from exc
        print(f"Math-Verify 已启用: version={math_verify_runtime['version']} | "
              f"mode={args.math_judge}")

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
    if args.bench == "omni_math":
        if not os.path.exists(args.omni_path):
            raise SystemExit(
                f"[错误] 缺 Omni-MATH 数据 {args.omni_path}, "
                "先跑 scripts/download_omni_math.py")
        try:
            omni_levels = parse_level_set(args.omni_levels)
            omni_items = load_omni_math_items(
                args.omni_path, seed=args.seed, limit=args.limit,
                levels=omni_levels,
                limit_per_level=args.omni_limit_per_level)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[错误] Omni-MATH 数据无效: {exc}") from exc
        level_name = args.omni_levels.replace(",", "_").replace(" ", "")
        benches.append((f"OMNI_MATH_L{level_name}", omni_items,
                        lambda r: str(r.get("answer", "")).strip(),
                        extract_boxed, answers_equal))
    if args.bench == "sft_val":
        try:
            sft_items = load_sft_validation(
                args.sft_val_path, args.seed, args.limit,
                args.sft_val_limit_per_stratum)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[错误] SFT 验证集无效: {exc}") from exc
        benches.append(("SFT_VAL", sft_items,
                        lambda r: str(r.get("answer", "")).strip(),
                        extract_boxed, answers_equal))
    if args.bench == "segments":
        # 三段可顺序运行，也可用 --segment_only 拆成独立进程并行运行。
        selected = args.segment_only
        if selected in ("all", "gsm8k"):
            benches.append(("GSM8K", load_gsm8k_test(args.seed, args.limit),
                            lambda r: extract_gold_gsm8k(r.get("answer", "")),
                            extract_pred_gsm8k, gsm8k_equal))
        if selected in ("all", "math_lo"):
            ensure_zip(args.zip)
            lo = parse_levels(args.math_lo)
            benches.append((f"MATH_lo{args.math_lo}",
                            load_test_problems(args.zip, args.seed, args.limit, levels=lo),
                            lambda r: extract_boxed(r.get("solution", "")),
                            extract_boxed, answers_equal))
        if selected in ("all", "math_hi"):
            ensure_zip(args.zip)
            hi = parse_levels(args.math_hi)
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
    cfg.fusion_injection_mode = args.injection_mode
    cfg.fusion_bypass_small = args.bypass_small
    dt = resolve_dtype(cfg.dtype)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    attn_kwargs = {"attn_implementation": args.attn_impl} if args.attn_impl else {}

    # 两个模型可同时挂载(旁路放 fusion 设备, LoRA 放另一张卡), 一次性并行验证两者正确率
    have_fusion = (not args.baseline_only) and (
        (not args.lora_ckpt) or bool(args.ckpt) or bool(args.multi_fusion_ckpt))
    have_lora = (not args.baseline_only) and bool(args.lora_ckpt)

    tokenizer = None
    large_f = large_l = large_b = None
    fusion = None
    fusions = []
    multi_payload = None
    multi_meta = None
    small_lora_model = None

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
        if args.multi_fusion_ckpt:
            multi_payload = torch.load(args.multi_fusion_ckpt, map_location="cpu")
            try:
                fusion_map, multi_meta = attach_multi_fusion_checkpoint(
                    large_f, small, multi_payload)
            except ValueError as exc:
                raise SystemExit(f"[错误] {args.multi_fusion_ckpt}: {exc}") from exc
            required_names = ("upstream", "downstream")
            fusions = [fusion_map[name] for name in required_names]
            fusion = fusions[0]  # 仅供旧的结果标签逻辑兼容；开关会控制全部旁路。
            scales = ", ".join(
                f"{name}={float(fusion_map[name].scale().detach()):.5f}"
                for name in required_names)
            print(f"双 Bridge scales: {scales} | meta={multi_meta} (设备 {fdev})")
            del multi_payload
            lora_meta = multi_meta.get("small_lora", {}) if isinstance(multi_meta, dict) else {}
            paired_lora = args.small_lora_ckpt
            if not paired_lora and lora_meta.get("enabled"):
                paired_lora = args.multi_fusion_ckpt + str(
                    lora_meta.get("suffix", ".small_lora"))
            if paired_lora:
                if not os.path.isdir(paired_lora):
                    raise SystemExit(
                        "[错误] 双 Bridge checkpoint 声明使用共享 small LoRA，但配套目录不存在: "
                        f"{paired_lora}")
                from peft import PeftModel
                # 两条 fusion 保存的 layer 引用指向同一个 small base；PEFT 原地
                # 替换并集层的线性模块，因此两条旁路会共同使用所加载的 LoRA。
                small_lora_model = PeftModel.from_pretrained(
                    small, paired_lora, is_trainable=False).eval()
                args.small_lora_ckpt = paired_lora
                print(f"已加载多 Bridge 共享 small LoRA: {paired_lora}")
        else:
            fusion = attach_fusion(large_f, small, cfg)
            fusions = [fusion]
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
        for branch in fusions:
            branch.enabled = on

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
            rec = {"id": r.get("benchmark_id", i),
                   "problem": problem, "gold": gold}
            math_strict = pred_fn is extract_boxed

            def record_prediction(label, generated_text, generation_meta=None):
                answer = pred_fn(generated_text)
                rec[f"{label}_pred"] = answer
                rec[f"{label}_correct"] = score_fn(answer, gold)
                if generation_meta is not None:
                    for field in ("generated_tokens", "finish_reason",
                                  "hit_max_new_tokens", "max_new_tokens"):
                        rec[f"{label}_{field}"] = generation_meta[field]
                if args.save_raw_output:
                    rec[f"{label}_text"] = generated_text
                if args.diagnose_math_format and math_strict:
                    relaxed = extract_math_marker_answer(generated_text)
                    rec[f"{label}_relaxed_pred"] = relaxed
                    rec[f"{label}_relaxed_correct"] = score_fn(relaxed, gold)
                if math_verify_runtime is not None and math_strict:
                    judged = judge_with_math_verify(
                        math_verify_runtime, generated_text, gold)
                    rec[f"{label}_math_verify_correct"] = judged["correct"]
                    rec[f"{label}_math_verify_extracted"] = judged["extracted"]
                    rec[f"{label}_math_verify_pred"] = judged["prediction"]
                    rec[f"{label}_math_verify_gold"] = judged["gold"]
                    if judged["error"] is not None:
                        rec[f"{label}_math_verify_error"] = judged["error"]

            # Omni-MATH 的分层信息进入逐题结果，供难度选择与领域诊断使用。
            for field in ("difficulty", "difficulty_band", "domain", "source",
                          "level", "stratum", "original_question"):
                if field in r:
                    rec[field] = r[field]

            # ① 纯 4B baseline(旁路关 = LoRA 关, 二者等价, 只算一次)
            if not args.fusion_only:
                try:
                    if large_f is not None:
                        set_fusion(False)
                        pred_base, meta_base = generate(
                            large_f, tokenizer, prompt, args.max_new,
                            return_meta=True)
                    elif large_l is not None:
                        set_lora(False)
                        pred_base, meta_base = generate(
                            large_l, tokenizer, prompt, args.max_new,
                            return_meta=True)
                    else:
                        pred_base, meta_base = generate(
                            large_b, tokenizer, prompt, args.max_new,
                            return_meta=True)
                finally:
                    # generate 抛异常也要恢复为开启, 防状态残留污染后续结果
                    set_fusion(True)
                    set_lora(True)
                record_prediction("baseline", pred_base, meta_base)

            # ② 旁路开
            if large_f is not None:
                set_fusion(True)
                pred_f, meta_f = generate(
                    large_f, tokenizer, prompt, args.max_new, return_meta=True)
                record_prediction("fusion", pred_f, meta_f)

            # ③ LoRA 开
            if large_l is not None:
                set_lora(True)
                pred_l, meta_l = generate(
                    large_l, tokenizer, prompt, args.max_new, return_meta=True)
                record_prediction("lora", pred_l, meta_l)

            results.append(rec)
            parts = [f"[{name} {i}/{n}]"]
            if large_f is not None:
                acc_f = sum(1 for x in results if x.get("fusion_correct")) / i
                parts.append(f"fusion={acc_f:.3f}")
            if large_l is not None:
                acc_l = sum(1 for x in results if x.get("lora_correct")) / i
                parts.append(f"lora={acc_l:.3f}")
                if "lora_math_verify_correct" in rec:
                    acc_l_mv = sum(
                        1 for x in results if x.get("lora_math_verify_correct")) / i
                    parts.append(f"lora_math_verify={acc_l_mv:.3f}")
            if not args.fusion_only:
                acc_b = sum(1 for x in results if x.get("baseline_correct")) / i
                parts.append(f"baseline={acc_b:.3f}")
            print(" | ".join(parts), flush=True)

        n_total = len(results)
        # 不只哈希题号，也纳入题面和标准答案。这样即使两个服务器上的
        # 数据文件路径/题号相同但内容版本不同，汇总检查仍能发现不一致。
        question_fingerprint = [
            {"id": row["id"], "problem": row["problem"], "gold": row["gold"]}
            for row in results
        ]
        question_set_sha256 = hashlib.sha256(json.dumps(
            question_fingerprint, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        summary = {"bench": name, "n": n_total,
                   "question_set_sha256": question_set_sha256}
        print("=" * 62)
        for label in ("baseline", "fusion", "lora"):
            key = f"{label}_correct"
            if any(key in x for x in results):
                c = sum(1 for x in results if x[key])
                summary[key] = c
                summary[f"{label}_acc"] = round(c / n_total, 4) if n_total else 0
                print(f"[{name}] {label} 准确率: {summary[f'{label}_acc']:.2%} ({c}/{n_total})")
                pred_key = f"{label}_pred"
                extracted = sum(1 for x in results if x.get(pred_key) is not None)
                summary[f"{label}_extract_rate"] = round(
                    extracted / n_total, 4) if n_total else 0
                print(f"[{name}] {label} 严格答案提取率: "
                      f"{summary[f'{label}_extract_rate']:.2%} ({extracted}/{n_total})")
                length_stops = sum(
                    1 for x in results
                    if x.get(f"{label}_hit_max_new_tokens") is True)
                if any(f"{label}_hit_max_new_tokens" in x for x in results):
                    summary[f"{label}_length_stops"] = length_stops
                    summary[f"{label}_length_stop_rate"] = round(
                        length_stops / n_total, 4) if n_total else 0
                    print(f"[{name}] {label} 命中生成上限: "
                          f"{summary[f'{label}_length_stop_rate']:.2%} "
                          f"({length_stops}/{n_total})")
                relaxed_key = f"{label}_relaxed_correct"
                if any(relaxed_key in x for x in results):
                    relaxed_correct = sum(1 for x in results if x.get(relaxed_key))
                    recovered = sum(
                        1 for x in results
                        if x.get(relaxed_key) and not x.get(key))
                    fallback_used = sum(
                        1 for x in results
                        if x.get(pred_key) is None
                        and x.get(f"{label}_relaxed_pred") is not None)
                    summary[relaxed_key] = relaxed_correct
                    summary[f"{label}_relaxed_acc"] = round(
                        relaxed_correct / n_total, 4) if n_total else 0
                    summary[f"{label}_format_recovered"] = recovered
                    summary[f"{label}_marker_fallback_used"] = fallback_used
                    print(f"[{name}] {label} 格式诊断准确率: "
                          f"{summary[f'{label}_relaxed_acc']:.2%} "
                          f"({relaxed_correct}/{n_total}) | 格式挽回 {recovered} 题 | "
                          f"标记兜底触发 {fallback_used} 题")
                math_verify_key = f"{label}_math_verify_correct"
                if any(math_verify_key in x for x in results):
                    mv_correct = sum(1 for x in results if x.get(math_verify_key))
                    mv_extracted = sum(
                        1 for x in results
                        if x.get(f"{label}_math_verify_extracted"))
                    mv_errors = sum(
                        1 for x in results
                        if x.get(f"{label}_math_verify_error") is not None)
                    mv_recovered = sum(
                        1 for x in results
                        if x.get(math_verify_key) and not x.get(key))
                    legacy_only = sum(
                        1 for x in results
                        if x.get(key) and not x.get(math_verify_key))
                    summary[math_verify_key] = mv_correct
                    summary[f"{label}_math_verify_acc"] = round(
                        mv_correct / n_total, 4) if n_total else 0
                    summary[f"{label}_math_verify_extract_rate"] = round(
                        mv_extracted / n_total, 4) if n_total else 0
                    summary[f"{label}_math_verify_errors"] = mv_errors
                    summary[f"{label}_math_verify_recovered"] = mv_recovered
                    summary[f"{label}_legacy_only_correct"] = legacy_only
                    print(f"[{name}] {label} Math-Verify 准确率: "
                          f"{summary[f'{label}_math_verify_acc']:.2%} "
                          f"({mv_correct}/{n_total}) | 提取率 "
                          f"{summary[f'{label}_math_verify_extract_rate']:.2%} | "
                          f"较 strict 挽回 {mv_recovered} 题 | "
                          f"strict-only {legacy_only} 题 | 解析异常 {mv_errors} 题")
        if name.startswith("OMNI_MATH"):
            summary["by_difficulty"] = summarize_by_difficulty(results)
            print(f"[{name}] 分难度结果:")
            for level, part in summary["by_difficulty"].items():
                values = []
                for label in ("baseline", "fusion", "lora"):
                    if f"{label}_acc" in part:
                        values.append(f"{label}={part[f'{label}_acc']:.2%}")
                print(f"  L{level} n={part['n']}: " + " | ".join(values))
        if name == "SFT_VAL":
            by_stratum = {}
            for row in results:
                stratum = str(row.get("stratum", "unknown"))
                part = by_stratum.setdefault(stratum, {"n": 0})
                part["n"] += 1
                for label in ("baseline", "fusion", "lora"):
                    for suffix in ("correct", "math_verify_correct"):
                        key = f"{label}_{suffix}"
                        if key in row:
                            part[key] = part.get(key, 0) + int(bool(row[key]))
            for part in by_stratum.values():
                for key, value in list(part.items()):
                    if key != "n" and key.endswith("correct"):
                        part[key.replace("correct", "acc")] = round(
                            value / part["n"], 4) if part["n"] else 0.0
            summary["by_stratum"] = dict(sorted(by_stratum.items()))
            for label in ("baseline", "fusion", "lora"):
                preferred_key = f"{label}_math_verify_acc"
                fallback_key = f"{label}_acc"
                values = [
                    part[preferred_key] if preferred_key in part else part[fallback_key]
                    for part in summary["by_stratum"].values()
                    if preferred_key in part or fallback_key in part
                ]
                if values:
                    summary[f"{label}_macro_acc"] = round(
                        sum(values) / len(values), 4)
            summary["recommended_selection_metric"] = (
                "math_verify_macro_accuracy_across_strata")
            print("[SFT_VAL] 分层结果:")
            for stratum, part in summary["by_stratum"].items():
                values = []
                for label in ("baseline", "fusion", "lora"):
                    key = f"{label}_math_verify_acc"
                    fallback = f"{label}_acc"
                    if key in part:
                        values.append(f"{label}={part[key]:.2%}")
                    elif fallback in part:
                        values.append(f"{label}={part[fallback]:.2%}")
                print(f"  {stratum} n={part['n']}: " + " | ".join(values))
            macro_values = [
                f"{label}={summary[f'{label}_macro_acc']:.2%}"
                for label in ("baseline", "fusion", "lora")
                if f"{label}_macro_acc" in summary
            ]
            print("[SFT_VAL] 分层宏平均: " + " | ".join(macro_values))
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
        path = (args.result_path or os.path.join(
            args.out_dir,
            f"eval_{name.lower()}{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary_path = f"{path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt,
                       "multi_fusion_ckpt": args.multi_fusion_ckpt,
                       "multi_fusion_meta": multi_meta,
                       "multi_fusion_scales": (
                           {str(getattr(branch, "fusion_name", index)):
                            float(branch.scale().detach())
                            for index, branch in enumerate(fusions)}
                           if args.multi_fusion_ckpt else None),
                       "small_lora_ckpt": args.small_lora_ckpt,
                       "lora_ckpt": args.lora_ckpt,
                       "suite_run_id": args.suite_run_id,
                       "small_start": args.small_start, "small_end": args.small_end,
                       "seed": args.seed, "segment_only": args.segment_only,
                       "omni_path": args.omni_path,
                       "omni_levels": args.omni_levels,
                       "omni_limit_per_level": args.omni_limit_per_level,
                       "sft_val_path": args.sft_val_path,
                       "sft_val_limit_per_stratum": args.sft_val_limit_per_stratum,
                       "diagnose_math_format": args.diagnose_math_format,
                       "math_judge": args.math_judge,
                       "math_verify_version": (
                           math_verify_runtime["version"]
                           if math_verify_runtime is not None else None),
                       "question_set_sha256": question_set_sha256,
                       "save_raw_output": args.save_raw_output,
                       "summary": summary,
                       "results": results}, f, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
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
