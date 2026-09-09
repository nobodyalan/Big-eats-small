# -*- coding: utf-8 -*-
"""
原始模型推理能力基线测试(对照实验)
用标准文本生成方式分别测试 0.6B 与 4B 的推理能力,结果保存到 eval_results/baseline_<时间戳>.json。
之后与 main.py(enable_eval=True)的融合管线结果对比,即可看出"潜空间协作"的增益。

用法(BES 环境):
  python test_baseline.py              # 两个模型都测(默认)
  python test_baseline.py --model 4b   # 只测 4B
  python test_baseline.py --model 0.6b # 只测 0.6B

说明: 0.6B 是基座模型(非 Instruct),回答质量天然偏低,这本身就是"基线"的一部分;
     如需更公平的对比,可在 main.py 的 Config 中把副脑也换成 -Instruct 版本。
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录 = 上一级(BES); 把 core_training/(main) 与 eval/(eval_questions) 加入导入路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from eval_questions import EVAL_QUESTIONS
from main import Config, build_prompt_text, resolve_dtype, resolve_model_path

# Windows 控制台默认 GBK 编码,打印中文可能报 UnicodeEncodeError,统一转为 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_model(model_id: str, local_override, dtype, device_map):
    """加载单个模型(与 main.py 相同的 modelscope 定位 + 精度策略)"""
    path = resolve_model_path(model_id, local_override)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device_map)
    model.eval()
    return tokenizer, model


def answer_one(model, tokenizer, question: str, config: Config) -> dict:
    """标准文本生成: 答一道题,返回 {model_answer, time_s, prompt_tokens}"""
    text = build_prompt_text(tokenizer, question)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=(config.temperature > 0),
            temperature=config.temperature,
            top_p=config.top_p,
        )
    dt = time.time() - t0
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
    return {"model_answer": reply.strip(),
            "time_s": round(dt, 2),
            "prompt_tokens": int(inputs["input_ids"].shape[1])}


def run_baseline(config: Config, model_names) -> str:
    """逐模型、逐题测试并保存结果,返回保存路径"""
    torch.manual_seed(config.seed)
    os.makedirs("eval_results", exist_ok=True)
    results = {}

    for name in model_names:
        model_id = config.model_small_id if name == "0.6B" else config.model_large_id
        local = config.model_small_local if name == "0.6B" else config.model_large_local
        print(f"\n{'=' * 60}\n加载 {name} 模型: {model_id}\n{'=' * 60}")
        tokenizer, model = load_model(model_id, local,
                                      resolve_dtype(config.dtype), config.device_map)
        records = []
        for i, q in enumerate(EVAL_QUESTIONS, 1):
            print(f"  [{i}/{len(EVAL_QUESTIONS)}] {q['question'][:30]} ...")
            try:
                rec = {"id": q["id"], "question": q["question"], "expected": q["answer"]}
                rec.update(answer_one(model, tokenizer, q["question"], config))
                print(f"      回答: {rec['model_answer'][:60]}... (耗时 {rec['time_s']}s)")
            except Exception as e:  # 单题失败不中断整批
                rec = {"id": q["id"], "question": q["question"],
                       "expected": q["answer"], "error": str(e)}
                print(f"      [失败] {e}")
            records.append(rec)
        results[name] = {"model_id": model_id, "records": records}
        # 及时释放显存,避免 12G 显卡上同时驻留两个模型
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = os.path.join("eval_results", f"baseline_{time.strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "meta": {
            "script": "test_baseline",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "params": {"max_new_tokens": config.max_new_tokens,
                       "temperature": config.temperature,
                       "top_p": config.top_p,
                       "seed": config.seed},
            "question_count": len(EVAL_QUESTIONS),
        },
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原始模型推理能力基线测试")
    parser.add_argument("--model", choices=["both", "0.6B", "4B"], default="both",
                        help="测试哪个模型(默认 both)")
    args = parser.parse_args()

    config = Config()
    names = ["0.6B", "4B"] if args.model == "both" else [args.model]
    saved = run_baseline(config, names)
    print(f"\n基线测试完成, 结果已保存: {saved}")
