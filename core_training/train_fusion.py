# -*- coding: utf-8 -*-
"""
门控残差融合适配器训练脚本
==========================
冻结 4B(主脑)与 0.6B(旁路中段宿主),训练门控残差旁路；可选地只对
实际使用的小模型层挂 LoRA。bridge、small LoRA 与 branch alpha 使用解耦学习率。

训练目标: 因果语言建模(下一 token 预测)损失。
梯度路径: loss → 4B 第 24 层注入点 → adapter2 → 0.6B 中段层 → adapter1。

针对 H100 等大显存 GPU 的默认设置:
  bf16 权重 + 关闭梯度检查点(更快) + batch=8 + 最大序列 512 + 3 epoch。
小显存(如 12GB)请改用: --grad_checkpoint 1 --batch_size 1 --max_len 256。
(注意: 梯度检查点要求模型处于 .train() 状态才会生效; Qwen3 的
 attention_dropout=0.0, 所以 train 模式不会引入 dropout 噪声。)

用法:
  python train_fusion.py                                          # 默认全量
  python train_fusion.py --max_samples 2000 --epochs 1            # 小规模试跑
  python train_fusion.py --resume cache/fusion_adapter.pt         # 断点续训
"""

import argparse
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录 = 上一级(BES); 把 core_training/(main) 与 eval/(main 的依赖) 加入导入路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from main import (Config, attach_fusion, save_fusion, load_fusion,
                  model_device, resolve_dtype, resolve_model_path,
                  resolve_small_range, resolve_large_range)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IGNORE = -100
SMALL_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ① 数据加载                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def load_records(path: str, max_samples: int):
    """读 jsonl, 每行返回 {"prompt":..., "response":...}。
    兼容旧格式 {"text": "题目\\n解题思路:答案"}(自动按 '解题思路:' 拆成 prompt/response)。"""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"text": line}
            if "prompt" in obj:
                recs.append({"prompt": obj.get("prompt", ""),
                             "response": obj.get("response", "")})
            else:
                t = obj.get("text", "") or line
                if "解题思路:" in t:
                    p, r = t.split("解题思路:", 1)
                    recs.append({"prompt": p + "解题思路:", "response": r})
                else:
                    recs.append({"prompt": t, "response": ""})
            if max_samples > 0 and len(recs) >= max_samples:
                break
    return [r for r in recs if r["response"].strip() or r["prompt"].strip()]


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def answer_start_char(response: str) -> int:
    """返回最靠后的标准答案标记位置；不存在则为 -1。"""
    positions = [response.rfind(marker) for marker in ("\\boxed", "####")]
    positions.extend(m.start() for m in re.finditer(
        r"(?:the\s+)?answer\s+is\s*:", response, flags=re.IGNORECASE))
    return max(positions, default=-1)


def collate(batch, tokenizer, max_len: int, answer_weight: float = 1.0):
    """SFT 式编码: 只对 response 算 loss(prompt 位置 label=IGNORE),
    并给最终答案段(\\boxed / #### 之后)加权。返回 (input_ids, attention_mask, labels, weights)。"""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    ids_list, labels_list, weights_list = [], [], []
    for rec in batch:
        prompt = rec.get("prompt", "")
        resp = rec.get("response", "")
        # 与评测一致: prompt 套 chat 模板(add_generation_prompt=True), response 直接拼在后面
        if getattr(tokenizer, "chat_template", None):
            p_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            p_text = prompt
        # prompt 编码(给答案至少留 1 个位置)
        p_ids = tokenizer(p_text, add_special_tokens=False).input_ids
        p_ids = p_ids[:max_len - 1]
        # response 编码 + 字符偏移(用于定位答案段)
        enc = tokenizer(resp, add_special_tokens=False, return_offsets_mapping=True)
        r_ids, r_off = enc.input_ids, enc.offset_mapping
        r_ids = r_ids[:max_len - len(p_ids)]
        r_off = r_off[:len(r_ids)]
        if not r_ids or r_ids[-1] != eos:
            r_ids = r_ids + [eos]
            r_off = r_off + [(0, 0)]
        r_ids = r_ids[:max_len - len(p_ids)]
        r_off = r_off[:len(r_ids)]
        # 定位最终答案段起点。MetaMathQA 常用 "The answer is:"，且同一回答
        # 可能同时含 boxed/####；必须取所有标记中最靠后的一个，不能按标记类型提前 break。
        ans_char = answer_start_char(resp)
        ans_tok = None
        if ans_char != -1:
            for k, (s, e) in enumerate(r_off):
                if s <= ans_char < e:
                    ans_tok = k
                    break
        w = [1.0] * len(r_ids)
        if ans_tok is not None:
            for k in range(ans_tok, len(r_ids)):
                w[k] = answer_weight
        ids_list.append(torch.tensor(p_ids + r_ids, dtype=torch.long))
        labels_list.append(torch.tensor([IGNORE] * len(p_ids) + r_ids, dtype=torch.long))
        weights_list.append(torch.tensor([0.0] * len(p_ids) + w, dtype=torch.float32))
    L = max(x.size(0) for x in ids_list)
    B = len(ids_list)
    input_ids = torch.full((B, L), pad, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    weights = torch.zeros((B, L), dtype=torch.float32)
    for i in range(B):
        n = ids_list[i].size(0)
        input_ids[i, :n] = ids_list[i]
        attention_mask[i, :n] = 1
        labels[i, :n] = labels_list[i]
        weights[i, :n] = weights_list[i]
    return input_ids, attention_mask, labels, weights


def causal_lm_loss(logits, labels, weights=None):
    """因果 LM: 每个位置预测下一个 token; padding/prompt 位置忽略(IGNORE)。
    weights 非空时对每个有效位置按权重加权(答案段加权用)。"""
    # 保持 bf16 传给 F.cross_entropy(内部按 fp32 算 log_softmax), 省掉整张 fp32 logits 拷贝
    # (batch×seq×151936 词表的 fp32 拷贝 ≈5GB, 是 4B 训练 OOM 的直接触发点)
    loss_sum, weight_sum = causal_lm_loss_parts(logits, labels, weights)
    return loss_sum / weight_sum.clamp(min=1)


def causal_lm_loss_parts(logits, labels, weights=None):
    """返回有效 token 的加权 CE 总和与权重总和，供跨 batch 无偏聚合。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                         shift_labels.view(-1), ignore_index=IGNORE, reduction="none")
    ce = ce.view(shift_labels.shape)
    mask = (shift_labels != IGNORE).float()
    if weights is None:
        w = mask
    else:
        w = weights[:, 1:].to(ce.device).float() * mask
    return (ce * w).sum(), w.sum()


def causal_lm_loss_per_example(logits, labels, weights=None):
    """返回每条样本的加权 token 平均 CE，供配对引导损失使用。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                         shift_labels.view(-1), ignore_index=IGNORE, reduction="none")
    ce = ce.view(shift_labels.shape)
    mask = (shift_labels != IGNORE).float()
    if weights is None:
        w = mask
    else:
        w = weights[:, 1:].to(ce.device).float() * mask
    return (ce * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def baseline_improvement_loss(correct_logits, zero_losses,
                              labels, weights=None, margin=0.02):
    """逐样本要求融合 CE 至少比冻结 4B baseline 低 margin。

    reference 被 detach，因此这是由真实标签监督的、baseline 感知的难例加权；
    它不会把错配分支推向任意坏分布，也不冒充对比学习。
    """
    correct = causal_lm_loss_per_example(correct_logits, labels, weights)
    per_example = F.relu(margin + correct - zero_losses.detach())
    active_ratio = (per_example > 0).float().mean()
    return per_example.mean(), active_ratio


def diagnose_train_branch(branch_ratio, branch_scale, bridge_grad,
                          alpha_grad=0.0, lora_grad=0.0,
                          alpha_trainable=True, lora_trainable=False,
                          min_ratio=1e-4, max_ratio=0.5,
                          grad_epsilon=1e-10):
    """返回分支数值/梯度健康告警，不改变训练过程。

    这里只能判断旁路是否“活着”；是否携带样本相关的有效信息，必须由验证集上的
    correct/zero/shuffled 对照判断。
    """
    values = {
        "branch/base RMS": float(branch_ratio),
        "branch_scale": float(branch_scale),
        "bridge_grad": float(bridge_grad),
        "alpha_grad": float(alpha_grad),
        "lora_grad": float(lora_grad),
    }
    nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
    if nonfinite:
        return ["非有限数值: " + ",".join(nonfinite)]

    warnings = []
    if values["branch/base RMS"] < min_ratio:
        warnings.append(
            f"旁路幅度过低({values['branch/base RMS']:.2e} < {min_ratio:.2e})")
    elif values["branch/base RMS"] > max_ratio:
        warnings.append(
            f"旁路幅度过高({values['branch/base RMS']:.2e} > {max_ratio:.2e})")
    if values["bridge_grad"] <= grad_epsilon:
        warnings.append("bridge 梯度近零")
    if alpha_trainable:
        if abs(values["branch_scale"]) < min_ratio:
            warnings.append("alpha 接近零，疑似关闭旁路")
        if values["alpha_grad"] <= grad_epsilon:
            warnings.append("alpha 梯度近零")
    if lora_trainable and values["lora_grad"] <= grad_epsilon:
        warnings.append("small LoRA 梯度近零")
    return warnings


def classify_branch_evaluation(fusion_loss, baseline_loss, shuffled_loss,
                               tolerance=1e-4):
    """分类验证集上是否存在有效、样本相关的旁路使用证据。"""
    values = [float(fusion_loss), float(baseline_loss), float(shuffled_loss)]
    if not all(math.isfinite(value) for value in values):
        return "NONFINITE", "评估损失出现非有限值"
    gain = baseline_loss - fusion_loss
    specificity = shuffled_loss - fusion_loss
    if gain > tolerance and specificity > tolerance:
        return "USEFUL", "正确旁路同时优于关闭与 shuffled"
    if gain < -tolerance:
        return "HARMFUL", "正确旁路劣于冻结 4B baseline"
    if specificity < -tolerance:
        return "SHUFFLED_BETTER", "shuffled 优于正确旁路，尚无内容特异性"
    if abs(gain) <= tolerance and abs(specificity) <= tolerance:
        return "DROPPED", "correct/zero/shuffled 几乎相同，疑似旁路被忽略"
    if gain > tolerance and specificity <= tolerance:
        return "GENERIC", "优于关闭但不优于 shuffled，可能只是通用扰动/容量收益"
    return "INCONCLUSIVE", "差异尚未超过容差，继续观察"


def mismatched_hidden(hidden, attention_mask=None):
    """构造长度感知的错配旁路，仅用于反事实诊断/可选旧 JS。

    batch>1 时按长度排序后选择无自配对，并把来源的有效 token 等比例重采样到
    目标有效长度，避免来源 padding 落入目标有效区。batch=1 只能退化为有效区
    内 token 滚动，因此正式 shuffled 诊断应使用 batch>1。
    """
    hidden = hidden.detach()
    if attention_mask is None:
        attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
    else:
        attention_mask = attention_mask.to(hidden.device)
    lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    if hidden.size(0) > 1:
        order = torch.argsort(lengths)
        # 两个方向都无自配对，选择总长度差更小的循环方向。
        prev = torch.roll(order, shifts=1)
        nxt = torch.roll(order, shifts=-1)
        prev_cost = (lengths[order] - lengths[prev]).abs().sum()
        nxt_cost = (lengths[order] - lengths[nxt]).abs().sum()
        sources_sorted = prev if prev_cost <= nxt_cost else nxt
        source_for = torch.empty_like(order)
        source_for[order] = sources_sorted
        wrong = torch.zeros_like(hidden)
        for target in range(hidden.size(0)):
            source = int(source_for[target])
            nt, ns = int(lengths[target]), int(lengths[source])
            # 用离散等比例索引填满目标有效区，不复制来源 padding。
            idx = torch.div(torch.arange(nt, device=hidden.device) * ns,
                            nt, rounding_mode="floor").clamp_max(ns - 1)
            wrong[target, :nt] = hidden[source, idx]
        return wrong
    n = int(lengths[0])
    wrong = torch.zeros_like(hidden)
    wrong[0, :n] = torch.roll(hidden[0, :n], shifts=1, dims=0) if n > 1 else -hidden[0, :n]
    return wrong


def attach_small_lora(model, s1, s2, rank, alpha, dropout, targets, resume=""):
    """只给实际使用的小模型片段挂 LoRA；调用前 fusion 已保存真实层引用。"""
    from peft import LoraConfig, PeftModel, get_peft_model

    if resume:
        return PeftModel.from_pretrained(model, resume, is_trainable=True)
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
        layers_to_transform=list(range(s1, s2 + 1)), layers_pattern="layers",
    )
    return get_peft_model(model, cfg)


def save_fusion_experiment(fusion, path, small_lora_model=None):
    save_fusion(fusion, path)
    if small_lora_model is not None:
        lora_path = path + ".small_lora"
        small_lora_model.save_pretrained(lora_path)
        print(f"    小模型 LoRA 已保存: {lora_path}")


JS_MARGIN = math.log(2.0)  # JS 散度使用 nats, 理论上界 ln2
EPS = 1e-8


def js_contrastive(logits_n, logits_r, labels, margin=JS_MARGIN):
    """
    InterLat 同款对比损失(JS 散度铰链):
      正确旁路 vs 错配旁路, 两边的输出分布应尽量不同(JS→ln2)。
    防止模型"无视这条旁路": 若旁路内容不影响预测, JS≈0 → 损失把 JS 往 ln2 推,
    逼模型去读旁路注入的隐状态。
    """
    # 与 CE 同对齐: logits[:, :-1] 预测 labels[:, 1:]
    shift_n = logits_n[:, :-1, :].contiguous()
    shift_r = logits_r[:, :-1, :].contiguous()
    shift_lab = labels[:, 1:].contiguous()
    valid = shift_lab != IGNORE
    # 只对有效 token(远小于 batch×seq)转 fp32 做 softmax, 避免整张 fp32 拷贝
    p = F.softmax(shift_n[valid].float(), dim=-1).clamp_min(EPS)
    q = F.softmax(shift_r[valid].float(), dim=-1).clamp_min(EPS)
    m = 0.5 * (p + q)
    # torch.kl_div(input, target) 计算 KL(target || exp(input))，因此 input
    # 必须是 log(m)，target 才是 p/q。旧实现把参数反了，实际算成反向 KL；
    # 同时又错误地做了 nats/bits 换算，导致对比项尺度不可解释。
    js_nats = 0.5 * F.kl_div(m.log(), p, reduction="batchmean") \
              + 0.5 * F.kl_div(m.log(), q, reduction="batchmean")
    return torch.clamp(margin - js_nats, min=0.0)


def plot_losses(train_ce_history, eval_history, path):
    """画训练 CE + 评估 correct/zero/shuffled 三路 loss。"""
    if not path:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("    [提示] 未安装 matplotlib, 跳过画图(可 pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if train_ce_history:
        steps = [s for s, _ in train_ce_history]
        ces = [c for _, c in train_ce_history]
        axes[0].plot(steps, ces, alpha=0.35, color="tab:blue", lw=0.6, label="per-step CE")
        if len(ces) >= 5:
            k = min(20, len(ces))
            sm = np.convolve(ces, np.ones(k) / k, mode="valid")
            axes[0].plot(steps[k - 1:], sm, color="tab:blue", lw=2, label=f"smoothed (win {k})")
        axes[0].set_title("Training CE loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("CE")
        axes[0].legend()
    if eval_history:
        steps = [row[0] for row in eval_history]
        f = [row[1] for row in eval_history]
        b = [row[2] for row in eval_history]
        axes[1].plot(steps, b, marker="o", ms=3, label="baseline (branch off)")
        if all(len(row) >= 4 for row in eval_history):
            shuffled = [row[3] for row in eval_history]
            axes[1].plot(steps, shuffled, marker="o", ms=3, label="shuffled branch")
            axes[1].plot(steps, f, marker="o", ms=3, label="fusion (branch on)")
            axes[1].set_title("Eval: correct vs zero vs shuffled")
        else:
            # train_lora.py 使用 (step, adapted, baseline) 三元组。
            axes[1].plot(steps, f, marker="o", ms=3, label="adapted model")
            axes[1].set_title("Eval: adapted model vs baseline")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("CE")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    loss 图已保存: {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ② 训练主流程                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    parser = argparse.ArgumentParser(description="训练门控残差融合适配器")
    parser.add_argument("--data", default="data/mix_all.jsonl")
    parser.add_argument("--max_samples", type=int, default=0, help="最多训练条数(0=全部; 评估集另算, 不被截断)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1024, help="单条最大 token 数(显存不足先调小)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="兼容参数；--bridge_lr 未设置时作为 bridge 学习率")
    parser.add_argument("--bridge_lr", type=float, default=None,
                        help="随机初始化 bridge 学习率；默认沿用 --lr (1e-4)")
    parser.add_argument("--small_lora_lr", type=float, default=2e-5,
                        help="小模型 LoRA 学习率；预训练参数增量应低于 bridge")
    parser.add_argument("--small_lora_delay_steps", type=int, default=200,
                        help="前 N 步只训练 bridge，之后才解冻小模型 LoRA；resume 自动跳过")
    parser.add_argument("--alpha_lr", type=float, default=5e-4,
                        help="branch_alpha 独立学习率；不使用 weight decay")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="bridge AdamW weight decay；LoRA/alpha 默认不衰减")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="学习率线性 warmup 步数(0=关闭; 之后余弦衰减到 10%%)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--small_lora_grad_clip", type=float, default=0.5,
                        help="小模型 LoRA 独立梯度裁剪；不受大 bridge 梯度范数牵连")
    parser.add_argument("--alpha_grad_clip", type=float, default=0.1,
                        help="标量 branch alpha 的独立梯度裁剪")
    parser.add_argument("--gate_init", type=float, default=None,
                        help="已弃用的旧 sigmoid gate 参数；新版训练忽略它以兼容旧运行脚本")
    parser.add_argument("--branch_alpha_init", type=float, default=0.0,
                        help="关闭 branch warmup 时的 ReZero 初值；默认 0")
    parser.add_argument("--branch_warmup_steps", type=int, default=400,
                        help="前 N 步固定非零 alpha 训练 bridge/LoRA；resume 时自动跳过")
    parser.add_argument("--branch_warmup_alpha", type=float, default=0.05,
                        help="branch warmup 阶段固定的残差系数")
    parser.add_argument("--branch_alpha_max", type=float, default=0.25,
                        help="释放后将 ReZero alpha 限制在 ±该值；0=不限制")
    parser.add_argument("--bridge_depth", type=int, default=1,
                        help="每个输入/输出 adapter 的深度；2≈88M bridge 参数")
    parser.add_argument("--bridge_mlp_dim", type=int, default=None,
                        help="bridge GLU 中间维度；None 使用 Config 默认 4096")
    parser.add_argument("--small_start", type=int, default=None,
                        help="0.6B 旁路起始层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--small_end", type=int, default=None,
                        help="0.6B 旁路结束层(含, 0-based); -1=最后一层; None=自动 2/3 位置")
    parser.add_argument("--large_start", type=int, default=None,
                        help="4B 取隐状态层(含, 0-based); None=自动 1/3 位置")
    parser.add_argument("--large_end", type=int, default=None,
                        help="4B 加回残差层(含, 0-based); None=自动 2/3 位置")
    parser.add_argument("--bypass_small", action="store_true",
                        help="跳过冻结小模型层，仅训练同容量 bridge（必要 control）")
    parser.add_argument("--small_lora_r", type=int, default=0,
                        help="小模型片段 LoRA rank；0=关闭，推荐实验值 32")
    parser.add_argument("--small_lora_alpha", type=int, default=0,
                        help="小模型 LoRA alpha；0=自动 2×rank")
    parser.add_argument("--small_lora_dropout", type=float, default=0.05)
    parser.add_argument("--small_lora_target", default=",".join(SMALL_LORA_TARGETS))
    parser.add_argument("--small_lora_resume", default="",
                        help="继续训练已保存的小模型 LoRA 目录")
    parser.add_argument("--grad_checkpoint", type=int, default=0, help="1=梯度检查点(省显存但更慢)")
    parser.add_argument("--batch_size", type=int, default=8, help="训练 batch(H100 可用 8~16)")
    parser.add_argument("--contrast_weight", type=float, default=0.0,
                        help="旧 JS 敏感性正则权重；不保证任务增益，正式实验建议 0")
    parser.add_argument("--guide_weight", type=float, default=0.0,
                        help="实验性 baseline 感知重加权；默认 0，正式训练仅用普通 CE")
    parser.add_argument("--guide_margin", type=float, default=0.02,
                        help="任务引导所需的每 token CE 优势(nats)")
    parser.add_argument("--guide_every", type=int, default=4,
                        help="每 N 个训练 step 计算一次引导，控制额外前向开销")
    parser.add_argument("--answer_weight", type=float, default=1.0,
                        help="最终答案段(\\boxed/#### 之后)的 loss 权重(SFT 答案加权)")
    parser.add_argument("--seed", type=int, default=None,
                        help="训练随机种子；None 使用 Config.seed")
    parser.add_argument("--eval_every", type=int, default=200, help="每隔 N 步对比一次 baseline/fusion loss")
    parser.add_argument("--eval_samples", type=int, default=64, help="从数据里留出多少条作评估集")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估时的 batch(越大评估越快)")
    parser.add_argument("--eval_max_samples", type=int, default=400,
                        help="每次评估最多用多少条(0=全部; 验证集大时应设小, 否则每步评估很慢)")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--branch_min_rms_ratio", type=float, default=1e-4,
                        help="训练诊断：低于此 branch/base RMS 时提示旁路过弱")
    parser.add_argument("--branch_max_rms_ratio", type=float, default=0.5,
                        help="训练诊断：高于此 branch/base RMS 时提示旁路过强")
    parser.add_argument("--branch_loss_tolerance", type=float, default=1e-4,
                        help="验证诊断：correct/zero/shuffled CE 的无差异容差")
    parser.add_argument("--branch_health_patience", type=int, default=3,
                        help="连续多少次验证无 USEFUL 证据后打印显著告警；0=关闭")
    parser.add_argument("--out", default=None,
                        help="最终权重路径(默认=自动 cache/fusion_s<起>_<止>_<时间戳>.pt)")
    parser.add_argument("--resume", default="", help="从已保存的旁路参数继续")
    parser.add_argument("--patience", type=int, default=0,
                        help="早停耐心: 连续 N 次评估 fusion loss 无改善就停(0=关闭)")
    parser.add_argument("--plot", default=None,
                        help="loss 图路径(默认=自动带层范围+时间戳; 传空字符串=不画)")
    parser.add_argument("--attn_impl", default="",
                        help="注意力实现(sdpa/flash_attention_2/eager; 空=自动)。旁路训练建议保持 sdpa")
    args = parser.parse_args()
    if args.branch_warmup_steps < 0:
        parser.error("--branch_warmup_steps 必须 >= 0")
    if args.small_lora_delay_steps < 0:
        parser.error("--small_lora_delay_steps 必须 >= 0")
    if args.guide_every < 1:
        parser.error("--guide_every 必须 >= 1")
    if args.branch_min_rms_ratio < 0 or args.branch_max_rms_ratio <= 0:
        parser.error("分支 RMS 阈值必须满足 min >= 0 且 max > 0")
    if args.branch_min_rms_ratio >= args.branch_max_rms_ratio:
        parser.error("--branch_min_rms_ratio 必须小于 --branch_max_rms_ratio")
    if args.branch_loss_tolerance < 0 or args.branch_health_patience < 0:
        parser.error("分支 loss 容差和 patience 必须 >= 0")
    if args.gate_init is not None:
        print(f"    [兼容] --gate_init {args.gate_init:g} 属于旧 sigmoid gate，"
              "新版 ReZero 训练忽略该值；请改用 --branch_alpha_init")

    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.small_start is not None:
        cfg.fusion_small_start = args.small_start
    if args.small_end is not None:
        cfg.fusion_small_end = args.small_end
    if args.large_start is not None:
        cfg.fusion_large_start = args.large_start
    if args.large_end is not None:
        cfg.fusion_large_end = args.large_end
    cfg.fusion_bridge_depth = args.bridge_depth
    if args.bridge_mlp_dim is not None:
        cfg.fusion_mlp_dim = args.bridge_mlp_dim
    cfg.fusion_bypass_small = args.bypass_small
    torch.manual_seed(cfg.seed)
    dt = resolve_dtype(cfg.dtype)
    print(f"精度: {dt} | CUDA: {torch.cuda.is_available()}")
    if dt not in (torch.bfloat16, torch.float16):
        print("    [提示] 建议用 bf16/fp16,4B 权重 fp32 会爆 12GB 显存")

    # ── 加载模型(全在 GPU,bf16;不用 device_map='auto' 的 offload) ──
    small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    # 以最终解码器 4B Instruct 的 chat template 为唯一训练格式。
    tokenizer = AutoTokenizer.from_pretrained(large_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # flash 只给大模型: 小模型的层是手写循环调用(传 4D 掩码), flash 不支持 4D 掩码
    attn_kwargs = {"attn_implementation": args.attn_impl} if args.attn_impl else {}
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dt).cuda()
    large = AutoModelForCausalLM.from_pretrained(large_path, dtype=dt, **attn_kwargs).cuda()
    # 基础权重先全部冻结；可选的小模型 LoRA 会在 attach_fusion 后单独启用。
    for m in (small, large):
        for p in m.parameters():
            p.requires_grad_(False)
    # 输出文件自动命名: 带大模型位置 + 旁路层范围 + 时间戳, 避免不同实验互相覆盖
    n_small = small.config.num_hidden_layers
    s1, s2 = resolve_small_range(cfg, n_small)
    l1, l2 = resolve_large_range(cfg, large.config.num_hidden_layers)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"cache/fusion_L{l1}-{l2}_s{s1}_{s2}_{ts}.pt"
    plot_path = args.plot if args.plot is not None else f"cache/train_fusion_L{l1}-{l2}_s{s1}_{s2}_{ts}.png"
    segment_desc = "bridge-only control" if args.bypass_small else f"0.6B 第{s1}~{s2}层"
    print(f"接入位置: 4B 第{l1}层取 → 第{l2}层加回 | {segment_desc}")
    print(f"输出权重: {out_path} | loss 图: {plot_path}")
    # 梯度检查点需要模型处于 train 模式才生效;Qwen3 attention_dropout=0 无噪声
    large.train()
    small.eval()
    if args.grad_checkpoint:
        large.gradient_checkpointing_enable()
        # 冻结主模型时仍需让隐藏状态保留梯度，确保外接 hook 中的 bridge 可回传；
        # Transformers 5.x 会自动设置，但显式调用可兼容较旧服务器版本。
        large.enable_input_require_grads()
        print("    梯度检查点已开启")

    # ── 挂载门控残差旁路 ──
    fusion = attach_fusion(large, small, cfg)
    small_lora_model = None
    if args.small_lora_r > 0 or args.small_lora_resume:
        if args.bypass_small:
            raise ValueError("--bypass_small 与 --small_lora_r/--small_lora_resume 不能同时使用")
        targets = [x.strip() for x in args.small_lora_target.split(",") if x.strip()]
        alpha = args.small_lora_alpha or 2 * max(1, args.small_lora_r)
        resume_path = args.small_lora_resume
        if not resume_path and args.resume and os.path.isdir(args.resume + ".small_lora"):
            resume_path = args.resume + ".small_lora"
        small_lora_model = attach_small_lora(
            small, s1, s2, max(1, args.small_lora_r), alpha,
            args.small_lora_dropout, targets, resume_path)
        small_lora_model.train()
    # 关键修复: 适配器参数用 fp32 主权重。attach_fusion 把它们搬到 bf16, 若直接
    # 让 AdamW 更新 bf16 参数, 梯度小到 bf16 精度(约 3 位有效数字)就归零,
    # 训练几十步后彻底冻结。这里转回 fp32, 前向用 autocast 做 bf16 计算。
    # 必须先转 fp32 再 resume: 否则 fp32 检查点会被先压成 bf16 再升回 fp32, 丢尾数。
    for p in fusion.parameters():
        p.data = p.data.float()
    if args.resume:
        load_fusion(fusion, args.resume)
    else:
        # 固定小非零 alpha 的 branch warmup 让 bridge 从第一步就获得梯度；
        # 小模型 LoRA 可延迟解冻，避免它替尚未成形的 bridge 补偿。
        # 若显式关闭 branch warmup，则仍可用纯 ReZero alpha=0 初始化。
        for m in fusion.modules():
            if hasattr(m, "up") and isinstance(m.up, torch.nn.Linear):
                torch.nn.init.normal_(m.up.weight, std=0.02)
                torch.nn.init.zeros_(m.up.bias)
        with torch.no_grad():
            initial_alpha = (args.branch_warmup_alpha if args.branch_warmup_steps > 0
                             else args.branch_alpha_init)
            fusion.branch_alpha.fill_(initial_alpha)

    # 随机初始化映射层、预训练小模型 LoRA 和标量 alpha 的优化尺度不同，不能共用
    # 一个 lr。alpha 不做 weight decay；LoRA 默认也不衰减，避免低秩更新被过早压小。
    scale_param = (fusion.gate_logit if fusion.gate_mode == "sigmoid"
                   else fusion.branch_alpha)
    bridge_params = [p for name, p in fusion.named_parameters()
                     if p.requires_grad and name not in {"branch_alpha", "gate_logit"}]
    small_lora_params = ([p for p in small_lora_model.parameters() if p.requires_grad]
                         if small_lora_model is not None else [])
    for p in small_lora_params:
        p.data = p.data.float()
    use_lora_delay = (bool(small_lora_params) and not args.resume
                      and args.small_lora_delay_steps > 0)
    if use_lora_delay:
        for p in small_lora_params:
            p.requires_grad_(False)
    use_branch_warmup = (not args.resume and fusion.gate_mode == "rezero"
                         and args.branch_warmup_steps > 0)
    if use_branch_warmup:
        scale_param.requires_grad_(False)
    scale_params = [scale_param]
    params = bridge_params + small_lora_params + scale_params
    n_bridge = sum(p.numel() for p in bridge_params)
    n_small_lora = sum(p.numel() for p in small_lora_params)
    print(f"可训练参数: bridge {n_bridge / 1e6:.2f}M | "
          f"small LoRA {n_small_lora / 1e6:.2f}M | "
          f"总计 {(n_bridge + n_small_lora) / 1e6:.2f}M")

    bridge_lr = args.bridge_lr if args.bridge_lr is not None else args.lr
    param_groups = [{"params": bridge_params, "lr": bridge_lr,
                     "weight_decay": args.weight_decay, "name": "bridge"}]
    if small_lora_params:
        param_groups.append({"params": small_lora_params, "lr": args.small_lora_lr,
                             "weight_decay": 0.0, "name": "small_lora"})
    param_groups.append({"params": scale_params, "lr": args.alpha_lr,
                         "weight_decay": 0.0, "name": "branch_alpha"})
    opt = torch.optim.AdamW(param_groups)
    print("优化器参数组: " + " | ".join(
        f"{g['name']} lr={g['lr']:.2e} wd={g['weight_decay']:g}"
        for g in param_groups))
    if use_branch_warmup:
        print(f"旁路 warmup: 前 {args.branch_warmup_steps} step 固定 "
              f"branch_alpha={args.branch_warmup_alpha:g}，随后释放 alpha")
    elif args.resume and args.branch_warmup_steps > 0:
        print("旁路 warmup: 检测到 --resume，保留检查点 scale 并直接联合优化")
    print("训练目标: response-only causal CE", end="")
    if args.guide_weight > 0:
        print(f" + 实验性 baseline 重加权(weight={args.guide_weight:g}, "
              f"margin={args.guide_margin:g}, every={args.guide_every})", end="")
    if args.contrast_weight > 0:
        print(f" + 旧 JS(weight={args.contrast_weight:g})", end="")
    print("；zero/shuffled 默认仅用于验证诊断")
    print(f"分支健康阈值: RMS ratio [{args.branch_min_rms_ratio:g}, "
          f"{args.branch_max_rms_ratio:g}] | loss tolerance "
          f"{args.branch_loss_tolerance:g} | 连续告警 {args.branch_health_patience} 次")
    if use_lora_delay:
        print(f"小模型 LoRA: 前 {args.small_lora_delay_steps} step 冻结，"
              "先让 bridge 对齐固定表征，随后解冻")
    dev = model_device(large)

    # ── 数据: 切出评估集(不参与训练, 用来对比 baseline/fusion loss) ──
    # 先全量加载, 从末尾切评估集, 再用 --max_samples 限制训练条数:
    # 这样 --max_samples 做快速测试时, 评估集仍来自文件末尾的真实验证集, 不会混进训练
    recs = load_records(args.data, 0)
    if not recs:
        raise SystemExit(f"[错误] 无训练数据: {args.data}")
    # 按 --eval_samples 从末尾切出评估集; 不再强行 10% 上限, 只保证至少留 1 条训练样本
    n_eval = min(args.eval_samples, len(recs) - 1) if args.eval_samples > 0 else 0
    eval_recs = recs[-n_eval:] if n_eval else []
    train_pool = recs[:-n_eval] if n_eval else recs
    train_recs = train_pool[:args.max_samples] if args.max_samples > 0 else train_pool
    print(f"训练样本: {len(train_recs)} 条 | 评估样本: {len(eval_recs)} 条")

    coll = lambda b: collate(b, tokenizer, args.max_len, args.answer_weight)
    loader = DataLoader(TextDataset(train_recs), batch_size=args.batch_size,
                        shuffle=True, collate_fn=coll)
    eval_loader = DataLoader(TextDataset(eval_recs), batch_size=args.eval_batch_size,
                             shuffle=False, collate_fn=coll) if eval_recs else None

    # 学习率调度: 线性 warmup → 余弦衰减到 10%×lr
    scheduler = None
    if args.warmup_steps > 0:
        total_steps = len(loader) * args.epochs
        def lr_lambda(step):
            if step < args.warmup_steps:
                # LambdaLR 在优化器第一次 step 前会先调用 step=0；使用 step+1，
                # 避免第一个 bridge warmup 更新的学习率恰好为 0。
                return (step + 1) / max(1, args.warmup_steps)
            p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            p = min(1.0, max(0.0, p))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def forward_logits(ids, mask):
        # bf16/fp16 权重用同精度 autocast(前向 bf16 + fp32 主权重, 梯度回传升回 fp32);
        # fp32 权重则不降精度直接算
        ctx = torch.autocast("cuda", dtype=dt) \
            if dt in (torch.bfloat16, torch.float16) and torch.cuda.is_available() \
            else nullcontext()
        with ctx:
            out = large.model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = large.lm_head(out.last_hidden_state)
        return logits

    @torch.no_grad()
    def evaluate():
        """评估正确旁路、关闭旁路、错配旁路的 token 加权 CE。"""
        large.eval()          # 关梯度检查点(只在 train 模式生效),no_grad 下干净前向
        if small_lora_model is not None:
            small_lora_model.eval()
        f_sum = b_sum = s_sum = 0.0
        f_weight = b_weight = s_weight = 0.0
        seen = 0
        for ids, mask, labels, weights in eval_loader:
            ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
            weights = weights.to(dev)
            fusion.enabled = False
            bs, bw = causal_lm_loss_parts(forward_logits(ids, mask), labels, weights)
            b_sum += bs.item()
            b_weight += bw.item()
            source = fusion.hook_state.get("h")
            fusion.enabled = True
            fusion.branch_override = mismatched_hidden(source, mask) if source is not None else None
            ss, sw = causal_lm_loss_parts(forward_logits(ids, mask), labels, weights)
            s_sum += ss.item()
            s_weight += sw.item()
            fusion.branch_override = None
            fs, fw = causal_lm_loss_parts(forward_logits(ids, mask), labels, weights)
            f_sum += fs.item()
            f_weight += fw.item()
            fusion.enabled = True
            seen += ids.size(0)
            if args.eval_max_samples > 0 and seen >= args.eval_max_samples:
                break
        large.train()         # 恢复训练模式(重新启用梯度检查点)
        if small_lora_model is not None:
            small_lora_model.train()
        return (f_sum / max(f_weight, 1.0), b_sum / max(b_weight, 1.0),
                s_sum / max(s_weight, 1.0))

    # ── 训练循环 ──
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fusion.train()
    train_ce_history = []   # (step, ce) 每一步训练 CE
    eval_history = []       # (step, fusion_loss, baseline_loss, shuffled_loss)
    best_fusion = float("inf")
    best_useful = float("inf")
    best_useful_step = 0
    best_step = 0
    patience_counter = 0
    branch_no_use_evals = 0
    stop = False
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for ids, mask, labels, weights in loader:
            ids = ids.to(dev)
            mask = mask.to(dev)
            labels = labels.to(dev)
            weights = weights.to(dev)

            if use_branch_warmup and step == args.branch_warmup_steps:
                scale_param.requires_grad_(True)
                print(f"    [阶段切换] step {step}: bridge warmup 完成，释放 branch_alpha")
            if use_lora_delay and step == args.small_lora_delay_steps:
                for p in small_lora_params:
                    p.requires_grad_(True)
                print(f"    [阶段切换] step {step}: bridge 已建立初始映射，解冻小模型 LoRA")

            opt.zero_grad()
            fusion.enabled = True
            large.train()                     # 确保梯度检查点生效
            if small_lora_model is not None:
                if use_lora_delay and step < args.small_lora_delay_steps:
                    small_lora_model.eval()
                else:
                    small_lora_model.train()

            # reference 全部 no_grad 且在正确分支之前执行。默认任务引导只需要
            # 冻结 4B baseline；shuffled 仅供显式开启的旧 JS 使用，不参与默认训练。
            guide_step = args.guide_weight > 0 and step % args.guide_every == 0
            need_reference = guide_step or args.contrast_weight > 0
            zero_losses = shuffled_logits = None
            if need_reference:
                large.eval()
                if small_lora_model is not None:
                    small_lora_model.eval()
                with torch.no_grad():
                    fusion.enabled = False
                    zero_logits = forward_logits(ids, mask)
                    if guide_step:
                        zero_losses = causal_lm_loss_per_example(
                            zero_logits, labels, weights)
                    del zero_logits
                    if args.contrast_weight > 0:
                        source = fusion.hook_state.get("h")
                        fusion.enabled = True
                        fusion.branch_override = (
                            mismatched_hidden(source, mask) if source is not None else None)
                        shuffled_logits = forward_logits(ids, mask)
                        fusion.branch_override = None
                large.train()
                if small_lora_model is not None:
                    if use_lora_delay and step < args.small_lora_delay_steps:
                        small_lora_model.eval()
                    else:
                        small_lora_model.train()

            fusion.enabled = True
            logits = forward_logits(ids, mask)
            ce = causal_lm_loss(logits, labels, weights)
            total = ce

            guide = guide_active = None
            if guide_step and zero_losses is not None:
                guide, guide_active = baseline_improvement_loss(
                    logits, zero_losses, labels, weights, margin=args.guide_margin)
                total = total + args.guide_weight * guide

            contrast = None
            if args.contrast_weight > 0 and shuffled_logits is not None:
                contrast = js_contrastive(logits, shuffled_logits, labels)
                total = total + args.contrast_weight * contrast
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"step {step + 1}: 训练 loss 非有限(CE={float(ce.detach())})")
            total.backward()
            # 分组裁剪，避免 44M/88M bridge 的总梯度范数把小 LoRA 的更新一并压小。
            bridge_grad = torch.nn.utils.clip_grad_norm_(
                bridge_params, args.grad_clip, error_if_nonfinite=True)
            lora_grad = (torch.nn.utils.clip_grad_norm_(
                small_lora_params, args.small_lora_grad_clip, error_if_nonfinite=True)
                if small_lora_params else torch.tensor(0.0))
            alpha_grad = torch.nn.utils.clip_grad_norm_(
                scale_params, args.alpha_grad_clip, error_if_nonfinite=True)
            opt.step()
            if fusion.gate_mode == "rezero" and args.branch_alpha_max > 0:
                with torch.no_grad():
                    fusion.branch_alpha.clamp_(
                        -args.branch_alpha_max, args.branch_alpha_max)
            if scheduler is not None:
                scheduler.step()

            step += 1
            train_ce_history.append((step, ce.item()))
            if step % args.log_every == 0:
                el = time.time() - t0
                gate = float(fusion.scale().detach())
                extra = f" | js对比 {contrast.item():.4f}" if contrast is not None else ""
                if guide is not None:
                    extra += (f" | baseline引导 {guide.item():.4f}"
                              f"(active={float(guide_active):.1%})")
                lrs = ",".join(f"{g['name']}={g['lr']:.2e}" for g in opt.param_groups)
                grads = (f"bridge={float(bridge_grad):.2e},"
                         f"lora={float(lora_grad):.2e},alpha={float(alpha_grad):.2e}")
                branch_ratio = fusion.hook_state.get("branch_rms_ratio", float("nan"))
                health = diagnose_train_branch(
                    branch_ratio, gate, bridge_grad,
                    alpha_grad=alpha_grad, lora_grad=lora_grad,
                    alpha_trainable=scale_param.requires_grad,
                    lora_trainable=any(p.requires_grad for p in small_lora_params),
                    min_ratio=args.branch_min_rms_ratio,
                    max_ratio=args.branch_max_rms_ratio)
                health_text = "OK" if not health else "WARN: " + "; ".join(health)
                print(f"epoch {ep + 1} step {step:>6} | CE {ce.item():.4f}{extra} "
                      f"| branch_scale {gate:.5f} | branch/base RMS {branch_ratio:.4f} "
                      f"| lr[{lrs}] | grad[{grads}] | health[{health_text}] | {el:.0f}s")

            if eval_loader is not None and step % args.eval_every == 0:
                f_loss, b_loss, s_loss = evaluate()
                eval_history.append((step, f_loss, b_loss, s_loss))
                gate = float(fusion.scale().detach())
                branch_state, branch_message = classify_branch_evaluation(
                    f_loss, b_loss, s_loss, args.branch_loss_tolerance)
                if branch_state == "USEFUL":
                    branch_no_use_evals = 0
                else:
                    branch_no_use_evals += 1
                print(f"    [评估] step {step:>6} | fusion {f_loss:.4f} | "
                      f"baseline {b_loss:.4f} | 增益 {b_loss - f_loss:+.4f} "
                      f"| shuffled {s_loss:.4f} | 对错配优势 {s_loss - f_loss:+.4f} "
                      f"| branch_scale {gate:.5f}")
                print(f"    [分支诊断] {branch_state}: {branch_message} "
                      f"| 连续无 USEFUL 证据 {branch_no_use_evals} 次")
                if (args.branch_health_patience > 0
                        and branch_no_use_evals >= args.branch_health_patience):
                    print("    [分支告警] 已连续多次未观察到正确旁路同时优于 zero/shuffled；"
                          "请检查 alpha、branch/base RMS、bridge 梯度及接入位置。"
                          "训练继续，不自动中止。")
                # 早停: 评估 fusion loss 连续 patience 次无改善(改善阈值 1e-4)就停
                if f_loss < best_fusion - 1e-4:
                    best_fusion, best_step = f_loss, step
                    patience_counter = 0
                    save_fusion_experiment(fusion, out_path + ".best", small_lora_model)
                else:
                    patience_counter += 1
                # “最低 fusion CE”不等于“旁路被有效使用”。另存一份同时优于关闭
                # 和错配旁路的 checkpoint，正式生成评测优先采用这份证据更强的权重。
                if f_loss < b_loss and f_loss < s_loss and f_loss < best_useful - 1e-4:
                    best_useful, best_useful_step = f_loss, step
                    save_fusion_experiment(
                        fusion, out_path + ".best_useful", small_lora_model)
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"    早停触发: {args.patience} 次评估无改善 "
                          f"(最优 fusion {best_fusion:.4f} @ step {best_step})")
                    stop = True
            if stop:
                break
        if stop:
            break

    save_fusion_experiment(fusion, out_path, small_lora_model)
    if plot_path:
        plot_losses(train_ce_history, eval_history, plot_path)
    print(f"\n训练完成, 共 {step} 步, 旁路参数已保存: {out_path}")
    if best_fusion < float("inf"):
        print(f"最优 fusion loss {best_fusion:.4f} @ step {best_step}, 已保存: {out_path}.best")
    if best_useful < float("inf"):
        print(f"最优有效旁路 loss {best_useful:.4f} @ step {best_useful_step}, "
              f"已保存: {out_path}.best_useful")
    elif eval_loader is not None:
        print("[警告] 本次训练没有 checkpoint 同时优于 zero 与 shuffled；"
              "不能据此声称小模型旁路有价值")
    print("验证: 在 test_fusion.py 里 load_fusion 后解码, 或直接跑 python main.py 看效果")


if __name__ == "__main__":
    main()
