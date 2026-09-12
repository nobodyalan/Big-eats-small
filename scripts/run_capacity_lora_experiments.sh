#!/usr/bin/env bash
# 三组独立实验：深 bridge / bridge+小模型 LoRA / 标准 4B LoRA baseline。
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
DATA="${DATA:-data/math_majority_all.jsonl}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-cache/capacity_lora_experiments}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LEN="${MAX_LEN:-1536}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-400}"
ACC_LIMIT="${ACC_LIMIT:-400}"
MATH_LO="${MATH_LO:-1-3}"
MATH_HI="${MATH_HI:-4-5}"
MAX_NEW="${MAX_NEW:-1024}"
SEED="${SEED:-42}"
RUN_EVAL="${RUN_EVAL:-1}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
WARMUP_STEPS="${WARMUP_STEPS:-400}"
GATE_INIT="${GATE_INIT:--2.0}"

# 可选位置覆盖。small_end 沿用 train_fusion 的 inclusive 语义。
POS_ARGS=()
[[ -n "${LARGE_START:-}" ]] && POS_ARGS+=(--large_start "$LARGE_START")
[[ -n "${LARGE_END:-}" ]] && POS_ARGS+=(--large_end "$LARGE_END")
[[ -n "${SMALL_START:-}" ]] && POS_ARGS+=(--small_start "$SMALL_START")
[[ -n "${SMALL_END:-}" ]] && POS_ARGS+=(--small_end "$SMALL_END")

COMMON=(--data "$DATA" --max_samples "$MAX_SAMPLES" --epochs "$EPOCHS"
        --batch_size "$BATCH_SIZE" --max_len "$MAX_LEN"
        --eval_samples "$EVAL_SAMPLES" --eval_every "$EVAL_EVERY"
        --eval_max_samples "$EVAL_MAX_SAMPLES" --warmup_steps "$WARMUP_STEPS"
        --grad_checkpoint 1 --attn_impl "$ATTN_IMPL"
        --contrast_weight 0 --gate_init "$GATE_INIT" --seed "$SEED")

mkdir -p "$OUT_ROOT"

echo "=================================================================="
echo "实验 1: 深 bridge，depth=2 / mlp=4096；大小模型全部冻结"
echo "预计可训练参数约 88.1M；程序会打印精确值"
echo "=================================================================="
DEEP_OUT="$OUT_ROOT/deep_bridge_seed${SEED}.pt"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" core_training/train_fusion.py \
  "${COMMON[@]}" "${POS_ARGS[@]}" \
  --bridge_depth 2 --bridge_mlp_dim 4096 --small_lora_r 0 \
  --out "$DEEP_OUT" --plot "$OUT_ROOT/deep_bridge_seed${SEED}.png"

DEEP_CKPT="$DEEP_OUT.best"
[[ -f "$DEEP_CKPT" ]] || DEEP_CKPT="$DEEP_OUT"
if [[ "$RUN_EVAL" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --fusion_only \
    --bridge_depth 2 --bridge_mlp_dim 4096 "${POS_ARGS[@]}" \
    --ckpt "$DEEP_CKPT" --out_dir "$OUT_ROOT/eval" \
    --tag "deep_bridge_seed${SEED}"
fi

echo "=================================================================="
echo "实验 2: 标准 depth=1 bridge + 小模型片段 LoRA r=32"
echo "预计 bridge 44.1M + small LoRA 约 7-8M；程序会打印精确值"
echo "=================================================================="
SMALL_OUT="$OUT_ROOT/small_lora_seed${SEED}.pt"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" core_training/train_fusion.py \
  "${COMMON[@]}" "${POS_ARGS[@]}" \
  --bridge_depth 1 --bridge_mlp_dim 4096 \
  --small_lora_r 32 --small_lora_alpha 64 --small_lora_dropout 0.05 \
  --out "$SMALL_OUT" --plot "$OUT_ROOT/small_lora_seed${SEED}.png"

SMALL_CKPT="$SMALL_OUT.best"
[[ -f "$SMALL_CKPT" ]] || SMALL_CKPT="$SMALL_OUT"
SMALL_LORA_CKPT="$SMALL_CKPT.small_lora"
if [[ "$RUN_EVAL" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --fusion_only \
    --bridge_depth 1 --bridge_mlp_dim 4096 "${POS_ARGS[@]}" \
    --ckpt "$SMALL_CKPT" --small_lora_ckpt "$SMALL_LORA_CKPT" \
    --out_dir "$OUT_ROOT/eval" --tag "small_lora_seed${SEED}"
fi

echo "=================================================================="
echo "实验 3: 纯 4B 标准 LoRA，r=64 / alpha=128 / 全 attention+MLP"
echo "预计可训练参数约 132M；不按 bridge 参数量缩放"
echo "=================================================================="
LARGE_OUT="$OUT_ROOT/large_lora_r64_seed${SEED}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" core_training/train_lora.py \
  --data "$DATA" --max_samples "$MAX_SAMPLES" --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" --max_len "$MAX_LEN" \
  --eval_samples "$EVAL_SAMPLES" --eval_every "$EVAL_EVERY" \
  --eval_max_samples "$EVAL_MAX_SAMPLES" --warmup_steps "$WARMUP_STEPS" \
  --grad_checkpoint 1 --attn_impl "$ATTN_IMPL" \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 --seed "$SEED" \
  --out "$LARGE_OUT" --plot "$OUT_ROOT/large_lora_r64_seed${SEED}.png"

LARGE_CKPT="$LARGE_OUT.best"
[[ -d "$LARGE_CKPT" ]] || LARGE_CKPT="$LARGE_OUT"
if [[ "$RUN_EVAL" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --fusion_only \
    --lora_ckpt "$LARGE_CKPT" --out_dir "$OUT_ROOT/eval" \
    --tag "large_lora_r64_seed${SEED}"

  echo "=================================================================="
  echo "共享对照: 纯 4B baseline（只评测一次，避免三组实验重复生成）"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" \
    --baseline_only --out_dir "$OUT_ROOT/eval" \
    --tag "base_4b_seed${SEED}"
fi

echo "全部完成。结果目录: $OUT_ROOT"
