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
RUN_CONTROLS="${RUN_CONTROLS:-0}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
WARMUP_STEPS="${WARMUP_STEPS:-400}"
BRIDGE_LR="${BRIDGE_LR:-1e-4}"
SMALL_LORA_LR="${SMALL_LORA_LR:-2e-5}"
SMALL_LORA_DELAY_STEPS="${SMALL_LORA_DELAY_STEPS:-200}"
ALPHA_LR="${ALPHA_LR:-5e-4}"
BRANCH_WARMUP_STEPS="${BRANCH_WARMUP_STEPS:-400}"
BRANCH_WARMUP_ALPHA="${BRANCH_WARMUP_ALPHA:-0.05}"
BRANCH_ALPHA_MAX="${BRANCH_ALPHA_MAX:-0.25}"
GUIDE_WEIGHT="${GUIDE_WEIGHT:-0}"
GUIDE_MARGIN="${GUIDE_MARGIN:-0.02}"
GUIDE_EVERY="${GUIDE_EVERY:-4}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-2.0}"

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
        --bridge_lr "$BRIDGE_LR" --small_lora_lr "$SMALL_LORA_LR" --alpha_lr "$ALPHA_LR"
        --small_lora_delay_steps "$SMALL_LORA_DELAY_STEPS"
        --branch_warmup_steps "$BRANCH_WARMUP_STEPS" --branch_warmup_alpha "$BRANCH_WARMUP_ALPHA"
        --branch_alpha_max "$BRANCH_ALPHA_MAX"
        --guide_weight "$GUIDE_WEIGHT" --guide_margin "$GUIDE_MARGIN" --guide_every "$GUIDE_EVERY"
        --small_lora_grad_clip 0.5 --alpha_grad_clip 0.1 \
        --contrast_weight 0 --answer_weight "$ANSWER_WEIGHT" --seed "$SEED")

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

DEEP_CKPT="$DEEP_OUT.best_useful"
[[ -f "$DEEP_CKPT" ]] || DEEP_CKPT="$DEEP_OUT.best"
[[ -f "$DEEP_CKPT" ]] || DEEP_CKPT="$DEEP_OUT"
if [[ "$RUN_EVAL" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --attn_impl "$ATTN_IMPL" --fusion_only \
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
  --small_lora_r 32 --small_lora_alpha 64 --small_lora_dropout 0.0 \
  --out "$SMALL_OUT" --plot "$OUT_ROOT/small_lora_seed${SEED}.png"

SMALL_CKPT="$SMALL_OUT.best_useful"
[[ -f "$SMALL_CKPT" ]] || SMALL_CKPT="$SMALL_OUT.best"
[[ -f "$SMALL_CKPT" ]] || SMALL_CKPT="$SMALL_OUT"
SMALL_LORA_CKPT="$SMALL_CKPT.small_lora"
if [[ "$RUN_EVAL" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --attn_impl "$ATTN_IMPL" --fusion_only \
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
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --answer_weight "$ANSWER_WEIGHT" --seed "$SEED" \
  --out "$LARGE_OUT" --plot "$OUT_ROOT/large_lora_r64_seed${SEED}.png"

LARGE_CKPT="$LARGE_OUT.best"
[[ -d "$LARGE_CKPT" ]] || LARGE_CKPT="$LARGE_OUT"
if [[ "$RUN_EVAL" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --attn_impl "$ATTN_IMPL" --fusion_only \
    --lora_ckpt "$LARGE_CKPT" --out_dir "$OUT_ROOT/eval" \
    --tag "large_lora_r64_seed${SEED}"

  echo "=================================================================="
  echo "共享对照: 纯 4B baseline（只评测一次，避免三组实验重复生成）"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
    --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
    --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --attn_impl "$ATTN_IMPL" \
    --baseline_only --out_dir "$OUT_ROOT/eval" \
    --tag "base_4b_seed${SEED}"
fi

if [[ "$RUN_CONTROLS" == "1" ]]; then
  echo "=================================================================="
  echo "附加完整训练 controls: 冻结小模型标准 bridge + depth1/depth2 bridge-only"
  echo "=================================================================="
  for SPEC in "frozen_small:1:0" "bridge_only_d1:1:1" "bridge_only_d2:2:1"; do
    IFS=: read -r CONTROL_NAME CONTROL_DEPTH CONTROL_BYPASS <<< "$SPEC"
    CONTROL_OUT="$OUT_ROOT/${CONTROL_NAME}_seed${SEED}.pt"
    CONTROL_ARGS=(--bridge_depth "$CONTROL_DEPTH" --bridge_mlp_dim 4096 --small_lora_r 0)
    if [[ "$CONTROL_BYPASS" == "1" ]]; then
      CONTROL_ARGS+=(--bypass_small)
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" core_training/train_fusion.py \
      "${COMMON[@]}" "${POS_ARGS[@]}" "${CONTROL_ARGS[@]}" \
      --out "$CONTROL_OUT" --plot "$OUT_ROOT/${CONTROL_NAME}_seed${SEED}.png"
    CONTROL_CKPT="$CONTROL_OUT.best_useful"
    [[ -f "$CONTROL_CKPT" ]] || CONTROL_CKPT="$CONTROL_OUT.best"
    [[ -f "$CONTROL_CKPT" ]] || CONTROL_CKPT="$CONTROL_OUT"
    if [[ "$RUN_EVAL" == "1" ]]; then
      CONTROL_EVAL_ARGS=()
      if [[ "$CONTROL_BYPASS" == "1" ]]; then
        CONTROL_EVAL_ARGS+=(--bypass_small)
      fi
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval/eval_math.py \
        --bench segments --limit "$ACC_LIMIT" --seed "$SEED" \
        --math_lo "$MATH_LO" --math_hi "$MATH_HI" --max_new "$MAX_NEW" --attn_impl "$ATTN_IMPL" --fusion_only \
        --bridge_depth "$CONTROL_DEPTH" --bridge_mlp_dim 4096 "${POS_ARGS[@]}" \
        "${CONTROL_EVAL_ARGS[@]}" --ckpt "$CONTROL_CKPT" \
        --out_dir "$OUT_ROOT/eval" --tag "${CONTROL_NAME}_seed${SEED}"
    fi
  done
fi

echo "全部完成。结果目录: $OUT_ROOT"
