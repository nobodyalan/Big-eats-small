#!/usr/bin/env bash
# 两阶段正式流程：GPU0 位置筛选；GPU1 串行训练三组模型并各评测三段 400 题。
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
SELECTION_GPU="${SELECTION_GPU:-0}"
EXPERIMENT_GPU="${EXPERIMENT_GPU:-1}"
STAGE="${STAGE:-all}"                    # all / selection / experiments
PREPARE_DATA="${PREPARE_DATA:-1}"
DATA="${DATA:-data/math_majority_all.jsonl}"
SELECTION_OUT="${SELECTION_OUT:-cache/two_gpu_pipeline/position_selection.json}"
EXPERIMENT_OUT="${EXPERIMENT_OUT:-cache/two_gpu_pipeline/experiments}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$SELECTION_OUT")" "$EXPERIMENT_OUT"

if [[ "$PREPARE_DATA" == "1" ]]; then
  echo "准备 MATH 为主的训练数据（MATH Level 1-5 + GSM8K）"
  "$PYTHON" scripts/prepare_math_majority.py \
    --math_zip data/.cache/math/MATH.zip \
    --gsm8k data/train_metamath.jsonl \
    --math_val_ratio 0.2 --gsm_train 2000 --gsm_val 500 --seed "$SEED" \
    --out_combined "$DATA"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "selection" ]]; then
  echo "=================================================================="
  echo "阶段 1/2: GPU $SELECTION_GPU 任务感知接入位置筛选"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES="$SELECTION_GPU" "$PYTHON" core_training/select_positions.py \
    --data "$DATA" --num_samples 300 --max_len 192 --batch_size 4 \
    --fit_ratio 0.7 --topk 9 --segment_lengths 1,2,4,8,12 \
    --inject_spans 6,12,18 --exit_topk 18 --exit_samples 64 \
    --exit_max_len 128 --intervention_ratios 0.01,0.03,0.10 \
    --rank_ratio 0.03 --out "$SELECTION_OUT"
  "$PYTHON" scripts/summarize_selection.py "$SELECTION_OUT" --topn 18
fi

if [[ "$STAGE" == "selection" ]]; then
  echo "阶段 1 完成: $SELECTION_OUT"
  exit 0
fi

if [[ "$STAGE" != "all" && "$STAGE" != "experiments" ]]; then
  echo "STAGE 必须是 all、selection 或 experiments" >&2
  exit 2
fi
if [[ ! -f "$SELECTION_OUT" ]]; then
  echo "缺少筛选结果: $SELECTION_OUT；请先运行 STAGE=selection" >&2
  exit 2
fi

# task_aware 已按“任务 ΔNLL 为负 + 优于 bridge-only control”排序。
read -r LARGE_START LARGE_END SMALL_START SMALL_END < <(
  "$PYTHON" scripts/best_position.py "$SELECTION_OUT"
)
echo "选中位置: large $LARGE_START -> $LARGE_END | small $SMALL_START..$SMALL_END (inclusive)"

echo "=================================================================="
echo "阶段 2/2: GPU $EXPERIMENT_GPU 串行训练与评测"
echo "三段评测: GSM8K / MATH Level 1-3 / MATH Level 4-5，各 400 题"
echo "=================================================================="
PYTHON="$PYTHON" DATA="$DATA" GPU="$EXPERIMENT_GPU" OUT_ROOT="$EXPERIMENT_OUT" \
MAX_SAMPLES=0 EVAL_SAMPLES=2000 ACC_LIMIT=400 MATH_LO=1-3 MATH_HI=4-5 \
SEED="$SEED" LARGE_START="$LARGE_START" LARGE_END="$LARGE_END" \
SMALL_START="$SMALL_START" SMALL_END="$SMALL_END" \
bash scripts/run_capacity_lora_experiments.sh

echo "两阶段流程完成"
echo "筛选结果: $SELECTION_OUT"
echo "训练与评测: $EXPERIMENT_OUT"
