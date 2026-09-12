#!/usr/bin/env bash
# 两阶段正式流程：GPU0 位置筛选；GPU1 串行训练三组模型并各评测三段 400 题。
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
SELECTION_GPU="${SELECTION_GPU:-0}"
EXPERIMENT_GPU="${EXPERIMENT_GPU:-1}"
STAGE="${STAGE:-all}"                    # all / selection / experiments
PREPARE_DATA="${PREPARE_DATA:-auto}"          # auto / 1 / 0
DATA="${DATA:-data/math_majority_all.jsonl}"
DATA_MANIFEST="${DATA_MANIFEST:-data/math_majority_manifest.json}"
SELECTION_OUT="${SELECTION_OUT:-cache/two_gpu_pipeline/position_selection.json}"
EXPERIMENT_OUT="${EXPERIMENT_OUT:-cache/two_gpu_pipeline/experiments}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$SELECTION_OUT")" "$EXPERIMENT_OUT"

prepare_training_data() {
  if [[ ! -f data/.cache/math/MATH.zip || ! -f data/metamath_math.jsonl || ! -f data/metamath_gsm8k.jsonl ]]; then
    echo "无法生成 $DATA：缺少 MATH.zip、metamath_math.jsonl 或 metamath_gsm8k.jsonl" >&2
    exit 2
  fi
  echo "准备 MATH 为主的增强数据（官方 MATH + MetaMathQA-MATH/GSM8K）"
  "$PYTHON" scripts/prepare_math_majority.py \
    --math_zip data/.cache/math/MATH.zip \
    --meta_math data/metamath_math.jsonl \
    --meta_gsm8k data/metamath_gsm8k.jsonl \
    --math_val_ratio 0.2 --meta_math_train 18000 \
    --meta_gsm_train 8000 --meta_gsm_val 500 --seed "$SEED" \
    --out_combined "$DATA" --manifest "$DATA_MANIFEST"
}

data_is_current() {
  [[ -f "$DATA" && -f "$DATA_MANIFEST" ]] && \
    "$PYTHON" -c "import json; assert json.load(open('$DATA_MANIFEST', encoding='utf-8')).get('recipe_version') == 'metamath_math_majority_v2'" \
    >/dev/null 2>&1
}

if [[ "$PREPARE_DATA" == "1" ]]; then
  prepare_training_data
elif [[ "$PREPARE_DATA" == "auto" ]]; then
  if data_is_current; then
    echo "当前版本训练数据已存在，自动跳过生成: $DATA"
  else
    echo "训练数据缺失或配方版本过旧，将从服务器本地数据池重新生成"
    prepare_training_data
  fi
elif [[ "$PREPARE_DATA" == "0" ]]; then
  if [[ ! -f "$DATA" ]]; then
    echo "PREPARE_DATA=0，但训练数据不存在: $DATA" >&2
    exit 2
  fi
else
  echo "PREPARE_DATA 必须是 auto、1 或 0" >&2
  exit 2
fi

if [[ "$STAGE" == "all" || "$STAGE" == "selection" ]]; then
  echo "=================================================================="
  echo "阶段 1/2: GPU $SELECTION_GPU 任务感知接入位置筛选"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES="$SELECTION_GPU" "$PYTHON" core_training/select_positions.py \
    --data "$DATA" --num_samples 384 --max_len 192 --batch_size 8 \
    --fit_ratio 0.7 --topk 9 --segment_lengths 1,2,4,8,12 \
    --inject_spans 6,12,18 --exit_topk 18 --exit_samples 128 \
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

if [[ -f "$DATA_MANIFEST" ]]; then
  MANIFEST_EVAL_SAMPLES=$("$PYTHON" -c \
    "import json; print(json.load(open('$DATA_MANIFEST', encoding='utf-8')).get('eval_samples', 2000))")
else
  MANIFEST_EVAL_SAMPLES=2000
fi
TRAIN_EVAL_SAMPLES="${EVAL_SAMPLES:-$MANIFEST_EVAL_SAMPLES}"

echo "=================================================================="
echo "阶段 2/2: GPU $EXPERIMENT_GPU 串行训练与评测"
echo "三段评测: GSM8K / MATH Level 1-3 / MATH Level 4-5，各 400 题"
echo "=================================================================="
PYTHON="$PYTHON" DATA="$DATA" GPU="$EXPERIMENT_GPU" OUT_ROOT="$EXPERIMENT_OUT" \
MAX_SAMPLES=0 EVAL_SAMPLES="$TRAIN_EVAL_SAMPLES" ACC_LIMIT=400 MATH_LO=1-3 MATH_HI=4-5 \
SEED="$SEED" LARGE_START="$LARGE_START" LARGE_END="$LARGE_END" \
SMALL_START="$SMALL_START" SMALL_END="$SMALL_END" \
bash scripts/run_capacity_lora_experiments.sh

echo "两阶段流程完成"
echo "筛选结果: $SELECTION_OUT"
echo "训练与评测: $EXPERIMENT_OUT"
