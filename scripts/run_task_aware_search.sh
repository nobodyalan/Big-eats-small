#!/usr/bin/env bash
# 多阶段任务感知位置搜索。所有命令在 BES 根目录运行。
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/mix_all.jsonl}"
GPUS="${GPUS:-0,1}"
PARALLEL="${PARALLEL:-2}"
STAGE1_GPU="${STAGE1_GPU:-0}"
ROOT_OUT="${ROOT_OUT:-cache/task_aware_search}"

mkdir -p "$ROOT_OUT"

echo "阶段 1/3: 分层结构诊断 + 等 RMS 任务 ΔNLL"
CUDA_VISIBLE_DEVICES="$STAGE1_GPU" python3 core_training/select_positions.py \
  --data "$DATA" --num_samples 300 --max_len 192 --fit_ratio 0.7 \
  --topk 9 --segment_lengths 1,2,4,8,12 --inject_spans 6,12,18 \
  --exit_topk 18 --exit_samples 64 --exit_max_len 128 \
  --intervention_ratios 0.01,0.03,0.10 --rank_ratio 0.03 \
  --out "$ROOT_OUT/position_selection.json"

echo "阶段 2/3: 3 seeds、等预算短训；以 held-out CE gain 排序"
for seed in 42 43 44; do
  python3 core_training/validate_positions.py \
    --candidates_json "$ROOT_OUT/position_selection.json" --topn 8 \
    --data "$DATA" --max_samples 2000 --epochs 1 --batch_size 4 --max_len 512 \
    --eval_samples 256 --eval_every 50 --warmup_steps 50 \
    --contrast_weight 0 --primary gain --skip_accuracy \
    --seed "$seed" --gpus "$GPUS" --parallel "$PARALLEL" \
    --out_dir "$ROOT_OUT/short_seed_${seed}"
done

python3 scripts/aggregate_position_trials.py \
  --inputs "$ROOT_OUT/short_seed_*/validation_summary.json" \
  --topn 3 --out "$ROOT_OUT/short_aggregate.json"

echo "阶段 3/3: 前 3 候选、3 seeds、三难度各 300 题"
for seed in 101 102 103; do
  python3 core_training/validate_positions.py \
    --candidates_json "$ROOT_OUT/short_aggregate.json" --topn 3 \
    --data "$DATA" --max_samples 6000 --epochs 1 --batch_size 4 --max_len 768 \
    --eval_samples 512 --eval_every 100 --warmup_steps 100 \
    --contrast_weight 0 --primary micro_accuracy --acc_limit 300 \
    --seed "$seed" --acc_seed "$seed" --gpus "$GPUS" --parallel "$PARALLEL" \
    --out_dir "$ROOT_OUT/medium_seed_${seed}"
done

python3 scripts/aggregate_position_trials.py \
  --inputs "$ROOT_OUT/medium_seed_*/validation_summary.json" \
  --topn 2 --out "$ROOT_OUT/finalists.json"

echo "完成。最终两个候选: $ROOT_OUT/finalists.json"
echo "下一步应对两个候选和默认方案各做至少 5 seeds 完整训练，再在未参与选型的完整测试集评测。"
