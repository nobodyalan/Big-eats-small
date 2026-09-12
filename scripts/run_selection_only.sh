#!/usr/bin/env bash
# 只筛选，不启动 adapter 短训或准确率 benchmark。
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/mix_all.jsonl}"
GPU="${GPU:-0}"
OUT="${OUT:-cache/task_aware_selection.json}"

CUDA_VISIBLE_DEVICES="$GPU" python3 core_training/select_positions.py \
  --data "$DATA" --num_samples 300 --max_len 192 --fit_ratio 0.7 \
  --topk 9 --segment_lengths 1,2,4,8,12 --inject_spans 6,12,18 \
  --exit_topk 18 --exit_samples 64 --exit_max_len 128 \
  --intervention_ratios 0.01,0.03,0.10 --rank_ratio 0.03 \
  --out "$OUT"

python3 scripts/summarize_selection.py "$OUT" --topn 18

echo "筛选完成：$OUT"
