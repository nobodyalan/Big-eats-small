#!/usr/bin/env bash
# 一键全流程: 数据驱动选位置(入口/片段/出口 Q) → 并行短训 → 三难度准确率定胜负
# 用法: bash scripts/run_position_search.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # 切到 BES 根目录

# ================= 可调参数 =================
DATA="data/mix_all.jsonl"
GPUS="0,1"              # 短训用哪些 GPU(逗号分隔)
PARALLEL=2              # 同时短训几个候选
NUM_SAMPLES=500         # ① 入口筛选的校准样本数
MAX_LEN=256             # 入口筛选的单条 token 上限
TOP_N=5                 # 出口筛选 + 短训验证的候选数
EXIT_SAMPLES=32         # 出口筛选(Q)的校准样本数
MAX_SAMPLES=1000        # 短训预算(样本数)
EPOCHS=1                # 短训 epoch
ACC_LIMIT=100            # ④ 最终准确率: 三难度各题数
# ============================================

echo "=================================================================="
echo "阶段 1/2: 数据驱动选位置(① 入口 CKA/E_in → ② 片段 E_seg → ③ 出口 Q)"
echo "=================================================================="
python3 core_training/select_positions.py \
    --data "$DATA" --num_samples "$NUM_SAMPLES" --max_len "$MAX_LEN" \
    --topk "$TOP_N" --exit_topk "$TOP_N" --exit_samples "$EXIT_SAMPLES" \
    --out cache/position_selection.json

echo ""
echo "=================================================================="
echo "阶段 2/2: 并行短训验证 + 三难度各 ${ACC_LIMIT} 题准确率(④ 最终标准)"
echo "=================================================================="
python3 core_training/validate_positions.py \
    --candidates_json cache/position_selection.json --topn "$TOP_N" \
    --max_samples "$MAX_SAMPLES" --epochs "$EPOCHS" \
    --gpus "$GPUS" --parallel "$PARALLEL" \
    --acc_limit "$ACC_LIMIT" \
    --out_dir cache/validate_positions

echo ""
echo "=================================================================="
echo "完成。选位置结果: cache/position_selection.json"
echo "短训汇总:       cache/validate_positions/validation_summary.json"
echo "=================================================================="
