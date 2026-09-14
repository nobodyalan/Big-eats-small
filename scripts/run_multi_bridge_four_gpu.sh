#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_ROOT="configs/experiments/multi_bridge/qwen3_4b__qwen3_06b"
mkdir -p logs/multi_bridge/qwen3_4b__qwen3_06b/launcher

configs=(
  "$CONFIG_ROOT/early_middle/staged/train_seed42.json"
  "$CONFIG_ROOT/middle_late/staged/train_seed42.json"
  "$CONFIG_ROOT/early_late/staged/train_seed42.json"
  "$CONFIG_ROOT/early_middle/joint/train_seed42.json"
)

pids=()
for config in "${configs[@]}"; do
  "$PYTHON_BIN" scripts/run_config.py "$config" &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[完成] ${configs[$index]}"
  else
    code=$?
    echo "[失败 code=$code] ${configs[$index]}" >&2
    status=1
  fi
done
exit "$status"
