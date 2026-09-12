# -*- coding: utf-8 -*-
"""Print the best screened position as shell-friendly integer fields."""

import json
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: best_position.py POSITION_SELECTION.json")
    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)
    candidates = payload.get("task_aware") or payload.get("exit") or []
    if not candidates:
        raise SystemExit("position selection contains no task-aware candidates")
    best = candidates[0]
    large_start = int(best["L"])
    large_end = int(best["l2"])
    small_start = int(best["a"])
    small_end_inclusive = int(best["b"]) - 1
    print(large_start, large_end, small_start, small_end_inclusive)


if __name__ == "__main__":
    main()
