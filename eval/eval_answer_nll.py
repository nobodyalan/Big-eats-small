# -*- coding: utf-8 -*-
"""Measure teacher-forced NLL on final-answer tokens only.

This is a diagnostic, not a generation-accuracy replacement.  For MATH and
GSM8K it conditions on the official gold rationale, then scores only the
answer content after the final ``\\boxed{...}`` or ``####`` marker.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core_training"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from main import (Config, attach_fusion, build_prompt_text, load_fusion,
                  resolve_dtype, resolve_model_path)
from eval_math import (ZIP_PATH, extract_boxed, extract_gold_gsm8k,
                       load_gsm8k_test, load_test_problems, math_prompt,
                       parse_levels)


def final_answer_span(response: str, kind: str):
    """Return the half-open character span of the final answer content."""
    if kind == "math":
        marker = response.rfind(r"\boxed")
        if marker < 0:
            return None
        cursor = marker + len(r"\boxed")
        while cursor < len(response) and response[cursor].isspace():
            cursor += 1
        if cursor >= len(response):
            return None
        if response[cursor] != "{":
            end = response.find("\n", cursor)
            end = len(response) if end < 0 else end
            while end > cursor and response[end - 1] in " \t。.;；$":
                end -= 1
            return (cursor, end) if end > cursor else None
        depth = 0
        start = cursor + 1
        for index in range(cursor, len(response)):
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
                if depth == 0:
                    return (start, index) if index > start else None
        return None

    if kind == "gsm8k":
        marker = response.rfind("####")
        if marker < 0:
            return None
        start = marker + len("####")
        while start < len(response) and response[start].isspace():
            start += 1
        end = response.find("\n", start)
        end = len(response) if end < 0 else end
        while end > start and response[end - 1].isspace():
            end -= 1
        return (start, end) if end > start else None

    raise ValueError(f"unknown answer kind: {kind}")


def encode_answer_example(tokenizer, prompt: str, response: str,
                          kind: str, max_len: int):
    """Encode prompt+gold response and label only answer-content tokens.

    Samples whose complete answer does not fit are reported as truncated and
    skipped rather than silently assigning them a misleading partial NLL.
    """
    span = final_answer_span(response, kind)
    if span is None:
        return None, "missing_answer_marker"

    prompt_text = build_prompt_text(tokenizer, prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = list(encoded.input_ids)
    offsets = list(encoded.offset_mapping)
    input_ids = prompt_ids + response_ids

    if len(input_ids) > max_len:
        return None, "answer_sequence_truncated"

    start, end = span
    selected = [
        index for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not selected:
        return None, "no_answer_tokens"
    # Subword tokenization can merge the last answer character with ``}``.
    # There is then no exact character-only token decomposition.  Include every
    # token overlapping the answer (the standard span-scoring convention), but
    # record the approximation instead of silently dropping difficult formats.
    boundary_merged_tokens = sum(
        offsets[index][0] < start or offsets[index][1] > end
        for index in selected
    )

    labels = [-100] * len(input_ids)
    prompt_len = len(prompt_ids)
    for index in selected:
        labels[prompt_len + index] = response_ids[index]

    # Token at position zero has no preceding logit and cannot be scored.
    if all(value == -100 for value in labels[1:]):
        return None, "no_predictable_answer_tokens"

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        "answer_tokens": len(selected),
        "answer_boundary_merged_tokens": boundary_merged_tokens,
        "sequence_tokens": len(input_ids),
    }, None


@torch.no_grad()
def score_example(model, encoded, device, dtype):
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    labels = encoded["labels"].to(device)
    ctx = (torch.autocast("cuda", dtype=dtype)
           if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
           else nullcontext())
    with ctx:
        outputs = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = model.lm_head(outputs.last_hidden_state)
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    return float(losses[valid].sum()), int(valid.sum())


def summarize(rows, label):
    valid = [row for row in rows if f"{label}_nll" in row]
    token_count = sum(row[f"{label}_answer_tokens"] for row in valid)
    loss_sum = sum(row[f"{label}_loss_sum"] for row in valid)
    token_nll = loss_sum / token_count if token_count else float("nan")
    example_nll = (
        sum(row[f"{label}_nll"] for row in valid) / len(valid)
        if valid else float("nan")
    )
    return {
        "scored_examples": len(valid),
        "answer_tokens": token_count,
        "boundary_merged_examples": sum(
            bool(row.get("answer_boundary_merged_tokens")) for row in valid
        ),
        "boundary_merged_tokens": sum(
            int(row.get("answer_boundary_merged_tokens", 0)) for row in valid
        ),
        "token_weighted_nll": token_nll,
        "question_mean_nll": example_nll,
        "token_weighted_perplexity": math.exp(token_nll) if math.isfinite(token_nll) else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score final-answer tokens under gold teacher forcing")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--segment", choices=["gsm8k", "math_lo", "math_hi"],
                        required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--math_lo", default="1-3")
    parser.add_argument("--math_hi", default="4-5")
    parser.add_argument("--zip", default=ZIP_PATH)
    parser.add_argument("--max_len", type=int, default=1536)
    parser.add_argument("--attn_impl", default="flash_attention_2")
    parser.add_argument("--bridge_depth", type=int, default=1)
    parser.add_argument("--bridge_mlp_dim", type=int, default=4096)
    parser.add_argument("--large_start", type=int, required=True)
    parser.add_argument("--large_end", type=int, required=True)
    parser.add_argument("--small_start", type=int, required=True)
    parser.add_argument("--small_end", type=int, required=True)
    parser.add_argument("--fusion_only", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    if args.segment == "gsm8k":
        items = load_gsm8k_test(args.seed, args.limit)
        answer_kind = "gsm8k"
        response_fn = lambda row: str(row.get("answer", ""))
        gold_fn = lambda row: extract_gold_gsm8k(row.get("answer", ""))
    else:
        levels = parse_levels(args.math_lo if args.segment == "math_lo" else args.math_hi)
        items = load_test_problems(args.zip, args.seed, args.limit, levels=levels)
        answer_kind = "math"
        response_fn = lambda row: str(row.get("solution", ""))
        gold_fn = lambda row: extract_boxed(row.get("solution", ""))

    cfg = Config()
    cfg.fusion_bridge_depth = args.bridge_depth
    cfg.fusion_mlp_dim = args.bridge_mlp_dim
    cfg.fusion_large_start = args.large_start
    cfg.fusion_large_end = args.large_end
    cfg.fusion_small_start = args.small_start
    cfg.fusion_small_end = args.small_end
    dtype = resolve_dtype(cfg.dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attention_kwargs = ({"attn_implementation": args.attn_impl}
                        if args.attn_impl else {})
    large_path = resolve_model_path(cfg.model_large_id, cfg.model_large_local)
    small_path = resolve_model_path(cfg.model_small_id, cfg.model_small_local)
    tokenizer = AutoTokenizer.from_pretrained(large_path)
    small = AutoModelForCausalLM.from_pretrained(small_path, dtype=dtype).to(device).eval()
    large = AutoModelForCausalLM.from_pretrained(
        large_path, dtype=dtype, **attention_kwargs).to(device).eval()
    fusion = attach_fusion(large, small, cfg)
    load_fusion(fusion, args.ckpt)
    # The fusion module is hook-owned rather than a registered child of the
    # large model, so ``large.eval()`` does not recursively switch it.
    fusion.eval()

    rows = []
    skipped = {}
    started = time.time()
    for index, item in enumerate(items, 1):
        response = response_fn(item)
        encoded, reason = encode_answer_example(
            tokenizer, math_prompt(item.get("problem", "")), response,
            answer_kind, args.max_len)
        row = {
            "id": item.get("benchmark_id", index),
            "problem": item.get("problem", ""),
            "gold": gold_fn(item),
        }
        if encoded is None:
            row["skipped"] = reason
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            row["sequence_tokens"] = encoded["sequence_tokens"]
            row["answer_boundary_merged_tokens"] = encoded[
                "answer_boundary_merged_tokens"]
            if not args.fusion_only:
                fusion.enabled = False
                loss_sum, tokens = score_example(large, encoded, device, dtype)
                row.update({
                    "baseline_loss_sum": loss_sum,
                    "baseline_answer_tokens": tokens,
                    "baseline_nll": loss_sum / tokens,
                })
            fusion.enabled = True
            loss_sum, tokens = score_example(large, encoded, device, dtype)
            row.update({
                "fusion_loss_sum": loss_sum,
                "fusion_answer_tokens": tokens,
                "fusion_nll": loss_sum / tokens,
            })
        rows.append(row)
        if args.log_every > 0 and index % args.log_every == 0:
            current = summarize(rows, "fusion")
            print(f"[{args.segment} {index}/{len(items)}] "
                  f"fusion answer NLL={current['token_weighted_nll']:.6f}",
                  flush=True)

    fingerprint = [
        {"id": row["id"], "problem": row["problem"], "gold": row["gold"]}
        for row in rows
    ]
    summary = {
        "segment": args.segment,
        "requested_examples": len(items),
        "seed": args.seed,
        "max_len": args.max_len,
        "answer_nll_definition": (
            "teacher-forced on official gold rationale; loss only on final "
            "tokens overlapping final-answer content, excluding EOS; tokenizer "
            "tokens crossing a marker/brace boundary are included and counted"
        ),
        "question_set_sha256": hashlib.sha256(json.dumps(
            fingerprint, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest(),
        "skipped": skipped,
        "fusion": summarize(rows, "fusion"),
        "elapsed_seconds": time.time() - started,
    }
    if not args.fusion_only:
        summary["baseline"] = summarize(rows, "baseline")
        summary["fusion_minus_baseline_token_nll"] = (
            summary["fusion"]["token_weighted_nll"]
            - summary["baseline"]["token_weighted_nll"]
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    temporary = f"{args.out}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump({
            "ckpt": args.ckpt,
            "positions": {
                "large_start": args.large_start,
                "large_end": args.large_end,
                "small_start": args.small_start,
                "small_end": args.small_end,
            },
            "summary": summary,
            "results": rows,
        }, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, args.out)

    print("\n===== 最终答案 token NLL =====")
    if "baseline" in summary:
        print(f"baseline: {summary['baseline']['token_weighted_nll']:.6f}")
    print(f"fusion:   {summary['fusion']['token_weighted_nll']:.6f}")
    if "fusion_minus_baseline_token_nll" in summary:
        print("fusion-baseline: "
              f"{summary['fusion_minus_baseline_token_nll']:+.6f}")
    print(f"scored/skipped: {summary['fusion']['scored_examples']}/{sum(skipped.values())}")
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
