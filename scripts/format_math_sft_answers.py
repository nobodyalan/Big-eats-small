# -*- coding: utf-8 -*-
"""把数学 SFT response 规范为唯一的 ``Final answer: \\boxed{...}``。

不修改题面或推理语义；从 response 中最后出现的明确答案标记
(``\\boxed`` / ``####`` / ``The answer is:``)提取答案，去掉旧 boxed 外框和
尾部重复答案行，再把唯一的规范答案追加到末尾。这样避免 Math-Verify 将两个
区间/坐标答案合并成集合。
输出采用原子替换，默认拒绝覆盖输入文件。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


CANONICAL_RE = re.compile(
    r"Final\s+answer\s*:\s*\\boxed\{", re.IGNORECASE)
ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|:)\s*([^\r\n]+)",
    re.IGNORECASE,
)
HASH_RE = re.compile(r"####\s*([^\r\n]+)")


def extract_boxed_span(text: str, start: int):
    """返回 ``(内容, 命令结束位置)``；兼容有/无花括号的 boxed。"""
    cursor = start + len(r"\boxed")
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return None
    if text[cursor] != "{":
        candidate = text[cursor:].splitlines()[0].strip()
        if candidate.startswith("$"):
            candidate = candidate[1:]
        math_end = candidate.find("$")
        if math_end >= 0:
            candidate = candidate[:math_end]
        candidate = candidate.rstrip("。.;；").strip()
        if not candidate:
            return None
        end = cursor + len(candidate)
        return candidate, end

    brace = cursor
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:index], index + 1
    return None


def extract_balanced_boxed(text: str, start: int):
    span = extract_boxed_span(text, start)
    return span[0] if span is not None else None


def unwrap_all_boxed(text: str) -> str:
    """只移除 boxed 外框，保留其中数学内容；支持少量嵌套。"""
    current = text
    for _ in range(10):
        pieces = []
        cursor = 0
        replaced = False
        while True:
            position = current.find(r"\boxed", cursor)
            if position < 0:
                pieces.append(current[cursor:])
                break
            pieces.append(current[cursor:position])
            span = extract_boxed_span(current, position)
            if span is None:
                pieces.append(current[position:position + len(r"\boxed")])
                cursor = position + len(r"\boxed")
                continue
            content, end = span
            pieces.append(content)
            cursor = end
            replaced = True
        updated = "".join(pieces)
        if not replaced or updated == current:
            return updated
        current = updated
    return current


def strip_trailing_answer_line(text: str) -> str:
    """移除末尾冗余 ``The answer is``/``####`` 行，不触碰前面的推理。"""
    candidates = []
    for match in HASH_RE.finditer(text):
        if not text[match.end():].strip():
            candidates.append(match.start())
    for match in ANSWER_RE.finditer(text):
        if not text[match.end():].strip():
            candidates.append(match.start())
    return text[:max(candidates)].rstrip() if candidates else text.rstrip()


def clean_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = re.sub(r"\s+$", "", answer)
    answer = answer.rstrip("。.;；")
    if len(answer) >= 2 and answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    if answer.startswith(r"\(") and answer.endswith(r"\)"):
        answer = answer[2:-2].strip()
    if answer.startswith(r"\[") and answer.endswith(r"\]"):
        answer = answer[2:-2].strip()
    return answer


def extract_final_answer(response: str):
    """返回 ``(answer, marker)``；只使用最后一个明确答案标记。"""
    candidates = []
    start = 0
    while True:
        position = response.find(r"\boxed", start)
        if position < 0:
            break
        answer = extract_balanced_boxed(response, position)
        if answer is not None:
            candidates.append((position, clean_answer(answer), "boxed"))
        start = position + len(r"\boxed")

    for match in HASH_RE.finditer(response):
        candidates.append((match.start(), clean_answer(match.group(1)), "hash"))
    for match in ANSWER_RE.finditer(response):
        candidates.append((match.start(), clean_answer(match.group(1)), "answer_is"))

    candidates = [item for item in candidates if item[1]]
    if not candidates:
        return None, None
    _, answer, marker = max(candidates, key=lambda item: item[0])
    return answer, marker


def canonicalize_response(response: str, answer_override: str | None = None):
    stripped = str(response).rstrip()
    # 对脚本自身的输出保持幂等；最后一个 canonical 标记必须位于末尾段。
    matches = list(CANONICAL_RE.finditer(stripped))
    if matches and answer_override is None:
        boxed_start = stripped.find(r"\boxed", matches[-1].start())
        final_answer = extract_balanced_boxed(stripped, boxed_start)
        if final_answer is not None:
            suffix = stripped[matches[-1].start():]
            if re.fullmatch(
                    r"Final\s+answer\s*:\s*\\boxed\{.*\}\s*",
                    suffix, flags=re.IGNORECASE | re.DOTALL):
                return stripped, clean_answer(final_answer), "already_canonical"

    answer, marker = extract_final_answer(stripped)
    if answer_override is not None:
        answer = clean_answer(answer_override)
        marker = "manual_correction"
    if answer is None:
        raise ValueError("response 中没有可识别的明确答案标记")
    body = unwrap_all_boxed(strip_trailing_answer_line(stripped)).rstrip()
    return body + f"\n\nFinal answer: \\boxed{{{answer}}}", answer, marker


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="规范数学 SFT 最终答案格式")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--corrections", default="",
        help="可选 JSON：按 prompt SHA-256 提供经人工核对的缺失答案")
    parser.add_argument("--expected_records", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if source == destination:
        parser.error("--output 不允许覆盖输入文件")
    if not source.is_file():
        parser.error(f"输入文件不存在: {source}")

    corrections = {}
    corrections_path = None
    if args.corrections:
        corrections_path = Path(args.corrections).resolve()
        payload = json.loads(corrections_path.read_text(encoding="utf-8"))
        corrections = payload.get("corrections", {})
        if not isinstance(corrections, dict):
            parser.error("corrections JSON 的 corrections 必须是对象")

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    counts = Counter()
    records = 0
    unresolved = []
    try:
        with source.open(encoding="utf-8") as reader, \
                temporary.open("w", encoding="utf-8") as writer:
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                response = str(row.get("response", ""))
                prompt_hash = sha256_text(str(row.get("prompt", "")))
                correction = corrections.get(prompt_hash)
                try:
                    formatted, answer, marker = canonicalize_response(
                        response,
                        answer_override=(correction or {}).get("answer"))
                except ValueError as exc:
                    if correction is None:
                        unresolved.append(
                            (line_number, str(row.get("source", "")), str(exc)))
                        continue
                    raise
                if correction is not None:
                    expected_source = str(correction.get("source", ""))
                    if (not answer or (
                            expected_source
                            and expected_source != str(row.get("source", "")))):
                        raise RuntimeError(
                            f"{source}:{line_number}: 人工修正项为空或 source 不匹配")
                row["response"] = formatted
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                records += 1
                counts[marker] += 1
                if not answer:
                    raise RuntimeError(f"{source}:{line_number}: 最终答案为空")
        if unresolved:
            preview = "; ".join(
                f"line {line_number} source={item_source!r}: {message}"
                for line_number, item_source, message in unresolved[:20]
            )
            raise RuntimeError(
                f"共有 {len(unresolved)} 条 response 无法规范化；{preview}")
        if args.expected_records and records != args.expected_records:
            raise RuntimeError(
                f"记录数不符: 实际 {records}, 预期 {args.expected_records}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "recipe_version": "math_sft_canonical_boxed_v2",
        "input": str(source),
        "output": str(destination),
        "records": records,
        "marker_counts": dict(sorted(counts.items())),
        "corrections": str(corrections_path) if corrections_path else None,
        "input_sha256": sha256(source),
        "output_sha256": sha256(destination),
        "transformation": (
            "Preserve reasoning semantics, unwrap previous boxed commands, remove "
            "a redundant trailing answer marker, and append exactly one canonical "
            "Final answer: \\boxed{...}."
        ),
    }
    temporary_manifest = manifest_path.with_name(
        manifest_path.name + f".tmp.{os.getpid()}")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    print(f"完成: {records} 条 -> {destination}")
    print("答案来源:", dict(sorted(counts.items())))
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()
