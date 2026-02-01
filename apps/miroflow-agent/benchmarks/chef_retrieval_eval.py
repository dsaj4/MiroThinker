#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
CHEF retrieval evaluation script.

This script evaluates retrieval effectiveness using CHEF data by matching
retrieved evidence texts against gold evidence texts.

It reports:
- Hit@K (examples with at least one gold evidence matched in top K)
- MRR (mean reciprocal rank of first matched evidence)
- Average gold coverage (matched gold count / total gold count)
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CJK_PUNCT = "，。！？…（）【】《》“”‘’；：、—·"
TRANS_TABLE = str.maketrans("", "", string.punctuation + CJK_PUNCT)


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.translate(TRANS_TABLE)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def load_examples(path: Path) -> List[Dict]:
    raw = path.read_text(encoding="utf-8")
    raw_strip = raw.lstrip()
    if raw_strip.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list at top-level.")
        return data
    # JSONL fallback
    examples = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def ordered_evidence_texts(evidence: Dict) -> List[str]:
    if not isinstance(evidence, dict):
        return []
    def sort_key(k: str) -> Tuple[int, str]:
        try:
            return (int(k), k)
        except Exception:
            return (10**9, k)

    texts = []
    for key in sorted(evidence.keys(), key=sort_key):
        item = evidence.get(key, {})
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = ""
        texts.append(text or "")
    return texts


def gold_texts_list(gold: Dict, min_len: int) -> List[str]:
    if not isinstance(gold, dict):
        return []
    texts = []
    for item in gold.values():
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = ""
        text = text or ""
        if len(text.strip()) >= min_len:
            texts.append(text)
    return texts


def match_ranks(evidence_texts: List[str], gold_texts: List[str]) -> List[int]:
    if not evidence_texts or not gold_texts:
        return []
    norm_evidence = [normalize(t) for t in evidence_texts]
    ranks = []
    for gold in gold_texts:
        norm_gold = normalize(gold)
        if not norm_gold:
            continue
        matched_rank = None
        for idx, ev in enumerate(norm_evidence, start=1):
            if norm_gold and norm_gold in ev:
                matched_rank = idx
                break
        if matched_rank is not None:
            ranks.append(matched_rank)
    return ranks


def evaluate(
    examples: Iterable[Dict],
    max_k: int,
    min_gold_len: int,
) -> Dict:
    hits_at = {k: 0 for k in range(1, max_k + 1)}
    mrr_total = 0.0
    coverage_total = 0.0
    count_with_gold = 0
    total_examples = 0
    gold_counts = []
    evidence_counts = []

    for ex in examples:
        total_examples += 1
        evidence_texts = ordered_evidence_texts(ex.get("evidence", {}))
        gold_texts = gold_texts_list(ex.get("gold evidence", {}), min_gold_len)

        evidence_counts.append(len(evidence_texts))
        gold_counts.append(len(gold_texts))

        if not gold_texts:
            continue

        count_with_gold += 1
        ranks = match_ranks(evidence_texts[:max_k], gold_texts)

        if ranks:
            first_rank = min(ranks)
            mrr_total += 1.0 / first_rank
            for k in range(1, max_k + 1):
                if first_rank <= k:
                    hits_at[k] += 1

        coverage_total += (len(ranks) / max(len(gold_texts), 1))

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    metrics = {
        "total_examples": total_examples,
        "examples_with_gold": count_with_gold,
        "avg_evidence_per_example": safe_div(sum(evidence_counts), total_examples),
        "avg_gold_per_example": safe_div(sum(gold_counts), total_examples),
        "hit_at": {f"hit@{k}": safe_div(v, count_with_gold) for k, v in hits_at.items()},
        "mrr": safe_div(mrr_total, count_with_gold),
        "avg_gold_coverage": safe_div(coverage_total, count_with_gold),
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEF retrieval evaluation")
    parser.add_argument("--data", required=True, help="Path to CHEF split json/jsonl")
    parser.add_argument("--max-k", type=int, default=5, help="Max K for hit@k")
    parser.add_argument(
        "--min-gold-len",
        type=int,
        default=1,
        help="Ignore gold evidence texts shorter than this length",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output json path for metrics",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    examples = load_examples(data_path)
    metrics = evaluate(examples, max_k=args.max_k, min_gold_len=args.min_gold_len)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
