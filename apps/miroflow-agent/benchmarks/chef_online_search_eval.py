#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
CHEF online search evaluation.

This script queries a live search API (Serper) for each claim and measures
retrieval effectiveness against CHEF gold evidence texts.

Matching is done via normalized substring match between gold evidence text and
retrieved result content (title+snippet or optionally fetched page content).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


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


def serper_search(
    query: str,
    api_key: str,
    base_url: str,
    num: int,
    gl: str,
    hl: str,
) -> Dict:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {
        "q": query,
        "gl": gl,
        "hl": hl,
        "num": num,
    }
    resp = requests.post(f"{base_url}/search", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def jina_fetch(url: str, api_key: str, base_url: str, max_chars: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    # Avoid double prefix
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]
    jina_url = f"{base_url.rstrip('/')}/{url}"
    resp = requests.get(jina_url, headers=headers, timeout=60)
    resp.raise_for_status()
    text = resp.text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def extract_result_text(result: Dict) -> str:
    title = result.get("title", "") or ""
    snippet = result.get("snippet", "") or ""
    return f"{title}\n{snippet}".strip()


def match_ranks(
    results_texts: List[str],
    gold_texts: List[str],
) -> Tuple[Optional[int], int]:
    if not results_texts or not gold_texts:
        return None, 0

    norm_results = [normalize(t) for t in results_texts]
    matches = set()
    first_rank = None
    for rank, res in enumerate(norm_results, start=1):
        if not res:
            continue
        for gi, gold in enumerate(gold_texts):
            if gi in matches:
                continue
            norm_gold = normalize(gold)
            if norm_gold and norm_gold in res:
                matches.add(gi)
                if first_rank is None:
                    first_rank = rank
    return first_rank, len(matches)


def evaluate(
    examples: Iterable[Dict],
    max_k: int,
    min_gold_len: int,
    search_k: int,
    serper_key: str,
    serper_base: str,
    gl: str,
    hl: str,
    fetch: bool,
    jina_key: str,
    jina_base: str,
    max_content_chars: int,
    cache_dir: Optional[Path],
    sleep_s: float,
    max_examples: Optional[int],
    out_rows: Optional[Path],
) -> Dict:
    hits_at = {k: 0 for k in range(1, max_k + 1)}
    mrr_total = 0.0
    coverage_total = 0.0
    count_with_gold = 0
    total_examples = 0
    error_count = 0

    out_f = None
    if out_rows:
        out_rows.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_rows.open("w", encoding="utf-8")

    for idx, ex in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break
        total_examples += 1
        claim = ex.get("claim", "") or ""
        claim_id = ex.get("claimId", idx)

        gold_texts = gold_texts_list(ex.get("gold evidence", {}), min_gold_len)
        if not gold_texts:
            if out_f:
                out_f.write(json.dumps({"claimId": claim_id, "skipped": True}) + "\n")
            continue
        count_with_gold += 1

        cache_key = f"{claim_id}.json"
        cached = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / cache_key
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))

        try:
            if cached:
                results = cached.get("results", [])
            else:
                data = serper_search(
                    query=claim,
                    api_key=serper_key,
                    base_url=serper_base,
                    num=search_k,
                    gl=gl,
                    hl=hl,
                )
                results = data.get("organic", []) or []
                if cache_dir:
                    cache_path.write_text(
                        json.dumps({"query": claim, "results": results}, ensure_ascii=False),
                        encoding="utf-8",
                    )
            # Build result texts
            results_texts = []
            for r in results[:max_k]:
                if fetch and jina_key:
                    url = r.get("link") or r.get("url")
                    if url:
                        try:
                            content = jina_fetch(url, jina_key, jina_base, max_content_chars)
                            results_texts.append(content)
                            continue
                        except Exception:
                            pass
                results_texts.append(extract_result_text(r))

            first_rank, matched_gold = match_ranks(results_texts, gold_texts)

            if first_rank is not None:
                mrr_total += 1.0 / first_rank
                for k in range(1, max_k + 1):
                    if first_rank <= k:
                        hits_at[k] += 1
            coverage_total += matched_gold / max(len(gold_texts), 1)

            if out_f:
                out_f.write(
                    json.dumps(
                        {
                            "claimId": claim_id,
                            "first_rank": first_rank,
                            "matched_gold": matched_gold,
                            "gold_count": len(gold_texts),
                            "query": claim,
                            "results": results[:max_k],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            error_count += 1
            if out_f:
                out_f.write(
                    json.dumps(
                        {"claimId": claim_id, "error": str(exc), "query": claim},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if sleep_s > 0:
            time.sleep(sleep_s)

    if out_f:
        out_f.close()

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    return {
        "total_examples": total_examples,
        "examples_with_gold": count_with_gold,
        "errors": error_count,
        "hit_at": {f"hit@{k}": safe_div(v, count_with_gold) for k, v in hits_at.items()},
        "mrr": safe_div(mrr_total, count_with_gold),
        "avg_gold_coverage": safe_div(coverage_total, count_with_gold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEF online search evaluation")
    parser.add_argument("--data", required=True, help="Path to CHEF split json/jsonl")
    parser.add_argument("--max-k", type=int, default=5, help="Max K for hit@k")
    parser.add_argument("--search-k", type=int, default=10, help="Serper results to fetch")
    parser.add_argument("--min-gold-len", type=int, default=20, help="Min gold evidence length")
    parser.add_argument("--gl", default="us", help="Serper country context")
    parser.add_argument("--hl", default="zh", help="Serper language")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between queries")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit examples (0=all)")
    parser.add_argument("--cache-dir", default="", help="Cache dir for search results")
    parser.add_argument("--fetch", action="store_true", help="Fetch page content via Jina")
    parser.add_argument("--max-content-chars", type=int, default=20000, help="Max chars per fetched page")
    parser.add_argument("--out-rows", default="", help="Per-query output jsonl path")
    parser.add_argument("--out", default="", help="Metrics output json path")
    args = parser.parse_args()

    serper_key = os.getenv("SERPER_API_KEY", "")
    serper_base = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
    if not serper_key:
        raise RuntimeError("SERPER_API_KEY is not set in environment.")

    fetch = args.fetch
    jina_key = os.getenv("JINA_API_KEY", "") if fetch else ""
    jina_base = os.getenv("JINA_BASE_URL", "https://r.jina.ai")
    if fetch and not jina_key:
        raise RuntimeError("JINA_API_KEY is not set but --fetch was enabled.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    max_examples = args.max_examples if args.max_examples > 0 else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    out_rows = Path(args.out_rows) if args.out_rows else None

    examples = load_examples(data_path)
    metrics = evaluate(
        examples=examples,
        max_k=args.max_k,
        min_gold_len=args.min_gold_len,
        search_k=args.search_k,
        serper_key=serper_key,
        serper_base=serper_base,
        gl=args.gl,
        hl=args.hl,
        fetch=fetch,
        jina_key=jina_key,
        jina_base=jina_base,
        max_content_chars=args.max_content_chars,
        cache_dir=cache_dir,
        sleep_s=args.sleep,
        max_examples=max_examples,
        out_rows=out_rows,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
