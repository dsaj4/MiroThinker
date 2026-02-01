#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
CHEF agent retrieval recall evaluation.

Runs the full agent on CHEF claims and evaluates how well the tool outputs
cover gold evidence texts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.core.pipeline import create_pipeline_components, execute_task_pipeline


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


def extract_tool_texts_from_log(log_data: Dict) -> str:
    history = log_data.get("main_agent_message_history", {}).get("message_history", [])
    if not history:
        return ""
    # first user message is the task itself; tool outputs start from later user messages
    tool_texts: List[str] = []
    first_user_seen = False
    for msg in history:
        if msg.get("role") != "user":
            continue
        if not first_user_seen:
            first_user_seen = True
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    tool_texts.append(item.get("text", ""))
        else:
            tool_texts.append(str(content))
    return "\n".join(tool_texts)


def match_gold_in_text(all_text: str, gold_texts: List[str]) -> List[int]:
    if not all_text or not gold_texts:
        return []
    norm_all = normalize(all_text)
    matched = []
    for i, gold in enumerate(gold_texts):
        if normalize(gold) and normalize(gold) in norm_all:
            matched.append(i)
    return matched


def build_cfg(overrides: List[str]) -> "OmegaConf":
    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(str(conf_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


async def evaluate(
    examples: Iterable[Dict],
    cfg,
    max_examples: Optional[int],
    min_gold_len: int,
    log_dir: Path,
    sleep_s: float,
    out_rows: Optional[Path],
) -> Dict:
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg)
    )

    hits = 0
    coverage_total = 0.0
    matched_gold_total = 0
    gold_total = 0
    total_examples = 0
    errors = 0

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
            continue

        task_description = (
            "请基于在线搜索检索与下述断言相关的证据句子，仅使用工具获取证据内容。\n"
            f"断言：{claim}"
        )

        try:
            _, _, log_file_path, _ = await execute_task_pipeline(
                cfg=cfg,
                task_id=f"chef_{claim_id}",
                task_file_name="",
                task_description=task_description,
                main_agent_tool_manager=main_agent_tool_manager,
                sub_agent_tool_managers=sub_agent_tool_managers,
                output_formatter=output_formatter,
                log_dir=str(log_dir),
            )

            log_data = json.loads(Path(log_file_path).read_text(encoding="utf-8"))
            tool_text = extract_tool_texts_from_log(log_data)
            matched = match_gold_in_text(tool_text, gold_texts)

            gold_total += len(gold_texts)
            matched_gold_total += len(matched)
            if matched:
                hits += 1
            coverage_total += len(matched) / max(len(gold_texts), 1)

            if out_f:
                out_f.write(
                    json.dumps(
                        {
                            "claimId": claim_id,
                            "matched_gold": len(matched),
                            "gold_count": len(gold_texts),
                            "hit": bool(matched),
                            "log_file_path": log_file_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            errors += 1
            if out_f:
                out_f.write(
                    json.dumps(
                        {"claimId": claim_id, "error": str(exc)}, ensure_ascii=False
                    )
                    + "\n"
                )
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

    if out_f:
        out_f.close()

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    return {
        "total_examples": total_examples,
        "errors": errors,
        "hit_rate": safe_div(hits, total_examples),
        "avg_gold_coverage": safe_div(coverage_total, total_examples),
        "gold_coverage_overall": safe_div(matched_gold_total, gold_total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEF agent retrieval recall eval")
    parser.add_argument("--data", required=True, help="Path to CHEF split json/jsonl")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit examples (0=all)")
    parser.add_argument("--min-gold-len", type=int, default=20, help="Min gold evidence length")
    parser.add_argument("--log-dir", default="../../logs/chef_agent_recall", help="Log directory")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between tasks")
    parser.add_argument("--out-rows", default="", help="Per-query output jsonl path")
    parser.add_argument("--out", default="", help="Metrics output json path")
    parser.add_argument("--llm", default="qwen-3", help="LLM config name")
    parser.add_argument("--agent", default="single_agent_keep5", help="Agent config name")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    overrides = [f"llm={args.llm}", f"agent={args.agent}"]
    cfg = build_cfg(overrides)

    examples = load_examples(data_path)
    max_examples = args.max_examples if args.max_examples > 0 else None
    log_dir = Path(args.log_dir)
    out_rows = Path(args.out_rows) if args.out_rows else None

    metrics = asyncio.run(
        evaluate(
            examples=examples,
            cfg=cfg,
            max_examples=max_examples,
            min_gold_len=args.min_gold_len,
            log_dir=log_dir,
            sleep_s=args.sleep,
            out_rows=out_rows,
        )
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
