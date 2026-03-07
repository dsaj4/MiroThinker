#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.llm.factory import ClientFactory
from src.logging.task_logger import TaskLog, bootstrap_logger, get_utc_plus_8_time

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    repair_json = None


JUDGE_PROMPT_TEMPLATE = """你是一个严格的事实核查评审。
请仅根据给定的 ground_truth 判断回答是否命中，不要按文风打分。

用户问题：
{query}

标准答案（ground_truth）：
{ground_truth}

模型回答：
{response}

请输出严格 JSON（只输出 JSON 对象，不要 markdown）：
{{
  "hit_label": "HIT|PARTIAL|MISS",
  "hit_score": 0-10,
  "hit_reason": "简短说明是否命中以及原因",
  "match_points": ["命中的关键点1", "命中的关键点2"],
  "missing_points": ["缺失点1", "缺失点2"]
}}

评分规则：
- HIT: 关键结论和关键事实基本完整命中，8-10 分
- PARTIAL: 命中部分关键点但缺关键事实或有偏差，4-7 分
- MISS: 未命中核心结论或事实错误明显，0-3 分
"""


@dataclass
class EvalRow:
    case_id: str
    query: str
    case_type: str
    latency: float
    token_used: int
    hit_label: str
    hit_score: int
    hit_reason: str
    ground_truth: str
    final_response: str


def _clamp_score(v: Any) -> int:
    try:
        x = int(round(float(v)))
    except Exception:
        return 0
    return max(0, min(10, x))


def _normalize_label(v: Any) -> str:
    x = str(v or "").strip().upper()
    if x in {"HIT", "PARTIAL", "MISS"}:
        return x
    return "MISS"


def _extract_json_block(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0).strip() if m else None


def _parse_judge_result(raw_text: str) -> Dict[str, Any]:
    payload = _extract_json_block(raw_text)
    if not payload:
        return {
            "hit_label": "MISS",
            "hit_score": 0,
            "hit_reason": "judge_non_json_output",
            "match_points": [],
            "missing_points": [],
        }

    parsed = None
    try:
        parsed = json.loads(payload)
    except Exception:
        if repair_json is not None:
            try:
                parsed = json.loads(repair_json(payload))
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return {
            "hit_label": "MISS",
            "hit_score": 0,
            "hit_reason": "judge_json_parse_failed",
            "match_points": [],
            "missing_points": [],
        }

    hit_label = _normalize_label(parsed.get("hit_label"))
    hit_score = _clamp_score(parsed.get("hit_score", 0))
    hit_reason = str(parsed.get("hit_reason", "")).strip()[:300]
    match_points = parsed.get("match_points", [])
    missing_points = parsed.get("missing_points", [])

    if not isinstance(match_points, list):
        match_points = []
    if not isinstance(missing_points, list):
        missing_points = []

    return {
        "hit_label": hit_label,
        "hit_score": hit_score,
        "hit_reason": hit_reason or "empty_reason",
        "match_points": [str(x) for x in match_points[:8]],
        "missing_points": [str(x) for x in missing_points[:8]],
    }


def _extract_token_used_from_task_log(log_path: str) -> int:
    try:
        data = json.loads(Path(log_path).read_text(encoding="utf-8"))
    except Exception:
        return 0

    max_in = 0
    max_out = 0
    for step in data.get("step_logs", []):
        msg = str(step.get("message", ""))
        m = re.search(r"Input:\s*(\d+)\s*,\s*Output:\s*(\d+)", msg)
        if not m:
            continue
        max_in = max(max_in, int(m.group(1)))
        max_out = max(max_out, int(m.group(2)))
    return max_in + max_out


def _extract_latency_from_task_log(log_path: str, fallback_s: float) -> float:
    try:
        data = json.loads(Path(log_path).read_text(encoding="utf-8"))
        elapsed_ms = int(data.get("trace_data", {}).get("elapsed_ms", 0))
        if elapsed_ms > 0:
            return round(elapsed_ms / 1000.0, 3)
    except Exception:
        pass
    return round(fallback_s, 3)


def _compose_cfg(
    llm_name: str,
    agent_name: str,
    max_turns: Optional[int] = None,
) -> DictConfig:
    overrides = [f"llm={llm_name}", f"agent={agent_name}"]
    if max_turns is not None and max_turns > 0:
        overrides.append(f"agent.main_agent.max_turns={max_turns}")
    with initialize_config_dir(config_dir=str(APP_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _prepare_eval_dataset(eval_dir: Path, source: Path) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    target = eval_dir / "eval_dataset.json"
    shutil.copy2(source, target)
    return target


async def _judge_case(
    judge_client: Any,
    query: str,
    ground_truth: str,
    response_text: str,
) -> Dict[str, Any]:
    if not str(ground_truth or "").strip():
        return {
            "hit_label": "MISS",
            "hit_score": 0,
            "hit_reason": "missing_ground_truth",
            "match_points": [],
            "missing_points": [],
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        ground_truth=ground_truth,
        response=response_text,
    )
    history = [{"role": "user", "content": prompt}]
    llm_resp, history = await judge_client.create_message(
        system_prompt="你是严格裁判，只输出 JSON 对象。",
        message_history=history,
        tool_definitions=[],
        keep_tool_result=-1,
        step_id=1,
        task_log=None,
        agent_type="judge",
    )
    assistant_text, _, _ = judge_client.process_llm_response(
        llm_resp, history, agent_type="judge"
    )
    return _parse_judge_result(assistant_text)


async def run_eval(args: argparse.Namespace):
    logger = bootstrap_logger()
    eval_dir = APP_ROOT / "eval"
    source_dataset = Path(args.dataset).resolve() if args.dataset else REPO_ROOT / "eval_dataset.json"
    if not source_dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {source_dataset}")

    dataset_path = _prepare_eval_dataset(eval_dir, source_dataset)
    logger.info(f"Dataset copied: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("eval_dataset.json must be a JSON array")

    if args.case_id:
        dataset = [x for x in dataset if str(x.get("id", "")) == args.case_id]
        if not dataset:
            raise ValueError(f"case_id not found: {args.case_id}")
    elif args.case_index > 0:
        if args.case_index > len(dataset):
            raise ValueError(
                f"case_index out of range: {args.case_index}, dataset size={len(dataset)}"
            )
        dataset = [dataset[args.case_index - 1]]

    cfg = _compose_cfg(
        llm_name=args.search_llm,
        agent_name=args.agent,
        max_turns=args.agent_max_turns if args.agent_max_turns > 0 else None,
    )
    main_tm, sub_tms, output_formatter = create_pipeline_components(cfg)

    judge_cfg = _compose_cfg(llm_name=args.judge_llm, agent_name="single_agent_keep5", max_turns=1)
    judge_log = TaskLog(
        log_dir=str((REPO_ROOT / "logs" / "eval").resolve()),
        task_id=f"judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=get_utc_plus_8_time(),
    )
    judge_client = ClientFactory(
        task_id=f"judge-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        cfg=judge_cfg,
        task_log=judge_log,
    )

    total = min(len(dataset), args.max_cases) if args.max_cases > 0 else len(dataset)
    rows: List[EvalRow] = []

    for i, case in enumerate(dataset[:total], start=1):
        case_id = str(case.get("id", f"case_{i:03d}"))
        query = str(case.get("query", "")).strip()
        case_type = str(case.get("type", "unknown"))
        task_file_name = str(case.get("file_name", "") or "")
        ground_truth = str(case.get("ground_truth", "") or "")
        if not query:
            continue

        logger.info(f"[{i}/{total}] case={case_id}")
        t0 = perf_counter()
        final_summary = ""
        log_file_path = ""
        try:
            final_summary, _, log_file_path, _ = await asyncio.wait_for(
                execute_task_pipeline(
                    cfg=cfg,
                    task_id=case_id,
                    task_description=query,
                    task_file_name=task_file_name,
                    main_agent_tool_manager=main_tm,
                    sub_agent_tool_managers=sub_tms,
                    output_formatter=output_formatter,
                    ground_truth=ground_truth,
                    log_dir=cfg.debug_dir,
                ),
                timeout=args.case_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.error(f"Case {case_id} timeout after {args.case_timeout_sec}s")
            final_summary = f"[Timeout] case exceeded {args.case_timeout_sec}s"
        except Exception as exc:
            logger.error(f"Case {case_id} failed: {exc}")
            final_summary = f"[ExecutionError] {exc}"

        fallback_latency = perf_counter() - t0
        latency = (
            _extract_latency_from_task_log(log_file_path, fallback_latency)
            if log_file_path
            else round(fallback_latency, 3)
        )
        token_used = _extract_token_used_from_task_log(log_file_path) if log_file_path else 0

        try:
            judge_res = await _judge_case(judge_client, query, ground_truth, str(final_summary))
        except Exception as exc:
            logger.error(f"Judge failed for {case_id}: {exc}")
            judge_res = {
                "hit_label": "MISS",
                "hit_score": 0,
                "hit_reason": f"judge_exception:{exc}",
                "match_points": [],
                "missing_points": [],
            }

        rows.append(
            EvalRow(
                case_id=case_id,
                query=query,
                case_type=case_type,
                latency=latency,
                token_used=token_used,
                hit_label=judge_res["hit_label"],
                hit_score=judge_res["hit_score"],
                hit_reason=judge_res["hit_reason"],
                ground_truth=ground_truth,
                final_response=str(final_summary),
            )
        )

    n = max(1, len(rows))
    avg_latency = sum(r.latency for r in rows) / n
    avg_token = sum(r.token_used for r in rows) / n
    avg_hit_score = sum(r.hit_score for r in rows) / n
    hit_rate = sum(1 for r in rows if r.hit_label == "HIT") / n
    partial_rate = sum(1 for r in rows if r.hit_label == "PARTIAL") / n
    miss_rate = sum(1 for r in rows if r.hit_label == "MISS") / n

    print("\n===== Eval Summary (Ground Truth Hit) =====")
    print(f"cases: {len(rows)}")
    print(f"avg_latency: {avg_latency:.3f}s")
    print(f"avg_token_used: {avg_token:.1f}")
    print(f"avg_hit_score: {avg_hit_score:.2f}/10")
    print(f"hit_rate: {hit_rate:.2%}")
    print(f"partial_rate: {partial_rate:.2%}")
    print(f"miss_rate: {miss_rate:.2%}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = eval_dir / f"eval_results_{ts}.csv"
    out_json = eval_dir / f"eval_results_{ts}.json"

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "query",
                "type",
                "latency",
                "token_used",
                "hit_label",
                "hit_score",
                "hit_reason",
                "ground_truth",
                "final_response",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.case_id,
                    r.query,
                    r.case_type,
                    r.latency,
                    r.token_used,
                    r.hit_label,
                    r.hit_score,
                    r.hit_reason,
                    r.ground_truth,
                    r.final_response,
                ]
            )
    print(f"csv_saved: {out_csv}")

    out_payload = [
        {
            "id": r.case_id,
            "query": r.query,
            "type": r.case_type,
            "latency": r.latency,
            "token_used": r.token_used,
            "hit_label": r.hit_label,
            "hit_score": r.hit_score,
            "hit_reason": r.hit_reason,
            "ground_truth": r.ground_truth,
            "final_response": r.final_response,
            "response": r.final_response,
        }
        for r in rows
    ]
    out_json.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"json_saved: {out_json}")

    try:
        judge_client.close()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate search quality by ground-truth hit criteria."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Source dataset path (default: repo_root/eval_dataset.json)",
    )
    parser.add_argument("--search-llm", type=str, default="qwen-3")
    parser.add_argument("--judge-llm", type=str, default="qwen-3")
    parser.add_argument(
        "--agent",
        type=str,
        default="mirothinker_v1.5_keep5_max200",
        help="Use full project agent config by default.",
    )
    parser.add_argument(
        "--agent-max-turns",
        type=int,
        default=0,
        help="Override max turns; 0 means keep agent config default.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases.")
    parser.add_argument(
        "--case-id",
        type=str,
        default="",
        help="Run exactly one case by its id (overrides --max-cases).",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Run exactly one case by 1-based index (overrides --max-cases).",
    )
    parser.add_argument(
        "--case-timeout-sec",
        type=int,
        default=120,
        help="Per-case timeout in seconds (default 120s = 2min).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_eval(parse_args()))
