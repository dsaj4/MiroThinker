#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hydra import compose, initialize_config_dir

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from src.llm.factory import ClientFactory
from src.logging.task_logger import TaskLog, get_utc_plus_8_time

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    repair_json = None


JUDGE_PROMPT_TEMPLATE = """你是严格的事实核查裁判。
请仅根据 ground_truth 判断回答是否命中，不要按文风打分。

用户问题：
{query}

ground_truth：
{ground_truth}

候选答案：
{response}

请仅输出 JSON：
{{
  "hit_label": "HIT|PARTIAL|MISS",
  "hit_score": 0-10,
  "hit_reason": "简要说明",
  "match_points": ["命中的关键点1"],
  "missing_points": ["缺失点1"]
}}
"""


@dataclass
class Row:
    case_id: str
    query: str
    hit_label: str
    hit_score: int
    hit_reason: str
    response: str


def _extract_json_block(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0).strip() if m else None


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


def _parse_hit(raw: str) -> Dict[str, Any]:
    block = _extract_json_block(raw)
    if not block:
        return {
            "hit_label": "MISS",
            "hit_score": 0,
            "hit_reason": "judge_non_json_output",
        }
    data = None
    try:
        data = json.loads(block)
    except Exception:
        if repair_json is not None:
            try:
                data = json.loads(repair_json(block))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return {
            "hit_label": "MISS",
            "hit_score": 0,
            "hit_reason": "judge_json_parse_failed",
        }
    return {
        "hit_label": _normalize_label(data.get("hit_label")),
        "hit_score": _clamp_score(data.get("hit_score", 0)),
        "hit_reason": str(data.get("hit_reason", "")).strip()[:300] or "empty_reason",
    }


def _compose_cfg(llm_name: str):
    with initialize_config_dir(config_dir=str(APP_ROOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=[f"llm={llm_name}", "agent=single_agent_keep5"])


def _load_dataset_map(dataset_path: Path) -> Dict[str, Dict[str, str]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for x in data:
        cid = str(x.get("id", "")).strip()
        if not cid:
            continue
        out[cid] = {
            "query": str(x.get("query", "")),
            "ground_truth": str(x.get("ground_truth", "")),
        }
    return out


def _load_candidates(path: Path, dataset_map: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for x in data:
            cid = str(x.get("id", x.get("case_id", "")))
            meta = dataset_map.get(cid, {})
            query = str(x.get("query", "")) or meta.get("query", "")
            gt = meta.get("ground_truth", "")
            resp = str(x.get("response", x.get("final_response", "")))
            if cid and query and resp:
                rows.append((cid, query, gt, resp))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for x in reader:
                cid = str(x.get("case_id", "") or x.get("id", ""))
                meta = dataset_map.get(cid, {})
                query = str(x.get("query", "")) or meta.get("query", "")
                gt = meta.get("ground_truth", "")
                resp = str(x.get("final_response", "") or x.get("response", ""))
                if cid and query and resp:
                    rows.append((cid, query, gt, resp))
        return rows
    raise ValueError(f"Unsupported file type: {path}")


async def _judge_case(judge_client: Any, query: str, ground_truth: str, response_text: str) -> Dict[str, Any]:
    if not str(ground_truth).strip():
        return {"hit_label": "MISS", "hit_score": 0, "hit_reason": "missing_ground_truth"}
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, ground_truth=ground_truth, response=response_text)
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
    assistant_text, _, _ = judge_client.process_llm_response(llm_resp, history, agent_type="judge")
    return _parse_hit(assistant_text)


async def _score_file(
    in_file: Path,
    out_dir: Path,
    dataset_map: Dict[str, Dict[str, str]],
    judge_client: Any,
) -> Path:
    candidates = _load_candidates(in_file, dataset_map)
    rows: List[Row] = []
    for cid, query, gt, resp in candidates:
        try:
            h = await _judge_case(judge_client, query, gt, resp)
        except Exception:
            h = {"hit_label": "MISS", "hit_score": 0, "hit_reason": "judge_exception"}
        rows.append(
            Row(
                case_id=cid,
                query=query,
                hit_label=h["hit_label"],
                hit_score=h["hit_score"],
                hit_reason=h["hit_reason"],
                response=resp,
            )
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"{in_file.stem}_judge_hit_scored_{ts}.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "query", "hit_label", "hit_score", "hit_reason", "response"])
        for r in rows:
            writer.writerow([r.case_id, r.query, r.hit_label, r.hit_score, r.hit_reason, r.response])

    if rows:
        n = len(rows)
        avg = sum(r.hit_score for r in rows) / n
        hit_rate = sum(1 for r in rows if r.hit_label == "HIT") / n
        partial_rate = sum(1 for r in rows if r.hit_label == "PARTIAL") / n
        miss_rate = sum(1 for r in rows if r.hit_label == "MISS") / n
        print(
            f"{in_file.name}: cases={n} avg_hit={avg:.2f} hit_rate={hit_rate:.2%} "
            f"partial_rate={partial_rate:.2%} miss_rate={miss_rate:.2%}"
        )
    else:
        print(f"{in_file.name}: no valid rows")
    print(f"saved: {out_csv}")
    return out_csv


async def main(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_map = _load_dataset_map(dataset_path)

    cfg = _compose_cfg(args.judge_llm)
    judge_log = TaskLog(
        log_dir=str((REPO_ROOT / "logs" / "eval").resolve()),
        task_id=f"judge_hit_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=get_utc_plus_8_time(),
    )
    judge_client = ClientFactory(
        task_id=f"judge-hit-batch-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        cfg=cfg,
        task_log=judge_log,
    )

    try:
        for p in args.inputs:
            await _score_file(Path(p).resolve(), out_dir, dataset_map, judge_client)
    finally:
        try:
            judge_client.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge existing result files by ground-truth hit criteria.")
    parser.add_argument("--dataset", type=str, default=str(REPO_ROOT / "eval_dataset.json"))
    parser.add_argument("--judge-llm", type=str, default="qwen-3")
    parser.add_argument("--out-dir", type=str, default=str(APP_ROOT / "eval"))
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files (.json/.csv) to score")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
