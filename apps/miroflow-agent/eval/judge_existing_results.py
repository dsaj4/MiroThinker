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


JUDGE_PROMPT_TEMPLATE = """你是一个严苛的计算机与商业分析师裁判。
用户问题：{query}
AI生成的答案：{response}

请按以下三个维度进行1-10分评估，并返回JSON格式（只返回JSON，不包含其他标记）：
{{
  "relevance": <答案解决问题的直接程度，过滤掉废话废链的比例>,
  "information_density": <事实和数据的饱满度，有效事实节点的数量>,
  "model_alignment": <答案是否有效地运用了诸如SWOT、金字塔等“思维模型”进行结构化输出，而非仅仅堆砌搜索结果>
}}
"""


@dataclass
class Row:
    case_id: str
    query: str
    response: str
    relevance: int
    density: int
    alignment: int


def _clamp(v: Any) -> int:
    try:
        x = int(round(float(v)))
    except Exception:
        return 1
    return max(1, min(10, x))


def _extract_json_block(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0).strip() if m else None


def _parse_scores(raw: str) -> Dict[str, int]:
    block = _extract_json_block(raw)
    if not block:
        return {"relevance": 1, "information_density": 1, "model_alignment": 1}
    try:
        data = json.loads(block)
    except Exception:
        return {"relevance": 1, "information_density": 1, "model_alignment": 1}
    if not isinstance(data, dict):
        return {"relevance": 1, "information_density": 1, "model_alignment": 1}
    return {
        "relevance": _clamp(data.get("relevance", 1)),
        "information_density": _clamp(data.get("information_density", 1)),
        "model_alignment": _clamp(data.get("model_alignment", 1)),
    }


def _compose_cfg(llm_name: str):
    with initialize_config_dir(config_dir=str(APP_ROOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=[f"llm={llm_name}", "agent=single_agent_keep5"])


def _load_query_map(dataset_path: Path) -> Dict[str, str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    qmap: Dict[str, str] = {}
    for x in data:
        qmap[str(x.get("id", ""))] = str(x.get("query", ""))
    return qmap


def _load_candidates(path: Path, query_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for x in data:
            cid = str(x.get("id", x.get("case_id", "")))
            query = str(x.get("query", "")) or query_map.get(cid, "")
            resp = str(x.get("response", x.get("final_response", "")))
            if cid and query and resp:
                rows.append((cid, query, resp))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for x in reader:
                cid = str(x.get("case_id", "") or x.get("id", ""))
                query = str(x.get("query", "")) or query_map.get(cid, "")
                resp = str(x.get("final_response", "") or x.get("response", ""))
                if cid and query and resp:
                    rows.append((cid, query, resp))
        return rows
    raise ValueError(f"Unsupported file type: {path}")


async def _judge_case(judge_client: Any, query: str, response_text: str) -> Dict[str, int]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, response=response_text)
    history = [{"role": "user", "content": prompt}]
    llm_resp, history = await judge_client.create_message(
        system_prompt="你是严格裁判，只输出JSON对象。",
        message_history=history,
        tool_definitions=[],
        keep_tool_result=-1,
        step_id=1,
        task_log=None,
        agent_type="judge",
    )
    assistant_text, _, _ = judge_client.process_llm_response(llm_resp, history, agent_type="judge")
    return _parse_scores(assistant_text)


async def _score_file(
    in_file: Path,
    out_dir: Path,
    query_map: Dict[str, str],
    judge_client: Any,
) -> Path:
    candidates = _load_candidates(in_file, query_map)
    rows: List[Row] = []
    for cid, query, resp in candidates:
        try:
            s = await _judge_case(judge_client, query, resp)
        except Exception:
            s = {"relevance": 1, "information_density": 1, "model_alignment": 1}
        rows.append(
            Row(
                case_id=cid,
                query=query,
                response=resp,
                relevance=s["relevance"],
                density=s["information_density"],
                alignment=s["model_alignment"],
            )
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"{in_file.stem}_judge_scored_{ts}.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "query",
                "relevance_score",
                "density_score",
                "alignment_score",
                "response",
            ]
        )
        for r in rows:
            writer.writerow([r.case_id, r.query, r.relevance, r.density, r.alignment, r.response])

    if rows:
        avg_rel = sum(r.relevance for r in rows) / len(rows)
        avg_den = sum(r.density for r in rows) / len(rows)
        avg_ali = sum(r.alignment for r in rows) / len(rows)
        print(f"{in_file.name}: cases={len(rows)} avg_rel={avg_rel:.2f} avg_den={avg_den:.2f} avg_align={avg_ali:.2f}")
    else:
        print(f"{in_file.name}: no valid rows")
    print(f"saved: {out_csv}")
    return out_csv


async def main(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    query_map = _load_query_map(dataset_path)

    cfg = _compose_cfg(args.judge_llm)
    judge_log = TaskLog(
        log_dir=str((REPO_ROOT / "logs" / "eval").resolve()),
        task_id=f"judge_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=get_utc_plus_8_time(),
    )
    judge_client = ClientFactory(
        task_id=f"judge-batch-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        cfg=cfg,
        task_log=judge_log,
    )

    try:
        for p in args.inputs:
            await _score_file(Path(p).resolve(), out_dir, query_map, judge_client)
    finally:
        try:
            judge_client.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge existing result files with a same-provider model.")
    parser.add_argument("--dataset", type=str, default=str(REPO_ROOT / "eval_dataset.json"))
    parser.add_argument("--judge-llm", type=str, default="qwen-3")
    parser.add_argument("--out-dir", type=str, default=str(APP_ROOT / "eval"))
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files (.json/.csv) to score")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
