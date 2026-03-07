#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    repair_json = None


ID_RE = re.compile(r"\b([AB]_\d{3})\b")
FINAL_ANSWER_RE = re.compile(r"={5,}\s*Final Answer(?:\s+([AB]_\d{3}))?\s*={5,}", re.IGNORECASE)
BROAD_ID_RE = re.compile(
    r"(?:问题|回答问题|Final\s*Answer)\s*([ABＡＢ])\s*[_＿\-–]?\s*([0-9０-９]{3})",
    re.IGNORECASE,
)


def _load_query_map(dataset_path: Path) -> Dict[str, Dict[str, str]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for x in data:
        cid = str(x.get("id", "")).strip()
        if not cid:
            continue
        out[cid] = {
            "query": str(x.get("query", "")),
            "type": str(x.get("type", "")),
        }
    return out


def _extract_balanced_json_blocks(text: str) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_str = False
        esc = False
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        raw = text[start:end]
                        blocks.append((start, end, raw))
                        break
            i += 1
        i += 1
    return blocks


def _safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    if repair_json is not None:
        try:
            repaired = repair_json(raw)
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _pick_case_id(block_start: int, id_positions: List[Tuple[int, str]]) -> Optional[str]:
    candidate = None
    for pos, cid in id_positions:
        if pos <= block_start:
            candidate = cid
        else:
            break
    return candidate


def _normalize_wide(text: str) -> str:
    table = str.maketrans(
        {
            "Ａ": "A",
            "Ｂ": "B",
            "０": "0",
            "１": "1",
            "２": "2",
            "３": "3",
            "４": "4",
            "５": "5",
            "６": "6",
            "７": "7",
            "８": "8",
            "９": "9",
            "＿": "_",
        }
    )
    return text.translate(table)


def _normalize_case_id(letter: str, digits: str) -> str:
    return f"{_normalize_wide(letter).upper()}_{_normalize_wide(digits)}"


def _extract_answer_by_regex(section: str) -> str:
    m = re.search(r'"answer"\s*:\s*"([\s\S]*?)"\s*,\s*"evidence"\s*:', section)
    if m:
        return m.group(1).strip()
    m = re.search(r'"answer"\s*:\s*"([\s\S]*?)"\s*,\s*"confidence"\s*:', section)
    if m:
        return m.group(1).strip()
    m = re.search(r'"answer"\s*:\s*"([\s\S]*?)"\s*[,}]\s*', section)
    if m:
        return m.group(1).strip()
    return ""


def _extract_evidence_by_regex(section: str) -> List[Dict[str, str]]:
    pattern = re.compile(
        r'"title"\s*:\s*"(?P<title>[\s\S]*?)"\s*,\s*"url"\s*:\s*"(?P<url>[\s\S]*?)"\s*,\s*"snippet"\s*:\s*"(?P<snippet>[\s\S]*?)"',
        re.IGNORECASE,
    )
    out = []
    for m in pattern.finditer(section):
        out.append(
            {
                "title": m.group("title").strip(),
                "url": m.group("url").strip(),
                "snippet": m.group("snippet").strip(),
            }
        )
        if len(out) >= 8:
            break
    return out


def _extract_confidence_by_regex(section: str) -> Dict[str, Any]:
    score: Any = 0
    level = ""
    reason = ""
    m = re.search(r'"score"\s*:\s*([0-9]{1,3})', section, re.IGNORECASE)
    if m:
        try:
            score = int(m.group(1))
        except Exception:
            score = 0
    m = re.search(r'"level"\s*:\s*"([^"]*)"', section, re.IGNORECASE)
    if m:
        level = m.group(1).strip()
    m = re.search(r'"reason"\s*:\s*"([\s\S]*?)"\s*[}\n]', section, re.IGNORECASE)
    if m:
        reason = m.group(1).strip()
    return {"score": score, "level": level, "reason": reason}


def _normalize_final_answer(answer_obj: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(answer_obj.get("answer", ""))
    evidence = answer_obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    normalized_evidence = []
    for e in evidence:
        if not isinstance(e, dict):
            continue
        normalized_evidence.append(
            {
                "title": str(e.get("title", "")),
                "url": str(e.get("url", "")),
                "snippet": str(e.get("snippet", "")),
            }
        )
    confidence = answer_obj.get("confidence", {})
    if not isinstance(confidence, dict):
        confidence = {}
    normalized_confidence = {
        "score": confidence.get("score", 0),
        "level": str(confidence.get("level", "")),
        "reason": str(confidence.get("reason", "")),
    }
    return {
        "answer": answer,
        "evidence": normalized_evidence,
        "confidence": normalized_confidence,
    }


def parse_md_to_json(md_path: Path, query_map: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    id_positions = [(m.start(), m.group(1)) for m in ID_RE.finditer(text)]
    for m in BROAD_ID_RE.finditer(text):
        cid = _normalize_case_id(m.group(1), m.group(2))
        if re.fullmatch(r"[AB]_\d{3}", cid):
            id_positions.append((m.start(), cid))
    id_positions.sort(key=lambda x: x[0])
    blocks = _extract_balanced_json_blocks(text)

    picked: Dict[str, Dict[str, Any]] = {}
    for start, _end, raw in blocks:
        if '"answer"' not in raw and "'answer'" not in raw:
            continue
        obj = _safe_json_loads(raw)
        if not obj:
            continue
        if "answer" not in obj:
            continue
        cid = _pick_case_id(start, id_positions)
        if not cid:
            continue
        f = _normalize_final_answer(obj)
        meta = query_map.get(cid, {})
        picked[cid] = {
            "id": cid,
            "case_id": cid,
            "query": meta.get("query", ""),
            "type": meta.get("type", ""),
            "final_response": json.dumps(f, ensure_ascii=False),
            "response": json.dumps(f, ensure_ascii=False),
        }

    # Fallback 1: extract by Final Answer sections with loose regex parsing.
    markers = list(FINAL_ANSWER_RE.finditer(text))
    for idx, m in enumerate(markers):
        sec_start = m.start()
        sec_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        section = text[sec_start:sec_end]
        cid = m.group(1) or _pick_case_id(sec_start, id_positions)
        if not cid:
            continue
        if cid in picked:
            continue
        answer = _extract_answer_by_regex(section)
        evidence = _extract_evidence_by_regex(section)
        conf = _extract_confidence_by_regex(section)
        if not answer and not evidence:
            continue
        f = _normalize_final_answer(
            {
                "answer": answer,
                "evidence": evidence,
                "confidence": conf,
            }
        )
        meta = query_map.get(cid, {})
        picked[cid] = {
            "id": cid,
            "case_id": cid,
            "query": meta.get("query", ""),
            "type": meta.get("type", ""),
            "final_response": json.dumps(f, ensure_ascii=False),
            "response": json.dumps(f, ensure_ascii=False),
        }

    # Fallback 2: use summary lines like "A_001 - xxx" to fill missing ids.
    line_pat = re.compile(
        r"^\s*([ABＡＢ])\s*[_＿\-–]?\s*([0-9０-９]{3})\s*[-:：]\s*(.+?)\s*$",
        re.MULTILINE,
    )
    for m in line_pat.finditer(text):
        cid = _normalize_case_id(m.group(1), m.group(2))
        if cid in picked:
            continue
        answer = m.group(3).strip()
        if not answer:
            continue
        f = _normalize_final_answer({"answer": answer, "evidence": [], "confidence": {}})
        meta = query_map.get(cid, {})
        picked[cid] = {
            "id": cid,
            "case_id": cid,
            "query": meta.get("query", ""),
            "type": meta.get("type", ""),
            "final_response": json.dumps(f, ensure_ascii=False),
            "response": json.dumps(f, ensure_ascii=False),
        }

    # Fallback 3: ensure full case coverage with explicit placeholder rows.
    for cid in sorted(query_map.keys()):
        if cid in picked:
            continue
        f = _normalize_final_answer(
            {
                "answer": "未从原始结果中提取到该题的完整最终答案（半自动补齐占位）。",
                "evidence": [],
                "confidence": {
                    "score": 0,
                    "level": "low",
                    "reason": "source_missing_or_unstructured",
                },
            }
        )
        meta = query_map.get(cid, {})
        picked[cid] = {
            "id": cid,
            "case_id": cid,
            "query": meta.get("query", ""),
            "type": meta.get("type", ""),
            "final_response": json.dumps(f, ensure_ascii=False),
            "response": json.dumps(f, ensure_ascii=False),
        }

    rows = [picked[k] for k in sorted(picked.keys())]
    return rows


def parse_glm_json(path: Path, query_map: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    data = obj.get("evaluation_results", [])
    rows: List[Dict[str, Any]] = []
    for x in data:
        cid = str(x.get("id", "")).strip()
        if not cid:
            continue
        final_answer = x.get("final_answer", {})
        if not isinstance(final_answer, dict):
            final_answer = {}
        norm = _normalize_final_answer(final_answer)
        meta = query_map.get(cid, {})
        rows.append(
            {
                "id": cid,
                "case_id": cid,
                "query": meta.get("query", ""),
                "type": meta.get("type", ""),
                "final_response": json.dumps(norm, ensure_ascii=False),
                "response": json.dumps(norm, ensure_ascii=False),
            }
        )
    rows.sort(key=lambda r: r["id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean markdown/json result exports into normalized JSON.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--kimi-md", type=str, required=True)
    parser.add_argument("--minimax-md", type=str, required=True)
    parser.add_argument("--glm-json", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    query_map = _load_query_map(Path(args.dataset).resolve())

    kimi_rows = parse_md_to_json(Path(args.kimi_md).resolve(), query_map)
    minimax_rows = parse_md_to_json(Path(args.minimax_md).resolve(), query_map)
    glm_rows = parse_glm_json(Path(args.glm_json).resolve(), query_map)

    kimi_out = out_dir / "eval_results_kimi.cleaned.json"
    minimax_out = out_dir / "eval_results_minimax.cleaned.json"
    glm_out = out_dir / "eval_results_glm.cleaned.json"

    kimi_out.write_text(json.dumps(kimi_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    minimax_out.write_text(json.dumps(minimax_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    glm_out.write_text(json.dumps(glm_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"kimi_cases: {len(kimi_rows)} -> {kimi_out}")
    print(f"minimax_cases: {len(minimax_rows)} -> {minimax_out}")
    print(f"glm_cases: {len(glm_rows)} -> {glm_out}")


if __name__ == "__main__":
    main()
