#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""FastAPI service wrapper for exposing the MiroSearch task pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.logging.task_logger import TaskLog, bootstrap_logger, get_utc_plus_8_time
from src.memory.manager import MemoryManager

APP_DIR = Path(__file__).resolve().parent
CONF_DIR = APP_DIR / "conf"
ENV_FILE = APP_DIR / ".env"

load_dotenv(ENV_FILE if ENV_FILE.exists() else None)
logger = bootstrap_logger()

API_TITLE = "MiroSearch API"
API_VERSION = "1.0.0"
DEFAULT_HOST = os.getenv("MIROSEARCH_API_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MIROSEARCH_API_PORT", "18080"))
DEFAULT_CONCURRENCY = max(1, int(os.getenv("MIROSEARCH_API_CONCURRENCY", "1")))
API_KEY = os.getenv("MIROSEARCH_API_KEY", "").strip()


class RunRequest(BaseModel):
    task: str = Field(..., description="Task/query text executed through the full pipeline.")
    task_id: Optional[str] = Field(
        default=None, description="Optional external task id. Auto-generated if absent."
    )
    agent: Optional[str] = Field(default=None, description="Hydra agent config override.")
    llm: Optional[str] = Field(default=None, description="Hydra llm config override.")
    log_dir: Optional[str] = Field(default=None, description="Optional log directory override.")
    task_file_name: str = Field(
        default="", description="Optional attached file path already accessible on the server."
    )
    output_mode: Optional[str] = Field(
        default=None, description="Optional output mode override: report | search_data | miro."
    )
    max_turns: Optional[int] = Field(
        default=None, ge=1, le=100, description="Optional main agent max turn override."
    )
    api_friendly: bool = Field(
        default=True,
        description="Enable API-friendly answer behavior with more tolerant final formatting.",
    )


class SearchMemoryRequest(BaseModel):
    query: str = Field(..., description="Semantic memory query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Recall top-k.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query executed through the full MiroSearch pipeline.")
    task_id: Optional[str] = Field(
        default=None, description="Optional external task id. Auto-generated if absent."
    )
    task_file_name: str = Field(
        default="", description="Optional attached file path already accessible on the server."
    )
    agent: Optional[str] = Field(default=None, description="Optional Hydra agent config override.")
    llm: Optional[str] = Field(default=None, description="Optional Hydra llm config override.")
    log_dir: Optional[str] = Field(default=None, description="Optional log directory override.")
    max_turns: Optional[int] = Field(
        default=None, ge=1, le=100, description="Optional main agent max turn override."
    )
    api_friendly: bool = Field(
        default=True,
        description="Enable API-friendly answer behavior with more tolerant final formatting.",
    )


class ApiTaskLog(TaskLog):
    """Lightweight task log holder for memory-only endpoints."""


def build_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(str(CONF_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def _make_task_id(prefix: str = "api") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _build_overrides(req: RunRequest) -> list[str]:
    overrides: list[str] = []
    if req.agent:
        overrides.append(f"agent={req.agent}")
    if req.llm:
        overrides.append(f"llm={req.llm}")
    if req.output_mode:
        overrides.append(f"agent.output_mode={req.output_mode}")
    if req.max_turns is not None:
        overrides.append(f"agent.main_agent.max_turns={req.max_turns}")
    overrides.append(f"agent.api_friendly={'true' if req.api_friendly else 'false'}")
    return overrides


async def _enforce_api_key(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _create_memory_task_log() -> TaskLog:
    debug_dir = (APP_DIR / "../../logs/debug").resolve()
    return ApiTaskLog(
        log_dir=str(debug_dir),
        task_id=_make_task_id(prefix="memory_api"),
        start_time=get_utc_plus_8_time(),
    )


def _build_response(
    cfg: DictConfig,
    task_id: str,
    summary: str,
    boxed: str,
    log_path: str,
    failure: Optional[str],
) -> Dict[str, Any]:
    output_mode = cfg.agent.get("output_mode", "report")
    response: Dict[str, Any] = {
        "service": "MiroSearch",
        "task_id": task_id,
        "output_mode": output_mode,
        "api_friendly": bool(cfg.agent.get("api_friendly", False)),
        "log_path": log_path,
        "failure_experience_summary": failure,
    }
    if output_mode == "search_data":
        try:
            response["search_data"] = json.loads(summary)
        except Exception:
            response["search_data"] = summary
    else:
        response["summary"] = summary
        response["answer"] = boxed
    return response


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _extract_miro_result_from_log(
    log_path: str, formatter
) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
    except Exception:
        return None

    history = (payload.get("main_agent_message_history", {}) or {}).get(
        "message_history", []
    ) or []
    final_answer_text = ""
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        final_answer_text = _message_content_to_text(message.get("content"))
        if final_answer_text:
            break

    if not final_answer_text:
        final_answer_text = str(payload.get("final_boxed_answer", "")).strip()

    tool_calls = payload.get("trace_data", {}).get("tool_calls", [])
    try:
        return formatter.build_miro_payload(final_answer_text, tool_calls)
    except Exception:
        return None


def _to_search_request(req: SearchRequest) -> RunRequest:
    return RunRequest(
        task=req.query,
        task_id=req.task_id,
        agent=req.agent,
        llm=req.llm,
        log_dir=req.log_dir,
        task_file_name=req.task_file_name,
        output_mode="miro",
        max_turns=req.max_turns,
        api_friendly=req.api_friendly,
    )


async def _run_pipeline_internal(
    req: RunRequest,
    semaphore: asyncio.Semaphore,
    default_cfg: DictConfig,
    default_main_tm,
    default_sub_tms,
    default_formatter,
) -> Dict[str, Any]:
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    overrides = _build_overrides(req)
    if overrides:
        cfg = build_cfg(overrides)
        main_tm, sub_tms, formatter = create_pipeline_components(cfg)
    else:
        cfg = default_cfg
        main_tm = default_main_tm
        sub_tms = default_sub_tms
        formatter = default_formatter

    task_id = req.task_id or _make_task_id()
    log_dir = req.log_dir or cfg.debug_dir

    async with semaphore:
        summary, boxed, log_path, failure = await execute_task_pipeline(
            cfg=cfg,
            task_id=task_id,
            task_file_name=req.task_file_name,
            task_description=req.task,
            main_agent_tool_manager=main_tm,
            sub_agent_tool_managers=sub_tms,
            output_formatter=formatter,
            log_dir=log_dir,
        )

    response = _build_response(cfg, task_id, summary, boxed, log_path, failure)
    if cfg.agent.get("output_mode", "report") == "miro":
        miro_result = _extract_miro_result_from_log(log_path, formatter)
        if miro_result is not None:
            response["miro_result"] = miro_result
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    app.state.cfg = build_cfg()
    (
        app.state.main_tool_manager,
        app.state.sub_tool_managers,
        app.state.output_formatter,
    ) = create_pipeline_components(app.state.cfg)
    app.state.memory_manager = MemoryManager(cfg=app.state.cfg, task_log=_create_memory_task_log())
    await app.state.memory_manager.warmup()
    logger.info(
        "MiroSearch API started with concurrency=%s, api_key_enabled=%s",
        DEFAULT_CONCURRENCY,
        bool(API_KEY),
    )
    yield
    try:
        app.state.memory_manager.close()
    except Exception:
        pass


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="External API for running the full VisionTree / MiroSearch pipeline.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    await _enforce_api_key(x_api_key)
    return {
        "status": "ok",
        "service": "MiroSearch",
        "version": API_VERSION,
        "concurrency": DEFAULT_CONCURRENCY,
        "api_key_enabled": bool(API_KEY),
    }


@app.get("/config")
async def config_snapshot(
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    await _enforce_api_key(x_api_key)
    cfg = app.state.cfg
    return {
        "service": "MiroSearch",
        "default_agent": str(cfg.get("agent")),
        "default_llm": str(cfg.get("llm")),
        "debug_dir": str(cfg.get("debug_dir")),
        "memory_enabled": bool(cfg.get("memory", {}).get("enabled", False)),
        "api_friendly_request_default": RunRequest.model_fields["api_friendly"].default,
    }


@app.post("/run")
async def run_pipeline(
    req: RunRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    await _enforce_api_key(x_api_key)
    return await _run_pipeline_internal(
        req=req,
        semaphore=app.state.semaphore,
        default_cfg=app.state.cfg,
        default_main_tm=app.state.main_tool_manager,
        default_sub_tms=app.state.sub_tool_managers,
        default_formatter=app.state.output_formatter,
    )


@app.post("/v1/search")
async def v1_search(
    req: SearchRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    await _enforce_api_key(x_api_key)
    response = await _run_pipeline_internal(
        req=_to_search_request(req),
        semaphore=app.state.semaphore,
        default_cfg=app.state.cfg,
        default_main_tm=app.state.main_tool_manager,
        default_sub_tms=app.state.sub_tool_managers,
        default_formatter=app.state.output_formatter,
    )

    miro_result = response.get("miro_result") or {
        "answer": response.get("answer", ""),
        "evidence": [],
        "confidence": {"score": 20, "level": "low", "reason": "Structured result unavailable."},
    }
    return {
        "service": response["service"],
        "task_id": response["task_id"],
        "mode": "miro",
        "api_friendly": response["api_friendly"],
        "log_path": response["log_path"],
        "failure_experience_summary": response["failure_experience_summary"],
        "result": miro_result,
    }


@app.post("/memory/search")
async def search_memory(
    req: SearchMemoryRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    await _enforce_api_key(x_api_key)
    memory_manager: MemoryManager = app.state.memory_manager
    recall = await memory_manager.recall(req.query, top_k=req.top_k)
    if recall is None:
        return {
            "service": "MiroSearch",
            "query": req.query,
            "top_k": req.top_k,
            "best_score": 0.0,
            "hits": [],
            "summary_text": "",
        }
    return {
        "service": "MiroSearch",
        "query": req.query,
        "top_k": req.top_k,
        "best_score": recall.best_score,
        "hits": recall.hits,
        "summary_text": recall.summary_text,
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=False)
