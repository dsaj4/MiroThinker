#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
FastAPI server wrapper for running the Miroflow agent as an API.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hydra import compose, initialize_config_dir

from src.core.pipeline import create_pipeline_components, execute_task_pipeline


load_dotenv()

app = FastAPI()


class RunRequest(BaseModel):
    task: str
    task_id: Optional[str] = None
    agent: Optional[str] = None
    llm: Optional[str] = None
    log_dir: Optional[str] = None


def build_cfg(overrides: list[str] | None = None):
    conf_dir = Path(__file__).resolve().parent / "conf"
    with initialize_config_dir(str(conf_dir), version_base=None):
        return compose(config_name="config", overrides=overrides or [])


@app.on_event("startup")
def startup():
    app.state.lock = asyncio.Lock()
    app.state.cfg = build_cfg()
    (
        app.state.main_tool_manager,
        app.state.sub_tool_managers,
        app.state.output_formatter,
    ) = create_pipeline_components(app.state.cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest):
    if not req.task:
        raise HTTPException(status_code=400, detail="task is required")

    overrides = []
    if req.agent:
        overrides.append(f"agent={req.agent}")
    if req.llm:
        overrides.append(f"llm={req.llm}")

    if overrides:
        cfg = build_cfg(overrides)
        main_tm, sub_tms, formatter = create_pipeline_components(cfg)
    else:
        cfg = app.state.cfg
        main_tm = app.state.main_tool_manager
        sub_tms = app.state.sub_tool_managers
        formatter = app.state.output_formatter

    task_id = req.task_id or "api_task"
    log_dir = req.log_dir or cfg.debug_dir

    async with app.state.lock:
        summary, boxed, log_path, failure = await execute_task_pipeline(
            cfg=cfg,
            task_id=task_id,
            task_file_name="",
            task_description=req.task,
            main_agent_tool_manager=main_tm,
            sub_agent_tool_managers=sub_tms,
            output_formatter=formatter,
            log_dir=log_dir,
        )

    if cfg.agent.get("output_mode", "report") == "search_data":
        try:
            search_data = json.loads(summary)
        except Exception:
            search_data = summary
        return {
            "search_data": search_data,
            "log_path": log_path,
            "failure_experience_summary": failure,
        }

    return {
        "summary": summary,
        "answer": boxed,
        "log_path": log_path,
        "failure_experience_summary": failure,
    }
