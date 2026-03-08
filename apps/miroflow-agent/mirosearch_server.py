# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from omegaconf import OmegaConf

from src.logging.task_logger import TaskLog, get_utc_plus_8_time
from src.memory.mirosearch_service import MirosearchService

APP_DIR = Path(__file__).resolve().parent
CONF_PATH = APP_DIR / "conf" / "config.yaml"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class MirosearchApp:
    def __init__(self):
        cfg = OmegaConf.load(CONF_PATH)
        memory_cfg = dict(cfg.get("memory", {}))
        root_dir = (APP_DIR / memory_cfg.get("root_dir", "../../memory")).resolve()
        index_dir = root_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        self.task_log = TaskLog(
            log_dir=str((APP_DIR / cfg.get("debug_dir", "../../logs/debug")).resolve()),
            task_id=f"mirosearch_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=get_utc_plus_8_time(),
        )
        self.service = MirosearchService(
            root_dir=root_dir,
            index_dir=index_dir,
            memory_cfg=memory_cfg,
            task_log=self.task_log,
            semantic_enabled=bool(memory_cfg.get("semantic_recall", True)),
        )

    async def warmup(self):
        await self.service.warmup()

    async def search(self, query: str, top_k: int):
        return await self.service.search(query, top_k=top_k)

    async def reindex(self):
        await self.service.reindex()

    def health(self):
        return self.service.health()

    def close(self):
        self.service.close()


app = FastAPI(title="Mirosearch Service", version="0.1.0")
miro = MirosearchApp()


@app.on_event("startup")
async def startup_event():
    await miro.warmup()


@app.on_event("shutdown")
def shutdown_event():
    miro.close()


@app.get("/health")
def health():
    return miro.health()


@app.post("/index")
async def index_memory():
    await miro.reindex()
    return {"ok": True, **miro.health()}


@app.post("/search")
async def search_memory(req: SearchRequest):
    hits = await miro.search(req.query, req.top_k)
    best_score = max((float(x.get("score", 0.0) or 0.0) for x in hits), default=0.0)
    return {
        "service": "Mirosearch",
        "backend": miro.health().get("backend"),
        "query": req.query,
        "top_k": req.top_k,
        "best_score": best_score,
        "hits": hits,
    }


if __name__ == "__main__":
    uvicorn.run("mirosearch_server:app", host="127.0.0.1", port=8765, reload=False)
