#!/usr/bin/env python
# Copyright (c) 2025 MiroMind
# Batch script to execute network search queries

import asyncio
import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from src.logging.task_logger import bootstrap_logger

logger = bootstrap_logger()


def _resolve_dataset_path() -> Path:
    candidates = [
        Path("data/network_search_dataset.jsonl"),
        Path("apps/miroflow-agent/data/network_search_dataset.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


async def amain(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg)
    )

    data_file = _resolve_dataset_path()
    if not data_file.exists():
        logger.error(f"Missing dataset {data_file}")
        return

    app_root = Path(__file__).resolve().parent
    debug_dir = Path(cfg.debug_dir)
    if not debug_dir.is_absolute():
        debug_dir = (app_root / debug_dir).resolve()

    out_file = debug_dir / "network_search_results.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(out_file, "a", encoding="utf-8") as out_f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_id = item["id"]
            question = item["question"]
            
            logger.info(f"--- Processing {task_id}: {question} ---")
            
            try:
                # Execute task using the pipeline
                final_summary, final_boxed_answer, log_file_path, failure_experience_summary = await execute_task_pipeline(
                    cfg=cfg,
                    task_id=task_id,
                    task_file_name="",
                    task_description=question,
                    main_agent_tool_manager=main_agent_tool_manager,
                    sub_agent_tool_managers=sub_agent_tool_managers,
                    output_formatter=output_formatter,
                    log_dir=str(debug_dir),
                )
                
                output_item = {
                    "id": task_id,
                    "question": question,
                    "report": str(final_summary),
                    "boxed_answer": str(final_boxed_answer),
                    "log_file": str(log_file_path)
                }
                out_f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                out_f.flush()
                
            except Exception as e:
                logger.error(f"Error processing {task_id}: {e}")

    logger.info(f"Finished processing. Results saved to {out_file}")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))

if __name__ == "__main__":
    main()
