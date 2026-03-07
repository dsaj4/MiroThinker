# Memory System Initialization Guide

## Purpose
This directory stores persistent memory for Miroflow Agent:
- `SOUL.md`: planner identity and iterative strategy reinforcement log
- `MEMORY.md`: long-term project preferences and constraints
- `daily/`: task execution memory by date
- `external/`: copied external attachments
- `index/`: semantic retrieval index cache
- `soul_versions/`: version backups of `SOUL.md`

## Current Reset State
- `SOUL.md` has been reset to default template.
- `MEMORY.md` has been reset to default template.
- `daily/`, `external/`, `index/`, `soul_versions/` are empty.

## Minimal Initialization Steps
1. Configure memory in `apps/miroflow-agent/conf/config.yaml`:
   - `memory.enabled: true`
   - `memory.semantic_recall: true`
   - `memory.root_dir: ../../memory`
2. (Optional but recommended) Configure Milvus for real memsearch backend:
   - set `MILVUS_URI`
   - set `MILVUS_TOKEN` if needed
3. Run one task through the main pipeline:
   - `python apps/miroflow-agent/main.py llm=qwen-3 agent=mirothinker_v1.5_keep5_max200`

## What Happens After First Run
- `daily/YYYY-MM-DD.md` is created and task memory is appended.
- `SOUL.md` gets a new reinforcement block.
- semantic index is warmed up/updated (Mirosearch backend).

## Health Check
Check task log (`logs/debug/task_*.json`) contains:
- `Mirosearch | Init`
- `Memory | Warmup`
- `Memory | Prompt Injection`
- `Memory | Persist`

If backend is `local_tfidf`, memory still works.
To switch to real memsearch backend, provide a valid remote Milvus endpoint.
