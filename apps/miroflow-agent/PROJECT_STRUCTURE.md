# Project Structure (miroflow-agent)

## 1) Runtime Entry
- `main.py`: CLI single-task entry (Hydra config + task pipeline).
- `api_server.py`: FastAPI service entry.
- `run_network_search.py`: batch-like network search runner.
- `mirosearch_server.py`: standalone Mirosearch HTTP service (`/health`, `/index`, `/search`).

## 2) Core Execution Path
- `src/core/pipeline.py`: component factory + end-to-end task execution pipeline.
- `src/core/orchestrator.py`: turn-level orchestration, tool-calling loop, rollback logic.
- `src/core/tool_executor.py`: tool invocation, duplicate-query handling, tool result post-process.
- `src/core/answer_generator.py`: final answer synthesis and formatting.

## 3) Cross-Cutting Modules
- `src/llm/*`: provider factory and concrete LLM clients.
- `src/memory/manager.py`: memory orchestration and persistence.
- `src/memory/mirosearch_service.py`: semantic retrieval service wrapper (`memsearch` + fallback).
- `src/logging/task_logger.py`: structured task logs.
- `src/io/*`: input conversion and output formatting.
- `src/utils/*`: prompt builders, parsing helpers, wrappers.

## 4) Config and Tests
- `conf/`: Hydra configs (`llm/`, `agent/`).
- `tests/`: pytest unit tests.
- `eval/`: published evaluation workspace:
  - `materials/`: dataset, questions, ground truth, prompts, cleaned answer sets
  - `results/quality_judge/`: relevance-density-alignment scoring outputs
  - `results/hitrate_judge/`: ground-truth hit-rate scoring outputs
  - `results/overview/`: cross-model summary tables
  - `docs/`: evaluation methodology and process docs
  - `archive/legacy_outputs/`: historical temporary outputs kept for traceability

## 5) Persistent Data Boundary
- Keep:
  - `memory/` (project memory + soul + daily logs)
  - `logs/` (debug traces used for diagnosis/comparison)
  - `apps/miroflow-agent/data/` (runtime datasets such as network search input)
- Safe to regenerate:
  - coverage and html reports
  - pytest cache and Python bytecode cache

## 6) Cleanup Rule of Thumb
- Delete generated artifacts first; keep source/config/data.
- Remove obsolete virtualenv backups only when a working `.venv` exists.
- Do not delete `conf/`, `src/`, `tests/` unless explicitly refactored.
