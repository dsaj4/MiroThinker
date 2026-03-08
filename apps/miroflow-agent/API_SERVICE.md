# MiroSearch API Service

## Overview
`api_server.py` exposes the full VisionTree / MiroSearch main pipeline as an external FastAPI service.

Endpoints:
- `GET /health`: service health check
- `GET /config`: current default config snapshot
- `POST /run`: execute the full search pipeline
- `POST /v1/search`: API-friendly concise search endpoint
- `POST /memory/search`: semantic recall against project memory
- `GET /docs`: Swagger UI

## Environment Variables
- `MIROSEARCH_API_HOST`: bind host, default `0.0.0.0`
- `MIROSEARCH_API_PORT`: bind port, default `18080`
- `MIROSEARCH_API_CONCURRENCY`: request concurrency limit, default `1`
- `MIROSEARCH_API_KEY`: optional API key; if set, clients must pass header `x-api-key`

## API-Friendly Mode
`POST /run` supports `api_friendly` and defaults to `true`.

Purpose:
- relax strict final formatting for external callers
- prefer short, directly consumable answers
- avoid treating simple tasks as failures only because `\\boxed{}` or verbose formatting is missing

This mode only affects API requests. Existing CLI and evaluation flows keep their default behavior unless `agent.api_friendly=true` is explicitly enabled.

## Start the Service
```bash
uv run python api_server.py
```

Production-style startup:
```bash
uv run uvicorn api_server:app --host 0.0.0.0 --port 18080 --workers 1
```

## Example Calls
Health check:
```bash
curl http://127.0.0.1:18080/health
```

Run a full task:
```bash
curl -X POST http://127.0.0.1:18080/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "基于2024-2025年公开财报，分析某公司能源业务增长是否足以对冲主业压力",
    "output_mode": "miro",
    "max_turns": 10,
    "api_friendly": true
  }'
```

Use the concise external search endpoint:
```bash
curl -X POST http://127.0.0.1:18080/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "基于2024-2025年跨境数据规则，哪个节点最适合作为AI区域总部桥梁市场",
    "max_turns": 10,
    "api_friendly": true
  }'
```

Memory search:
```bash
curl -X POST http://127.0.0.1:18080/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "新加坡 数据 跨境 合规 结构洞",
    "top_k": 5
  }'
```

If `MIROSEARCH_API_KEY` is configured:
```bash
curl -H "x-api-key: <your-key>" http://127.0.0.1:18080/health
```

## Response Shape
`POST /run` returns:
- `service`
- `task_id`
- `output_mode`
- `api_friendly`
- `log_path`
- `failure_experience_summary`
- `summary` + `answer` for `report` / `miro`
- `miro_result` for `miro` mode, shaped as `{answer, evidence, confidence}`
- `search_data` for `search_data`

`POST /v1/search` returns:
- `service`
- `task_id`
- `mode`
- `api_friendly`
- `log_path`
- `failure_experience_summary`
- `result`
  - `answer`
  - `evidence`
  - `confidence`

## Deployment Notes
- Keep `memory/` and `logs/` mounted on persistent storage.
- If you need stronger semantic recall on Windows, configure a remote Milvus endpoint.
- Default concurrency is conservative because the tool chain is long-running and IO heavy.
