# Linux Deployment

## Scope
This document covers deploying the API service on Linux. The recommended entrypoint is:

- `api_server.py` for the full search API
- `mirosearch_server.py` only if you want standalone memory search

## Runtime Requirements
- Python 3.12+
- `uv` installed on the server
- outbound network access for your configured LLM and search providers
- optional Milvus service for semantic memory backend

## Install
From `apps/miroflow-agent/`:

```bash
uv sync --extra dev
```

If you do not need test dependencies:

```bash
uv sync
```

## Environment
Create `.env` from `.env.example` and set at minimum:

- `DASHSCOPE_API_KEY` or your configured LLM provider key
- `SERPER_API_KEY`
- `JINA_API_KEY`
- `E2B_API_KEY` if sandbox tools are enabled

Optional:

- `MIROSEARCH_API_KEY`
- `MILVUS_URI`
- `MILVUS_TOKEN`

## Start
Recommended production command:

```bash
uv run uvicorn api_server:app --host 0.0.0.0 --port 18080 --workers 1
```

Health check:

```bash
curl http://127.0.0.1:18080/health
```

## Systemd Example
See:

- `deploy/linux/mirosearch-api.service`

Before enabling it, replace:

- `__APP_DIR__`
- `__UV_BIN__`
- `__USER__`
- `__GROUP__`

## Notes
- API responses are JSON-first. External integrations should prefer `POST /v1/search`.
- Relative paths in runtime entrypoints are resolved from the application directory, so the service is safe to run from any working directory.
- If Milvus is not configured, memory falls back to local TF-IDF recall.
