from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.logging.task_logger import TaskLog, get_utc_plus_8_time
from src.memory.manager import MemoryManager


def _build_cfg(memory_root: Path, semantic_recall: bool = False):
    return OmegaConf.create(
        {
            "llm": {
                "provider": "openai",
                "model_name": "gpt-5",
                "temperature": 0.0,
                "top_p": 1.0,
                "min_p": 0.0,
                "top_k": -1,
                "max_context_length": 65536,
                "max_tokens": 2048,
                "async_client": False,
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
            },
            "agent": {
                "keep_tool_result": -1,
                "main_agent": {"max_turns": 3},
            },
            "memory": {
                "enabled": True,
                "semantic_recall": semantic_recall,
                "update_soul": True,
                "short_circuit_search_on_hit": False,
                "recall_top_k": 5,
                "inject_top_k": 4,
                "recall_min_score": 0.45,
                "root_dir": str(memory_root),
                "soul_file": "SOUL.md",
                "memory_file": "MEMORY.md",
                "milvus_uri": str(memory_root / "index" / "memsearch.db"),
                "collection": "test_memory",
                "embedding_provider": "openai",
                "embedding_model": None,
            },
        }
    )


def _build_log(log_dir: Path):
    return TaskLog(
        log_dir=str(log_dir),
        task_id="test-task",
        start_time=get_utc_plus_8_time(),
    )


@pytest.mark.asyncio
async def test_memory_bootstrap_and_context_injection(tmp_path: Path):
    cfg = _build_cfg(tmp_path / "memory")
    log = _build_log(tmp_path / "logs")
    manager = MemoryManager(cfg, log)

    assert manager.enabled is True
    assert manager.ready is True
    assert (tmp_path / "memory" / "SOUL.md").exists()
    assert (tmp_path / "memory" / "MEMORY.md").exists()
    assert (tmp_path / "memory" / "daily").exists()
    assert (tmp_path / "memory" / "external").exists()

    context = await manager.build_system_memory_context(
        task_description="Find latest benchmark setup",
        task_file_name="",
    )
    assert "Planner Memory Priority Context" in context
    assert "Soul File" in context
    assert "Persistent Memory" in context
    manager.close()


@pytest.mark.asyncio
async def test_memory_persist_task_writes_daily_and_soul(tmp_path: Path):
    cfg = _build_cfg(tmp_path / "memory")
    log = _build_log(tmp_path / "logs")
    manager = MemoryManager(cfg, log)

    log.trace_data["tool_calls"] = [
        {
            "tool_name": "google_search",
            "arguments": {"q": "GAIA benchmark latest setting"},
            "result": {"result": '{"organic":[{"title":"x"}]}'},
        },
        {
            "tool_name": "scrape_and_extract_info",
            "arguments": {
                "url": "https://example.com",
                "info_to_extract": "official score table",
            },
            "result": {"result": '{"text":"evidence"}'},
        },
    ]

    await manager.persist_task_memory(
        task_id="task-001",
        task_description="Summarize benchmark setup",
        task_file_name="",
        final_summary="done summary",
        final_boxed_answer="42",
    )

    daily_files = list((tmp_path / "memory" / "daily").glob("*.md"))
    assert daily_files, "daily memory log should be created"
    daily_content = daily_files[0].read_text(encoding="utf-8")
    assert "Task task-001" in daily_content
    assert "Final Boxed Answer" in daily_content
    assert "google_search" in daily_content

    soul_content = (tmp_path / "memory" / "SOUL.md").read_text(encoding="utf-8")
    assert "Iterative Reinforcement Log" in soul_content
    assert "task=task-001" in soul_content
    manager.close()


@pytest.mark.asyncio
async def test_memory_recall_short_circuit_disabled_without_semantic(tmp_path: Path):
    cfg = _build_cfg(tmp_path / "memory")
    log = _build_log(tmp_path / "logs")
    manager = MemoryManager(cfg, log)

    result = await manager.maybe_short_circuit_search(
        tool_name="google_search",
        arguments={"q": "python unittest"},
    )
    assert result is None
    manager.close()


@pytest.mark.asyncio
async def test_memory_semantic_fallback_uses_local_index_when_memsearch_unavailable(
    tmp_path: Path, monkeypatch
):
    memory_root = tmp_path / "memory"
    daily = memory_root / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    (daily / "2026-03-01.md").write_text(
        "# 2026-03-01\n\n## 规划优化\n- 通过多轮搜索来强化规划能力与证据覆盖。\n",
        encoding="utf-8",
    )

    from src.memory import mirosearch_service as ms

    monkeypatch.setattr(ms, "MemSearch", None)
    cfg = _build_cfg(memory_root, semantic_recall=True)
    log = _build_log(tmp_path / "logs")
    manager = MemoryManager(cfg, log)

    assert manager.semantic_backend == "local_tfidf"
    await manager.warmup()
    recall = await manager.recall("如何在多轮任务中强化搜索规划能力", top_k=3)
    assert recall is not None
    assert recall.hits
    assert recall.best_score > 0
    manager.close()


@pytest.mark.asyncio
async def test_memory_semantic_recall_matches_chinese_heading(tmp_path: Path, monkeypatch):
    memory_root = tmp_path / "memory"
    note = memory_root / "daily" / "2026-03-01.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# 2026-03-01\n\n## 搜索规划能力强化\n记录：优先复用历史查询并减少重复搜索。\n",
        encoding="utf-8",
    )

    from src.memory import mirosearch_service as ms

    monkeypatch.setattr(ms, "MemSearch", None)
    cfg = _build_cfg(memory_root, semantic_recall=True)
    log = _build_log(tmp_path / "logs")
    manager = MemoryManager(cfg, log)

    recall = await manager.recall("搜索规划能力应该如何强化", top_k=3)
    assert recall is not None
    assert recall.hits
    top = recall.hits[0]
    assert "2026-03-01.md" in str(top.get("source", ""))
    assert top.get("score", 0) > 0
    manager.close()
