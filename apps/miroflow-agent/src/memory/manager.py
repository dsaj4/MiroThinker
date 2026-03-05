# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Persistent memory manager for planner persona and iterative search reinforcement.

This module integrates markdown-first memory storage with optional semantic recall
powered by Mirosearch (memsearch backend + Windows fallback). It supports:
- Soul file bootstrap and iterative reinforcement updates
- Task-level memory recall for system prompt augmentation
- Search-time memory checks before external retrieval
- End-of-task memory write-back from tool traces and final answers
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from ..logging.task_logger import TaskLog, get_utc_plus_8_time
from .mirosearch_service import MirosearchService


DEFAULT_SOUL_TEMPLATE = """# Planner Soul

## Identity
You are the Planning Architect for deep research. Your job is to maximize answer correctness and verifiability through iterative search planning.

## Core Principles
- Evidence first: every key claim should be supported by retrieved evidence.
- Progressive narrowing: start broad, then narrow by entities, dates, and constraints.
- Multi-source verification: prefer cross-source consistency over single-source confidence.
- Anti-looping: avoid repeating equivalent queries unless new constraints are introduced.
- Cost-awareness: stop when marginal information gain is low.

## Search Planning Playbook
1. Decompose task into answerable sub-questions.
2. Produce at least one broad query and one constrained query.
3. Validate with source quality and temporal relevance.
4. Aggregate evidence and identify unresolved gaps.
5. If gaps remain, run targeted follow-up queries.

## Evidence Quality Rubric
- High: official docs, primary sources, reputable publications, direct data.
- Medium: secondary summaries with citations.
- Low: opinionated or unattributed content.

## Iterative Reinforcement Log
"""


DEFAULT_MEMORY_TEMPLATE = """# MEMORY

## Project Memory
- This file stores long-term project-level constraints and persistent preferences.
- Task-specific execution traces should be stored in daily logs under `daily/`.
"""


@dataclass
class MemoryRecallResult:
    query: str
    hits: List[Dict[str, Any]]
    best_score: float
    summary_text: str


class MemoryManager:
    """Persistent memory and planner soul management."""

    SEARCH_TOOL_NAMES = {
        "google_search",
        "sogou_search",
        "search_and_browse",
        "scrape_website",
        "scrape_and_extract_info",
    }

    def __init__(self, cfg: DictConfig, task_log: TaskLog):
        self.cfg = cfg
        self.task_log = task_log
        self.memory_cfg = cfg.get("memory", {})
        self.enabled = bool(self.memory_cfg.get("enabled", False))
        self.semantic_enabled = bool(self.memory_cfg.get("semantic_recall", True))
        self.update_soul_enabled = bool(self.memory_cfg.get("update_soul", True))
        self.short_circuit_on_hit = bool(
            self.memory_cfg.get("short_circuit_search_on_hit", False)
        )
        self.recall_top_k = int(self.memory_cfg.get("recall_top_k", 5))
        self.inject_top_k = int(self.memory_cfg.get("inject_top_k", 4))
        self.recall_min_score = float(self.memory_cfg.get("recall_min_score", 0.45))
        self.max_hit_chars = int(self.memory_cfg.get("max_hit_chars", 420))
        self.max_context_chars = int(self.memory_cfg.get("max_context_chars", 3200))

        self.root_dir = Path(self.memory_cfg.get("root_dir", "../../memory")).resolve()
        self.daily_dir = self.root_dir / "daily"
        self.external_dir = self.root_dir / "external"
        self.index_dir = self.root_dir / "index"
        self.soul_file = self.root_dir / str(self.memory_cfg.get("soul_file", "SOUL.md"))
        self.memory_file = self.root_dir / str(
            self.memory_cfg.get("memory_file", "MEMORY.md")
        )
        self.soul_versions_dir = self.root_dir / "soul_versions"

        self.semantic_backend = "disabled"
        self.mirosearch: Optional[MirosearchService] = None
        self.ready = False

        if not self.enabled:
            return

        self._bootstrap_dirs_and_files()
        self._initialize_mirosearch()

    def _bootstrap_dirs_and_files(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.soul_versions_dir.mkdir(parents=True, exist_ok=True)

        if not self.soul_file.exists():
            self.soul_file.write_text(DEFAULT_SOUL_TEMPLATE, encoding="utf-8")
        if not self.memory_file.exists():
            self.memory_file.write_text(DEFAULT_MEMORY_TEMPLATE, encoding="utf-8")

    def _initialize_mirosearch(self):
        self.mirosearch = MirosearchService(
            root_dir=self.root_dir,
            index_dir=self.index_dir,
            memory_cfg=self.memory_cfg,
            task_log=self.task_log,
            semantic_enabled=self.semantic_enabled,
        )
        self.semantic_backend = self.mirosearch.backend
        self.ready = self.mirosearch.ready

    async def warmup(self):
        """Initial indexing to ensure recall is available before execution."""
        if not self.enabled or not self.ready or not self.semantic_enabled:
            return
        try:
            if self.mirosearch is not None:
                await self.mirosearch.warmup()
            self.task_log.log_step(
                "info",
                "Memory | Warmup",
                f"Memory index warmup completed (backend={self.semantic_backend}).",
            )
        except Exception as exc:
            self.task_log.log_step(
                "warning",
                "Memory | Warmup",
                f"Memory warmup failed: {exc}",
            )

    def _safe_read_text(self, path: Path, default: str = "") -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return default

    async def build_system_memory_context(
        self, task_description: str, task_file_name: str = ""
    ) -> str:
        """Build memory context to inject into system prompt."""
        if not self.enabled or not self.ready:
            return ""

        soul = self._safe_read_text(self.soul_file, "")
        memory = self._safe_read_text(self.memory_file, "")

        query = task_description
        if task_file_name:
            query += f"\nAttached file: {task_file_name}"

        recalls = await self.recall(query, top_k=self.inject_top_k)
        recall_text = recalls.summary_text if recalls else "No relevant memory hits."

        packed = (
            "## Planner Memory Priority Context\n"
            "The following memory context is persistent project memory and should be prioritized when planning search strategy.\n\n"
            "### Soul File\n"
            f"{soul[: self.max_context_chars]}\n\n"
            "### Persistent Memory\n"
            f"{memory[: max(800, self.max_context_chars // 2)]}\n\n"
            "### Recalled Relevant Memory\n"
            f"{recall_text[: self.max_context_chars]}\n"
        )
        return packed

    def _extract_query_from_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "google_search":
            return str(arguments.get("q", "")).strip()
        if tool_name == "sogou_search":
            return str(arguments.get("Query", "")).strip()
        if tool_name == "search_and_browse":
            return str(arguments.get("subtask", "")).strip()
        if tool_name == "scrape_website":
            return str(arguments.get("url", "")).strip()
        if tool_name == "scrape_and_extract_info":
            url = str(arguments.get("url", "")).strip()
            what = str(arguments.get("info_to_extract", "")).strip()
            return f"{url} {what}".strip()
        return ""

    def is_search_tool(self, tool_name: str) -> bool:
        return tool_name in self.SEARCH_TOOL_NAMES or "search" in tool_name

    async def recall(self, query: str, top_k: Optional[int] = None) -> Optional[MemoryRecallResult]:
        if not self.enabled or not self.ready or not query:
            return None

        if not self.semantic_enabled:
            return None

        k = top_k or self.recall_top_k
        try:
            if self.mirosearch is not None:
                hits = await self.mirosearch.search(query, top_k=k)
            else:
                hits = []
        except Exception as exc:
            self.task_log.log_step(
                "warning",
                "Memory | Recall",
                f"Semantic recall failed (backend={self.semantic_backend}): {exc}",
            )
            return None

        if not hits:
            return MemoryRecallResult(query=query, hits=[], best_score=0.0, summary_text="")

        lines = []
        best_score = 0.0
        for i, hit in enumerate(hits, start=1):
            score = float(hit.get("score", 0.0) or 0.0)
            best_score = max(best_score, score)
            content = str(hit.get("content", "")).strip().replace("\n", " ")
            content = content[: self.max_hit_chars]
            source = str(hit.get("source", "unknown"))
            lines.append(f"{i}. [score={score:.4f}] ({source}) {content}")

        return MemoryRecallResult(
            query=query,
            hits=hits,
            best_score=best_score,
            summary_text="\n".join(lines),
        )

    async def maybe_short_circuit_search(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use memory recall as a pre-search source.

        If a strong memory match exists and short-circuit is enabled, return a
        synthetic tool result to avoid external retrieval.
        """
        if not self.enabled or not self.is_search_tool(tool_name):
            return None

        query = self._extract_query_from_tool_call(tool_name, arguments)
        if not query:
            return None

        recalls = await self.recall(query, top_k=self.recall_top_k)
        if recalls is None:
            return None

        if not recalls.hits:
            self.task_log.log_step(
                "info",
                "Memory | Pre-Search",
                f"No reusable memory for query: {query}",
            )
            return None

        self.task_log.log_step(
            "info",
            "Memory | Pre-Search",
            f"Memory recall hit for query: {query} (best_score={recalls.best_score:.4f})",
        )

        if not self.short_circuit_on_hit or recalls.best_score < self.recall_min_score:
            return None

        synthetic = {
            "result": json.dumps(
                {
                    "memory_recall": True,
                    "query": query,
                    "best_score": recalls.best_score,
                    "hits": [
                        {
                            "content": h.get("content", ""),
                            "source": h.get("source", ""),
                            "score": h.get("score", 0.0),
                            "heading": h.get("heading", ""),
                        }
                        for h in recalls.hits
                    ],
                },
                ensure_ascii=False,
            )
        }
        return synthetic

    def _summarize_tool_calls(self, tool_calls: List[Dict[str, Any]], limit: int = 20) -> str:
        if not tool_calls:
            return "No tool calls were recorded."

        lines = []
        for idx, call in enumerate(tool_calls[:limit], start=1):
            tool_name = call.get("tool_name", "unknown")
            args = call.get("arguments", {})
            result = call.get("result")
            if isinstance(result, dict):
                result_preview = str(result.get("result", result.get("error", "")))
            else:
                result_preview = str(result)
            result_preview = result_preview.replace("\n", " ")[:260]
            lines.append(
                f"{idx}. tool={tool_name}; args={json.dumps(args, ensure_ascii=False)[:180]}; result={result_preview}"
            )
        if len(tool_calls) > limit:
            lines.append(f"... truncated {len(tool_calls) - limit} additional tool calls ...")
        return "\n".join(lines)

    def _extract_iteration_insights(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        insights = []
        search_calls = [c for c in tool_calls if self.is_search_tool(str(c.get("tool_name", "")))]
        if search_calls:
            insights.append(
                f"Search depth this run: {len(search_calls)} search-related tool calls."
            )

        successful_queries = 0
        failed_queries = 0
        for call in search_calls:
            result = call.get("result")
            if isinstance(result, dict):
                raw = str(result.get("result", ""))
            else:
                raw = str(result)
            if '"organic": []' in raw or "Unknown tool:" in raw or "Error executing tool" in raw:
                failed_queries += 1
            else:
                successful_queries += 1

        if successful_queries > 0:
            insights.append(
                f"Effective retrieval pattern observed in {successful_queries} search calls."
            )
        if failed_queries > 0:
            insights.append(
                f"{failed_queries} search calls appeared weak or empty; reinforce query rewriting before repeating."
            )
        if not insights:
            insights.append("No search-specific signal captured; prioritize explicit decomposition next run.")
        return insights

    def _append_soul_update(self, task_id: str, insights: List[str]):
        if not self.enabled or not self.update_soul_enabled:
            return

        ts = get_utc_plus_8_time()
        update_block = [f"\n### {ts} | task={task_id}"]
        for ins in insights:
            update_block.append(f"- {ins}")
        update_text = "\n".join(update_block) + "\n"

        try:
            if self.soul_file.exists():
                previous = self.soul_file.read_text(encoding="utf-8")
                version_path = self.soul_versions_dir / f"SOUL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                version_path.write_text(previous, encoding="utf-8")
            with open(self.soul_file, "a", encoding="utf-8") as f:
                f.write(update_text)
            self.task_log.log_step(
                "info",
                "Memory | Soul Update",
                f"Soul file updated with {len(insights)} insights.",
            )
        except Exception as exc:
            self.task_log.log_step(
                "warning",
                "Memory | Soul Update",
                f"Failed to update soul file: {exc}",
            )

    def _capture_external_file(self, task_file_name: str):
        if not task_file_name:
            return
        src = Path(task_file_name)
        if not src.exists():
            return
        try:
            dst = self.external_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            note_path = self.daily_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
            with open(note_path, "a", encoding="utf-8") as f:
                f.write(
                    "\n\n## External Attachment\n"
                    f"- File: {src.name}\n"
                    f"- Stored at: {dst}\n"
                    f"- Captured at: {get_utc_plus_8_time()}\n"
                )
            self.task_log.log_step(
                "info",
                "Memory | External File",
                f"Captured external attachment: {src.name}",
            )
        except Exception as exc:
            self.task_log.log_step(
                "warning",
                "Memory | External File",
                f"Failed to capture external file: {exc}",
            )

    async def persist_task_memory(
        self,
        task_id: str,
        task_description: str,
        task_file_name: str,
        final_summary: str,
        final_boxed_answer: str,
    ):
        """Write task memory and update semantic index and soul file."""
        if not self.enabled:
            return

        tool_calls = self.task_log.trace_data.get("tool_calls", [])
        now = datetime.now().strftime("%Y-%m-%d")
        daily_file = self.daily_dir / f"{now}.md"

        memory_block = (
            f"\n\n## Task {task_id} | {get_utc_plus_8_time()}\n"
            f"### Task\n{task_description}\n\n"
            f"### Final Boxed Answer\n{final_boxed_answer}\n\n"
            f"### Final Summary\n{str(final_summary)[:2200]}\n\n"
            "### Search and Tool Trace Summary\n"
            f"{self._summarize_tool_calls(tool_calls)}\n"
        )
        with open(daily_file, "a", encoding="utf-8") as f:
            if daily_file.stat().st_size == 0:
                f.write(f"# {now}\n")
            f.write(memory_block)

        self._capture_external_file(task_file_name)

        insights = self._extract_iteration_insights(tool_calls)
        self._append_soul_update(task_id, insights)

        if self.semantic_enabled:
            try:
                if self.mirosearch is not None:
                    await self.mirosearch.reindex()
                self.task_log.log_step(
                    "info",
                    "Memory | Persist",
                    f"Task memory persisted and semantic index updated (backend={self.semantic_backend}).",
                )
            except Exception as exc:
                self.task_log.log_step(
                    "warning",
                    "Memory | Persist",
                    f"Memory persisted but index update failed: {exc}",
                )

    def close(self):
        if self.mirosearch is not None:
            self.mirosearch.close()
