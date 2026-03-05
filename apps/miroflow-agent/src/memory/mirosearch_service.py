# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging.task_logger import TaskLog

try:
    from memsearch import MemSearch
except Exception:  # pragma: no cover - optional dependency
    MemSearch = None  # type: ignore


class LocalSemanticMemoryIndex:
    """Lightweight local semantic-like index for Windows fallback."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.docs: List[Dict[str, Any]] = []
        self._idf: Dict[str, float] = {}
        self._doc_vectors: List[Dict[str, float]] = []
        self._doc_norms: List[float] = []

    @staticmethod
    def _cjk_ngrams(span: str, min_n: int = 2, max_n: int = 3) -> List[str]:
        chars = [c for c in span if "\u4e00" <= c <= "\u9fff"]
        if not chars:
            return []
        out = [f"c1:{c}" for c in chars]
        for n in range(min_n, max_n + 1):
            if len(chars) < n:
                continue
            for i in range(0, len(chars) - n + 1):
                out.append(f"c{n}:{''.join(chars[i : i + n])}")
        return out

    @staticmethod
    def _latin_ngrams(token: str, min_n: int = 3, max_n: int = 4) -> List[str]:
        token = token.strip()
        if len(token) < min_n:
            return []
        out: List[str] = []
        upper = min(max_n, len(token))
        for n in range(min_n, upper + 1):
            for i in range(0, len(token) - n + 1):
                out.append(f"w{n}:{token[i : i + n]}")
        return out

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if not normalized:
            return []

        tokens: List[str] = []
        latin_tokens = re.findall(r"[a-z0-9_]+", normalized)
        for tok in latin_tokens:
            tokens.append(f"w:{tok}")
            tokens.extend(cls._latin_ngrams(tok))

        cjk_spans = re.findall(r"[\u4e00-\u9fff]+", normalized)
        for span in cjk_spans:
            tokens.extend(cls._cjk_ngrams(span))
        return tokens

    @staticmethod
    def _split_chunks(content: str, max_chars: int = 800) -> List[Tuple[str, int, int]]:
        chunks = []
        if not content.strip():
            return chunks
        lines = content.splitlines()
        buf = []
        start_line = 1
        current_len = 0
        for i, line in enumerate(lines, start=1):
            line_len = len(line) + 1
            if current_len + line_len > max_chars and buf:
                text = "\n".join(buf).strip()
                if text:
                    chunks.append((text, start_line, i - 1))
                buf = [line]
                start_line = i
                current_len = line_len
            else:
                if not buf:
                    start_line = i
                buf.append(line)
                current_len += line_len
        if buf:
            text = "\n".join(buf).strip()
            if text:
                chunks.append((text, start_line, len(lines)))
        return chunks

    @staticmethod
    def _extract_headings(lines: List[str]) -> List[Tuple[int, str, int]]:
        headings: List[Tuple[int, str, int]] = []
        for i, line in enumerate(lines, start=1):
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not m:
                continue
            level = len(m.group(1))
            text = m.group(2).strip()
            headings.append((i, text, level))
        return headings

    @staticmethod
    def _heading_for_chunk(
        headings: List[Tuple[int, str, int]], start_line: int
    ) -> Tuple[str, int]:
        chosen = ("", 0)
        for line_no, text, level in headings:
            if line_no <= start_line:
                chosen = (text, level)
            else:
                break
        return chosen

    def _collect_markdown_docs(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for path in self.root_dir.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            lines = text.splitlines()
            headings = self._extract_headings(lines)
            for chunk_text, start, end in self._split_chunks(text):
                heading, level = self._heading_for_chunk(headings, start)
                docs.append(
                    {
                        "content": chunk_text,
                        "source": str(path),
                        "heading": heading,
                        "heading_level": level,
                        "chunk_hash": f"{path}:{start}:{end}",
                        "start_line": start,
                        "end_line": end,
                    }
                )
        return docs

    def build(self):
        self.docs = self._collect_markdown_docs()
        if not self.docs:
            self._idf = {}
            self._doc_vectors = []
            self._doc_norms = []
            return

        tokenized_docs = []
        for d in self.docs:
            content_tokens = self._tokenize(d["content"])
            heading_tokens = self._tokenize(d.get("heading", ""))
            tokenized_docs.append(content_tokens + heading_tokens + heading_tokens)

        df = Counter()
        for tokens in tokenized_docs:
            for t in set(tokens):
                df[t] += 1

        n_docs = len(self.docs)
        self._idf = {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}

        self._doc_vectors = []
        self._doc_norms = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            vec = {t: (tf[t] / max(1, len(tokens))) * self._idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self._doc_vectors.append(vec)
            self._doc_norms.append(norm)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.docs:
            return []

        q_tf = Counter(q_tokens)
        q_vec = {
            t: (q_tf[t] / max(1, len(q_tokens))) * self._idf.get(t, 0.0)
            for t in q_tf
            if t in self._idf
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        scores: List[Tuple[float, int]] = []
        for i, d_vec in enumerate(self._doc_vectors):
            d_norm = self._doc_norms[i]
            if d_norm == 0:
                continue
            dot = 0.0
            if len(q_vec) <= len(d_vec):
                for t, qv in q_vec.items():
                    dv = d_vec.get(t)
                    if dv is not None:
                        dot += qv * dv
            else:
                for t, dv in d_vec.items():
                    qv = q_vec.get(t)
                    if qv is not None:
                        dot += qv * dv
            score = dot / (q_norm * d_norm)
            if score > 0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            item = dict(self.docs[idx])
            item["score"] = float(score)
            results.append(item)
        return results


class MirosearchService:
    """
    Unified semantic memory service.

    Backend strategy:
    - `memsearch` when available and backend init succeeds
    - `local_tfidf` fallback for Windows/no-Milvus cases
    """

    def __init__(
        self,
        root_dir: Path,
        index_dir: Path,
        memory_cfg: Dict[str, Any],
        task_log: TaskLog,
        semantic_enabled: bool = True,
    ):
        self.root_dir = root_dir
        self.index_dir = index_dir
        self.memory_cfg = memory_cfg
        self.task_log = task_log
        self.semantic_enabled = semantic_enabled
        self.memsearch = None
        self.local_index: Optional[LocalSemanticMemoryIndex] = None
        self.backend = "disabled"
        self.ready = False
        self.status_message = ""

        self._initialize()

    @staticmethod
    def _is_remote_milvus_uri(uri: str) -> bool:
        return uri.startswith("http://") or uri.startswith("https://")

    def _initialize(self):
        if not self.semantic_enabled:
            self.backend = "disabled"
            self.ready = True
            self.status_message = "Semantic recall disabled by config."
            self.task_log.log_step("info", "Mirosearch | Init", self.status_message)
            return

        if MemSearch is None:
            self._use_local_fallback(
                "memsearch package unavailable, using local semantic fallback."
            )
            return

        provider = str(self.memory_cfg.get("embedding_provider", "openai"))
        model_name = self.memory_cfg.get("embedding_model", None)
        collection = str(self.memory_cfg.get("collection", "mirothinker_memory"))
        configured_uri = str(
            self.memory_cfg.get("milvus_uri", str(self.index_dir / "memsearch.db"))
        )
        milvus_uri = configured_uri
        if not self._is_remote_milvus_uri(configured_uri):
            milvus_uri = str(Path(configured_uri).resolve())

        milvus_token = self.memory_cfg.get("milvus_token", None) or os.getenv(
            "MILVUS_TOKEN", None
        )

        try:
            self.memsearch = MemSearch(
                paths=[str(self.root_dir)],
                embedding_provider=provider,
                embedding_model=model_name,
                milvus_uri=milvus_uri,
                milvus_token=milvus_token,
                collection=collection,
            )
            self.backend = "memsearch"
            self.ready = True
            self.status_message = (
                f"Mirosearch initialized with memsearch backend "
                f"(provider={provider}, collection={collection}, milvus_uri={milvus_uri})"
            )
            self.task_log.log_step("info", "Mirosearch | Init", self.status_message)
        except Exception as exc:
            self._use_local_fallback(
                f"memsearch init failed; fallback to local_tfidf. reason: {exc}"
            )

    def _use_local_fallback(self, reason: str):
        self.memsearch = None
        self.local_index = LocalSemanticMemoryIndex(self.root_dir)
        self.local_index.build()
        self.backend = "local_tfidf"
        self.ready = True
        self.status_message = reason
        self.task_log.log_step("warning", "Mirosearch | Init", reason)

    async def warmup(self):
        if not self.ready or not self.semantic_enabled:
            return
        if self.backend == "memsearch" and self.memsearch is not None:
            await self.memsearch.index()
        elif self.backend == "local_tfidf" and self.local_index is not None:
            self.local_index.build()

    async def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.ready or not self.semantic_enabled or not query:
            return []
        if self.backend == "memsearch" and self.memsearch is not None:
            return await self.memsearch.search(query, top_k=top_k)
        if self.backend == "local_tfidf" and self.local_index is not None:
            return self.local_index.search(query, top_k=top_k)
        return []

    async def reindex(self):
        await self.warmup()

    def health(self) -> Dict[str, Any]:
        return {
            "service": "Mirosearch",
            "ready": self.ready,
            "semantic_enabled": self.semantic_enabled,
            "backend": self.backend,
            "status": self.status_message,
        }

    def close(self):
        if self.memsearch is not None:
            try:
                self.memsearch.close()
            except Exception:
                pass
