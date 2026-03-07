# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""Output formatting utilities for agent responses."""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.prompt_utils import FORMAT_ERROR_MESSAGE

# Maximum length for tool results before truncation (100k chars ≈ 25k tokens)
TOOL_RESULT_MAX_LENGTH = 100_000


class OutputFormatter:
    """Formatter for processing and formatting agent outputs."""

    def _extract_boxed_content(self, text: str) -> str:
        r"""
        Extract the content of the last \boxed{...} occurrence in the given text.

        Supports:
          - Arbitrary levels of nested braces
          - Escaped braces (\{ and \})
          - Whitespace between \boxed and the opening brace
          - Empty content inside braces
          - Incomplete boxed expressions (extracts to end of string as fallback)

        Args:
            text: Input text that may contain \boxed{...} expressions

        Returns:
            The extracted boxed content, or empty string if no match is found.
        """
        if not text:
            return ""

        _BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)

        last_result = None  # Track the last boxed content (complete or incomplete)
        i = 0
        n = len(text)

        while True:
            # Find the next \boxed occurrence
            m = _BOXED_RE.search(text, i)
            if not m:
                break
            j = m.end()

            # Skip any whitespace after \boxed
            while j < n and text[j].isspace():
                j += 1

            # Require that the next character is '{'
            if j >= n or text[j] != "{":
                i = j
                continue

            # Parse the brace content manually to handle nesting and escapes
            depth = 0
            k = j
            escaped = False
            found_closing = False
            while k < n:
                ch = text[k]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    # When depth returns to zero, the boxed content ends
                    if depth == 0:
                        last_result = text[j + 1 : k]
                        i = k + 1
                        found_closing = True
                        break
                k += 1

            # If we didn't find a closing brace, this is an incomplete boxed
            # Store it as the last result (will be overwritten if we find more boxed later)
            if not found_closing and depth > 0:
                last_result = text[j + 1 : n]
                i = k  # Continue from where we stopped
            elif not found_closing:
                i = j + 1  # Move past this invalid boxed

        # Return the last boxed content found (complete or incomplete)
        black_list = ["?", "??", "???", "？", "……", "…", "...", "unknown", None]
        return last_result.strip() if last_result not in black_list else ""

    def format_tool_result_for_user(self, tool_call_execution_result: dict) -> dict:
        """
        Format tool execution results to be fed back to LLM as user messages.

        Only includes necessary information (results or errors). Long results
        are truncated to TOOL_RESULT_MAX_LENGTH to prevent context overflow.

        Args:
            tool_call_execution_result: Dict containing server_name, tool_name,
                and either 'result' or 'error'.

        Returns:
            Dict with 'type' and 'text' keys suitable for LLM message content.
        """
        server_name = tool_call_execution_result["server_name"]
        tool_name = tool_call_execution_result["tool_name"]

        if "error" in tool_call_execution_result:
            # Provide concise error information to LLM
            content = f"Tool call to {tool_name} on {server_name} failed. Error: {tool_call_execution_result['error']}"
        elif "result" in tool_call_execution_result:
            # Provide the original output result of the tool
            content = tool_call_execution_result["result"]
            # Truncate overly long results to prevent context overflow
            if len(content) > TOOL_RESULT_MAX_LENGTH:
                content = content[:TOOL_RESULT_MAX_LENGTH] + "\n... [Result truncated]"
        else:
            content = f"Tool call to {tool_name} on {server_name} completed, but produced no specific output or result."

        return {"type": "text", "text": content}

    def format_final_summary_and_log(
        self, final_answer_text: str, client=None
    ) -> Tuple[str, str, str]:
        """
        Format final summary information, including answers and token statistics.

        Args:
            final_answer_text: The final answer text from the agent
            client: Optional LLM client for token usage statistics

        Returns:
            Tuple of (summary_text, boxed_result, usage_log)
        """
        summary_lines = []
        summary_lines.append("\n" + "=" * 30 + " Final Answer " + "=" * 30)
        summary_lines.append(final_answer_text)

        # Extract boxed result - find the last match using safer regex patterns
        boxed_result = self._extract_boxed_content(final_answer_text)

        # Add extracted result section
        summary_lines.append("\n" + "-" * 20 + " Extracted Result " + "-" * 20)

        if boxed_result:
            summary_lines.append(boxed_result)
        elif final_answer_text:
            summary_lines.append("No \\boxed{} content found.")
            boxed_result = FORMAT_ERROR_MESSAGE

        # Token usage statistics and cost estimation - use client method
        if client and hasattr(client, "format_token_usage_summary"):
            token_summary_lines, log_string = client.format_token_usage_summary()
            summary_lines.extend(token_summary_lines)
        else:
            # If no client or client doesn't support it, use default format
            summary_lines.append("\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20)
            summary_lines.append("Token usage information not available.")
            summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
            log_string = "Token usage information not available."

        return "\n".join(summary_lines), boxed_result, log_string

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        candidates = [text]
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidates.append(m.group(0))
        for c in candidates:
            try:
                parsed = json.loads(c)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _collect_allowed_sources(self, tool_calls: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        if not tool_calls:
            return []

        seen: Set[str] = set()
        sources: List[Dict[str, str]] = []

        def add_source(url: str, title: str = "", snippet: str = ""):
            url = (url or "").strip()
            if not url or url in seen:
                return
            seen.add(url)
            sources.append(
                {
                    "url": url,
                    "title": (title or "").strip(),
                    "snippet": (snippet or "").strip(),
                }
            )

        for call in tool_calls:
            args = call.get("arguments") or {}
            if isinstance(args, dict) and isinstance(args.get("url"), str):
                add_source(args["url"])

            result = call.get("result") or {}
            if not isinstance(result, dict):
                continue
            raw = result.get("result")
            if not isinstance(raw, str):
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            for item in parsed.get("organic", []) or []:
                if not isinstance(item, dict):
                    continue
                add_source(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                )
        return sources

    def _build_default_miro_payload(
        self, final_answer_text: str, allowed_sources: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        answer = self._extract_boxed_content(final_answer_text) or final_answer_text.strip()
        answer = answer if answer else "No reliable answer."
        return {
            "answer": answer,
            "evidence": allowed_sources[:3],
            "confidence": {
                "score": 35 if allowed_sources else 20,
                "level": "low",
                "reason": "Fallback payload due to invalid JSON output or weak evidence.",
            },
        }

    def format_miro_summary_and_log(
        self,
        final_answer_text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        client=None,
    ) -> Tuple[str, str, str]:
        allowed_sources = self._collect_allowed_sources(tool_calls)
        payload = self._extract_json_block(final_answer_text) or self._build_default_miro_payload(
            final_answer_text, allowed_sources
        )

        answer = str(payload.get("answer", "")).strip() or "No reliable answer."
        evidence = payload.get("evidence", [])
        confidence = payload.get("confidence", {})

        # Keep only evidence URLs that were actually retrieved in tool calls.
        allowed_urls = {x["url"] for x in allowed_sources if x.get("url")}
        normalized_evidence: List[Dict[str, str]] = []
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url or (allowed_urls and url not in allowed_urls):
                    continue
                normalized_evidence.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "url": url,
                        "snippet": str(item.get("snippet", "")).strip(),
                    }
                )
        if not normalized_evidence:
            normalized_evidence = allowed_sources[:3]

        score = confidence.get("score", 20)
        try:
            score = max(0, min(100, int(float(score))))
        except Exception:
            score = 20
        level = str(confidence.get("level", "low")).strip().lower()
        if level not in {"high", "medium", "low"}:
            level = "high" if score >= 80 else ("medium" if score >= 50 else "low")
        reason = str(confidence.get("reason", "")).strip() or "Auto-estimated from evidence quality."

        summary_lines = []
        summary_lines.append("\n" + "=" * 30 + " Final Answer " + "=" * 30)
        summary_lines.append(answer)
        summary_lines.append("\n" + "-" * 20 + " Evidence " + "-" * 20)
        if normalized_evidence:
            for idx, item in enumerate(normalized_evidence, 1):
                title = item.get("title", "") or "(untitled)"
                summary_lines.append(f"[{idx}] {title}")
                summary_lines.append(f"URL: {item.get('url', '')}")
                snippet = item.get("snippet", "")
                if snippet:
                    summary_lines.append(f"Snippet: {snippet}")
        else:
            summary_lines.append("No evidence sources available.")

        summary_lines.append("\n" + "-" * 20 + " Confidence " + "-" * 20)
        summary_lines.append(f"{score}/100 ({level})")
        summary_lines.append(f"Reason: {reason}")

        if client and hasattr(client, "format_token_usage_summary"):
            token_summary_lines, log_string = client.format_token_usage_summary()
            summary_lines.extend(token_summary_lines)
        else:
            summary_lines.append("\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20)
            summary_lines.append("Token usage information not available.")
            summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
            log_string = "Token usage information not available."

        return "\n".join(summary_lines), answer, log_string
