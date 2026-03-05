from src.utils.parsing_utils import parse_llm_response_for_tool_calls


def test_parse_mcp_tool_call_relaxed_server_name_equals():
    content = """
I should fetch the page first.

<use_mcp_tool>
<server_name=jina_scrape_llm_summary</server_name>
<tool_name>scrape_and_extract_info</tool_name>
<arguments>
{"url":"https://arxiv.org/list/cs/recent","info_to_extract":"first title"}
</arguments>
</use_mcp_tool>
"""
    calls = parse_llm_response_for_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["server_name"] == "jina_scrape_llm_summary"
    assert calls[0]["tool_name"] == "scrape_and_extract_info"
    assert calls[0]["arguments"]["url"] == "https://arxiv.org/list/cs/recent"


def test_parse_mcp_tool_call_arguments_with_code_fence():
    content = """
<use_mcp_tool>
<server_name>search_and_scrape_webpage</server_name>
<tool_name>google_search</tool_name>
<arguments>
```json
{"q":"arxiv cs today","gl":"us","hl":"en"}
```
</arguments>
</use_mcp_tool>
"""
    calls = parse_llm_response_for_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["tool_name"] == "google_search"
    assert calls[0]["arguments"]["q"] == "arxiv cs today"
