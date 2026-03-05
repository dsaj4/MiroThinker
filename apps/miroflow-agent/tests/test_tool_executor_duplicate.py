from src.core.tool_executor import ToolExecutor


def test_diversify_duplicate_query_google_search():
    original = {"q": "arxiv cs today", "gl": "us", "hl": "en"}
    diversified = ToolExecutor.diversify_duplicate_query(
        "google_search", original, attempt_index=2
    )
    assert diversified["q"] != original["q"]
    assert "[refined attempt 2]" in diversified["q"]


def test_diversify_duplicate_query_scrape_extract():
    original = {
        "url": "https://arxiv.org/list/cs/new",
        "info_to_extract": "first paper title",
    }
    diversified = ToolExecutor.diversify_duplicate_query(
        "scrape_and_extract_info", original, attempt_index=2
    )
    assert diversified["info_to_extract"] != original["info_to_extract"]
    assert "corroborating detail" in diversified["info_to_extract"]
