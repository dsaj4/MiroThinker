from miroflow_tools.dev_mcp_servers import jina_scrape_llm_summary as mod


def test_build_chat_completions_url_normalization():
    assert (
        mod._build_chat_completions_url("https://api.openai.com/v1")
        == "https://api.openai.com/v1/chat/completions"
    )
    assert (
        mod._build_chat_completions_url("https://x.example/v1/chat/completions")
        == "https://x.example/v1/chat/completions"
    )
    assert mod._build_chat_completions_url("your_summary_llm_base_url") == ""
    assert mod._build_chat_completions_url("") == ""


def test_resolve_summary_llm_config_fallback(monkeypatch):
    monkeypatch.setattr(mod, "SUMMARY_LLM_BASE_URL", "your_summary_llm_base_url")
    monkeypatch.setattr(mod, "SUMMARY_LLM_API_KEY", "")
    monkeypatch.setattr(mod, "SUMMARY_LLM_MODEL_NAME", "your_summary_llm_model_name")
    monkeypatch.setattr(mod, "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(mod, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(mod, "DASHSCOPE_BASE_URL", "")
    monkeypatch.setattr(mod, "DASHSCOPE_API_KEY", "")

    model, url, api_key = mod._resolve_summary_llm_config("your_summary_llm_model_name")
    assert model == "qwen3-max"
    assert url.endswith("/chat/completions")
    assert api_key == "sk-test"
