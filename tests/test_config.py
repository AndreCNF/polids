from polids.config import Settings  # type: ignore[import]


def test_settings_defaults_for_rate_limit_and_gemini_validation():
    cfg = Settings(_env_file=None)

    assert cfg.llm_validation_max_workers == 2
    assert cfg.llm_rate_limit_max_retries == 4
    assert cfg.llm_rate_limit_base_sleep_seconds == 2.0
    assert cfg.llm_rate_limit_max_sleep_seconds == 60.0
    assert cfg.gemini_validation_model_name == "gemini-3-flash-preview"
    assert cfg.gemini_validation_search_context_size == "high"
    assert cfg.gemini_validation_thinking_level == "high"


def test_settings_reads_gemini_validation_env_aliases(monkeypatch):
    monkeypatch.setenv("POLIDS_GEMINI_VALIDATION_MODEL", "gemini-custom")
    monkeypatch.setenv("POLIDS_GEMINI_VALIDATION_SEARCH_CONTEXT_SIZE", "low")
    monkeypatch.setenv("POLIDS_GEMINI_VALIDATION_THINKING_LEVEL", "low")

    cfg = Settings(_env_file=None)

    assert cfg.gemini_validation_model_name == "gemini-custom"
    assert cfg.gemini_validation_search_context_size == "low"
    assert cfg.gemini_validation_thinking_level == "low"
