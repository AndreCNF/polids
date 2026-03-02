from polids.config import Settings  # type: ignore[import]


def test_settings_defaults_for_rate_limit_controls():
    cfg = Settings(_env_file=None)

    assert cfg.llm_analysis_max_workers == 2
    assert cfg.llm_rate_limit_max_retries == 4
    assert cfg.llm_rate_limit_base_sleep_seconds == 2.0
    assert cfg.llm_rate_limit_max_sleep_seconds == 60.0
