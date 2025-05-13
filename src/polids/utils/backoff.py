"""
Provides a reusable exponential backoff decorator for LLM API calls (OpenAI, Perplexity, etc).
"""

import backoff

llm_backoff = backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=120,
)
