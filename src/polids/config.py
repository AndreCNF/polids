import os
from pathlib import Path
from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_PATH = Path("data")
if not DATA_PATH.exists():
    DATA_PATH.mkdir()


class LangfuseConfig(BaseModel):
    """
    Configuration for Langfuse integration.

    When defining these as environment variables or in a `.env` file,
    use the following format:
    ```
    LANGFUSE__{key}={value}
    ```

    Note the double underscore `__` which is used to separate the keys.

    For example:
    ```.env
    LANGFUSE__SECRET_KEY=your_secret_key
    LANGFUSE__PUBLIC_KEY=your_public_key
    LANGFUSE__HOST=https://your_langfuse_host
    LANGFUSE__LOG_TO_LANGFUSE=true
    ```

    This will automatically populate the `LangfuseConfig` fields.
    """

    secret_key: str | None = Field(
        default=os.getenv("LANGFUSE_SECRET_KEY"),
        description="Langfuse secret key",
    )
    public_key: str | None = Field(
        default=os.getenv("LANGFUSE_PUBLIC_KEY"),
        description="Langfuse public key",
    )
    host: AnyUrl | None = Field(
        default=os.getenv("LANGFUSE_HOST"),
        description="Langfuse host URL",
    )  # ty:ignore[invalid-assignment]
    log_to_langfuse: bool = Field(
        default=True, description="Flag to enable or disable logging to Langfuse"
    )

    def model_post_init(self, __context: Any) -> None:
        # Update environment variables for Langfuse configuration
        if self.secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = self.secret_key
        if self.public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.public_key
        if self.host:
            os.environ["LANGFUSE_HOST"] = str(self.host)


class Settings(BaseSettings):
    """Settings for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",  # Allow extra fields in the environment
    )

    openai_api_key: str | None = Field(default=None, description="OpenAI API Key")
    perplexity_api_key: str | None = Field(
        default=None, description="Perplexity API Key"
    )
    mistral_api_key: str | None = Field(default=None, description="Mistral API key")
    google_api_key: str | None = Field(
        default=None, description="Google / Gemini API key"
    )
    llm_analysis_max_workers: int = Field(
        default=2,
        ge=1,
        validation_alias="POLIDS_ANALYSIS_MAX_WORKERS",
        description="Maximum concurrent workers for structured chunk analysis.",
    )
    llm_validation_max_workers: int = Field(
        default=2,
        ge=1,
        validation_alias="POLIDS_VALIDATION_MAX_WORKERS",
        description="Maximum concurrent workers for proposal scientific validation.",
    )
    llm_rate_limit_max_retries: int = Field(
        default=4,
        ge=0,
        validation_alias="POLIDS_RATE_LIMIT_MAX_RETRIES",
        description="Maximum retries for rate-limited tasks.",
    )
    llm_rate_limit_base_sleep_seconds: float = Field(
        default=2.0,
        gt=0,
        validation_alias="POLIDS_RATE_LIMIT_BASE_SLEEP_SECONDS",
        description="Base sleep duration in seconds for rate-limit backoff.",
    )
    llm_rate_limit_max_sleep_seconds: float = Field(
        default=60.0,
        gt=0,
        validation_alias="POLIDS_RATE_LIMIT_MAX_SLEEP_SECONDS",
        description="Maximum sleep duration in seconds for rate-limit backoff.",
    )
    gemini_validation_model_name: str = Field(
        default="gemini-3-flash-preview",
        validation_alias="POLIDS_GEMINI_VALIDATION_MODEL",
        description="Gemini model name used for scientific validation.",
    )
    gemini_validation_search_context_size: Literal["low", "medium", "high"] = Field(
        default="high",
        validation_alias="POLIDS_GEMINI_VALIDATION_SEARCH_CONTEXT_SIZE",
        description="Gemini web search context size used for scientific validation.",
    )
    gemini_validation_thinking_level: Literal["low", "high"] = Field(
        default="high",
        validation_alias="POLIDS_GEMINI_VALIDATION_THINKING_LEVEL",
        description="Gemini thinking level used for scientific validation.",
    )
    langfuse: LangfuseConfig = LangfuseConfig()

    @model_validator(mode="after")
    def validate_rate_limit_settings(self) -> "Settings":
        """
        Validate dependent rate-limit settings.

        Returns:
            Settings: The validated settings instance.

        Raises:
            ValueError: If max sleep is lower than base sleep.
        """
        if (
            self.llm_rate_limit_max_sleep_seconds
            < self.llm_rate_limit_base_sleep_seconds
        ):
            raise ValueError(
                "llm_rate_limit_max_sleep_seconds must be greater than or equal to llm_rate_limit_base_sleep_seconds."
            )
        return self


settings = Settings()
