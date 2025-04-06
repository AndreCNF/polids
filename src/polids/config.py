from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field

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

    secret_key: str | None = Field(default=None, description="Langfuse secret key")
    public_key: str | None = Field(default=None, description="Langfuse public key")
    host: str | None = Field(default=None, description="Langfuse host URL")
    log_to_langfuse: bool = Field(
        default=True, description="Flag to enable or disable logging to Langfuse"
    )


class Settings(BaseSettings):
    """Settings for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    openai_api_key: str | None = Field(default=None, description="OpenAI API Key")
    langfuse: LangfuseConfig = LangfuseConfig()


settings = Settings()
