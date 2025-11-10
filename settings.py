import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str = ""
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "agent-book"

    # LM Studio 接続設定
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_MODEL: str = ""

    # for Application
    openai_smart_model: str = ""
    openai_embedding_model: str = ""
    anthropic_smart_model: str = ""


    default_reflection_db_path: str = "tmp/reflection_db.json"
    language: str = "日本語"

    def __init__(self, **values):
        super().__init__(**values)
        self._set_env_variables()

    def _set_env_variables(self):
        for key in self.__annotations__.keys():
            if key.isupper():
                os.environ[key] = str(getattr(self, key))
