"""Application configuration.

All settings are environment driven so the same image can run locally,
in Docker Compose and in a managed environment without code changes.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Exception Engine"
    environment: str = "local"
    debug: bool = True

    # Storage -----------------------------------------------------------
    # SQLite keeps the developer experience zero-dependency; Docker Compose
    # overrides this with the PostgreSQL DSN.
    database_url: str = "sqlite+pysqlite:///./ai_exception_engine.db"
    redis_url: str = "redis://localhost:6379/0"

    # HTTP --------------------------------------------------------------
    api_prefix: str = "/v1"
    cors_origins: str = "http://localhost:3000"

    # Multi tenancy -----------------------------------------------------
    # Development convenience: requests without an API key are attributed to
    # the demo tenant. Set to False for anything resembling production.
    allow_anonymous_demo_tenant: bool = True

    # AI layer ----------------------------------------------------------
    # "offline" is a deterministic, dependency free provider used for tests
    # and demos. "openai_compatible" talks to any OpenAI style /chat/completions
    # endpoint (hosted gateways, vLLM, Ollama, LM Studio, ...).
    ai_provider: str = "offline"
    ai_base_url: str = "http://localhost:11434/v1"
    ai_api_key: str = ""
    ai_model: str = "local-model"
    ai_timeout_seconds: float = 30.0
    ai_redact_pii: bool = True

    # Demo --------------------------------------------------------------
    seed_demo_on_startup: bool = True
    demo_random_seed: int = 20260830

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
