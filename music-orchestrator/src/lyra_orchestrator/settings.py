from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model_intent: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL_INTENT")
    openai_model_conductor: str = Field(default="gpt-4.1", alias="OPENAI_MODEL_CONDUCTOR")
    openai_model_stem: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL_STEM")
    openai_model_qc_judge: str = Field(default="gpt-4.1", alias="OPENAI_MODEL_QC_JUDGE")

    suno_api_base_url: str = Field(default="http://localhost:3000", alias="SUNO_API_BASE_URL")
    suno_model: str = Field(default="chirp-v3-5", alias="SUNO_MODEL")
    suno_timeout_seconds: float = Field(default=120.0, alias="SUNO_TIMEOUT_SECONDS")
    suno_poll_timeout_seconds: float = Field(default=420.0, alias="SUNO_POLL_TIMEOUT_SECONDS")
    min_credits_threshold: int = Field(default=3, alias="MIN_CREDITS_THRESHOLD")

    artifacts_root: Path = Field(default=Path(r"d:\Git\lyra\artifacts"), alias="ARTIFACTS_ROOT")
    db_path: Path = Field(default=Path("./data/runs.db"), alias="RUNS_DB_PATH")

    max_run_retries: int = Field(default=2, alias="MAX_RUN_RETRIES")
    qc_min_pass_ratio: float = Field(default=1.0, alias="QC_MIN_PASS_RATIO")
    qc_duration_tolerance_ms: int = Field(default=250, alias="QC_DURATION_TOLERANCE_MS")
    qc_target_lufs: float = Field(default=-16.0, alias="QC_TARGET_LUFS")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.artifacts_root.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
