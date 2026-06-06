"""Centralized configuration via environment variables."""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MNEMO_", env_file=".env", extra="ignore")

    # Storage
    db_path: Path = Field(default=Path("/data/mnemo.db"))
    work_dir: Path = Field(default=Path("/data/work"))
    cookies_path: Path = Field(default=Path("/data/youtube_cookies.txt"))

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Worker
    worker_id: str = Field(default="worker-local")
    worker_poll_interval_seconds: float = Field(default=5.0)

    # Pipeline
    frame_sample_rate_fps: float = Field(default=1.0)
    audio_segment_seconds: float = Field(default=1.0)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
