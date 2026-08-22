"""Environment-backed application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "SIH 26034 Legal Metrology API"
    environment: str = "development"
    api_prefix: str = "/api"
    storage_dir: Path = Path("data")
    max_upload_size_bytes: int = 10 * 1024 * 1024
    min_image_width: int = 320
    min_image_height: int = 240

    model_config = SettingsConfigDict(env_prefix="SIH_", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
