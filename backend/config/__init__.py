"""
config.py — Application Configuration
======================================
Why this file exists:
    Centralises ALL runtime configuration in one place using Pydantic BaseSettings.
    Settings are loaded from environment variables or a .env file, making the app
    12-factor compliant and safe to deploy across dev / staging / production without
    changing code.

How it connects:
    - Imported by every service and the FastAPI app via `get_settings()`.
    - Never import directly from `os.environ`; always use `get_settings()`.

Possible improvements:
    - Add a `SECRET_KEY` for JWT auth when authentication is introduced.
    - Add Redis / PostgreSQL connection strings for future phases.
    - Add Sentry DSN for error tracking.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    All fields have sensible defaults so the app runs out-of-the-box in
    development. Override via environment variables in production.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = Field(default="AI Investment Advisor", description="Human-readable application name.")
    APP_VERSION: str = Field(default="1.0.0", description="Semantic version of the application.")
    DEBUG: bool = Field(default=False, description="Enable debug mode. NEVER True in production.")

    # --- Server ---
    HOST: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    PORT: int = Field(default=8000, description="Uvicorn bind port.")

    # --- CORS ---
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "chrome-extension://*"],
        description=(
            "List of allowed CORS origins. In production, restrict this to "
            "your Chrome extension ID and any web dashboard domain."
        ),
    )

    # --- Logging ---
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Python logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )

    # --- yfinance Data Fetching ---
    YFINANCE_PERIOD: str = Field(
        default="5y",
        description=(
            "Historical data period for yfinance. '5y' gives us 5 years of "
            "financials which is ideal for growth metric calculations."
        ),
    )
    YFINANCE_INTERVAL: str = Field(
        default="1d",
        description="Data interval for price history (1d = daily candles).",
    )
    YFINANCE_TIMEOUT: int = Field(
        default=30,
        description="Timeout in seconds for yfinance HTTP requests.",
    )

    # --- Analysis ---
    NSE_SUFFIX: str = Field(
        default=".NS",
        description="yfinance suffix for NSE-listed Indian stocks.",
    )
    BSE_SUFFIX: str = Field(
        default=".BO",
        description="yfinance suffix for BSE-listed Indian stocks.",
    )

    # --- Scoring ---
    SCORE_MIN: float = Field(default=0.0, description="Minimum possible metric score.")
    SCORE_MAX: float = Field(default=10.0, description="Maximum possible metric score.")

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, v) -> bool:
        """Accept common environment labels as production debug=False."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
                return False
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}. Got: {v!r}")
        return upper

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Ensure port is in a valid range."""
        if not (1024 <= v <= 65535):
            raise ValueError(f"PORT must be between 1024 and 65535. Got: {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings singleton.

    Using lru_cache ensures the .env file is read exactly once per process
    lifecycle, which is efficient and avoids repeated disk I/O.

    Usage:
        from config import get_settings
        settings = get_settings()
        print(settings.APP_NAME)

    Testing:
        To override settings in tests, call `get_settings.cache_clear()` before
        each test and set environment variables with `monkeypatch.setenv()`.
    """
    return Settings()
