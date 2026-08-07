"""
config/settings.py — Application Configuration & Security Settings
====================================================================
Manages environment variables, security credentials, CORS settings,
and rate limits using Pydantic Settings.
"""

from functools import lru_cache
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with secure defaults."""

    APP_NAME: str = "Axion AI Investment Advisor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Stock Exchange Settings
    NSE_SUFFIX: str = ".NS"
    BSE_SUFFIX: str = ".BO"

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
    ]

    # Security & Auth Settings
    REQUIRE_API_KEY: bool = False
    API_KEYS: List[str] = ["axion-dev-key-12345"]
    JWT_SECRET_KEY: str = "axion-super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"

    # Anti-Abuse & Limits
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_PAYLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit (allows base64 webcam & face photo images)
    ENABLE_SECURITY_HEADERS: bool = True

    # MFA Settings
    MFA_ENABLED: bool = False
    MFA_OTP_EXPIRY_SECONDS: int = 300  # 5 minutes

    # SMTP Email Settings (for MFA OTP)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "Axion AI <noreply@axion.ai>"

    # WebAuthn Settings
    WEBAUTHN_RP_ID: str = "localhost"
    WEBAUTHN_RP_NAME: str = "Axion AI Investment Advisor"
    WEBAUTHN_ORIGIN: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("ALLOWED_ORIGINS", "API_KEYS", mode="before")
    @classmethod
    def assemble_list_from_str(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if not v.strip():
                return []
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
