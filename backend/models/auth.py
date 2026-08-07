"""
models/auth.py — Authentication & History Pydantic Models
==========================================================
Data validation schemas for user registration, login, profile, and search history.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, description="Unique username.")
    email: str = Field(..., description="Valid email address.")
    password: str = Field(..., min_length=6, max_length=100, description="Raw user password.")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        cleaned = v.strip()
        if not re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
            raise ValueError("Username can only contain alphanumeric characters, underscores, and hyphens.")
        return cleaned

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", cleaned):
            raise ValueError("Invalid email address format.")
        return cleaned


class UserLoginRequest(BaseModel):
    username_or_email: str = Field(..., description="Username or email address.")
    password: str = Field(..., description="User password.")


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class SearchHistoryItemResponse(BaseModel):
    id: int
    ticker: str
    exchange: str
    company_name: str
    score: float
    grade: str
    recommendation: str
    searched_at: str


class SearchHistoryListResponse(BaseModel):
    history: List[SearchHistoryItemResponse]
    total_count: int
