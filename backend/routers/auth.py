"""
routers/auth.py — Authentication Router
========================================
Endpoints for user account registration, login, MFA, and user profile.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from config import get_settings
from database.db import (
    create_mfa_code,
    create_user,
    get_latest_mfa_code,
    get_user_by_id,
    get_user_by_username_or_email,
    mark_mfa_code_used,
)
from models.auth import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse
from security.auth import get_current_user
from security.email import generate_otp, send_otp_email
from security.jwt_tokens import create_access_token
from security.passwords import hash_password, verify_password

logger = logging.getLogger(__name__)

auth_router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"],
)


# ==============================================================================
# Registration
# ==============================================================================

@auth_router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register_user(request: UserRegisterRequest) -> TokenResponse:
    """Creates a new user account with hashed password storage."""
    existing_user = get_user_by_username_or_email(request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username is already taken.")
    existing_email = get_user_by_username_or_email(request.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email address is already registered.")

    pwd_hash, salt = hash_password(request.password)
    user_record = create_user(
        username=request.username,
        email=request.email,
        password_hash=pwd_hash,
        salt=salt,
    )

    token = create_access_token(user_id=user_record["id"], username=user_record["username"])
    user_response = UserResponse(**user_record)
    logger.info("User registered: username=%s id=%d", user_record["username"], user_record["id"])
    return TokenResponse(access_token=token, user=user_response)


# ==============================================================================
# Login (with optional MFA gate)
# ==============================================================================

class LoginResponse(BaseModel):
    """Login can return a full token OR signal that MFA is required."""
    access_token: Optional[str] = None
    token_type: str = "bearer"
    user: Optional[UserResponse] = None
    mfa_required: bool = False
    pending_user_id: Optional[int] = None


@auth_router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate user and obtain session token",
)
def login_user(request: UserLoginRequest) -> LoginResponse:
    """
    Authenticates credentials. If MFA is enabled, returns mfa_required=True
    and a pending_user_id so the client can request an OTP.
    """
    user_record = get_user_by_username_or_email(request.username_or_email)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    is_valid = verify_password(
        plain_password=request.password,
        stored_hash=user_record["password_hash"],
        salt=user_record["salt"],
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings = get_settings()
    if settings.MFA_ENABLED:
        # Gate login behind MFA — do NOT issue token yet
        logger.info("MFA required for user: %s", user_record["username"])
        return LoginResponse(mfa_required=True, pending_user_id=user_record["id"])

    # MFA not enabled — issue token immediately
    token = create_access_token(user_id=user_record["id"], username=user_record["username"])
    user_response = UserResponse(
        id=user_record["id"],
        username=user_record["username"],
        email=user_record["email"],
        created_at=user_record["created_at"],
    )
    logger.info("User logged in: username=%s", user_record["username"])
    return LoginResponse(access_token=token, user=user_response, mfa_required=False)


# ==============================================================================
# Multi-Factor Authentication (Email OTP)
# ==============================================================================

class MFASendRequest(BaseModel):
    pending_user_id: int


class MFAVerifyRequest(BaseModel):
    pending_user_id: int
    otp_code: str


@auth_router.post(
    "/mfa/send",
    summary="Send a one-time password to the user's registered email",
)
def send_mfa_code(request: MFASendRequest):
    """Generates a 6-digit OTP, hashes it, stores it, and emails the user."""
    user = get_user_by_id(request.pending_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    settings = get_settings()
    otp = generate_otp(6)
    # Hash OTP before storing (never store plaintext secrets)
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=settings.MFA_OTP_EXPIRY_SECONDS)
    ).isoformat()

    create_mfa_code(user_id=user["id"], code_hash=otp_hash, expires_at=expires_at)
    sent = send_otp_email(to_address=user["email"], username=user["username"], otp_code=otp)

    if not sent:
        raise HTTPException(status_code=500, detail="Failed to send OTP email. Check SMTP settings.")

    # Mask email for privacy (e.g. u***r@example.com)
    email = user["email"]
    parts = email.split("@")
    masked = parts[0][0] + "***" + parts[0][-1] + "@" + parts[1] if len(parts[0]) > 2 else "***@" + parts[1]
    return {"status": "sent", "email_hint": masked}


@auth_router.post(
    "/mfa/verify",
    response_model=TokenResponse,
    summary="Verify OTP and issue session token",
)
def verify_mfa_code(request: MFAVerifyRequest) -> TokenResponse:
    """Verifies the submitted OTP and issues a full session token on success."""
    user = get_user_by_id(request.pending_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    code_record = get_latest_mfa_code(user["id"])
    if not code_record:
        raise HTTPException(status_code=400, detail="No active code found. Please request a new code.")

    submitted_hash = hashlib.sha256(request.otp_code.strip().encode()).hexdigest()
    if not secrets.compare_digest(submitted_hash, code_record["code_hash"]):
        raise HTTPException(status_code=400, detail="Invalid or expired code.")

    mark_mfa_code_used(code_record["id"])

    token = create_access_token(user_id=user["id"], username=user["username"])
    user_response = UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        created_at=user["created_at"],
    )
    logger.info("MFA verified for user: %s", user["username"])
    return TokenResponse(access_token=token, user=user_response)


# ==============================================================================
# User Profile
# ==============================================================================

@auth_router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_user_profile(current_user: dict = Depends(get_current_user)) -> UserResponse:
    """Returns profile information for the authenticated user."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"],
    )
