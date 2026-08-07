"""
security/auth.py — API Key & Token Authentication Module
=========================================================
Provides dependency injection function `verify_api_key` for FastAPI routes.
Checks:
  1. `X-API-Key` HTTP request header
  2. `Authorization: Bearer <token>` HTTP header
  3. `api_key` query parameter
"""

import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Settings, get_settings

logger = logging.getLogger(__name__)
security_bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    api_key_query: Optional[str] = Query(None, alias="api_key"),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    FastAPI dependency that enforces API key authentication when configured.

    If `settings.REQUIRE_API_KEY` is False, allows the request to pass.
    Otherwise, checks X-API-Key header, Bearer token, or query param.
    """
    if not settings.REQUIRE_API_KEY:
        return "unauthenticated-dev-mode"

    extracted_key: Optional[str] = None

    if x_api_key:
        extracted_key = x_api_key.strip()
    elif auth_credentials and auth_credentials.credentials:
        extracted_key = auth_credentials.credentials.strip()
    elif api_key_query:
        extracted_key = api_key_query.strip()

    if not extracted_key:
        logger.warning("Access denied: Missing API key or Authorization header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401",
                "title": "Unauthorized",
                "status": 401,
                "detail": "API key required. Provide via 'X-API-Key' header or 'Authorization: Bearer <key>'.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if extracted_key not in settings.API_KEYS:
        logger.warning("Access denied: Invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401",
                "title": "Unauthorized",
                "status": 401,
                "detail": "Invalid API key supplied.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return extracted_key


def get_current_user_optional(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> Optional[dict]:
    """
    FastAPI dependency that extracts the current logged-in user from a Bearer token if provided.
    Returns None if no token or an invalid token is supplied (allows guest access).
    """
    if not auth_credentials or not auth_credentials.credentials:
        return None
        
    token = auth_credentials.credentials.strip()
    from security.jwt_tokens import decode_access_token
    from database.db import get_user_by_id
    
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None
        
    try:
        user_id = int(payload["sub"])
        return get_user_by_id(user_id)
    except Exception:
        return None


def get_current_user(
    current_user: Optional[dict] = Depends(get_current_user_optional),
) -> dict:
    """
    FastAPI dependency that enforces a valid user session.
    Raises HTTP 401 if unauthenticated.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

