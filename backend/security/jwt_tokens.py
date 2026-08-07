"""
security/jwt_tokens.py — Authentication Token Module
===================================================
Generates and decodes signed authentication tokens for session management.
"""

import base64
import json
import time
from typing import Dict, Optional

from config import get_settings
from security.passwords import hmac

DEFAULT_EXPIRE_SECONDS = 86400 * 7  # 7 days


def create_access_token(user_id: int, username: str, expires_in: int = DEFAULT_EXPIRE_SECONDS) -> str:
    """
    Creates a cryptographically signed authentication token containing user claims.
    Format: base64(payload).signature_hex
    """
    settings = get_settings()
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in,
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    
    signature = hmac.new(
        settings.JWT_SECRET_KEY.encode("utf-8"),
        payload_b64.encode("utf-8"),
        "sha256",
    ).hexdigest()
    
    return f"{payload_b64}.{signature}"


def decode_access_token(token: str) -> Optional[Dict]:
    """
    Validates token signature and expiration, returning claims if valid.
    """
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts[0], parts[1]
        
        settings = get_settings()
        expected_sig = hmac.new(
            settings.JWT_SECRET_KEY.encode("utf-8"),
            payload_b64.encode("utf-8"),
            "sha256",
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_sig):
            return None
            
        # Restore padding for b64 decoding
        padded_b64 = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(padded_b64).decode("utf-8")
        payload = json.loads(payload_json)
        
        if payload.get("exp", 0) < time.time():
            return None
            
        return payload
    except Exception:
        return None
