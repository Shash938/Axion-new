"""
routers/webauthn.py — WebAuthn (Passkey / Face ID) Authentication Router
==========================================================================
Implements the WebAuthn protocol for biometric authentication:
  - Registration: ties a device's Face ID / fingerprint to a user account.
  - Authentication: allows passwordless login using biometrics.

Uses the 'webauthn' library (https://pypi.org/project/webauthn/).
"""

import base64
import logging
from typing import Optional

import webauthn
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from config import get_settings
from database.db import (
    get_user_by_username_or_email,
    get_webauthn_challenge,
    get_webauthn_credential_by_credential_id,
    get_webauthn_credentials_by_user,
    store_webauthn_challenge,
    store_webauthn_credential,
    update_webauthn_sign_count,
)
from security.auth import get_current_user
from security.jwt_tokens import create_access_token

logger = logging.getLogger(__name__)

webauthn_router = APIRouter(
    prefix="/api/v1/auth/webauthn",
    tags=["WebAuthn"],
)

# ==============================================================================
# Helpers
# ==============================================================================

def _get_webauthn_rp_and_origin(request: Request):
    """Resolve RP ID and origin from the request. WebAuthn rejects raw IPs, so 127.0.0.1 maps to localhost."""
    hostname = request.url.hostname or "localhost"
    # WebAuthn spec forbids IP addresses as RP ID — map loopback IP to localhost
    if hostname in ("127.0.0.1", "::1"):
        hostname = "localhost"
    rp_id = hostname
    origin = request.headers.get("origin")
    if not origin:
        scheme = request.url.scheme or "http"
        port = request.url.port
        if port and port not in (80, 443):
            origin = f"{scheme}://{hostname}:{port}"
        else:
            origin = f"{scheme}://{hostname}"
    else:
        # Also fix origin if it contains 127.0.0.1
        origin = origin.replace("127.0.0.1", "localhost").replace("::1", "localhost")
    return rp_id, origin


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _get_expiry(seconds: int = 300) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


# ==============================================================================
# Registration — Begin
# ==============================================================================

@webauthn_router.post(
    "/register/begin",
    summary="Start Face ID / biometric registration for the current user",
)
def register_begin(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Generates a WebAuthn registration challenge.
    Returns PublicKeyCredentialCreationOptions for the browser's
    navigator.credentials.create() call.
    """
    settings = get_settings()
    rp_id, origin = _get_webauthn_rp_and_origin(request)
    existing_creds = get_webauthn_credentials_by_user(current_user["id"])

    exclude_credentials = [
        webauthn.helpers.structs.PublicKeyCredentialDescriptor(
            id=base64.urlsafe_b64decode(cred["credential_id"] + "=="),
        )
        for cred in existing_creds
    ]

    options = webauthn.generate_registration_options(
        rp_id=rp_id,
        rp_name=settings.WEBAUTHN_RP_NAME,
        user_id=str(current_user["id"]).encode(),
        user_name=current_user["username"],
        user_display_name=current_user["username"],
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        exclude_credentials=exclude_credentials,
        timeout=60000,
    )

    # Store challenge for later verification (one-time use)
    challenge_b64 = _b64url_encode(options.challenge)
    store_webauthn_challenge(
        user_id=current_user["id"],
        challenge=challenge_b64,
        expires_at=_get_expiry(120),
    )

    return Response(content=webauthn.options_to_json(options), media_type="application/json")


# ==============================================================================
# Registration — Complete
# ==============================================================================

class RegistrationCompleteRequest(BaseModel):
    credential: dict


@webauthn_router.post(
    "/register/complete",
    summary="Complete Face ID registration and store the public key",
)
def register_complete(
    request: Request,
    payload: RegistrationCompleteRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Verifies the browser's attestation and stores the public key credential.
    """
    rp_id, origin = _get_webauthn_rp_and_origin(request)

    try:
        # Extract challenge from clientDataJSON
        import json
        client_data_raw = payload.credential.get("response", {}).get("clientDataJSON", "")
        client_data_json = base64.urlsafe_b64decode(client_data_raw + "==")
        client_data = json.loads(client_data_json)
        challenge_b64 = client_data.get("challenge", "")
        # Normalize base64url padding
        challenge_b64 = challenge_b64.replace("+", "-").replace("/", "_").rstrip("=")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed credential: {e}")

    stored = get_webauthn_challenge(challenge_b64)
    if not stored or stored.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=400, detail="Invalid or expired challenge.")

    try:
        # Build the RegistrationCredential object using the v3.x helper
        reg_credential = webauthn.helpers.parse_registration_credential_json(
            payload.credential
        )

        verified = webauthn.verify_registration_response(
            credential=reg_credential,
            expected_challenge=base64.urlsafe_b64decode(challenge_b64 + "=="),
            expected_rp_id=rp_id,
            expected_origin=origin,
            require_user_verification=False,
        )
    except Exception as e:
        logger.error("WebAuthn registration verification failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Registration failed: {e}")

    # Store the public key
    cred_id_b64 = _b64url_encode(verified.credential_id)
    pub_key_b64 = base64.b64encode(verified.credential_public_key).decode()
    store_webauthn_credential(
        user_id=current_user["id"],
        credential_id=cred_id_b64,
        public_key=pub_key_b64,
    )

    logger.info("WebAuthn credential registered for user: %s", current_user["username"])
    return {"status": "registered", "message": "Face ID / biometric credential saved successfully."}


# ==============================================================================
# Authentication — Begin (Passwordless Login)
# ==============================================================================

class LoginBeginRequest(BaseModel):
    username: str


@webauthn_router.post(
    "/login/begin",
    summary="Start Face ID login — returns challenge for the browser",
)
def login_begin(request: Request, payload: LoginBeginRequest):
    """
    Generates an authentication challenge for a given username.
    The browser will use this to call navigator.credentials.get()
    and trigger the biometric prompt.
    """
    rp_id, origin = _get_webauthn_rp_and_origin(request)
    user = get_user_by_username_or_email(payload.username)
    if not user:
        raise HTTPException(status_code=404, detail="No account found with that username.")

    creds = get_webauthn_credentials_by_user(user["id"])
    if not creds:
        raise HTTPException(
            status_code=404,
            detail="No biometric credentials registered. Please register Face ID first.",
        )

    allow_credentials = [
        webauthn.helpers.structs.PublicKeyCredentialDescriptor(
            id=base64.urlsafe_b64decode(cred["credential_id"] + "=="),
        )
        for cred in creds
    ]

    options = webauthn.generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=allow_credentials,
        user_verification=UserVerificationRequirement.PREFERRED,
        timeout=60000,
    )

    challenge_b64 = _b64url_encode(options.challenge)
    store_webauthn_challenge(
        user_id=user["id"],
        challenge=challenge_b64,
        expires_at=_get_expiry(120),
    )

    return Response(content=webauthn.options_to_json(options), media_type="application/json")


# ==============================================================================
# Authentication — Complete
# ==============================================================================

class LoginCompleteRequest(BaseModel):
    username: str
    credential: dict


@webauthn_router.post(
    "/login/complete",
    summary="Verify Face ID assertion and issue session token",
)
def login_complete(request: Request, payload: LoginCompleteRequest):
    """
    Verifies the biometric signature and issues a JWT on success.
    """
    rp_id, origin = _get_webauthn_rp_and_origin(request)
    user = get_user_by_username_or_email(payload.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    try:
        import json
        client_data_raw = payload.credential.get("response", {}).get("clientDataJSON", "")
        client_data_json = base64.urlsafe_b64decode(client_data_raw + "==")
        client_data = json.loads(client_data_json)
        challenge_b64 = client_data.get("challenge", "").replace("+", "-").replace("/", "_").rstrip("=")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed credential: {e}")

    stored = get_webauthn_challenge(challenge_b64)
    if not stored or stored.get("user_id") != user["id"]:
        raise HTTPException(status_code=400, detail="Invalid or expired challenge.")

    # Look up the matching stored credential
    raw_cred_id = payload.credential.get("id", "")
    cred_id_normalized = raw_cred_id.replace("+", "-").replace("/", "_").rstrip("=")
    db_cred = get_webauthn_credential_by_credential_id(cred_id_normalized)
    if not db_cred or db_cred["user_id"] != user["id"]:
        raise HTTPException(status_code=400, detail="Credential not found or does not belong to this user.")

    try:
        auth_credential = webauthn.helpers.parse_authentication_credential_json(
            payload.credential
        )

        verified = webauthn.verify_authentication_response(
            credential=auth_credential,
            expected_challenge=base64.urlsafe_b64decode(challenge_b64 + "=="),
            expected_rp_id=rp_id,
            expected_origin=origin,
            credential_public_key=base64.b64decode(db_cred["public_key"]),
            credential_current_sign_count=db_cred["sign_count"],
            require_user_verification=False,
        )
    except Exception as e:
        logger.error("WebAuthn login verification failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Authentication failed: {e}")

    # Update signature counter (replay attack prevention)
    update_webauthn_sign_count(cred_id_normalized, verified.new_sign_count)

    token = create_access_token(user_id=user["id"], username=user["username"])
    from models.auth import UserResponse, TokenResponse
    user_response = UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        created_at=user["created_at"],
    )
    logger.info("WebAuthn login successful for user: %s", user["username"])
    return {"access_token": token, "token_type": "bearer", "user": user_response.dict()}


# ==============================================================================
# List registered credentials for current user
# ==============================================================================

@webauthn_router.get(
    "/credentials",
    summary="List all registered biometric credentials for the current user",
)
def list_credentials(current_user: dict = Depends(get_current_user)):
    creds = get_webauthn_credentials_by_user(current_user["id"])
    return {"credentials": [{"id": c["id"], "created_at": c.get("created_at")} for c in creds]}
