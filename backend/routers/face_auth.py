"""
routers/face_auth.py — Face Recognition Authentication Router
=============================================================
Endpoints for registering and authenticating via webcam face recognition
using a multi-layered OpenCV pipeline (DNN detection, structural validation,
HOG+LBP encoding, cosine matching, anti-spoofing).

Endpoints:
    POST /api/v1/auth/face/register   — Register face (requires auth + webcam image)
    POST /api/v1/auth/face/login      — Login via face recognition (username + webcam image)
    GET  /api/v1/auth/face/status     — Check if current user has registered face
    DELETE /api/v1/auth/face/reset    — Remove all face data for current user
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from security.auth import get_current_user
from security.face_recognition_service import (
    compute_face_encoding,
    detect_face_detailed,
    detect_face_in_image,
    find_matching_user,
)
from security.jwt_tokens import create_access_token
from database.db import (
    get_user_by_username_or_email,
    get_user_by_id,
    store_face_encoding,
    get_face_encodings_by_user,
    get_all_face_encodings,
    delete_face_encodings_by_user,
)

logger = logging.getLogger(__name__)

face_auth_router = APIRouter(
    prefix="/api/v1/auth/face",
    tags=["Face Recognition Auth"],
)


# ==============================================================================
# Request Models
# ==============================================================================

class FaceRegisterRequest(BaseModel):
    image: str  # Base64-encoded webcam image


class FaceLoginRequest(BaseModel):
    username: str
    image: str  # Base64-encoded webcam image


# ==============================================================================
# Register Face
# ==============================================================================

@face_auth_router.post(
    "/register",
    summary="Register your face for face recognition login",
    status_code=status.HTTP_201_CREATED,
)
def register_face(
    payload: FaceRegisterRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Captures a face encoding from the provided webcam image and stores it
    for the authenticated user. Uses the multi-layered validation pipeline
    to ensure only genuine face images are accepted.
    """
    user_id = current_user["id"]

    # Run full face detection + structural validation with detailed feedback
    detection = detect_face_detailed(payload.image)

    if not detection["face_detected"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detection["validation"]["reason"],
        )

    if not detection["validation"]["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detection["validation"]["reason"],
        )

    # Compute multi-feature face encoding (HOG + LBP)
    encoding = compute_face_encoding(payload.image)
    if encoding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Could not extract face features. The image may have failed "
                "anti-spoof checks (flat surface detected) or structural validation. "
                "Please try again with your face clearly visible and well-lit."
            ),
        )

    # Store encoding
    encoding_json = json.dumps(encoding)
    record_id = store_face_encoding(user_id=user_id, encoding_json=encoding_json)

    # Count total registered faces for this user
    all_encodings = get_face_encodings_by_user(user_id)
    count = len(all_encodings)

    logger.info(
        "Face registered for user %s (id=%d), total encodings: %d, quality=%s",
        current_user["username"],
        user_id,
        count,
        detection.get("quality", {}),
    )
    return {
        "status": "registered",
        "message": f"Face registered successfully! You now have {count} face sample(s) stored.",
        "encoding_id": record_id,
        "total_samples": count,
        "quality_metrics": detection.get("quality", {}),
    }


# ==============================================================================
# Login with Face
# ==============================================================================

@face_auth_router.post(
    "/login",
    summary="Authenticate via face recognition",
)
def login_with_face(payload: FaceLoginRequest):
    """
    Authenticates a user by comparing their webcam face against stored encodings.
    Uses DNN detection, structural validation, edge-based features, and
    cosine similarity with a strict 0.75 threshold.
    """
    # 1. Verify the username exists
    user = get_user_by_username_or_email(payload.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with that username.",
        )

    # 2. Check the user has registered a face
    user_encodings = get_face_encodings_by_user(user["id"])
    if not user_encodings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face data registered for this account. Please register your face first via Settings.",
        )

    # 3. Run full face detection + validation with detailed feedback
    detection = detect_face_detailed(payload.image)

    if not detection["face_detected"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detection["validation"]["reason"],
        )

    if not detection["validation"]["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detection["validation"]["reason"],
        )

    # 4. Compute encoding from the login image
    probe_encoding = compute_face_encoding(payload.image)
    if probe_encoding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Could not extract face features. The image failed validation checks. "
                "Please ensure your face is clearly visible, well-lit, and not obstructed."
            ),
        )

    # 5. Compare against this user's stored encodings
    match_result = find_matching_user(probe_encoding, user_encodings)

    if match_result is None:
        # Check if all stored encodings were incompatible (old format)
        import json as _json
        import numpy as _np
        incompatible = 0
        for stored in user_encodings:
            try:
                db_enc = _json.loads(stored["encoding"])
                if len(db_enc) != len(probe_encoding):
                    incompatible += 1
            except Exception:
                incompatible += 1

        if incompatible == len(user_encodings):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Your stored face data uses an outdated format and is incompatible "
                    "with the improved recognition system. Please go to Settings and "
                    "re-register your face for better security."
                ),
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Face does not match. Authentication failed. Please try again or use password login.",
        )

    matched_user_id, similarity = match_result

    # 6. Issue JWT token
    token = create_access_token(user_id=user["id"], username=user["username"])

    from models.auth import UserResponse
    user_response = UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        created_at=user["created_at"],
    )

    logger.info(
        "Face login successful for user=%s (similarity=%.4f, quality=%s)",
        user["username"],
        similarity,
        detection.get("quality", {}),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_response.dict(),
        "face_similarity": round(similarity, 4),
    }


# ==============================================================================
# Face Status & Reset
# ==============================================================================

@face_auth_router.get(
    "/status",
    summary="Check if the current user has registered face data",
)
def face_status(current_user: dict = Depends(get_current_user)):
    encodings = get_face_encodings_by_user(current_user["id"])
    return {
        "has_face": len(encodings) > 0,
        "sample_count": len(encodings),
    }


@face_auth_router.delete(
    "/reset",
    summary="Remove all face recognition data for the current user",
)
def reset_face(current_user: dict = Depends(get_current_user)):
    count = delete_face_encodings_by_user(current_user["id"])
    logger.info("Deleted %d face encoding(s) for user=%s", count, current_user["username"])
    return {"status": "cleared", "deleted_count": count}
