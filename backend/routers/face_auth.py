"""
routers/face_auth.py — Face Recognition Authentication Router
=============================================================
Endpoints for registering and authenticating via webcam face recognition
using OpenCV. Replaces the WebAuthn/passkey approach with actual
computer-vision-based face matching.

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
    for the authenticated user. The user can register multiple angles/lighting
    for better accuracy.
    """
    user_id = current_user["id"]

    # Check if face is present in the image
    if not detect_face_in_image(payload.image):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please ensure your face is clearly visible and well-lit.",
        )

    # Compute face encoding
    encoding = compute_face_encoding(payload.image)
    if encoding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract face features. Please try again with better lighting.",
        )

    # Store encoding
    encoding_json = json.dumps(encoding)
    record_id = store_face_encoding(user_id=user_id, encoding_json=encoding_json)

    # Count total registered faces for this user
    all_encodings = get_face_encodings_by_user(user_id)
    count = len(all_encodings)

    logger.info("Face registered for user %s (id=%d), total encodings: %d", current_user["username"], user_id, count)
    return {
        "status": "registered",
        "message": f"Face registered successfully! You now have {count} face sample(s) stored.",
        "encoding_id": record_id,
        "total_samples": count,
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
    Requires a username to narrow the search (or match against all users).
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

    # 3. Detect face in the login image
    if not detect_face_in_image(payload.image):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please position your face clearly in front of the camera.",
        )

    # 4. Compute encoding from the login image
    probe_encoding = compute_face_encoding(payload.image)
    if probe_encoding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract face features. Please try again.",
        )

    # 5. Compare against this user's stored encodings
    match_result = find_matching_user(probe_encoding, user_encodings)

    if match_result is None:
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
        "Face login successful for user=%s (similarity=%.4f)",
        user["username"],
        similarity,
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_response.model_dump() if hasattr(user_response, "model_dump") else user_response.dict(),
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
