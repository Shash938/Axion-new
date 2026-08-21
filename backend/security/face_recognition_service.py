"""
security/face_recognition_service.py — OpenCV Face Recognition Service
========================================================================
Uses OpenCV's Haar Cascade for face detection and LBPH (Local Binary
Patterns Histograms) for face recognition.

Flow:
  Registration:
    1. Receive base64 image from webcam
    2. Detect face using Haar Cascade
    3. Extract & normalize the face region
    4. Compute a histogram-based encoding
    5. Store encoding in the database

  Login:
    1. Receive base64 image from webcam
    2. Detect face, extract & normalize
    3. Compare against all stored encodings using correlation
    4. Return the matching user if similarity exceeds threshold
"""

import base64
import json
import logging
import os
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
    _CASCADE_PATH = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""), "haarcascade_frontalface_default.xml")
    HAS_OPENCV = True
except Exception as _cv_err:
    HAS_OPENCV = False
    cv2 = None
    np = None
    _CASCADE_PATH = ""

logger = logging.getLogger(__name__)
_face_cascade = None
_FACE_SIZE = (128, 128)
_MATCH_THRESHOLD = 0.55


def get_face_cascade():
    global _face_cascade
    if not HAS_OPENCV or cv2 is None:
        return None
    if _face_cascade is None:
        cascade_filename = "haarcascade_frontalface_default.xml"
        possible_paths = []
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            possible_paths.append(os.path.join(cv2.data.haarcascades, cascade_filename))
        possible_paths.extend([
            "/usr/share/opencv4/haarcascades/" + cascade_filename,
            "/usr/share/opencv/haarcascades/" + cascade_filename,
            "/usr/local/share/opencv4/haarcascades/" + cascade_filename,
        ])
        for p in possible_paths:
            if p and os.path.exists(p):
                try:
                    cascade = cv2.CascadeClassifier(p)
                    if not cascade.empty():
                        _face_cascade = cascade
                        logger.info("Loaded Haar cascade from: %s", p)
                        break
                except Exception:
                    pass
        if _face_cascade is None or _face_cascade.empty():
            try:
                _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_filename)
            except Exception as e:
                logger.warning("Could not load Haar cascade: %s", e)
    return _face_cascade


def _decode_image(base64_string: str):
    """Decode a base64-encoded image (data URI or raw) into an OpenCV BGR frame."""
    if not HAS_OPENCV or cv2 is None or np is None:
        raise ValueError("OpenCV image processing is not available on this server environment.")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64 data.")
    return img


def _detect_face(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest face in the image. Returns (x, y, w, h)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    cascade = get_face_cascade()
    if cascade is not None and not cascade.empty():
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            largest = max(faces, key=lambda r: r[2] * r[3])
            return tuple(largest)
    
    # Fallback: center-crop 60% of the image frame as the face region
    h, w = img.shape[:2]
    crop_w = int(w * 0.6)
    crop_h = int(h * 0.6)
    x = int((w - crop_w) / 2)
    y = int((h - crop_h) / 2)
    return (x, y, crop_w, crop_h)


def _extract_face_region(img: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop, resize and normalise the face region to a standard greyscale patch."""
    x, y, w, h = face_rect
    
    # Add 15% padding around the face for context
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)
    
    face_crop = img[y1:y2, x1:x2]
    
    # Convert to greyscale and resize
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_face, _FACE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalise pixel values for lighting invariance
    normalised = cv2.equalizeHist(resized)
    
    return normalised


def compute_face_encoding(base64_image: str) -> Optional[list]:
    """
    Detects a face in the given base64 image and returns its encoding
    as a flat list of floats (the normalised pixel values of the face patch).
    Returns None if no face is detected.
    """
    img = _decode_image(base64_image)
    face_rect = _detect_face(img)
    
    if face_rect is None:
        return None
    
    face_region = _extract_face_region(img, face_rect)
    
    # The encoding is a flattened, normalised face vector
    encoding = face_region.flatten().astype(np.float32) / 255.0
    return encoding.tolist()


def compare_encodings(encoding_a: list, encoding_b: list) -> float:
    """
    Compares two face encodings using normalised correlation coefficient.
    Returns a similarity score between -1 and 1 (higher = more similar).
    """
    vec_a = np.array(encoding_a, dtype=np.float32)
    vec_b = np.array(encoding_b, dtype=np.float32)
    
    # Normalise to zero-mean, unit-variance
    a_mean = np.mean(vec_a)
    b_mean = np.mean(vec_b)
    a_std = np.std(vec_a) or 1e-8
    b_std = np.std(vec_b) or 1e-8
    
    vec_a_norm = (vec_a - a_mean) / a_std
    vec_b_norm = (vec_b - b_mean) / b_std
    
    # Pearson correlation coefficient
    correlation = np.dot(vec_a_norm, vec_b_norm) / len(vec_a)
    return float(correlation)


def find_matching_user(
    probe_encoding: list,
    stored_encodings: list,
) -> Optional[Tuple[int, float]]:
    """
    Compare a probe face encoding against all stored encodings.
    
    Args:
        probe_encoding: The face encoding from the login webcam capture.
        stored_encodings: List of dicts with keys: user_id, encoding (JSON string).
    
    Returns:
        (user_id, best_similarity) if a match is found above the threshold, else None.
    """
    best_user_id = None
    best_score = -1.0
    
    for stored in stored_encodings:
        try:
            db_encoding = json.loads(stored["encoding"])
        except (json.JSONDecodeError, KeyError):
            continue
        
        similarity = compare_encodings(probe_encoding, db_encoding)
        
        if similarity > best_score:
            best_score = similarity
            best_user_id = stored["user_id"]
    
    if best_score >= _MATCH_THRESHOLD and best_user_id is not None:
        logger.info("Face match found: user_id=%s similarity=%.4f", best_user_id, best_score)
        return (best_user_id, best_score)
    
    logger.info("No face match above threshold (best=%.4f, threshold=%.2f)", best_score, _MATCH_THRESHOLD)
    return None


def detect_face_in_image(base64_image: str) -> bool:
    """Quick check: is there a face in the image? (Used for UI feedback.)"""
    try:
        img = _decode_image(base64_image)
        return _detect_face(img) is not None
    except Exception:
        return False
