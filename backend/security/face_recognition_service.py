"""
security/face_recognition_service.py — Enterprise Face Recognition Service
============================================================================
Multi-layered face validation and recognition pipeline using OpenCV YuNet ONNX,
structural landmark validation, HOG/LBP feature extraction, and anti-spoofing checks.

Pipeline Layers:
  1. DNN-based face detection (YuNet ONNX) — NO fallback to center-crop
  2. Structural & 5-point facial landmark validation (eyes, nose, mouth)
  3. Edge-based feature extraction — Canny edges + HOG descriptors
  4. Multi-feature encoding — HOG (1764) + LBP (512) texture features = 2276 dims
  5. Cosine similarity matching (strict threshold 0.75) + anti-spoof quality gate
"""

import base64
import json
import logging
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

_FACE_SIZE = (128, 128)                # Normalised face patch size
_MATCH_THRESHOLD = 0.75                # Cosine similarity threshold (strict)
_DNN_CONFIDENCE_THRESHOLD = 0.65       # Minimum DNN detection confidence
_MIN_FACE_PIXELS = 80                  # Minimum face width/height in pixels
_ASPECT_RATIO_MIN = 0.6                # Face bounding box w/h ratio lower bound
_ASPECT_RATIO_MAX = 1.5                # Face bounding box w/h ratio upper bound
_MIN_EDGE_DENSITY = 0.03               # Anti-spoof: min fraction of edge pixels
_HOG_ORIENTATIONS = 9
_HOG_PIXELS_PER_CELL = (16, 16)
_HOG_CELLS_PER_BLOCK = (2, 2)

# ==============================================================================
# YuNet ONNX Model Management
# ==============================================================================

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
_YUNET_PATH = os.path.join(_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

_yunet_detector = None
_haar_cascade = None


def _ensure_yunet_model() -> str:
    """Download YuNet ONNX model if not already present."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_YUNET_PATH) or os.path.getsize(_YUNET_PATH) < 10000:
        logger.info("Downloading OpenCV YuNet ONNX face detector (~230KB)...")
        urllib.request.urlretrieve(_YUNET_URL, _YUNET_PATH)
        logger.info("YuNet model downloaded to: %s", _YUNET_PATH)
    return _YUNET_PATH


def _get_yunet_detector(input_size: Tuple[int, int] = (320, 320)):
    """Lazy-load OpenCV YuNet FaceDetectorYN instance."""
    global _yunet_detector
    try:
        model_path = _ensure_yunet_model()
        if _yunet_detector is None:
            _yunet_detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=input_size,
                score_threshold=_DNN_CONFIDENCE_THRESHOLD,
                nms_threshold=0.3,
                top_k=5000,
            )
        else:
            _yunet_detector.setInputSize(input_size)
        return _yunet_detector
    except Exception as e:
        logger.warning("YuNet initialization failed: %s. Falling back to Haar Cascade.", e)
        return None


def _get_haar_cascade():
    """Lazy-load Haar cascade as secondary detector."""
    global _haar_cascade
    if _haar_cascade is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _haar_cascade = cv2.CascadeClassifier(cascade_path)
    return _haar_cascade


# ==============================================================================
# Image Decoding
# ==============================================================================


def _decode_image(base64_string: str) -> np.ndarray:
    """Decode a base64-encoded image (data URI or raw) into an OpenCV BGR frame."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64 data.")
    return img


# ==============================================================================
# Layer 1 — Strict Face Detection (YuNet ONNX + Strict Haar, NO fallback)
# ==============================================================================


def _detect_face_yunet(img: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    """
    Detect face with YuNet ONNX.
    Returns: ((x, y, w, h), score, landmarks) or None.
    Landmarks array has 5 points: right_eye, left_eye, nose, right_mouth, left_mouth.
    """
    h, w = img.shape[:2]
    detector = _get_yunet_detector((w, h))
    if detector is None:
        return None

    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None

    # Pick highest scoring face
    best_face = max(faces, key=lambda f: float(f[-1]))
    score = float(best_face[-1])
    if score < _DNN_CONFIDENCE_THRESHOLD:
        return None

    x, y, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
    x = max(0, x)
    y = max(0, y)
    fw = min(w - x, fw)
    fh = min(h - y, fh)

    if fw < _MIN_FACE_PIXELS or fh < _MIN_FACE_PIXELS:
        return None

    landmarks = best_face[4:14].reshape((5, 2))
    return ((x, y, fw, fh), score, landmarks)


def _detect_face_haar(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Strict Haar cascade detection. NO fallback to center-crop."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = _get_haar_cascade()
    if cascade is None or cascade.empty():
        return None

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(_MIN_FACE_PIXELS, _MIN_FACE_PIXELS),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    largest = max(faces, key=lambda r: r[2] * r[3])
    return tuple(largest)


def _detect_face_strict(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Multi-method strict face detection.
    Tries YuNet first, then strict Haar. NEVER falls back to center-crop.
    """
    yunet_res = _detect_face_yunet(img)
    if yunet_res is not None:
        return yunet_res[0]

    haar_res = _detect_face_haar(img)
    if haar_res is not None:
        return haar_res

    return None


# ==============================================================================
# Layer 2 — Face Structural Validation
# ==============================================================================


def _validate_face_structure(
    img: np.ndarray, face_rect: Tuple[int, int, int, int]
) -> Dict[str, any]:
    """
    Validates face structure (aspect ratio, landmarks/eyes, skin tone distribution).
    """
    x, y, w, h = face_rect
    result = {
        "valid": True,
        "reason": "",
        "eye_count": 0,
        "aspect_ratio": 0.0,
    }

    if w < _MIN_FACE_PIXELS or h < _MIN_FACE_PIXELS:
        result["valid"] = False
        result["reason"] = f"Face region too small ({w}x{h}px). Minimum is {_MIN_FACE_PIXELS}px."
        return result

    aspect_ratio = w / h
    result["aspect_ratio"] = round(aspect_ratio, 3)

    if aspect_ratio < _ASPECT_RATIO_MIN or aspect_ratio > _ASPECT_RATIO_MAX:
        result["valid"] = False
        result["reason"] = (
            f"Aspect ratio {aspect_ratio:.2f} is outside human face range "
            f"({_ASPECT_RATIO_MIN}–{_ASPECT_RATIO_MAX})."
        )
        return result

    # Check for facial landmarks or eyes
    yunet_res = _detect_face_yunet(img)
    if yunet_res is not None:
        landmarks = yunet_res[2]
        # Landmarks: 0: right eye, 1: left eye, 2: nose, 3: right mouth, 4: left mouth
        r_eye, l_eye, nose, r_mouth, l_mouth = landmarks
        # Eyes must be above nose, nose above mouth
        if (r_eye[1] < nose[1] or l_eye[1] < nose[1]) and (nose[1] < r_mouth[1] or nose[1] < l_mouth[1]):
            result["eye_count"] = 2
        else:
            result["valid"] = False
            result["reason"] = "Facial landmark geometry invalid (eyes/nose/mouth inverted)."
            return result
    else:
        # Fallback eye cascade check
        face_roi_gray = cv2.cvtColor(img[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        face_roi_gray = cv2.equalizeHist(face_roi_gray)
        upper_half = face_roi_gray[0 : h // 2, :]

        eye_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(
                upper_half,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(w * 0.08), int(h * 0.08)),
            )
            result["eye_count"] = len(eyes)

        if result["eye_count"] == 0:
            result["valid"] = False
            result["reason"] = "No eyes detected in face region."
            return result

    # Skin tone heuristic check using HSV color space
    face_roi_hsv = cv2.cvtColor(img[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(face_roi_hsv, lower_skin, upper_skin)
    skin_ratio = np.count_nonzero(skin_mask) / (w * h)

    if skin_ratio < 0.15:
        result["valid"] = False
        result["reason"] = f"Face region has low skin-tone presence ({skin_ratio:.1%})."
        return result

    return result


# ==============================================================================
# Layer 3 — Edge-Based Feature Extraction (Canny & HOG)
# ==============================================================================


def _compute_canny_edges(gray_face: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection to the normalised face patch."""
    blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
    median_val = np.median(blurred)
    lower = int(max(0, 0.6 * median_val))
    upper = int(min(255, 1.4 * median_val))
    return cv2.Canny(blurred, lower, upper)


def _compute_hog_features(gray_face: np.ndarray) -> np.ndarray:
    """Compute Histogram of Oriented Gradients (HOG) descriptor."""
    try:
        from skimage.feature import hog
        hog_features = hog(
            gray_face,
            orientations=_HOG_ORIENTATIONS,
            pixels_per_cell=_HOG_PIXELS_PER_CELL,
            cells_per_block=_HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return hog_features.astype(np.float32)
    except Exception:
        # Fallback gradient features
        gx = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        angle = cv2.phase(gx, gy, angleInDegrees=True)

        cell_size = 16
        n_bins = 9
        h, w = gray_face.shape
        features = []

        for cy in range(0, h - cell_size + 1, cell_size):
            for cx in range(0, w - cell_size + 1, cell_size):
                cell_mag = magnitude[cy : cy + cell_size, cx : cx + cell_size]
                cell_ang = angle[cy : cy + cell_size, cx : cx + cell_size]
                hist = np.zeros(n_bins, dtype=np.float32)
                for bin_idx in range(n_bins):
                    bin_lower = bin_idx * (180.0 / n_bins)
                    bin_upper = (bin_idx + 1) * (180.0 / n_bins)
                    mask = (cell_ang % 180 >= bin_lower) & (cell_ang % 180 < bin_upper)
                    hist[bin_idx] = np.sum(cell_mag[mask])
                norm = np.linalg.norm(hist)
                if norm > 1e-6:
                    hist /= norm
                features.extend(hist)

        return np.array(features, dtype=np.float32)


# ==============================================================================
# Layer 4 — Multi-Feature Texture Encoding (LBP)
# ==============================================================================


def _compute_lbp_vectorized(gray_face: np.ndarray) -> np.ndarray:
    """Vectorized Local Binary Pattern (LBP) texture descriptor."""
    padded = np.pad(gray_face, 1, mode="edge")
    h, w = gray_face.shape
    center = padded[1 : h + 1, 1 : w + 1].astype(np.int16)

    neighbors = [
        padded[0:h, 0:w],
        padded[0:h, 1:w + 1],
        padded[0:h, 2:w + 2],
        padded[1:h + 1, 2:w + 2],
        padded[2:h + 2, 2:w + 2],
        padded[2:h + 2, 1:w + 1],
        padded[2:h + 2, 0:w],
        padded[1:h + 1, 0:w],
    ]

    lbp = np.zeros((h, w), dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor.astype(np.int16) >= center).astype(np.uint8) << (7 - bit))

    n_regions_x = 4
    n_regions_y = 4
    region_h = h // n_regions_y
    region_w = w // n_regions_x
    features = []

    for ry in range(n_regions_y):
        for rx in range(n_regions_x):
            region = lbp[
                ry * region_h : (ry + 1) * region_h,
                rx * region_w : (rx + 1) * region_w,
            ]
            hist, _ = np.histogram(region.ravel(), bins=32, range=(0, 256))
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist /= norm
            features.extend(hist)

    return np.array(features, dtype=np.float32)


# ==============================================================================
# Layer 5 — Anti-Spoof Quality Gate
# ==============================================================================


def _compute_anti_spoof_score(gray_face: np.ndarray) -> Dict[str, float]:
    """Computes quality metrics to filter flat objects and non-faces."""
    edges = _compute_canny_edges(gray_face)
    edge_density = np.count_nonzero(edges) / edges.size
    texture_variance = float(np.var(gray_face.astype(np.float32) / 255.0))
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    return {
        "edge_density": round(edge_density, 4),
        "texture_variance": round(texture_variance, 4),
        "laplacian_variance": round(laplacian_var, 4),
    }


def _extract_face_region(
    img: np.ndarray, face_rect: Tuple[int, int, int, int]
) -> np.ndarray:
    """Crop, resize to 128x128 and normalise with CLAHE."""
    x, y, w, h = face_rect
    pad_w = int(w * 0.10)
    pad_h = int(h * 0.10)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)

    face_crop = img[y1:y2, x1:x2]
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_face, _FACE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


# ==============================================================================
# Public API
# ==============================================================================


def detect_face_in_image(base64_image: str) -> bool:
    """Quick face validation check."""
    try:
        img = _decode_image(base64_image)
        face_rect = _detect_face_strict(img)
        if face_rect is None:
            return False
        validation = _validate_face_structure(img, face_rect)
        return validation["valid"]
    except Exception:
        return False


def detect_face_detailed(base64_image: str) -> Dict:
    """Detailed face detection and validation with rich error reasons."""
    result = {
        "face_detected": False,
        "validation": {"valid": False, "reason": "No face detected in image."},
        "quality": {},
    }

    try:
        img = _decode_image(base64_image)
        face_rect = _detect_face_strict(img)

        if face_rect is None:
            result["validation"]["reason"] = (
                "No face detected in the image. Please ensure your face is clearly "
                "visible, well-lit, and centered in the camera frame."
            )
            return result

        result["face_detected"] = True
        validation = _validate_face_structure(img, face_rect)
        result["validation"] = validation

        if not validation["valid"]:
            return result

        gray_face = _extract_face_region(img, face_rect)
        quality = _compute_anti_spoof_score(gray_face)
        result["quality"] = quality

        if quality["edge_density"] < _MIN_EDGE_DENSITY:
            result["validation"]["valid"] = False
            result["validation"]["reason"] = (
                f"Face region has abnormally low edge complexity ({quality['edge_density']:.3f}). "
                "This may be a flat surface, not a real human face."
            )

    except Exception as e:
        logger.warning("Detailed face detection failed: %s", e)
        result["validation"]["reason"] = f"Image processing error: {str(e)}"

    return result


def compute_face_encoding(base64_image: str) -> Optional[list]:
    """Computes 2276-dim L2-normalized HOG+LBP face vector."""
    try:
        img = _decode_image(base64_image)
        face_rect = _detect_face_strict(img)
        if face_rect is None:
            return None

        validation = _validate_face_structure(img, face_rect)
        if not validation["valid"]:
            return None

        gray_face = _extract_face_region(img, face_rect)
        quality = _compute_anti_spoof_score(gray_face)
        if quality["edge_density"] < _MIN_EDGE_DENSITY:
            return None

        hog_features = _compute_hog_features(gray_face)
        lbp_features = _compute_lbp_vectorized(gray_face)
        encoding = np.concatenate([hog_features, lbp_features])

        norm = np.linalg.norm(encoding)
        if norm > 1e-8:
            encoding = encoding / norm

        return encoding.tolist()
    except Exception as e:
        logger.error("Face encoding failed: %s", e)
        return None


def compare_encodings(encoding_a: list, encoding_b: list) -> float:
    """Cosine similarity between two face encodings."""
    vec_a = np.array(encoding_a, dtype=np.float32)
    vec_b = np.array(encoding_b, dtype=np.float32)

    if len(vec_a) != len(vec_b):
        return -1.0

    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))


def find_matching_user(
    probe_encoding: list,
    stored_encodings: list,
) -> Optional[Tuple[int, float]]:
    """Compares probe face against all stored encodings with threshold 0.75."""
    best_user_id = None
    best_score = -1.0
    incompatible_count = 0

    for stored in stored_encodings:
        try:
            db_encoding = json.loads(stored["encoding"])
        except (json.JSONDecodeError, KeyError):
            continue

        similarity = compare_encodings(probe_encoding, db_encoding)
        if similarity == -1.0:
            incompatible_count += 1
            continue

        if similarity > best_score:
            best_score = similarity
            best_user_id = stored["user_id"]

    if incompatible_count > 0:
        logger.warning("Skipped %d incompatible (old format) encodings.", incompatible_count)

    if best_score >= _MATCH_THRESHOLD and best_user_id is not None:
        logger.info("Face match found: user_id=%s similarity=%.4f", best_user_id, best_score)
        return (best_user_id, best_score)

    return None
