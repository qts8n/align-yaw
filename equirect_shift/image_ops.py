import math
from typing import Tuple

import cv2
import numpy as np
from fastapi import HTTPException, status


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to decode image bytes",
        )
    return img


def robust_resize_for_features(img: np.ndarray, max_w: int = 2048) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def circular_shift_equirect(img: np.ndarray, shift_px: int) -> np.ndarray:
    return np.roll(img, shift=shift_px, axis=1)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def yaw_px_from_rad(yaw_rad: float, width: int) -> int:
    return int(round((yaw_rad * width) / (2.0 * math.pi)))
