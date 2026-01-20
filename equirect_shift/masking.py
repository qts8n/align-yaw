import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO

from .image_ops import clamp_int


def add_bottom_mask(mask: np.ndarray, bottom_frac: float) -> None:
    if bottom_frac <= 0:
        return
    h, w = mask.shape[:2]
    y0 = int(round(h * (1.0 - bottom_frac)))
    y0 = clamp_int(y0, 0, h)
    mask[y0:h, :] = 255


def yolo_person_mask_seg(
        model: YOLO,
        img_bgr_small: np.ndarray,
        conf: float,
        iou: float,
        seam_shifts: int,
        person_class_id: int = 0) -> np.ndarray:
    """
    Returns uint8 mask (255 where person pixels are) in the same size as img_bgr_small.
    Runs on a few horizontally rolled copies to catch people near the seam.
    """
    h, w = img_bgr_small.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if seam_shifts <= 1:
        shifts = [0]
    else:
        step = max(1, w // seam_shifts)
        shifts = [i * step for i in range(seam_shifts)]

    for sx in shifts:
        shifted = np.roll(img_bgr_small, shift=sx, axis=1)

        # Ultralytics expects RGB
        rgb = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
        results = model.predict(
            source=rgb,
            conf=conf,
            iou=iou,
            classes=[person_class_id],
            verbose=False,
        )

        if not results:
            continue

        r = results[0]
        if r.masks is None:
            continue

        # r.masks.data is (N, mh, mw) as a torch tensor. Convert to numpy.
        m = r.masks.data

        if isinstance(m, np.ndarray):
            m_np = np.asarray(m)
        else:
            m_np = m.cpu().numpy()

        # Combine all person instances
        # Masks may be in model input resolution, but Ultralytics returns them already aligned to the original image size.
        # Still, be defensive and resize to (h,w) if needed.
        combined = (np.max(m_np, axis=0) > 0.5).astype(np.uint8) * 255
        if combined.shape[0] != h or combined.shape[1] != w:
            combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)

        # Unshift back
        unshifted = np.roll(combined, shift=-sx, axis=1)
        mask = cv2.bitwise_or(mask, unshifted)

    return mask


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    k = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)
