import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO

from .image_ops import clamp_int


def _seam_shifts(width: int, seam_shifts: int) -> list[int]:
    if seam_shifts <= 1:
        return [0]
    step = max(1, width // seam_shifts)
    return [i * step for i in range(seam_shifts)]


def _combine_instance_masks(result, h: int, w: int) -> np.ndarray | None:
    if result.masks is None:
        return None

    m = result.masks.data
    if isinstance(m, np.ndarray):
        m_np = np.asarray(m)
    else:
        m_np = m.cpu().numpy()

    if m_np.size == 0:
        return None

    combined = (np.max(m_np, axis=0) > 0.5).astype(np.uint8) * 255
    if combined.shape[0] != h or combined.shape[1] != w:
        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
    return combined


def add_bottom_mask(mask: np.ndarray, bottom_frac: float) -> None:
    if bottom_frac <= 0:
        return
    h, w = mask.shape[:2]
    y0 = int(round(h * (1.0 - bottom_frac)))
    y0 = clamp_int(y0, 0, h)
    mask[y0:h, :] = 255


def yolo_person_mask_seg_batch(
    model: YOLO,
    imgs_bgr_small: list[np.ndarray],
    conf: float,
    iou: float,
    seam_shifts: int,
    person_class_id: int = 0,
) -> list[np.ndarray]:
    """
    Returns uint8 masks (255 where person pixels are) for each image, same size as inputs.
    Runs batch inference on horizontally rolled copies to catch people near the seam.
    """
    if not imgs_bgr_small:
        return []

    masks: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    meta: list[tuple[int, int, int, int]] = []

    for idx, img_bgr_small in enumerate(imgs_bgr_small):
        h, w = img_bgr_small.shape[:2]
        masks.append(np.zeros((h, w), dtype=np.uint8))
        shifts = _seam_shifts(w, seam_shifts)

        for sx in shifts:
            shifted = np.roll(img_bgr_small, shift=sx, axis=1)
            rgb = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
            sources.append(rgb)
            meta.append((idx, sx, h, w))

    if not sources:
        return masks

    results = list(
        model.predict(
            source=sources,
            conf=conf,
            iou=iou,
            batch=2,
            classes=[person_class_id],
            verbose=False,
        )
    )

    for res, (img_idx, sx, h, w) in zip(results, meta):
        combined = _combine_instance_masks(res, h, w)
        if combined is None:
            continue
        unshifted = np.roll(combined, shift=-sx, axis=1)
        masks[img_idx] = cv2.bitwise_or(masks[img_idx], unshifted)

    return masks


def yolo_person_mask_seg(
    model: YOLO,
    img_bgr_small: np.ndarray,
    conf: float,
    iou: float,
    seam_shifts: int,
    person_class_id: int = 0,
) -> np.ndarray:
    """
    Returns uint8 mask (255 where person pixels are) in the same size as img_bgr_small.
    Runs batch inference over horizontally rolled copies to catch people near the seam.
    """
    masks = yolo_person_mask_seg_batch(
        model=model,
        imgs_bgr_small=[img_bgr_small],
        conf=conf,
        iou=iou,
        seam_shifts=seam_shifts,
        person_class_id=person_class_id,
    )
    if masks:
        return masks[0]
    h, w = img_bgr_small.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    k = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)
