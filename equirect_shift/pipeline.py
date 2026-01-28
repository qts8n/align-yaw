import logging
import math
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO

from .config import AlignConfig
from .features import build_detector, match_descriptors
from .geometry import equirect_to_unit, ransac_rotation, yaw_from_rotation
from .image_ops import circular_shift_equirect, clamp_int, robust_resize_for_features, to_gray, yaw_px_from_rad
from .masking import add_bottom_mask, dilate_mask, yolo_person_mask_seg

logger = logging.getLogger(__name__)

_YOLO_CACHE: dict[str, YOLO] = {}
_YOLO_LOCK = threading.Lock()
_DETECTOR_CACHE: dict[bool, tuple[str, cv2.Feature2D]] = {}
_DETECTOR_LOCK = threading.Lock()


def get_yolo_model(model_path: str) -> YOLO:
    key = os.fspath(model_path)
    cached = _YOLO_CACHE.get(key)
    if cached is not None:
        return cached
    with _YOLO_LOCK:
        cached = _YOLO_CACHE.get(key)
        if cached is not None:
            return cached
        model = YOLO(key)
        _YOLO_CACHE[key] = model
        return model


def get_detector(prefer_sift: bool) -> tuple[str, cv2.Feature2D]:
    cached = _DETECTOR_CACHE.get(prefer_sift)
    if cached is not None:
        return cached
    with _DETECTOR_LOCK:
        cached = _DETECTOR_CACHE.get(prefer_sift)
        if cached is not None:
            return cached
        cached = build_detector(prefer_sift=prefer_sift)
        _DETECTOR_CACHE[prefer_sift] = cached
        return cached


def warmup_models(config: AlignConfig) -> None:
    """Preload heavy models/detectors into memory."""
    get_detector(config.prefer_sift)
    if config.mask_people:
        get_yolo_model(config.yolo_model)


class AlignmentError(RuntimeError):
    """Raised when the alignment pipeline cannot complete."""


@dataclass
class AlignmentResult:
    aligned: np.ndarray
    mask_full: np.ndarray
    mask_small: np.ndarray
    det_name: str
    matches: int
    correspondences: int
    inliers: int
    inlier_ratio: float
    median_error_deg: float
    yaw_rad: float
    shift_px: int
    width: int
    height: int


def align_panoramas(config: AlignConfig) -> AlignmentResult:
    img1 = cv2.imread(config.pano_ref, cv2.IMREAD_COLOR)
    img2 = cv2.imread(config.pano_late, cv2.IMREAD_COLOR)
    return align_panoramas_images(img1, img2, config)


def align_panoramas_images(
    img1: np.ndarray | None,
    img2: np.ndarray | None,
    config: AlignConfig,
) -> AlignmentResult:
    if img1 is None or img2 is None:
        raise AlignmentError("Failed to read one of the input images.")

    H, W = img1.shape[:2]
    if img2.shape[:2] != (H, W):
        raise AlignmentError(f"Input sizes differ: ref={img1.shape[:2]} late={img2.shape[:2]} (must match)")

    img1_small, s1 = robust_resize_for_features(img1, max_w=config.maxw)
    img2_small, s2 = robust_resize_for_features(img2, max_w=config.maxw)

    hS, wS = img1_small.shape[:2]
    mask_small = np.zeros((hS, wS), dtype=np.uint8)

    add_bottom_mask(mask_small, config.bottom_mask_frac)

    if config.mask_people:
        model: YOLO = get_yolo_model(config.yolo_model)

        pm1 = yolo_person_mask_seg(
            model=model,
            img_bgr_small=img1_small,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            seam_shifts=config.yolo_seam_shifts,
            person_class_id=0,
        )
        pm2 = yolo_person_mask_seg(
            model=model,
            img_bgr_small=img2_small,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            seam_shifts=config.yolo_seam_shifts,
            person_class_id=0,
        )

        pm = cv2.bitwise_or(pm1, pm2)
        pm = dilate_mask(pm, config.mask_dilate_px)
        mask_small = cv2.bitwise_or(mask_small, pm)

    # OpenCV feature mask expects nonzero as allowed, so invert.
    use_mask_small = cv2.bitwise_not(mask_small)

    g1 = to_gray(img1_small)
    g2 = to_gray(img2_small)

    det_name, detector = get_detector(prefer_sift=config.prefer_sift)

    k1, d1 = detector.detectAndCompute(g1, use_mask_small)
    k2, d2 = detector.detectAndCompute(g2, use_mask_small)

    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        raise AlignmentError("Not enough keypoints/descriptors. Try increasing --maxw, lowering masks, or using SIFT.")

    matches = match_descriptors(det_name, d1, d2, ratio=config.ratio)
    if len(matches) < 80:
        raise AlignmentError(f"Not enough good matches after ratio test: {len(matches)}. Try increasing --maxw or loosening --ratio.")

    mask_full = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    A_list = []
    B_list = []

    for m in matches:
        x1, y1 = k1[m.queryIdx].pt
        x2, y2 = k2[m.trainIdx].pt

        u1 = x1 / s1
        v1 = y1 / s1
        u2 = x2 / s2
        v2 = y2 / s2

        iu1 = clamp_int(int(round(u1)), 0, W - 1)
        iv1 = clamp_int(int(round(v1)), 0, H - 1)
        iu2 = clamp_int(int(round(u2)), 0, W - 1)
        iv2 = clamp_int(int(round(v2)), 0, H - 1)

        if mask_full[iv1, iu1] != 0:
            continue
        if mask_full[iv2, iu2] != 0:
            continue

        # Pole filter (still useful)
        if v1 < 0.08 * H or v1 > 0.92 * H:
            continue
        if v2 < 0.08 * H or v2 > 0.92 * H:
            continue

        A_list.append(equirect_to_unit(u2, v2, W, H))  # late
        B_list.append(equirect_to_unit(u1, v1, W, H))  # ref

    if len(A_list) < 60:
        raise AlignmentError(f"Too few correspondences after filtering: {len(A_list)}. Try increasing --maxw, lowering --mask-dilate-px, or using a bigger YOLO model.")

    A = np.vstack(A_list)
    B = np.vstack(B_list)

    res = ransac_rotation(A, B, iters=config.ransac_iters, thresh_deg=config.thresh_deg, seed=config.seed)
    yaw = -yaw_from_rotation(res.R)

    shift_px = yaw_px_from_rad(yaw, W)
    aligned = circular_shift_equirect(img2, shift_px)

    inlier_count = int(np.sum(res.inliers))
    total = int(len(res.inliers))
    inlier_ratio = inlier_count / max(1, total)

    logger.info("Detector: %s", det_name)
    logger.info("Matches after ratio test: %d", len(matches))
    logger.info("Used correspondences (after masking and pole filter): %d", total)
    logger.info("RANSAC inliers: %d/%d (%.3f)", inlier_count, total, inlier_ratio)
    logger.info("Median inlier angular error: %.3f deg", res.best_error_median_deg)
    logger.info("Estimated yaw: %.3f deg", math.degrees(yaw))
    logger.info("Pixel shift (wrap): %d px (of width %d)", shift_px, W)

    return AlignmentResult(
        aligned=aligned,
        mask_full=mask_full,
        mask_small=mask_small,
        det_name=det_name,
        matches=len(matches),
        correspondences=total,
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        median_error_deg=res.best_error_median_deg,
        yaw_rad=yaw,
        shift_px=shift_px,
        width=W,
        height=H,
    )
