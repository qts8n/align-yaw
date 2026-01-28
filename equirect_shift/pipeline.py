import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO

from .config import AlignConfig
from .features import build_detector, match_descriptors
from .geometry import ransac_rotation, yaw_from_rotation
from .image_ops import circular_shift_equirect, robust_resize_for_features, to_gray, yaw_px_from_rad
from .masking import add_bottom_mask, dilate_mask, yolo_person_mask_seg_batch

logger = logging.getLogger(__name__)

_YOLO_CACHE: dict[str, YOLO] = {}
_YOLO_LOCK = threading.Lock()
_DETECTOR_CACHE: dict[bool, tuple[str, cv2.Feature2D]] = {}
_DETECTOR_LOCK = threading.Lock()


def _run_pair(fn1, fn2):
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(fn1)
        f2 = executor.submit(fn2)
        return f1.result(), f2.result()


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
    img1, img2 = _run_pair(
        lambda: cv2.imread(config.pano_ref, cv2.IMREAD_COLOR),
        lambda: cv2.imread(config.pano_late, cv2.IMREAD_COLOR),
    )
    return align_panoramas_images(img1, img2, config)


def align_panoramas_images(
    img1: np.ndarray | None,
    img2: np.ndarray | None,
    config: AlignConfig,
    return_aligned: bool = True,
    return_mask_full: bool = True,
) -> AlignmentResult:
    if img1 is None or img2 is None:
        raise AlignmentError("Failed to read one of the input images.")

    H, W = img1.shape[:2]
    if img2.shape[:2] != (H, W):
        raise AlignmentError(f"Input sizes differ: ref={img1.shape[:2]} late={img2.shape[:2]} (must match)")

    (img1_small, s1), (img2_small, s2) = _run_pair(
        lambda: robust_resize_for_features(img1, max_w=config.maxw),
        lambda: robust_resize_for_features(img2, max_w=config.maxw),
    )

    hS, wS = img1_small.shape[:2]
    mask_small = np.zeros((hS, wS), dtype=np.uint8)
    add_bottom_mask(mask_small, config.bottom_mask_frac)

    if config.mask_people:
        model: YOLO = get_yolo_model(config.yolo_model)

        pm_list = yolo_person_mask_seg_batch(
            model=model,
            imgs_bgr_small=[img1_small, img2_small],
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            seam_shifts=config.yolo_seam_shifts,
            person_class_id=0,
        )
        pm1, pm2 = pm_list

        pm = cv2.bitwise_or(pm1, pm2)
        pm = dilate_mask(pm, config.mask_dilate_px)
        mask_small = cv2.bitwise_or(mask_small, pm)

    has_mask = bool(np.any(mask_small))
    # OpenCV feature mask expects nonzero as allowed, so invert.
    use_mask_small = None if not has_mask else cv2.bitwise_not(mask_small)

    g1, g2 = _run_pair(
        lambda: to_gray(img1_small),
        lambda: to_gray(img2_small),
    )

    det_name, detector1 = get_detector(prefer_sift=config.prefer_sift)
    det_name2, detector2 = build_detector(prefer_sift=config.prefer_sift)
    if det_name2 != det_name:
        logger.warning("Detector mismatch between cached and new detector: %s vs %s", det_name, det_name2)
        det_name = det_name2

    (k1, d1), (k2, d2) = _run_pair(
        lambda: detector1.detectAndCompute(g1, use_mask_small),
        lambda: detector2.detectAndCompute(g2, use_mask_small),
    )

    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        raise AlignmentError("Not enough keypoints/descriptors. Try increasing --maxw, lowering masks, or using SIFT.")

    matches = match_descriptors(det_name, d1, d2, ratio=config.ratio)
    if len(matches) < 80:
        raise AlignmentError(f"Not enough good matches after ratio test: {len(matches)}. Try increasing --maxw or loosening --ratio.")

    if return_mask_full:
        mask_full = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        mask_full = np.empty((0, 0), dtype=np.uint8)

    k1_pts = np.array([kp.pt for kp in k1], dtype=np.float32)
    k2_pts = np.array([kp.pt for kp in k2], dtype=np.float32)
    q_idx = np.fromiter((m.queryIdx for m in matches), dtype=np.int32, count=len(matches))
    t_idx = np.fromiter((m.trainIdx for m in matches), dtype=np.int32, count=len(matches))

    pts1 = k1_pts[q_idx]
    pts2 = k2_pts[t_idx]

    inv_s1 = 1.0 / s1
    inv_s2 = 1.0 / s2

    v1_full = pts1[:, 1] * inv_s1
    v2_full = pts2[:, 1] * inv_s2
    pole_ok = (
        (v1_full >= 0.08 * H)
        & (v1_full <= 0.92 * H)
        & (v2_full >= 0.08 * H)
        & (v2_full <= 0.92 * H)
    )

    if has_mask:
        x1 = np.rint(pts1[:, 0]).astype(np.int32)
        y1 = np.rint(pts1[:, 1]).astype(np.int32)
        x2 = np.rint(pts2[:, 0]).astype(np.int32)
        y2 = np.rint(pts2[:, 1]).astype(np.int32)
        np.clip(x1, 0, wS - 1, out=x1)
        np.clip(y1, 0, hS - 1, out=y1)
        np.clip(x2, 0, wS - 1, out=x2)
        np.clip(y2, 0, hS - 1, out=y2)
        mask_ok = (mask_small[y1, x1] == 0) & (mask_small[y2, x2] == 0)
    else:
        mask_ok = np.ones(len(matches), dtype=bool)

    valid = pole_ok & mask_ok
    valid_count = int(np.sum(valid))
    if valid_count < 60:
        raise AlignmentError(
            f"Too few correspondences after filtering: {valid_count}. "
            "Try increasing --maxw, lowering --mask-dilate-px, or using a bigger YOLO model."
        )

    u1 = pts1[valid, 0] * inv_s1
    v1 = pts1[valid, 1] * inv_s1
    u2 = pts2[valid, 0] * inv_s2
    v2 = pts2[valid, 1] * inv_s2

    def _equirect_to_unit_batch(u: np.ndarray, v: np.ndarray, w: int, h: int) -> np.ndarray:
        two_pi = 2.0 * math.pi
        lam = (u / w) * two_pi - math.pi
        phi = (math.pi / 2.0) - (v / h) * math.pi
        c = np.cos(phi)
        x = c * np.cos(lam)
        y = np.sin(phi)
        z = c * np.sin(lam)
        vecs = np.stack((x, y, z), axis=1)
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        np.divide(vecs, n, out=vecs, where=n != 0)
        return vecs

    A = _equirect_to_unit_batch(u2, v2, W, H)
    B = _equirect_to_unit_batch(u1, v1, W, H)

    res = ransac_rotation(A, B, iters=config.ransac_iters, thresh_deg=config.thresh_deg, seed=config.seed)
    yaw = -yaw_from_rotation(res.R)

    shift_px = yaw_px_from_rad(yaw, W)
    if return_aligned:
        aligned = img2 if (shift_px % W) == 0 else circular_shift_equirect(img2, shift_px)
    else:
        aligned = img2

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
