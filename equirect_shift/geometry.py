import math
import random
from dataclasses import dataclass

import numpy as np


def equirect_to_unit(u: float, v: float, w: int, h: int) -> np.ndarray:
    lam = (u / w) * (2.0 * math.pi) - math.pi
    phi = (math.pi / 2.0) - (v / h) * math.pi
    c = math.cos(phi)
    x = c * math.cos(lam)
    y = math.sin(phi)
    z = c * math.sin(lam)
    vec = np.array([x, y, z], dtype=np.float64)
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def angular_errors(R: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    RA = (R @ A.T).T
    dots = np.sum(RA * B, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


@dataclass
class RansacResult:
    R: np.ndarray
    inliers: np.ndarray
    best_error_median_deg: float


def ransac_rotation(A: np.ndarray, B: np.ndarray, iters: int = 2000, thresh_deg: float = 1.5, seed: int = 0) -> RansacResult:
    assert A.shape == B.shape and A.shape[1] == 3
    n = A.shape[0]
    if n < 6:
        raise ValueError(f"Not enough correspondences for RANSAC: {n}")

    rng = random.Random(seed)
    thresh = math.radians(thresh_deg)

    best_inliers = None
    best_R = None
    best_med = 1e9

    idxs = list(range(n))

    for _ in range(iters):
        sample = rng.sample(idxs, 3)
        As = A[sample]
        Bs = B[sample]

        if np.linalg.norm(np.cross(As[1] - As[0], As[2] - As[0])) < 1e-6:
            continue

        R = kabsch_rotation(As, Bs)
        errs = angular_errors(R, A, B)
        inliers = errs < thresh
        k = int(np.sum(inliers))
        if k < 6:
            continue

        R_refit = kabsch_rotation(A[inliers], B[inliers])
        errs_refit = angular_errors(R_refit, A[inliers], B[inliers])
        med = float(np.median(errs_refit))

        if best_inliers is None:
            best_inliers = inliers
            best_R = R_refit
            best_med = med
            continue

        if k > int(np.sum(best_inliers)) or (k == int(np.sum(best_inliers)) and med < best_med):
            best_inliers = inliers
            best_R = R_refit
            best_med = med

    if best_R is None or best_inliers is None:
        raise RuntimeError("RANSAC failed to find a valid rotation. Try more features, looser threshold, or different detector.")

    return RansacResult(R=best_R, inliers=best_inliers, best_error_median_deg=math.degrees(best_med))


def yaw_from_rotation(R: np.ndarray) -> float:
    return math.atan2(R[0, 2], R[0, 0])
