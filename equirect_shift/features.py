import logging

import cv2

logger = logging.getLogger(__name__)


def build_detector(prefer_sift: bool = True):
    if prefer_sift:
        sift = getattr(cv2, "SIFT_create", None)
        if sift is not None:
            return "SIFT", cv2.SIFT_create(nfeatures=8000)  # type: ignore
        logger.warning("SIFT was requested but not available")
    return "ORB", cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8, fastThreshold=10)  # type: ignore


def match_descriptors(det_name: str, des1, des2, ratio: float = 0.75):
    if det_name == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good
