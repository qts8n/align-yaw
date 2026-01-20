import argparse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Estimate yaw offset between two equirectangular panoramas and align the later one, masking people with YOLO segmentation.")
    ap.add_argument("pano_ref", nargs="?", help="Reference panorama (earlier)")
    ap.add_argument("pano_late", nargs="?", help="Later panorama to be shifted")
    ap.add_argument("--config-yaml", dest="config_yaml", default=None, help="YAML file with AlignConfig values (CLI flags override)")
    ap.add_argument("--out", default=None, help="Output filename for aligned later panorama")

    ap.add_argument("--maxw", type=int, default=None, help="Max width used for feature extraction (downscale)")
    ap.add_argument("--ransac-iters", type=int, default=None, help="RANSAC iterations")
    ap.add_argument("--thresh-deg", type=float, default=None, help="Inlier threshold in degrees")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--prefer-sift", action="store_true", default=None, help="Prefer SIFT if available")
    ap.add_argument("--ratio", type=float, default=None, help="Lowe ratio test threshold")

    ap.add_argument("--bottom-mask-frac", type=float, default=None, help="Mask bottom fraction of image (tripod etc.)")

    # YOLO controls
    ap.add_argument("--yolo-model", default=None, help="YOLO segmentation model path or name")
    ap.add_argument("--yolo-conf", type=float, default=None, help="YOLO confidence threshold")
    ap.add_argument("--yolo-iou", type=float, default=None, help="YOLO NMS IoU threshold")
    ap.add_argument("--yolo-seam-shifts", type=int, default=None, help="How many horizontal rolls to run YOLO on (seam robustness)")
    ap.add_argument("--mask-people", action="store_true", default=None, help="Enable YOLO person masking")
    ap.add_argument("--mask-dilate-px", type=int, default=None, help="Dilate mask by N pixels (in small image space)")

    ap.add_argument("--save-mask", default=None, help="Optional path to save the final full-res mask as image (debug)")
    return ap
