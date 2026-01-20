# equirect-shift

Aligns the yaw between two equirectangular panoramas. The later panorama is rolled horizontally to match the reference, with optional YOLO person masking to avoid transient people from driving the estimate. Ships with a ready-to-run sample config.

## How it works
- Load both panoramas (must be identical dimensions) and optionally downscale to `maxw` for feature work.
- Build a person + bottom mask (YOLO segmentation rolled across the seam, optional dilation) and ignore masked pixels.
- Detect + describe features (SIFT when available, otherwise ORB), apply Lowe ratio filtering, and keep non-polar matches.
- Project matches to the unit sphere, solve for rotation with RANSAC, extract yaw, then roll the later panorama by the corresponding pixel shift.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
The default weights file `yolo26s-seg.pt` lives in the repo root. Point `--yolo-model` to another YOLO segmentation model if you prefer.

## Quick start (CLI)
```bash
python align_yaw.py --config-yaml default_config.yaml
```
That uses the bundled sample panoramas in `assets/` and writes the aligned image to `--out`.

Minimal explicit invocation:
```bash
python align_yaw.py ref.jpg late.jpg \
  --out aligned.png \
  --mask-people --save-mask mask.png
```

Key flags / config fields (CLI flags override YAML):
- `pano_ref`, `pano_late`, `out`: input/output paths.
- `maxw` (downscale width), `prefer_sift`, `ratio` (Lowe threshold).
- `bottom_mask_frac` (tripod/ground mask), `mask_people`, `mask_dilate_px`, `yolo_model`, `yolo_conf`, `yolo_iou`, `yolo_seam_shifts`.
- `ransac_iters`, `thresh_deg`, `seed`.
- `save_mask`: optional path for the final full-res mask (debug/inspection).

## Panorama viewer (web)
Split-screen WebGL viewer for inspecting equirectangular panoramas.

- Serve `viewer/` (e.g., `python -m http.server 8000 -d viewer`) and open `http://localhost:8000/`.
- If `viewer/pano.jpg` exists it loads automatically; otherwise an on-page prompt asks you to pick a panorama.
- `Left pano` is required; enable Split view and `Right pano` to compare two scenes side by side.
- Drag to look around; use the mouse wheel or pinch to zoom.

## Library usage
```python
import cv2
from equirect_shift import AlignConfig, align_panoramas

cfg = AlignConfig(
    pano_ref="ref.jpg",
    pano_late="late.jpg",
    out="aligned.png",
    mask_people=True,
    save_mask="mask.png",
)
result = align_panoramas(cfg)
cv2.imwrite(cfg.out, result.aligned)
```
`AlignmentResult` also exposes metadata such as detector type, match counts, inlier ratio, yaw (radians), pixel shift, and the masks used.

## Tips
- Both panoramas must have the same resolution; mismatches abort early.
- If you see "Not enough keypoints/descriptors," try increasing `maxw`, enabling `--prefer-sift`, reducing `mask_dilate_px`, or loosening `--ratio`.
- For people-rich scenes, keep `--mask-people` on and consider a larger YOLO seg model for stronger masks.
