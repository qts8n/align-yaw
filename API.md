# Equirect Shift API

FastAPI service for estimating yaw alignment between two equirectangular panoramas stored in S3.

## Overview
- Reads two S3 objects (`pano_ref`, `pano_late`) from the same bucket.
- Runs the alignment pipeline and returns yaw + shift metadata.
- Does not upload or persist output images.

## Authentication
Authentication is optional. If `SERVICE_TOKEN` is set, every request must include the token
in the header defined by `SERVICE_TOKEN_HEADER` (defaults to `x-service-token`).

Example header:
```
x-service-token: your-shared-token
```

## Endpoints

### GET /healthz
Simple readiness check.

Response:
```json
{"status":"ok"}
```

### POST /align/yaw
Compute yaw shift between two panoramas in S3.

Request body:
```json
{
  "bucket": "my-panos",
  "pano_ref": "week1/pano.jpg",
  "pano_late": "week2/pano.jpg"
}
```

Response body:
```json
{
  "yaw_rad": -0.042,
  "shift_px": -256,
  "width": 8192,
  "height": 4096,
  "median_error_deg": 0.83,
  "det_name": "SIFT"
}
```

Field notes:
- `yaw_rad`: estimated yaw in radians (late -> ref).
- `shift_px`: horizontal pixel roll to align the late panorama (`np.roll` shift on width).
- `width`, `height`: dimensions of the input panoramas.
- `median_error_deg`: median angular error (in degrees) of RANSAC inliers.
- `det_name`: feature detector used (SIFT or ORB).

## Errors
The service uses standard HTTP errors with a JSON `detail` string. Common cases:
- `401 Unauthorized`: missing or invalid service token.
- `404 Not Found`: S3 object missing.
- `422 Unprocessable Entity`: invalid request, decode failure, or alignment failure.
- `502 Bad Gateway`: S3 read failed.
- `500 Internal Server Error`: configuration errors (e.g., missing `ALIGN_CONFIG_YAML`).

## Configuration
Environment variables used by the service:
- `ALIGN_CONFIG_YAML`: path to YAML config file with pipeline settings (required).
- `SERVICE_TOKEN`: if set, enables token auth.
- `SERVICE_TOKEN_HEADER`: header name for the token (default `x-service-token`).
- `S3_REGION`, `S3_ENDPOINT_URL`, `S3_ADDRESSING_STYLE`: S3 client config.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`: explicit AWS creds.

Notes:
- `pano_ref` and `pano_late` in the YAML are ignored by the API; they are overridden by the request.
- All other YAML fields control the pipeline (e.g., `maxw`, `ratio`, `mask_people`, `yolo_model`).

## Method overview
Alignment steps performed by the service:
1. Decode both panoramas and ensure they share identical dimensions.
2. Downscale for feature detection (`maxw`) and build a mask:
   - Bottom mask to ignore tripod/ground.
   - Optional person mask from YOLO segmentation, dilated and seam-rolled.
3. Detect and describe features (SIFT when available, otherwise ORB).
4. Match descriptors using Lowe ratio filtering; discard matches near the poles
   and any masked pixels.
5. Project matched pixels to unit-sphere vectors and solve for rotation using RANSAC.
6. Extract yaw from the rotation, convert to a pixel shift, and return metrics.
