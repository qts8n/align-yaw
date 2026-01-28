import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import AlignConfig
from .pipeline import AlignmentError, align_panoramas_images, warmup_models


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


@dataclass(frozen=True)
class ServiceConfig:
    service_token: Optional[str]
    service_token_header: str
    s3_region: Optional[str]
    s3_endpoint_url: Optional[str]
    s3_addressing_style: Optional[str]
    tmp_dir: Optional[str]
    align_config_yaml: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_session_token: Optional[str]


def load_service_config() -> ServiceConfig:
    return ServiceConfig(
        service_token=_env_str("SERVICE_TOKEN"),
        service_token_header=str(_env_str("SERVICE_TOKEN_HEADER", "x-service-token")),
        s3_region=_env_str("S3_REGION"),
        s3_endpoint_url=_env_str("S3_ENDPOINT_URL"),
        s3_addressing_style=_env_str("S3_ADDRESSING_STYLE"),
        tmp_dir=_env_str("TMP_DIR"),
        align_config_yaml=_env_str("ALIGN_CONFIG_YAML"),
        aws_access_key_id=_env_str("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env_str("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=_env_str("AWS_SESSION_TOKEN"),
    )


class AlignRequest(BaseModel):
    bucket: str = Field(..., min_length=1)
    pano_ref: str = Field(..., min_length=1)
    pano_late: str = Field(..., min_length=1)


class AlignResponse(BaseModel):
    yaw_rad: float
    shift_px: int
    width: int
    height: int
    median_error_deg: float
    det_name: str


def _s3_client(settings: ServiceConfig):
    kwargs = {
        "region_name": settings.s3_region,
        "endpoint_url": settings.s3_endpoint_url,
        "config": Config(s3={"addressing_style": "virtual"})
    }

    if settings.s3_addressing_style:
        kwargs["config"] = Config(
            s3={"addressing_style": settings.s3_addressing_style})
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        kwargs["aws_session_token"] = settings.aws_session_token
    return boto3.client("s3", **kwargs)


def _fetch_s3_object_bytes(client, bucket: str, key: str) -> bytes:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"S3 object not found: s3://{bucket}/{key}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download s3://{bucket}/{key}",
        ) from exc
    body = response.get("Body")
    if body is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"S3 object body missing: s3://{bucket}/{key}",
        )
    try:
        data = body.read()
    finally:
        body.close()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Empty S3 object: s3://{bucket}/{key}",
        )
    return data


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to decode image bytes",
        )
    return img


def _make_align_config(
    settings: ServiceConfig,
    pano_ref: str,
    pano_late: str,
    base_config: dict | None = None,
) -> AlignConfig:
    if base_config is None:
        if not settings.align_config_yaml:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ALIGN_CONFIG_YAML is not configured",
            )
        try:
            return AlignConfig.from_yaml(
                settings.align_config_yaml,
                overrides={"pano_ref": pano_ref, "pano_late": pano_late},
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    try:
        return AlignConfig.from_base(
            base_config,
            overrides={"pano_ref": pano_ref, "pano_late": pano_late},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def create_app() -> FastAPI:
    settings = load_service_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s3_client = _s3_client(settings)
        if settings.align_config_yaml:
            base_config = AlignConfig.load_base_from_yaml(settings.align_config_yaml)
            app.state.align_config_base = base_config
            warm_cfg = AlignConfig.from_base(
                base_config,
                overrides={"pano_ref": "__warmup__", "pano_late": "__warmup__"},
            )
            warmup_models(warm_cfg)
        yield

    app = FastAPI(title="Equirect Shift Service", lifespan=lifespan)

    @app.get("/healthz")
    def healthcheck():
        return {"status": "ok"}

    @app.post("/align/yaw", response_model=AlignResponse)
    def align_yaw(request: Request, payload: AlignRequest):
        if settings.service_token:
            token = request.headers.get(settings.service_token_header)
            if not token or token != settings.service_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid service token")

        client = getattr(request.app.state, "s3_client", None) or _s3_client(settings)
        ref_bytes = _fetch_s3_object_bytes(client, payload.bucket, payload.pano_ref)
        late_bytes = _fetch_s3_object_bytes(client, payload.bucket, payload.pano_late)
        img_ref = _decode_image_bytes(ref_bytes)
        img_late = _decode_image_bytes(late_bytes)

        base_config = getattr(request.app.state, "align_config_base", None)
        config = _make_align_config(settings, "__s3_ref__", "__s3_late__", base_config=base_config)
        try:
            result = align_panoramas_images(img_ref, img_late, config)
        except AlignmentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc)) from exc

        return AlignResponse(
            yaw_rad=result.yaw_rad,
            shift_px=result.shift_px,
            width=result.width,
            height=result.height,
            median_error_deg=result.median_error_deg,
            det_name=result.det_name)

    return app


app = create_app()
