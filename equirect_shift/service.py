from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import AlignConfig, _env_str
from .image_ops import decode_image_bytes
from .pipeline import AlignmentError, align_panoramas_images, warmup_models
from .storage import fetch_s3_object_bytes, s3_client


@dataclass
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


class AlignUrlRequest(BaseModel):
    pano_ref_url: str = Field(..., min_length=1)
    pano_late_url: str = Field(..., min_length=1)


class AlignResponse(BaseModel):
    yaw_rad: float
    shift_px: int
    width: int
    height: int
    median_error_deg: float
    det_name: str


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


def _require_service_token(settings: ServiceConfig, request: Request) -> None:
    if settings.service_token:
        token = request.headers.get(settings.service_token_header)
        if not token or token != settings.service_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service token",
            )


def _get_align_base_config(request: Request) -> dict | None:
    return getattr(request.app.state, "align_config_base", None)


def _get_s3_client(request: Request, settings: ServiceConfig):
    client = getattr(request.app.state, "s3_client", None)
    if client is None:
        client = s3_client(settings)
        request.app.state.s3_client = client
    return client


def _fetch_url_bytes(url: str, timeout_s: float = 20.0) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid URL: {url}",
        )
    req = UrlRequest(url, headers={"User-Agent": "equirect-shift/1.0"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            status_code = getattr(resp, "status", None) or resp.getcode()
            if status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"URL not found: {url}",
                )
            if status_code and status_code >= 400:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to download URL (status {status_code}): {url}",
                )
            data = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"URL not found: {url}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download URL (status {exc.code}): {url}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download URL: {url}",
        ) from exc

    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Empty response body: {url}",
        )
    return data


def _align_from_images(
    settings: ServiceConfig,
    base_config: dict | None,
    img_ref,
    img_late,
    ref_label: str,
    late_label: str,
) -> AlignResponse:
    config = _make_align_config(settings, ref_label, late_label, base_config=base_config)
    try:
        result = align_panoramas_images(
            img_ref,
            img_late,
            config,
            return_aligned=False,
            return_mask_full=False,
        )
    except AlignmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return AlignResponse(
        yaw_rad=result.yaw_rad,
        shift_px=result.shift_px,
        width=result.width,
        height=result.height,
        median_error_deg=result.median_error_deg,
        det_name=result.det_name,
    )


def _align_from_bytes(
    settings: ServiceConfig,
    base_config: dict | None,
    ref_bytes: bytes,
    late_bytes: bytes,
    ref_label: str,
    late_label: str,
) -> AlignResponse:
    img_ref = decode_image_bytes(ref_bytes)
    img_late = decode_image_bytes(late_bytes)
    return _align_from_images(
        settings,
        base_config,
        img_ref,
        img_late,
        ref_label,
        late_label,
    )


def create_app() -> FastAPI:
    settings = load_service_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
        _require_service_token(settings, request)

        client = _get_s3_client(request, settings)
        ref_bytes = fetch_s3_object_bytes(client, payload.bucket, payload.pano_ref)
        late_bytes = fetch_s3_object_bytes(client, payload.bucket, payload.pano_late)
        base_config = _get_align_base_config(request)
        return _align_from_bytes(
            settings,
            base_config,
            ref_bytes,
            late_bytes,
            "__s3_ref__",
            "__s3_late__",
        )

    @app.post("/align/yaw/url", response_model=AlignResponse)
    def align_yaw_url(request: Request, payload: AlignUrlRequest):
        _require_service_token(settings, request)

        ref_bytes = _fetch_url_bytes(payload.pano_ref_url)
        late_bytes = _fetch_url_bytes(payload.pano_late_url)
        base_config = _get_align_base_config(request)
        return _align_from_bytes(
            settings,
            base_config,
            ref_bytes,
            late_bytes,
            "__url_ref__",
            "__url_late__",
        )

    return app


app = create_app()
