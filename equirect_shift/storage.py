from typing import Optional, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException, status


class S3Settings(Protocol):
    s3_region: Optional[str]
    s3_endpoint_url: Optional[str]
    s3_addressing_style: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_session_token: Optional[str]


def s3_client(settings: S3Settings):
    kwargs = {
        "region_name": settings.s3_region,
        "endpoint_url": settings.s3_endpoint_url,
        "config": Config(s3={"addressing_style": "virtual"}),
    }

    if settings.s3_addressing_style:
        kwargs["config"] = Config(s3={"addressing_style": settings.s3_addressing_style})
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        kwargs["aws_session_token"] = settings.aws_session_token
    return boto3.client("s3", **kwargs)


def fetch_s3_object_bytes(client, bucket: str, key: str) -> bytes:
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
