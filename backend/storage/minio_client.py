"""MinIO object storage configuration and profile upload helpers for Auctus v2."""

from __future__ import annotations

import io
import json
import os

try:
    from minio import Minio
    from minio.error import S3Error
except Exception:  # pragma: no cover - allow backend startup without minio installed
    Minio = None
    S3Error = Exception

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
PROFILE_BUCKET_NAME = "auctus-dataset-profiles"


def get_storage_client() -> Minio:
    """Create and return a MinIO client for local development."""
    if Minio is None:
        raise RuntimeError("minio package is not available in this environment")
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """Create the bucket if it does not already exist."""
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def upload_heavy_profile(client: Minio, dataset_id: str, comprehensive_profile: dict) -> str:
    """Upload a full dataset profile JSON document to MinIO.

    The profile is serialized to JSON bytes and stored as
    ``{dataset_id}_profile.json`` in ``PROFILE_BUCKET_NAME``.
    """
    ensure_bucket_exists(client, PROFILE_BUCKET_NAME)

    profile_bytes = json.dumps(comprehensive_profile, ensure_ascii=False, indent=2).encode("utf-8")
    profile_stream = io.BytesIO(profile_bytes)
    object_name = f"{dataset_id}_profile.json"

    client.put_object(
        bucket_name=PROFILE_BUCKET_NAME,
        object_name=object_name,
        data=profile_stream,
        length=len(profile_bytes),
        content_type="application/json",
    )
    return object_name


def get_heavy_profile_object(client: Minio, dataset_id: str):
    """Return the raw MinIO object stream for a stored dataset profile JSON file."""
    object_name = f"{dataset_id}_profile.json"
    try:
        return client.get_object(PROFILE_BUCKET_NAME, object_name), object_name
    except S3Error:
        raise
