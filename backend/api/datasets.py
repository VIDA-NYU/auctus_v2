from __future__ import annotations

import logging
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

try:
    from minio.error import S3Error
except Exception:  # pragma: no cover - keep route importable even when minio is missing
    S3Error = Exception

from storage.minio_client import PROFILE_BUCKET_NAME, get_heavy_profile_object, get_storage_client

router = APIRouter()
logger = logging.getLogger(__name__)


def _stream_minio_object(minio_object) -> Iterator[bytes]:
    try:
        while True:
            chunk = minio_object.read(64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            minio_object.close()
        finally:
            minio_object.release_conn()


def _resolve_profile_object_name(client, dataset_id: str) -> str:
    """Resolve the most likely profile object name for a dataset."""
    exact_name = f"{dataset_id}_profile.json"

    try:
        client.stat_object(PROFILE_BUCKET_NAME, exact_name)
        return exact_name
    except S3Error:
        pass

    # Fallback to prefix scan in case the object was stored with a slightly different suffix.
    for obj in client.list_objects(PROFILE_BUCKET_NAME, prefix=dataset_id, recursive=True):
        if getattr(obj, "object_name", "").endswith("_profile.json"):
            return obj.object_name

    return exact_name


@router.get("/api/datasets/{dataset_id}/profile")
async def get_dataset_profile(dataset_id: str):
    """Stream the stored MinIO profile JSON for a dataset."""
    try:
        client = get_storage_client()
    except Exception as exc:
        logger.warning("MinIO client unavailable for %s: %s", dataset_id, exc)
        raise HTTPException(status_code=503, detail="Profile storage unavailable")
    object_name = _resolve_profile_object_name(client, dataset_id)

    try:
        minio_object, _ = get_heavy_profile_object(client, object_name.removesuffix("_profile.json"))
    except S3Error as exc:
        error_code = getattr(exc, "code", None) or getattr(exc, "errno", None)
        if error_code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise HTTPException(status_code=404, detail=f"Profile not found for dataset {dataset_id}")
        logger.exception("MinIO profile fetch failed for %s", dataset_id)
        raise HTTPException(status_code=503, detail="Profile storage unavailable")

    return StreamingResponse(
        _stream_minio_object(minio_object),
        media_type="application/json",
        headers={"Content-Disposition": f'inline; filename="{object_name}"'},
    )