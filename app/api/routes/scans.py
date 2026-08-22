"""Package-image ingestion endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.scans import ScanCreateResponse
from app.services.scans import ImageValidationError, ScanService

router = APIRouter(prefix="/scans", tags=["scans"])


def get_scan_service() -> ScanService:
    """Build the service used by scan routes."""
    return ScanService(get_settings())


@router.post("", response_model=ScanCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_scan(
    image: Annotated[UploadFile, File(description="Package image to validate and store")],
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanCreateResponse:
    """Validate and store one package image for later processing."""
    try:
        return await scan_service.create_scan(image)
    except ImageValidationError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "message": error.message},
        ) from error
