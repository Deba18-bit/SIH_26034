"""Package-image ingestion endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.history import ScanHistoryResponse, ScanStatsResponse, ScanSummaryItem
from app.schemas.scans import ScanCreateResponse, EdgeScanRequest, InspectorDecisionRequest
from app.services.scans import ImageProcessingError, ImageValidationError, ScanService

router = APIRouter(prefix="/scans", tags=["scans"])


def get_scan_service() -> ScanService:
    """Build the service used by scan routes."""
    return ScanService(get_settings())


@router.get("/history", response_model=ScanHistoryResponse, status_code=status.HTTP_200_OK)
async def get_scan_history(
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    officer_id: str | None = None,
    search: str | None = None,
) -> ScanHistoryResponse:
    """Retrieve paginated inspection history with optional filtering."""
    return scan_service.list_scans(
        limit=limit, offset=offset, status=status, officer_id=officer_id, search=search
    )


@router.get("/stats", response_model=ScanStatsResponse, status_code=status.HTTP_200_OK)
async def get_scan_stats(
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanStatsResponse:
    """Retrieve aggregate inspection metrics across all scans."""
    return scan_service.get_scan_stats()


@router.get("/{scan_id}", response_model=ScanCreateResponse, status_code=status.HTTP_200_OK)
async def get_scan_details(
    scan_id: str,
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanCreateResponse:
    """Retrieve full audit report details for a specific scan."""
    try:
        return scan_service.get_scan_details(scan_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/edge", response_model=ScanCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_edge_scan(
    request: EdgeScanRequest,
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanCreateResponse:
    """Process a lightweight JSON payload of native Edge OCR bounding boxes."""
    return await scan_service.create_edge_scan(request)


@router.post("/{scan_id}/fallback", response_model=ScanCreateResponse, status_code=status.HTTP_200_OK)
async def fallback_edge_scan(
    scan_id: str,
    image: Annotated[UploadFile, File(description="Fallback image for Gemini Vision processing")],
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanCreateResponse:
    """Process a fallback image for an incomplete Edge OCR scan."""
    try:
        return await scan_service.fallback_edge_scan(scan_id, image)
    except ImageValidationError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "message": error.message},
        ) from error
    except ImageProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "image_processing_failed", "message": error.message},
        ) from error

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
    except ImageProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "image_processing_failed", "message": error.message},
        ) from error


@router.post("/{scan_id}/identify", response_model=ScanSummaryItem, status_code=status.HTTP_200_OK)
async def identify_scan(
    scan_id: str,
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanSummaryItem:
    """Trigger on-demand AI visual inspection to identify brand, product, and manufacturer."""
    try:
        return await scan_service.identify_scan(scan_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "image_not_available", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "identification_failed", "message": str(e)},
        )


@router.post("/{scan_id}/decision", response_model=ScanCreateResponse, status_code=status.HTTP_200_OK)
async def submit_inspector_decision(
    scan_id: str,
    request: InspectorDecisionRequest,
    scan_service: Annotated[ScanService, Depends(get_scan_service)],
) -> ScanCreateResponse:
    """Submit an authoritative Legal Metrology officer enforcement decision (compliant/violation)."""
    try:
        return scan_service.apply_inspector_decision(scan_id, request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "decision_failed", "message": str(e)},
        )


