"""Pydantic schemas for package-image scans."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ScanStatus(str, Enum):
    """Lifecycle state of an ingested scan."""

    RECEIVED = "received"


class ImageQualityStatus(str, Enum):
    """Outcome of engineering checks for downstream processing readiness."""

    PASS = "PASS"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class DimensionsQualityCheck(BaseModel):
    """Measured dimensions and the configured minimum requirements."""

    passed: bool
    width: int
    height: int
    minimum_width: int
    minimum_height: int


class SharpnessQualityCheck(BaseModel):
    """Laplacian-variance sharpness measurement."""

    passed: bool
    laplacian_variance: float
    minimum_variance: float


class BrightnessQualityCheck(BaseModel):
    """Mean grayscale-intensity exposure measurement."""

    passed: bool
    mean_intensity: float
    minimum_intensity: float
    maximum_intensity: float


class ReadabilityQualityCheck(BaseModel):
    """Whether OpenCV could decode the validated image content."""

    passed: bool
    method: str


class ImageQualityChecks(BaseModel):
    """Deterministic quality checks performed on the uploaded image."""

    dimensions: DimensionsQualityCheck
    sharpness: SharpnessQualityCheck
    brightness: BrightnessQualityCheck
    readability: ReadabilityQualityCheck


class ImageQualityAssessment(BaseModel):
    """Explainable engineering assessment for future OCR preparation."""

    status: ImageQualityStatus
    checks: ImageQualityChecks
    messages: list[str]


class ScanCreateResponse(BaseModel):
    """Public metadata returned after a package image is ingested."""

    scan_id: str
    status: ScanStatus
    filename: str
    content_type: str
    file_size: int
    width: int
    height: int
    created_at: datetime
    quality: ImageQualityAssessment
