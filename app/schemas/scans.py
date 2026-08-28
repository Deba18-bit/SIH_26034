"""Pydantic schemas for package-image scans."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class OcrStatus(str, Enum):
    """Outcome of an OCR attempt on a processed scan derivative."""

    COMPLETED = "completed"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    NO_TEXT = "no_text"
    FAILED = "failed"


class OCRTextEvidence(BaseModel):
    """One OCR finding measured against the processed derivative image."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]
    engine: Literal["paddleocr"]
    source: Literal["processed"]
    source_image_id: str
    extraction_method: Literal["paddleocr_pp_ocr", "ai_fallback"]

    @model_validator(mode="after")
    def validate_bbox(self) -> "OCRTextEvidence":
        """Require an axis-aligned left, top, right, bottom bounding box."""
        left, top, right, bottom = self.bbox
        if right <= left or bottom <= top:
            raise ValueError("bbox must be [left, top, right, bottom] with positive area")
        return self


class OCRResult(BaseModel):
    """Structured evidence from OCR without any compliance interpretation."""

    status: OcrStatus
    items: list[OCRTextEvidence]
    source_image_id: str
    source_width: int
    source_height: int
    error: str | None = None


class ExtractionStatus(str, Enum):
    """Outcome of the structured extraction process."""

    COMPLETED = "completed"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    FAILED = "failed"


class ExtractedField(BaseModel):
    """A single structured fact extracted from OCR evidence."""

    field_name: str
    value: float | str
    unit: str | None = None
    confidence: float
    source_evidence: OCRTextEvidence


class ExtractionResult(BaseModel):
    """The complete set of facts extracted from a scan."""

    status: ExtractionStatus
    fields: list[ExtractedField]


class ComplianceStatus(str, Enum):
    """Overall or rule-specific compliance outcome."""

    COMPLIANT = "compliant"
    VIOLATION = "violation"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    NOT_APPLICABLE = "not_applicable"


class ComplianceFinding(BaseModel):
    """Outcome of a single deterministic rule evaluation."""

    rule_id: str
    status: ComplianceStatus
    message: str
    legal_reference: str
    evidence: list[ExtractedField]


class ComplianceResult(BaseModel):
    """The complete set of compliance findings for a scan."""

    status: ComplianceStatus
    findings: list[ComplianceFinding]


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
    ocr: OCRResult
    extraction: ExtractionResult
    compliance: ComplianceResult
