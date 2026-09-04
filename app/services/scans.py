"""Validation and local storage for package-image scans."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import time
import logging

logger = logging.getLogger(__name__)

import cv2
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.schemas.scans import ScanCreateResponse, ScanStatus
from app.services.image_preprocessing import ImagePreprocessingService, ProcessedImage
from app.services.image_quality import ImageQualityService
from app.services.ocr import OcrService
from app.services.extraction import ExtractionService
from app.services.ai_extraction import AiExtractionService
from app.services.compliance import ComplianceService

ALLOWED_IMAGE_FORMATS = {
    "image/jpeg": ("JPEG", ".jpg"),
    "image/png": ("PNG", ".png"),
    "image/webp": ("WEBP", ".webp"),
}


@dataclass(frozen=True)
class ImageValidationError(Exception):
    """A client-correctable image validation failure."""

    status_code: int
    code: str
    message: str


@dataclass(frozen=True)
class ImageProcessingError(Exception):
    """A processing failure that must not expose internal implementation details."""

    message: str = "The image could not be prepared for downstream processing."


@dataclass(frozen=True)
class ValidatedImage:
    """Validated image data and metadata retained for storage."""

    content: bytes
    content_type: str
    filename: str
    file_size: int
    width: int
    height: int
    suffix: str


def generate_scan_id() -> str:
    """Create an opaque identifier for a newly received scan."""
    return str(uuid4())


class ScanService:
    """Create scans by validating and storing original image uploads."""

    def __init__(
        self,
        settings: Settings,
        ocr_service: OcrService | None = None,
        extraction_service: ExtractionService | None = None,
        ai_extraction_service: AiExtractionService | None = None,
        compliance_service: ComplianceService | None = None
    ) -> None:
        self._settings = settings
        self._quality_service = ImageQualityService(settings)
        self._preprocessing_service = ImagePreprocessingService(settings)
        self._ocr_service = ocr_service or OcrService()
        self._extraction_service = extraction_service or ExtractionService()
        if ai_extraction_service is None:
            if settings.gemini_api_key:
                from app.services.ai_extraction import GeminiAiProvider
                provider = GeminiAiProvider(api_key=settings.gemini_api_key)
                self._ai_extraction_service = AiExtractionService(provider=provider)
            else:
                self._ai_extraction_service = AiExtractionService()
        else:
            self._ai_extraction_service = ai_extraction_service
        self._compliance_service = compliance_service or ComplianceService()

    async def create_scan(self, image: UploadFile) -> ScanCreateResponse:
        """Validate, persist, and describe one uploaded package image."""
        total_start = time.time()
        
        # 1. Validation & Storage
        t0 = time.time()
        validated_image = await self._validate_image(image)
        scan_id = generate_scan_id()
        self._store_original(scan_id, validated_image)
        t_upload = time.time() - t0

        try:
            # 2. Quality Check
            t0 = time.time()
            quality = self._quality_service.assess(
                validated_image.content, validated_image.width, validated_image.height
            )
            t_quality = time.time() - t0
            
            # 3. Preprocessing
            t0 = time.time()
            processed_image = await asyncio.to_thread(self._preprocessing_service.preprocess, validated_image.content)
            processed_image_path = await asyncio.to_thread(self._store_processed, scan_id, processed_image)
            t_preprocessing = time.time() - t0
        except (cv2.error, OSError, ValueError) as error:
            raise ImageProcessingError() from error
            
        created_at = datetime.now(timezone.utc)
        
        # 4. OCR
        t0 = time.time()
        ocr = await asyncio.to_thread(
            self._ocr_service.extract,
            processed_image_path=processed_image_path,
            source_image_id=f"scan:{scan_id}:processed",
            source_width=processed_image.width,
            source_height=processed_image.height,
        )
        t_ocr = time.time() - t0
        
        # Rescale OCR bounding boxes to original image dimensions
        scale_x = validated_image.width / processed_image.width
        scale_y = validated_image.height / processed_image.height
        if scale_x != 1.0 or scale_y != 1.0:
            for item in ocr.items:
                l, t, r, b = item.bbox
                item.bbox = (l * scale_x, t * scale_y, r * scale_x, b * scale_y)
            ocr.source_width = validated_image.width
            ocr.source_height = validated_image.height

        # 5. Extraction
        t0 = time.time()
        extraction = self._extraction_service.extract(ocr)
        t_extraction = time.time() - t0
        
        # 6. AI Fallback
        t0 = time.time()
        
        # Compress image for AI to drastically reduce network upload latency
        img = Image.open(BytesIO(validated_image.content)).convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=60)
        ai_image_bytes = out.getvalue()
        
        extraction = await self._ai_extraction_service.enrich(
            ocr, 
            extraction,
            image_bytes=ai_image_bytes,
            source_width=validated_image.width,
            source_height=validated_image.height
        )
        t_ai = time.time() - t0
        
        # 7. Compliance Engine
        t0 = time.time()
        compliance = self._compliance_service.evaluate(extraction)
        t_compliance = time.time() - t0
        
        total_time = time.time() - total_start
        
        # Print Performance Report
        print("\n" + "="*16 + " SCAN PERFORMANCE " + "="*16)
        print(f"Image upload handling:        {t_upload:.3f} sec")
        print(f"Quality check:                {t_quality:.3f} sec")
        print(f"Preprocessing:                {t_preprocessing:.3f} sec")
        print(f"OCR inference:                {t_ocr:.3f} sec")
        print(f"Deterministic extraction:     {t_extraction:.3f} sec")
        print(f"AI fallback:                  {t_ai:.3f} sec")
        print(f"Compliance evaluation:        {t_compliance:.3f} sec")
        print(f"\nTOTAL SCAN TIME:              {total_time:.3f} sec")
        print("="*50)

        # Print Engine Breakdown
        print(f"\nTotal fields found: {len(extraction.fields)}\n" + "-"*50)
        for field in extraction.fields:
            if field.source_evidence.engine == "paddleocr":
                engine_tag = "[ DETERMINISTIC (Regex + PaddleOCR)]"
            else:
                engine_tag = "[ VLM AI FALLBACK (Gemini Vision)]"
                
            print(f"{engine_tag:<40} {field.field_name.upper():<20} = {field.value}")
        print("-" * 50 + "\n")

        return ScanCreateResponse(
            scan_id=scan_id,
            status=ScanStatus.RECEIVED,
            filename=validated_image.filename,
            content_type=validated_image.content_type,
            file_size=validated_image.file_size,
            width=validated_image.width,
            height=validated_image.height,
            created_at=created_at,
            quality=quality,
            ocr=ocr,
            extraction=extraction,
            compliance=compliance
        )

    async def _validate_image(self, image: UploadFile) -> ValidatedImage:
        content_type = image.content_type
        if content_type not in ALLOWED_IMAGE_FORMATS:
            raise ImageValidationError(
                status_code=415,
                code="unsupported_media_type",
                message="Only JPEG, PNG, and WebP images are supported.",
            )

        content = await image.read(self._settings.max_upload_size_bytes + 1)
        if len(content) > self._settings.max_upload_size_bytes:
            raise ImageValidationError(
                status_code=413,
                code="image_too_large",
                message="The uploaded image exceeds the maximum allowed size.",
            )

        width, height, image_format = self._read_image_dimensions(content)
        expected_format, suffix = ALLOWED_IMAGE_FORMATS[content_type]
        if image_format != expected_format:
            raise ImageValidationError(
                status_code=422,
                code="content_type_mismatch",
                message="The image content does not match its declared content type.",
            )

        if width < self._settings.min_image_width or height < self._settings.min_image_height:
            raise ImageValidationError(
                status_code=422,
                code="image_dimensions_too_small",
                message=(
                    "The image must be at least "
                    f"{self._settings.min_image_width}x{self._settings.min_image_height} pixels."
                ),
            )

        filename = Path(image.filename or "upload").name
        return ValidatedImage(
            content=content,
            content_type=content_type,
            filename=filename,
            file_size=len(content),
            width=width,
            height=height,
            suffix=suffix,
        )

    @staticmethod
    def _read_image_dimensions(content: bytes) -> tuple[int, int, str]:
        """Decode an image sufficiently to confirm it is readable."""
        try:
            with Image.open(BytesIO(content)) as decoded_image:
                decoded_image.verify()
            with Image.open(BytesIO(content)) as decoded_image:
                decoded_image.load()
                width, height = decoded_image.size
                image_format = decoded_image.format
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
            raise ImageValidationError(
                status_code=422,
                code="invalid_image",
                message="The uploaded file is not a readable image.",
            ) from error

        if image_format is None:
            raise ImageValidationError(
                status_code=422,
                code="invalid_image",
                message="The uploaded file is not a readable image.",
            )
        return width, height, image_format

    def _store_original(self, scan_id: str, image: ValidatedImage) -> None:
        """Store a validated original using an opaque server-generated filename."""
        self._write_bytes_atomically(
            self._settings.storage_dir / "scans" / f"{scan_id}{image.suffix}", image.content
        )

    def _store_processed(self, scan_id: str, image: ProcessedImage) -> Path:
        """Store the derivative alongside, but separately from, its original image."""
        destination = self._settings.storage_dir / "scans" / "processed" / f"{scan_id}.png"
        self._write_bytes_atomically(destination, image.content)
        return destination

    @staticmethod
    def _write_bytes_atomically(destination: Path, content: bytes) -> None:
        """Write a derivative or original without leaving partially written output."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary_destination.write_bytes(content)
            temporary_destination.replace(destination)
        finally:
            if temporary_destination.exists():
                temporary_destination.unlink()
