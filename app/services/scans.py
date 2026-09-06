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
from app.schemas.history import ScanHistoryResponse, ScanStatsResponse, ScanSummaryItem
from app.schemas.scans import (
    ScanCreateResponse, ScanStatus, OCRResult, ExtractionResult, ExtractedField,
    ComplianceResult, EdgeScanRequest, EdgeOcrElement, OCRTextEvidence, OcrStatus, OCRSource, OCREngine,
    InspectorDecisionRequest
)
from app.services.image_preprocessing import ImagePreprocessingService, ProcessedImage
from app.services.image_quality import ImageQualityService
from app.services.ocr import OcrService
from app.services.extraction import ExtractionService
from app.services.ai_extraction import AiExtractionService
from app.services.compliance import ComplianceService
from app.services.provenance import build_scan_provenance, format_terminal_trace
from app.services.database import DatabaseService

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
        compliance_service: ComplianceService | None = None,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._settings = settings
        self._quality_service = ImageQualityService(settings)
        self._preprocessing_service = ImagePreprocessingService(settings)
        self._ocr_service = ocr_service or OcrService()
        self._extraction_service = extraction_service or ExtractionService()
        self._db_service = database_service or DatabaseService(settings)
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
        self.last_deterministic_extraction: ExtractionResult | None = None

    async def process_ocr_evidence(
        self,
        ocr: OCRResult,
        ai_fallback_image_bytes: bytes | None = None
    ) -> tuple[ExtractionResult, ComplianceResult]:
        """Run the unified deterministic and compliance engine on any normalized OCR evidence."""
        # 5. Extraction
        deterministic_extraction = self._extraction_service.extract(ocr)
        self.last_deterministic_extraction = deterministic_extraction
        extraction = deterministic_extraction
        
        # 6. AI Fallback
        if ai_fallback_image_bytes is not None:
            extraction = await self._ai_extraction_service.enrich(
                ocr, 
                extraction,
                image_bytes=ai_fallback_image_bytes,
                source_width=ocr.source_width,
                source_height=ocr.source_height
            )
            
        # 7. Compliance Engine
        compliance = self._compliance_service.evaluate(extraction)
        
        return extraction, compliance

    async def create_edge_scan(self, request: EdgeScanRequest) -> ScanCreateResponse:
        """Process lightweight OCR JSON payload natively produced on an edge device."""
        # Check idempotency for offline sync
        if request.client_scan_id:
            existing = self._db_service.get_scan_by_client_id(request.client_scan_id)
            if existing:
                try:
                    return self._load_scan_state(existing["scan_id"])
                except Exception:
                    pass

        t0 = time.time()
        scan_id = generate_scan_id()
        created_at = datetime.now(timezone.utc)
        
        # 1. Map Edge request to normalized OCRResult schema
        items = []
        for item in request.items:
            left, top, right, bottom = item.bbox
            if right <= left:
                right = left + 1.0
            if bottom <= top:
                bottom = top + 1.0
            clean_elements = []
            for el in item.elements:
                e_left, e_top, e_right, e_bottom = el.bbox
                if e_right <= e_left:
                    e_right = e_left + 1.0
                if e_bottom <= e_top:
                    e_bottom = e_top + 1.0
                clean_elements.append(
                    EdgeOcrElement(
                        text=el.text,
                        confidence=max(0.0, min(1.0, float(el.confidence))),
                        bbox=(e_left, e_top, e_right, e_bottom),
                    )
                )
            items.append(
                OCRTextEvidence(
                    text=item.text,
                    confidence=max(0.0, min(1.0, float(item.confidence))),
                    bbox=(left, top, right, bottom),
                    engine=request.engine,
                    source=OCRSource.EDGE,
                    source_image_id=f"scan:{scan_id}:edge",
                    elements=clean_elements,
                )
            )
        
        ocr = OCRResult(
            status=OcrStatus.COMPLETED if items else OcrStatus.NO_TEXT,
            items=items,
            source_image_id=f"scan:{scan_id}:edge",
            source_width=request.source_width,
            source_height=request.source_height,
        )
        
        # 2. Run the unified processing pipeline (Extraction -> Compliance)
        # We pass None for ai_fallback_image_bytes because we don't have the image yet!
        extraction, compliance = await self.process_ocr_evidence(ocr, ai_fallback_image_bytes=None)
        
        t_pipeline = time.time() - t0
        
        # 3. Decision Logic: If extraction is incomplete, transition to PENDING_FALLBACK
        needs_fallback, _ = self._ai_extraction_service.should_fallback(extraction)
        fallback_reasons = self._ai_extraction_service.get_fallback_reasons(extraction)
        status = ScanStatus.PENDING_FALLBACK if needs_fallback else ScanStatus.RECEIVED

        provenance = build_scan_provenance(
            ocr=ocr,
            deterministic_extraction=self.last_deterministic_extraction or extraction,
            final_extraction=extraction,
            compliance=compliance,
            fallback_triggered=needs_fallback,
            fallback_reasons=fallback_reasons,
            ai_telemetry=self._ai_extraction_service.last_telemetry,
        )

        # Print Performance Report
        print("\n" + "="*16 + " EDGE SCAN PERFORMANCE " + "="*16)
        print(f"Unified Logic Pipeline:       {t_pipeline:.3f} sec")
        print("="*50)

        # Print OCR Provenance Trace
        print(format_terminal_trace(provenance))

        response = ScanCreateResponse(
            scan_id=scan_id,
            status=status,
            width=request.source_width,
            height=request.source_height,
            created_at=created_at,
            ocr=ocr,
            extraction=extraction,
            compliance=compliance,
            provenance=provenance,
            client_scan_id=request.client_scan_id,
            officer_id=request.officer_id,
        )
        self._store_scan_state(response)
        self._db_service.save_scan(
            response,
            client_scan_id=request.client_scan_id,
            officer_id=request.officer_id,
            sync_source="offline_sync" if request.is_offline_sync else "realtime",
        )
        return response

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

        # Compress image for AI to drastically reduce network upload latency
        img = Image.open(BytesIO(validated_image.content)).convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=60)
        ai_image_bytes = out.getvalue()
        
        t0 = time.time()
        extraction, compliance = await self.process_ocr_evidence(ocr, ai_image_bytes)
        t_pipeline = time.time() - t0
        
        total_time = time.time() - total_start
        
        ai_telemetry = self._ai_extraction_service.last_telemetry
        provenance = build_scan_provenance(
            ocr=ocr,
            deterministic_extraction=self.last_deterministic_extraction or extraction,
            final_extraction=extraction,
            compliance=compliance,
            fallback_triggered=ai_telemetry.get("triggered", False),
            fallback_reasons=ai_telemetry.get("reasons", []),
            ai_telemetry=ai_telemetry,
        )

        # Print Performance Report
        print("\n" + "="*16 + " SCAN PERFORMANCE " + "="*16)
        print(f"Image upload handling:        {t_upload:.3f} sec")
        print(f"Quality check:                {t_quality:.3f} sec")
        print(f"Preprocessing:                {t_preprocessing:.3f} sec")
        print(f"OCR inference:                {t_ocr:.3f} sec")
        print(f"Unified Logic Pipeline:       {t_pipeline:.3f} sec")
        print(f"\nTOTAL SCAN TIME:              {total_time:.3f} sec")
        print("="*50)

        # Print OCR Provenance Trace
        print(format_terminal_trace(provenance))

        response = ScanCreateResponse(
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
            compliance=compliance,
            provenance=provenance,
        )
        self._store_scan_state(response)
        self._db_service.save_scan(
            response,
            officer_id="OFFICER-DEFAULT",
            sync_source="realtime",
            image_path=str(self._settings.storage_dir / "scans" / f"{scan_id}{validated_image.suffix}"),
        )
        return response

    async def fallback_edge_scan(self, scan_id: str, image: UploadFile) -> ScanCreateResponse:
        """Process an uploaded image as a fallback for an incomplete edge scan."""
        t0 = time.time()
        
        # 1. Load scan state
        try:
            state = self._load_scan_state(scan_id)
        except ValueError as e:
            raise ImageProcessingError(message=str(e))
            
        if state.status != ScanStatus.PENDING_FALLBACK:
            raise ImageProcessingError(message="Scan is not pending a fallback.")
            
        # 2. Validate & Store image
        validated_image = await self._validate_image(image)
        self._store_original(scan_id, validated_image)
        
        # 3. Compress image for AI
        img = Image.open(BytesIO(validated_image.content)).convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=60)
        ai_image_bytes = out.getvalue()
        
        # 4. Re-run unified processing pipeline with fallback image
        extraction, compliance = await self.process_ocr_evidence(state.ocr, ai_image_bytes)
        
        t_fallback = time.time() - t0
        
        ai_telemetry = self._ai_extraction_service.last_telemetry
        provenance = build_scan_provenance(
            ocr=state.ocr,
            deterministic_extraction=state.extraction,
            final_extraction=extraction,
            compliance=compliance,
            fallback_triggered=True,
            fallback_reasons=ai_telemetry.get("reasons", []),
            ai_telemetry=ai_telemetry,
        )

        # 5. Update state and return
        state.status = ScanStatus.RECEIVED
        state.filename = validated_image.filename
        state.content_type = validated_image.content_type
        state.file_size = validated_image.file_size
        state.extraction = extraction
        state.compliance = compliance
        state.provenance = provenance
        
        self._store_scan_state(state)
        self._db_service.save_scan(
            state,
            client_scan_id=state.client_scan_id,
            officer_id=state.officer_id or "OFFICER-DEFAULT",
            sync_source="realtime",
            image_path=str(self._settings.storage_dir / "scans" / f"{scan_id}{validated_image.suffix}"),
        )
        
        print("\n" + "="*16 + " FALLBACK SCAN PERFORMANCE " + "="*16)
        print(f"Fallback Execution Time:      {t_fallback:.3f} sec")
        print("="*50)

        # Print OCR Provenance Trace
        print(format_terminal_trace(provenance))
        
        return state

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

    def _store_scan_state(self, response: ScanCreateResponse) -> None:
        """Persist the scan metadata to disk as JSON."""
        destination = self._settings.storage_dir / "scans" / "states" / f"{response.scan_id}.json"
        self._write_bytes_atomically(destination, response.model_dump_json(indent=2).encode("utf-8"))

    def _load_scan_state(self, scan_id: str) -> ScanCreateResponse:
        """Load scan metadata from disk."""
        source = self._settings.storage_dir / "scans" / "states" / f"{scan_id}.json"
        if not source.exists():
            raise ValueError(f"Scan {scan_id} not found.")
        return ScanCreateResponse.model_validate_json(source.read_text("utf-8"))

    def list_scans(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        officer_id: str | None = None,
        search: str | None = None,
    ) -> ScanHistoryResponse:
        """Query paginated inspection history."""
        return self._db_service.list_scans(
            limit=limit, offset=offset, status=status, officer_id=officer_id, search=search
        )

    def get_scan_details(self, scan_id: str) -> ScanCreateResponse:
        """Fetch full scan report details."""
        return self._load_scan_state(scan_id)

    def get_scan_stats(self) -> ScanStatsResponse:
        """Fetch dashboard statistics."""
        return self._db_service.get_stats()

    async def identify_scan(self, scan_id: str) -> ScanSummaryItem:
        """Run on-demand AI identification for brand, product, and manufacturer on an existing scan."""
        # 1. Look up scan in DB or disk state
        scan_record = self._db_service.get_scan(scan_id)
        state: ScanCreateResponse | None = None
        try:
            state = self._load_scan_state(scan_id)
        except ValueError:
            pass

        if not scan_record and not state:
            raise ValueError(f"Scan {scan_id} not found.")

        # 2. Locate the stored original image file
        image_path: Path | None = None
        if scan_record and scan_record.get("image_path"):
            cand = Path(scan_record["image_path"])
            if cand.exists():
                image_path = cand

        if not image_path:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                cand = self._settings.storage_dir / "scans" / f"{scan_id}{ext}"
                if cand.exists():
                    image_path = cand
                    break

        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Original image for scan {scan_id} is not available on server.")

        # Compress image for AI to reduce network upload latency
        img = Image.open(image_path).convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=65)
        ai_image_bytes = out.getvalue()

        # 3. Retrieve OCR text context if available
        ocr_context: str | None = None
        if state and state.ocr and state.ocr.items:
            ocr_context = "\n".join(item.text for item in state.ocr.items if item.text)
        elif scan_record and scan_record.get("ocr_data"):
            ocr_data = scan_record["ocr_data"]
            if isinstance(ocr_data, dict) and "items" in ocr_data:
                ocr_context = "\n".join(item.get("text", "") for item in ocr_data["items"] if item.get("text"))

        # 4. Invoke AI extraction service
        ai_result = await self._ai_extraction_service.identify_package(
            image_bytes=ai_image_bytes,
            ocr_context=ocr_context,
        )

        # 5. Persist the identification fields in the database
        updated_dict = self._db_service.update_scan_identification(
            scan_id=scan_id,
            brand_name=ai_result.brand_name,
            product_name=ai_result.product_name,
            manufacturer_name=ai_result.manufacturer_name,
            identification_status=ai_result.identification_status.value,
            identification_method=ai_result.identification_method.value,
            identification_confidence=ai_result.confidence,
            identification_evidence=ai_result.identification_evidence,
        )

        # 6. Update disk state JSON if present to keep it synchronized
        if state:
            try:
                existing_field_names = {f.field_name for f in state.extraction.fields}
                dummy_evidence = OCRTextEvidence(
                    text=ai_result.brand_name or ai_result.product_name or "AI Identified",
                    confidence=ai_result.confidence,
                    bbox=(0.0, 0.0, float(state.width or 100), float(state.height or 100)),
                    engine=OCREngine.GOOGLE_MLKIT,
                    source=OCRSource.EDGE,
                    source_image_id=f"scan:{scan_id}:ai",
                )
                if ai_result.brand_name and "brand_name" not in existing_field_names:
                    state.extraction.fields.append(
                        ExtractedField(
                            field_name="brand_name",
                            value=ai_result.brand_name,
                            confidence=ai_result.confidence,
                            source_evidence=dummy_evidence,
                        )
                    )
                if ai_result.product_name and "product_name" not in existing_field_names:
                    state.extraction.fields.append(
                        ExtractedField(
                            field_name="product_name",
                            value=ai_result.product_name,
                            confidence=ai_result.confidence,
                            source_evidence=dummy_evidence,
                        )
                    )
                self._store_scan_state(state)
            except Exception as e:
                logger.warning("Could not update scan state JSON: %s", e)

        # 7. Return updated summary
        if updated_dict:
            return self._db_service._row_to_scan_summary_item(updated_dict)

        summary = self._db_service.get_scan_summary(scan_id)
        if summary:
            return summary

        raise ValueError(f"Scan identification failed to save for {scan_id}.")

    def apply_inspector_decision(self, scan_id: str, request: InspectorDecisionRequest) -> ScanCreateResponse:
        """Apply an authoritative Legal Metrology officer decision to a scan."""
        self._db_service.apply_inspector_decision(
            scan_id=scan_id,
            decision=request.decision.value,
            officer_id=request.officer_id,
            notes=request.notes,
        )
        return self._load_scan_state(scan_id)

