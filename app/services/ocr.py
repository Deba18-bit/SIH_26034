"""PaddleOCR integration and evidence normalization."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.schemas.scans import OCRResult, OCRTextEvidence, OcrStatus


@dataclass(frozen=True)
class RawOcrDetection:
    """Engine-neutral OCR output before application normalization."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


class OcrEngine(Protocol):
    """Minimal boundary that allows OCR inference to be tested without models."""

    def extract(self, image_path: Path) -> Iterable[RawOcrDetection]:
        """Extract text detections from one image path."""


class PaddleOcrEngine:
    """Lazy CPU PaddleOCR adapter for processed package images."""

    def __init__(self) -> None:
        self._ocr: Any | None = None

    def extract(self, image_path: Path) -> Iterable[RawOcrDetection]:
        """Run PaddleOCR and normalize its page-level output."""
        for page_result in self._get_ocr().predict(str(image_path)):
            yield from normalize_paddle_page_output(self._result_to_payload(page_result))

    def _get_ocr(self) -> Any:
        """Initialize PaddleOCR only when a real inference request arrives."""
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="paddle",
            )
        return self._ocr

    @staticmethod
    def _result_to_payload(page_result: Any) -> dict[str, Any]:
        """Handle the documented PaddleOCR result object and plain test mappings."""
        if isinstance(page_result, dict):
            return page_result
        if hasattr(page_result, "json"):
            return page_result.json
        if hasattr(page_result, "to_dict"):
            return page_result.to_dict()
        raise ValueError("PaddleOCR returned an unsupported result format.")


def normalize_paddle_page_output(payload: dict[str, Any]) -> list[RawOcrDetection]:
    """Convert PaddleOCR recognition arrays into stable application-neutral detections."""
    result = payload.get("res", payload)
    texts = result.get("rec_texts", [])
    confidences = result.get("rec_scores", [])
    boxes = result.get("rec_boxes", [])
    if not (len(texts) == len(confidences) == len(boxes)):
        raise ValueError("PaddleOCR text, confidence, and bounding-box counts differ.")

    detections: list[RawOcrDetection] = []
    for text, confidence, box in zip(texts, confidences, boxes, strict=True):
        if len(box) != 4:
            raise ValueError("PaddleOCR bounding boxes must have four coordinates.")
        left, top, right, bottom = (float(coordinate) for coordinate in box)
        detections.append(
            RawOcrDetection(
                text=str(text),
                confidence=float(confidence),
                bbox=(left, top, right, bottom),
            )
        )
    return detections


class OcrService:
    """Create evidence records from OCR while retaining failures as explicit states."""

    def __init__(self, engine: OcrEngine | None = None) -> None:
        self._engine = engine or PaddleOcrEngine()

    def extract(
        self,
        processed_image_path: Path,
        source_image_id: str,
        source_width: int,
        source_height: int,
    ) -> OCRResult:
        """Run OCR without letting an engine failure erase or misstate evidence."""
        try:
            items = [
                OCRTextEvidence(
                    text=detection.text,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    engine="paddleocr",
                    source="processed",
                    source_image_id=source_image_id,
                    extraction_method="paddleocr_pp_ocr",
                )
                for detection in self._engine.extract(processed_image_path)
            ]
        except Exception:
            return OCRResult(
                status=OcrStatus.FAILED,
                items=[],
                source_image_id=source_image_id,
                source_width=source_width,
                source_height=source_height,
                error="OCR extraction could not be completed.",
            )

        if not items:
            status = OcrStatus.NO_TEXT
        elif any(item.confidence < 0.60 for item in items):
            status = OcrStatus.MANUAL_REVIEW_REQUIRED
        else:
            status = OcrStatus.COMPLETED

        return OCRResult(
            status=status,
            items=items,
            source_image_id=source_image_id,
            source_width=source_width,
            source_height=source_height,
        )
