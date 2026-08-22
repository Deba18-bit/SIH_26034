"""Unit tests for OCR evidence normalization and failure behavior."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.scans import OCRTextEvidence, OcrStatus
from app.services.ocr import (
    OcrService,
    RawOcrDetection,
    normalize_paddle_page_output,
)


class StaticOcrEngine:
    """A deterministic OCR-engine fake used without model downloads."""

    def __init__(self, detections: list[RawOcrDetection]) -> None:
        self._detections = detections

    def extract(self, image_path: Path) -> list[RawOcrDetection]:
        return self._detections


class FailingOcrEngine:
    """An OCR fake that models an inference failure."""

    def extract(self, image_path: Path) -> list[RawOcrDetection]:
        raise RuntimeError("model unavailable")


def extract_with(engine: object):
    """Run one model-free OCR service request."""
    return OcrService(engine=engine).extract(Path("processed.png"), "scan:test:processed", 640, 480)


def test_ocr_text_evidence_rejects_out_of_range_confidence() -> None:
    """Confidence is represented only on the inclusive zero-to-one scale."""
    with pytest.raises(ValidationError):
        OCRTextEvidence(
            text="MRP Rs. 99",
            confidence=1.01,
            bbox=(1, 2, 30, 20),
            engine="paddleocr",
            source="processed",
            source_image_id="scan:test:processed",
            extraction_method="paddleocr_pp_ocr",
        )


def test_ocr_text_evidence_rejects_invalid_bbox() -> None:
    """Bounding boxes use a positive-area left, top, right, bottom convention."""
    with pytest.raises(ValidationError):
        OCRTextEvidence(
            text="MRP Rs. 99",
            confidence=0.9,
            bbox=(30, 20, 1, 2),
            engine="paddleocr",
            source="processed",
            source_image_id="scan:test:processed",
            extraction_method="paddleocr_pp_ocr",
        )


def test_normalizes_representative_paddle_output() -> None:
    """PaddleOCR's recognition arrays become engine-neutral evidence records."""
    detections = normalize_paddle_page_output(
        {
            "res": {
                "rec_texts": ["Manufacturer: Example Co.", "Net Quantity: 500 g", "MRP Rs. 99"],
                "rec_scores": [0.99, 0.88, 0.96],
                "rec_boxes": [[10, 20, 210, 50], [10, 60, 180, 90], [10, 100, 150, 130]],
            }
        }
    )

    assert detections[2] == RawOcrDetection("MRP Rs. 99", 0.96, (10.0, 100.0, 150.0, 130.0))


def test_empty_ocr_result_is_preserved() -> None:
    """No recognized text is distinct from an inference failure."""
    result = extract_with(StaticOcrEngine([]))

    assert result.status is OcrStatus.NO_TEXT
    assert result.items == []
    assert result.error is None


def test_low_confidence_ocr_result_is_preserved() -> None:
    """Low-confidence evidence remains available for later manual review."""
    result = extract_with(StaticOcrEngine([RawOcrDetection("Consumer care", 0.08, (1, 2, 70, 20))]))

    assert result.status is OcrStatus.COMPLETED
    assert result.items[0].confidence == 0.08


def test_ocr_failure_returns_structured_failure_state() -> None:
    """Engine errors produce no fabricated evidence or internal stack trace."""
    result = extract_with(FailingOcrEngine())

    assert result.status is OcrStatus.FAILED
    assert result.items == []
    assert result.error == "OCR extraction could not be completed."
