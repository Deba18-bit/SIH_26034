"""Tests for deterministic structured extraction from OCR."""

from app.schemas.scans import OCRResult, OCRTextEvidence, OcrStatus
from app.services.extraction import ExtractionService

def make_evidence(text: str, confidence: float = 0.95) -> OCRTextEvidence:
    return OCRTextEvidence(
        text=text,
        confidence=confidence,
        bbox=(0, 0, 10, 10),
        engine="paddleocr",
        source="processed",
        source_image_id="scan:test",
        extraction_method="paddleocr_pp_ocr"
    )

def test_extracts_mrp_successfully():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("MRP Rs. 150.00")],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 1
    assert result.fields[0].field_name == "mrp"
    assert result.fields[0].value == 150.0

def test_extracts_net_quantity_successfully():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("Net Wt: 500 g")],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 1
    assert result.fields[0].field_name == "net_quantity"
    assert result.fields[0].value == 500.0
    assert result.fields[0].unit == "g"

def test_ambiguous_fields_flag_manual_review():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            make_evidence("MRP Rs 150.00"),
            make_evidence("MRP 200")
        ],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "manual_review_required"
    assert len(result.fields) == 2

def test_propagates_ocr_manual_review_status():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.MANUAL_REVIEW_REQUIRED,
        items=[make_evidence("MRP Rs 150.00", 0.4)],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "manual_review_required"

def test_handles_failed_ocr():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.FAILED,
        items=[],
        source_image_id="scan:test",
        source_width=100,
        source_height=100,
        error="OCR Error"
    )
    result = service.extract(ocr)
    assert result.status == "failed"
    assert len(result.fields) == 0


def test_extracts_manufacturing_and_packing_dates():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            make_evidence("Mfg: 12/2023"),
            make_evidence("Mfd: 11/2023"),
            make_evidence("Packed: 10/2023"),
            make_evidence("Pkd: 09/2023"),
        ],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    
    # We should have 4 fields extracted
    assert len(result.fields) == 4
    
    mfg_fields = [f for f in result.fields if f.field_name == "manufacturing_date"]
    pkd_fields = [f for f in result.fields if f.field_name == "packing_date"]
    
    assert len(mfg_fields) == 2
    assert {f.value for f in mfg_fields} == {"12/2023", "11/2023"}
    
    assert len(pkd_fields) == 2
    assert {f.value for f in pkd_fields} == {"10/2023", "09/2023"}

def test_extracts_mrp_with_colon():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("MRP: Rs. 150.00 (Incl. of all taxes)")],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 1
    assert result.fields[0].field_name == "mrp"
    assert result.fields[0].value == 150.0

def test_extracts_standalone_quantity():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("Quantity: 250g")],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 1
    assert result.fields[0].field_name == "net_quantity"
    assert result.fields[0].value == 250.0
    assert result.fields[0].unit == "g"

def test_extracts_consumer_care_variants():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("CR Details: Call 1800-123-4567")],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 1
    assert result.fields[0].field_name == "consumer_care_phone"
    assert result.fields[0].value == "1800-123-4567"

def test_extracts_alphabetic_dates():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            make_evidence("Mfg: 15 OCT 2023"),
            make_evidence("Packed 12-Jan-24")
        ],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 2
    
    mfg = next(f for f in result.fields if f.field_name == "manufacturing_date")
    assert mfg.value == "15 OCT 2023"
    
    pkd = next(f for f in result.fields if f.field_name == "packing_date")
    assert pkd.value == "12-Jan-24"

def test_ignores_unprefixed_date():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("15 OCT 2023")],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 0

def test_low_confidence_unrelated_item_does_not_poison_extraction():
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            make_evidence("MRP Rs. 150.00", 0.98),
            make_evidence("Some random blurry text", 0.10)
        ],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    result = service.extract(ocr)
    assert result.status == "completed"
    assert len(result.fields) == 1
    assert result.fields[0].field_name == "mrp"
