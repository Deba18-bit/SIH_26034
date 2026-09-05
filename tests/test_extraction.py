"""Tests for deterministic structured extraction from OCR."""

from app.schemas.scans import ExtractionStatus, OCRResult, OCRTextEvidence, OcrStatus
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


def test_distant_items_are_not_merged_spatially():
    """Verify that vertically distant items (e.g. 2000px apart) are not falsely merged."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            OCRTextEvidence(
                text="MRP:",
                confidence=0.95,
                bbox=(1678.0, 2358.0, 1886.0, 2422.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
            OCRTextEvidence(
                text="0.0",
                confidence=0.95,
                bbox=(2393.0, 270.0, 2452.0, 306.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
        ],
        source_image_id="scan:test",
        source_width=3072,
        source_height=4096,
    )
    result = service.extract(ocr)
    assert not any(f.field_name == "mrp" for f in result.fields)


def test_spatially_adjacent_split_tokens_are_merged():
    """Verify that physically adjacent key and value tokens are properly merged into a tight box."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            OCRTextEvidence(
                text="MRP:",
                confidence=0.95,
                bbox=(100.0, 200.0, 150.0, 230.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
            OCRTextEvidence(
                text="Rs. 250",
                confidence=0.95,
                bbox=(160.0, 200.0, 240.0, 230.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
        ],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    mrp = result.fields[0]
    assert mrp.field_name == "mrp"
    assert mrp.value == 250.0
    assert mrp.source_evidence.bbox == (100.0, 200.0, 240.0, 230.0)


def test_contributing_elements_produce_tight_bbox():
    """Verify that element tokens restrict the field bbox to the matching text span only."""
    from app.schemas.scans import EdgeOcrElement

    service = ExtractionService()
    line_item = OCRTextEvidence(
        text="Net Qty: 500 g Made in India",
        confidence=0.95,
        bbox=(50.0, 100.0, 500.0, 140.0),
        engine="google_mlkit",
        source="edge",
        source_image_id="scan:test",
        elements=[
            EdgeOcrElement(text="Net", bbox=(50.0, 100.0, 90.0, 140.0)),
            EdgeOcrElement(text="Qty:", bbox=(95.0, 100.0, 140.0, 140.0)),
            EdgeOcrElement(text="500", bbox=(145.0, 100.0, 180.0, 140.0)),
            EdgeOcrElement(text="g", bbox=(185.0, 100.0, 200.0, 140.0)),
            EdgeOcrElement(text="Made", bbox=(210.0, 100.0, 260.0, 140.0)),
            EdgeOcrElement(text="in", bbox=(265.0, 100.0, 285.0, 140.0)),
            EdgeOcrElement(text="India", bbox=(290.0, 100.0, 350.0, 140.0)),
        ],
    )
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[line_item],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    qty = result.fields[0]
    assert qty.field_name == "net_quantity"
    assert qty.value == 500.0
    assert qty.unit == "g"
    # Scoped to Net Qty: 500 g (50.0 to 200.0), NOT the entire line (50.0 to 500.0)
    assert qty.source_evidence.bbox == (50.0, 100.0, 200.0, 140.0)


def test_extracts_consumer_care_with_colon_country_code():
    """Verify phone extraction when formatted with newline and colon prefix like :91 9739093912."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("Phone:\n:91 9739093912. Emait crm@akpro.cam")],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    phone = result.fields[0]
    assert phone.field_name == "consumer_care_phone"
    assert phone.value == "9739093912"


def test_extracts_mrp_with_slash_dash_and_usp():
    """Verify MRP extraction with slash-dash and unit sale price suffix like 620/-(6.2/GM)."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("MRP: Rs. 620/-(6.2/GM) (Inclusive of all taxes)")],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    mrp = result.fields[0]
    assert mrp.field_name == "mrp"
    assert mrp.value == 620.0


def test_extracts_net_quantity_with_unit_confusion_digit_nine():
    """Verify net quantity extraction when 'g' is misrecognized as '9' in dot-matrix font."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("Net Quantity: 100 9")],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    qty = result.fields[0]
    assert qty.field_name == "net_quantity"
    assert qty.value == 100.0
    assert qty.unit == "g"


def test_two_column_table_row_alignment_date():
    """Verify two-column table alignment pairs label item in Col 1 with date in Col 2 at same row height."""
    service = ExtractionService()
    item_col1 = OCRTextEvidence(
        text="Batch No. 3\nMfg. Date :\nExp. Date :",
        confidence=0.95,
        bbox=(1518.0, 2700.0, 1742.0, 2898.0),
        engine="google_mlkit",
        source="edge",
        source_image_id="scan:test",
    )
    item_col2 = OCRTextEvidence(
        text="CREO0g038\nü8.08.28",
        confidence=0.95,
        bbox=(1969.0, 2715.0, 2252.0, 2824.0),
        engine="google_mlkit",
        source="edge",
        source_image_id="scan:test",
    )
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[item_col1, item_col2],
        source_image_id="scan:test",
        source_width=3000,
        source_height=4000,
    )
    result = service.extract(ocr)
    mfg = next((f for f in result.fields if f.field_name == "manufacturing_date"), None)
    assert mfg is not None
    assert mfg.value == "08.08.28"
    # Row bounding box should be constrained to row 1 height (~2766 to ~2832)
    assert mfg.source_evidence.bbox[0] == 1518.0
    assert mfg.source_evidence.bbox[2] == 2252.0
    assert mfg.source_evidence.bbox[1] >= 2750.0
    assert mfg.source_evidence.bbox[3] <= 2850.0


def test_extracts_mrp_with_dot_matrix_digit_e_confusion():
    """Verify dot-matrix 'E' misrecognition of '6' in MRP: E20:E.2SM! -> 620.0."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("MRP: E20:E.2SM!")],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    mrp = result.fields[0]
    assert mrp.field_name == "mrp"
    assert mrp.value == 620.0


def test_extracts_date_with_trailing_e_dot_matrix():
    """Verify dot-matrix trailing 'E' in year: 8.08.2E -> 8.08.28."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("Mfg. Date : 8.08.2E")],
        source_image_id="scan:test",
        source_width=1000,
        source_height=1000,
    )
    result = service.extract(ocr)
    assert len(result.fields) == 1
    d = result.fields[0]
    assert d.field_name == "manufacturing_date"
    assert d.value == "8.08.28"


def test_tabular_alignment_does_not_confuse_mfg_and_exp_date_rows():
    """Verify that multi-row table alignment pairs Mfg. Date with its exact row and does not pick Exp. Date."""
    service = ExtractionService()
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            OCRTextEvidence(
                text="Mfg. Date :",
                confidence=0.95,
                bbox=(1527.0, 3038.0, 1761.0, 3094.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
            OCRTextEvidence(
                text="Exp. Date",
                confidence=0.95,
                bbox=(1528.0, 3114.0, 1729.0, 3169.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
            OCRTextEvidence(
                text="0S.08.28",
                confidence=0.95,
                bbox=(2016.0, 3018.0, 2294.0, 3087.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
            OCRTextEvidence(
                text="0s.02.28",
                confidence=0.95,
                bbox=(2017.0, 3076.0, 2296.0, 3150.0),
                engine="google_mlkit",
                source="edge",
                source_image_id="scan:test",
            ),
        ],
        source_image_id="scan:test",
        source_width=4096,
        source_height=3072,
    )
    result = service.extract(ocr)
    assert result.status == ExtractionStatus.COMPLETED
    mfg_fields = [f for f in result.fields if f.field_name == "manufacturing_date"]
    assert len(mfg_fields) == 1
    assert mfg_fields[0].value == "08.08.28"




