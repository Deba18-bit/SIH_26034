"""Unit and integration tests for Brand & Product Identification."""

import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app
from app.schemas.scans import (
    ComplianceResult,
    ComplianceStatus,
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    OCREngine,
    OCRResult,
    OCRSource,
    OCRTextEvidence,
    OcrStatus,
    ScanCreateResponse,
    ScanStatus,
)
from app.services.ai_extraction import AiExtractionService, FakeAiProvider
from app.services.database import DatabaseService
from app.services.identification import (
    IdentificationMethod,
    IdentificationResult,
    IdentificationService,
    IdentificationStatus,
)
from app.services.scans import ScanService


def _create_dummy_ocr(text_lines: list[tuple[str, tuple[float, float, float, float]]]) -> OCRResult:
    """Helper to build OCRResult from a list of (text, bbox) tuples."""
    items = []
    for text, bbox in text_lines:
        items.append(
            OCRTextEvidence(
                text=text,
                confidence=0.95,
                bbox=bbox,
                engine=OCREngine.GOOGLE_MLKIT,
                source=OCRSource.EDGE,
                source_image_id="test_image",
            )
        )
    return OCRResult(
        status=OcrStatus.COMPLETED,
        items=items,
        source_image_id="test_image",
        source_width=1000,
        source_height=1000,
    )


# =========================================================================
# 1. Deterministic Brand & Product Identification Unit Tests
# =========================================================================

def test_deterministic_identification_with_known_brand_and_commodity():
    """Direct match of known FMCG brand and known commodity in OCR items."""
    ocr = _create_dummy_ocr([
        ("NAKPRO NUTRITION", (50, 50, 400, 120)),
        ("100% PURE CREATINE MONOHYDRATE", (50, 140, 500, 200)),
        ("MRP Rs. 699 incl. of all taxes", (50, 220, 300, 260)),
        ("Net Qty: 100g", (50, 280, 200, 310)),
    ])
    result = IdentificationService.identify(ocr)

    assert result.brand_name == "NAKPRO"
    assert result.product_name == "Creatine Monohydrate"
    assert result.identification_status == IdentificationStatus.IDENTIFIED
    assert result.identification_method == IdentificationMethod.DETERMINISTIC
    assert result.confidence >= 0.9
    assert any("Brand 'NAKPRO'" in e for e in result.identification_evidence)


def test_brand_identification_from_website_domain():
    """Identification of brand via mandatory corporate website or email domain."""
    ocr = _create_dummy_ocr([
        ("PREMIUM WHEY PROTEIN ISOLATE", (50, 50, 500, 120)),
        ("For feedback visit www.myprotein.co.in", (50, 300, 450, 340)),
        ("Net Weight: 1 kg", (50, 360, 250, 390)),
    ])
    result = IdentificationService.identify(ocr)

    assert result.brand_name == "MYPROTEIN"
    assert "Whey Protein" in result.product_name
    assert result.identification_status == IdentificationStatus.IDENTIFIED
    assert any("Website domain" in e for e in result.identification_evidence)


def test_brand_identification_from_consumer_care_email():
    """Identification of brand via Rule 6(1)(g) consumer care email domain."""
    ocr = _create_dummy_ocr([
        ("ROASTED PEANUT BUTTER CRUNCHY", (50, 50, 500, 120)),
        ("Consumer Care: care@pintola.in or call 1800-123-456", (50, 400, 600, 440)),
    ])
    result = IdentificationService.identify(ocr)

    assert result.brand_name == "PINTOLA"
    assert result.product_name == "Peanut Butter"
    assert result.identification_status == IdentificationStatus.IDENTIFIED
    assert any("Consumer care email" in e for e in result.identification_evidence)


def test_strict_manufacturer_separation_no_brand_hallucination():
    """Legal Rule 6(1)(a) requires manufacturer. It must NOT be converted to brand without evidence."""
    ocr = _create_dummy_ocr([
        ("Manufactured by: Sri Balaji Food & Agro Products Pvt Ltd", (50, 200, 600, 250)),
        ("Plot 42, Industrial Area, Phase II, Bengaluru", (50, 260, 550, 290)),
        ("IODIZED SALT", (50, 50, 300, 100)),
    ])
    result = IdentificationService.identify(ocr)

    assert result.manufacturer_name == "Sri Balaji Food & Agro Products Pvt Ltd"
    assert result.brand_name is None  # Never falsely set brand to manufacturer
    assert result.product_name == "Iodized Salt"
    assert result.identification_status == IdentificationStatus.PARTIALLY_IDENTIFIED


def test_unidentified_scan_honest_reporting():
    """Unidentifiable OCR noise honestly returns UNIDENTIFIED with zero hallucination."""
    ocr = _create_dummy_ocr([
        ("BATCH NO: X992-B", (50, 50, 200, 80)),
        ("12/26 EXP", (50, 90, 150, 120)),
        ("KEEP IN COOL PLACE", (50, 130, 250, 160)),
    ])
    result = IdentificationService.identify(ocr)

    assert result.brand_name is None
    assert result.product_name is None
    assert result.manufacturer_name is None
    assert result.identification_status == IdentificationStatus.UNIDENTIFIED
    assert result.confidence == 0.0


# =========================================================================
# 2. AI Extraction Service & On-Demand Fallback Tests
# =========================================================================

@pytest.mark.anyio
async def test_ai_extraction_service_identify_package():
    """Verify on-demand AI identification service returns typed IdentificationResult."""
    fake_ai_response = {
        "brand_name": "AMUL",
        "product_name": "Pasteurised Butter",
        "manufacturer_name": "Gujarat Cooperative Milk Marketing Federation Ltd",
        "brand_confidence": 0.98,
        "product_confidence": 0.96,
        "manufacturer_confidence": 0.95,
        "evidence": ["Visible Amul girl logo and header text", "Product text 'Pasteurised Butter'"],
    }
    provider = FakeAiProvider(stub_id_response=fake_ai_response)
    service = AiExtractionService(provider=provider)

    res = await service.identify_package(b"fake_image_bytes", ocr_context="Amul Butter")
    assert res.brand_name == "AMUL"
    assert res.product_name == "Pasteurised Butter"
    assert res.manufacturer_name == "Gujarat Cooperative Milk Marketing Federation Ltd"
    assert res.identification_status == IdentificationStatus.IDENTIFIED
    assert res.identification_method == IdentificationMethod.AI_ASSISTED
    assert res.confidence == 0.98
    assert len(res.identification_evidence) == 2


# =========================================================================
# 3. Database Persistence & Update Scan Identification Tests
# =========================================================================

def test_database_update_scan_identification():
    """Updating identification in DB persists new identity fields without altering compliance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(storage_dir=Path(tmpdir))
        db = DatabaseService(settings)

        scan_id = str(uuid4())
        evidence = OCRTextEvidence(
            text="MRP Rs 100",
            confidence=0.9,
            bbox=(10, 10, 100, 50),
            engine=OCREngine.GOOGLE_MLKIT,
            source=OCRSource.EDGE,
            source_image_id="test",
        )
        scan_resp = ScanCreateResponse(
            scan_id=scan_id,
            status=ScanStatus.RECEIVED,
            width=1000,
            height=1000,
            created_at=datetime.now(timezone.utc),
            ocr=OCRResult(status=OcrStatus.COMPLETED, items=[evidence], source_image_id="t", source_width=1000, source_height=1000),
            extraction=ExtractionResult(status=ExtractionStatus.COMPLETED, fields=[]),
            compliance=ComplianceResult(status=ComplianceStatus.COMPLIANT, findings=[]),
        )
        db.save_scan(scan_resp)

        # Before update: UNIDENTIFIED
        initial_summary = db.get_scan_summary(scan_id)
        assert initial_summary is not None
        assert initial_summary.brand_name is None
        assert initial_summary.identification_status == "UNIDENTIFIED"
        assert initial_summary.status == "compliant"

        # Update identification via AI fallback
        updated = db.update_scan_identification(
            scan_id=scan_id,
            brand_name="TATA",
            product_name="Salt",
            manufacturer_name="Tata Consumer Products Ltd",
            identification_status="IDENTIFIED",
            identification_method="AI_ASSISTED",
            identification_confidence=0.95,
            identification_evidence=["TATA logo identified via Gemini Vision"],
        )
        assert updated is not None
        assert updated["brand_name"] == "TATA"
        assert updated["product_name"] == "Salt"
        assert updated["manufacturer_name"] == "Tata Consumer Products Ltd"
        assert updated["identification_status"] == "IDENTIFIED"
        assert updated["identification_method"] == "AI_ASSISTED"

        # Verify summary reflects changes and compliance remains intact
        updated_summary = db.get_scan_summary(scan_id)
        assert updated_summary.brand_name == "TATA"
        assert updated_summary.manufacturer_name == "Tata Consumer Products Ltd"
        assert updated_summary.identification_method == "AI_ASSISTED"
        assert updated_summary.status == "compliant"  # Compliance untouched


# =========================================================================
# 4. End-to-End API Route Tests: POST /api/scans/{scan_id}/identify
# =========================================================================

def test_api_identify_scan_endpoint_success():
    """POST /api/scans/{scan_id}/identify runs AI identification and returns updated ScanSummaryItem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)
        settings = Settings(storage_dir=storage_path)

        # Setup fake AI provider
        fake_ai = FakeAiProvider(
            stub_id_response={
                "brand_name": "FORTUNE",
                "product_name": "Mustard Oil",
                "manufacturer_name": "Adani Wilmar Limited",
                "brand_confidence": 0.94,
                "product_confidence": 0.92,
                "manufacturer_confidence": 0.91,
                "evidence": ["Fortune sun logo visible", "Label text 'Kachi Ghani Mustard Oil'"],
            }
        )
        ai_service = AiExtractionService(provider=fake_ai)

        scan_service = ScanService(
            settings=settings,
            ai_extraction_service=ai_service,
        )

        app = create_app()
        from app.api.routes.scans import get_scan_service
        app.dependency_overrides[get_scan_service] = lambda: scan_service

        client = TestClient(app)

        # 1. Create a dummy scan with a real image file on disk
        scan_id = str(uuid4())
        scans_dir = storage_path / "scans"
        scans_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (200, 200), color="yellow")
        img_path = scans_dir / f"{scan_id}.jpg"
        img.save(img_path, format="JPEG")

        evidence = OCRTextEvidence(
            text="Net Qty 1L",
            confidence=0.9,
            bbox=(10, 10, 100, 50),
            engine=OCREngine.GOOGLE_MLKIT,
            source=OCRSource.EDGE,
            source_image_id="test",
        )
        scan_resp = ScanCreateResponse(
            scan_id=scan_id,
            status=ScanStatus.RECEIVED,
            width=200,
            height=200,
            created_at=datetime.now(timezone.utc),
            ocr=OCRResult(status=OcrStatus.COMPLETED, items=[evidence], source_image_id="t", source_width=200, source_height=200),
            extraction=ExtractionResult(status=ExtractionStatus.COMPLETED, fields=[]),
            compliance=ComplianceResult(status=ComplianceStatus.COMPLIANT, findings=[]),
        )
        scan_service._store_scan_state(scan_resp)
        scan_service._db_service.save_scan(scan_resp, image_path=str(img_path))

        # 2. Call POST /api/scans/{scan_id}/identify
        resp = client.post(f"/api/scans/{scan_id}/identify")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scan_id"] == scan_id
        assert data["brand_name"] == "FORTUNE"
        assert data["product_name"] == "Mustard Oil"
        assert data["manufacturer_name"] == "Adani Wilmar Limited"
        assert data["identification_status"] == "IDENTIFIED"
        assert data["identification_method"] == "AI_ASSISTED"
        assert data["identification_confidence"] == 0.94
        assert len(data["identification_evidence"]) == 2


def test_api_identify_scan_not_found():
    """POST /api/scans/{scan_id}/identify returns 404 for unknown scan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app()
        client = TestClient(app)

        resp = client.post("/api/scans/non-existent-scan-id/identify")
        assert resp.status_code == 404
