"""Tests for persistent SQLite database service and history endpoints."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

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
from app.services.database import DatabaseService


def _make_dummy_scan_response(
    scan_id: str,
    status: ScanStatus = ScanStatus.RECEIVED,
    comp_status: ComplianceStatus = ComplianceStatus.COMPLIANT,
    mrp: float = 620.0,
    net_qty: str = "100 g",
    client_scan_id: str | None = None,
    officer_id: str = "OFFICER-001",
) -> ScanCreateResponse:
    """Create a minimal valid ScanCreateResponse for DB testing."""
    evidence = OCRTextEvidence(
        text=f"MRP: {mrp}",
        confidence=0.95,
        bbox=(10.0, 10.0, 100.0, 50.0),
        engine=OCREngine.GOOGLE_MLKIT,
        source=OCRSource.EDGE,
        source_image_id="test",
    )
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[evidence],
        source_image_id="test_image_1",
        source_width=1000,
        source_height=1000,
    )
    fields = [
        ExtractedField(field_name="mrp", value=mrp, confidence=0.95, source_evidence=evidence),
        ExtractedField(field_name="net_quantity", value=100.0, unit="g", confidence=0.95, source_evidence=evidence),
        ExtractedField(field_name="manufacturing_date", value="08.08.26", confidence=0.95, source_evidence=evidence),
    ]
    extraction = ExtractionResult(status=ExtractionStatus.COMPLETED, fields=fields)
    compliance = ComplianceResult(status=comp_status, findings=[])

    return ScanCreateResponse(
        scan_id=scan_id,
        status=status,
        width=1000,
        height=1000,
        created_at=datetime.now(timezone.utc),
        ocr=ocr,
        extraction=extraction,
        compliance=compliance,
        client_scan_id=client_scan_id,
        officer_id=officer_id,
    )


def test_database_save_and_retrieve() -> None:
    """A scan record can be saved and retrieved with all fields intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(storage_dir=Path(tmpdir), database_url="sqlite:///:memory:")
        db = DatabaseService(settings)

        scan_id = str(uuid4())
        scan_resp = _make_dummy_scan_response(scan_id, client_scan_id="client-123")

        db.save_scan(scan_resp, client_scan_id="client-123", officer_id="OFFICER-001")

        record = db.get_scan(scan_id)
        assert record is not None
        assert record["scan_id"] == scan_id
        assert record["client_scan_id"] == "client-123"
        assert record["officer_id"] == "OFFICER-001"
        assert record["mrp"] == 620.0
        assert record["manufacturing_date"] == "08.08.26"
        assert record["ocr_data"]["status"] == "completed"

        # Check lookup by client UUID
        client_record = db.get_scan_by_client_id("client-123")
        assert client_record is not None
        assert client_record["scan_id"] == scan_id


def test_database_list_and_stats() -> None:
    """Listing scans supports filtering and stats report aggregates accurately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(storage_dir=Path(tmpdir), database_url="sqlite:///:memory:")
        db = DatabaseService(settings)

        s1 = str(uuid4())
        s2 = str(uuid4())

        db.save_scan(_make_dummy_scan_response(s1, comp_status=ComplianceStatus.COMPLIANT))
        db.save_scan(_make_dummy_scan_response(s2, comp_status=ComplianceStatus.MANUAL_REVIEW_REQUIRED))

        stats = db.get_stats()
        assert stats.total_scans == 2
        assert stats.compliant_count == 1
        assert stats.manual_review_count == 1

        history = db.list_scans(limit=10)
        assert history.total == 2
        assert len(history.items) == 2

        # Filter by status
        filtered = db.list_scans(status="compliant")
        assert filtered.total == 1
        assert filtered.items[0].scan_id == s1


def test_api_history_and_stats_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FastAPI app exposes /api/scans/history, /stats, and /{scan_id}."""
    from app.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    app = create_app()
    client = TestClient(app)

    # Test history endpoint
    resp = client.get("/api/scans/history?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)

    # Test stats endpoint
    stats_resp = client.get("/api/scans/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "total_scans" in stats
    assert "compliant_count" in stats

    # If scans exist, check details endpoint
    if data["items"]:
        scan_id = data["items"][0]["scan_id"]
        detail_resp = client.get(f"/api/scans/{scan_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["scan_id"] == scan_id

    # Non-existent scan returns 404
    missing_resp = client.get("/api/scans/non-existent-scan-id")
    assert missing_resp.status_code == 404
