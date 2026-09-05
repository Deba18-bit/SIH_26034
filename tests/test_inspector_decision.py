"""Tests for Inspector Decision Workflow (Confirm Compliant & Confirm Violation)."""

from io import BytesIO
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.api.routes.scans import get_scan_service
from app.core.config import Settings
from app.main import app
from app.services.ocr import OcrService, RawOcrDetection
from app.services.scans import ScanService


class StubOcrEngine:
    def extract(self, image_path: Path) -> list[RawOcrDetection]:
        # Only MRP detected, missing net_quantity, mfg date, consumer care -> triggers manual review
        return [RawOcrDetection("MRP Rs. 99", 0.94, (20, 30, 180, 60))]


@pytest.fixture(autouse=True)
def scan_storage(tmp_path: Path) -> Path:
    settings = Settings(storage_dir=tmp_path)
    app.dependency_overrides[get_scan_service] = lambda: ScanService(
        settings, ocr_service=OcrService(engine=StubOcrEngine())
    )
    yield tmp_path
    app.dependency_overrides.clear()


def make_label_image() -> bytes:
    output = BytesIO()
    image = Image.new("RGB", (640, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((30, 40), "MRP Rs. 99", fill=(0, 0, 0))
    image.save(output, format="JPEG")
    return output.getvalue()


def test_inspector_can_confirm_compliant():
    client = TestClient(app)
    # 1. Upload scan that has review items
    upload_res = client.post(
        "/api/scans",
        files={"image": ("sample.jpg", make_label_image(), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    scan_id = upload_res.json()["scan_id"]
    initial_status = upload_res.json()["compliance"]["status"]
    assert initial_status == "manual_review_required"

    # Check stats before decision
    stats_pre = client.get("/api/scans/stats").json()
    assert stats_pre["manual_review_count"] >= 1

    # 2. Inspector confirms compliant
    decision_res = client.post(
        f"/api/scans/{scan_id}/decision",
        json={
            "decision": "compliant",
            "officer_id": "OFFICER-007",
            "notes": "Verified declaration on back panel personally.",
        },
    )
    assert decision_res.status_code == 200
    data = decision_res.json()
    assert data["compliance"]["status"] == "compliant"
    assert data["inspector_decision"] is not None
    assert data["inspector_decision"]["decision"] == "compliant"
    assert data["inspector_decision"]["officer_id"] == "OFFICER-007"
    assert data["inspector_decision"]["notes"] == "Verified declaration on back panel personally."

    # 3. Verify history and stats updated
    history_res = client.get("/api/scans/history?status=compliant")
    assert history_res.status_code == 200
    items = history_res.json()["items"]
    matched = [it for it in items if it["scan_id"] == scan_id]
    assert len(matched) == 1
    assert matched[0]["status"] == "compliant"
    assert matched[0]["review_rules_count"] == 0
    assert matched[0]["inspector_decision"] == "compliant"
    assert matched[0]["inspector_id"] == "OFFICER-007"

    # Review list should not contain this scan
    review_res = client.get("/api/scans/history?status=manual_review_required")
    assert not any(it["scan_id"] == scan_id for it in review_res.json()["items"])


def test_inspector_can_confirm_violation():
    client = TestClient(app)
    # 1. Upload scan
    upload_res = client.post(
        "/api/scans",
        files={"image": ("sample.jpg", make_label_image(), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    scan_id = upload_res.json()["scan_id"]

    # 2. Inspector confirms violation
    decision_res = client.post(
        f"/api/scans/{scan_id}/decision",
        json={
            "decision": "violation",
            "officer_id": "OFFICER-101",
            "notes": "Mandatory net quantity declaration missing across entire package.",
        },
    )
    assert decision_res.status_code == 200
    data = decision_res.json()
    assert data["compliance"]["status"] == "violation"
    assert data["inspector_decision"]["decision"] == "violation"
    assert data["inspector_decision"]["officer_id"] == "OFFICER-101"

    # 3. Verify in history query under violation
    history_res = client.get("/api/scans/history?status=violation")
    assert history_res.status_code == 200
    items = history_res.json()["items"]
    matched = [it for it in items if it["scan_id"] == scan_id]
    assert len(matched) == 1
    assert matched[0]["status"] == "violation"
    assert matched[0]["violation_rules_count"] >= 1
    assert matched[0]["review_rules_count"] == 0
    assert matched[0]["inspector_decision"] == "violation"


def test_inspector_decision_invalid_scan_404():
    client = TestClient(app)
    res = client.post(
        "/api/scans/non-existent-scan-id/decision",
        json={"decision": "compliant", "officer_id": "OFFICER-001"},
    )
    assert res.status_code == 404


def test_inspector_decision_invalid_payload_422():
    client = TestClient(app)
    res = client.post(
        "/api/scans/some-id/decision",
        json={"decision": "invalid_decision_type"},
    )
    assert res.status_code == 422

