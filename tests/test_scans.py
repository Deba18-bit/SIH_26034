"""Tests for package-image ingestion."""

import asyncio
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.api.routes.scans import get_scan_service
from app.core.config import Settings
from app.main import app
from app.services.ocr import OcrService, RawOcrDetection
from app.services.scans import ScanService, generate_scan_id


class StubOcrEngine:
    """Model-free OCR boundary for scan API tests."""

    def extract(self, image_path: Path) -> list[RawOcrDetection]:
        return [RawOcrDetection("MRP Rs. 99", 0.94, (20, 30, 180, 60))]


@pytest.fixture(autouse=True)
def scan_storage(tmp_path: Path) -> Path:
    """Use isolated local storage for every scan test."""
    app.dependency_overrides[get_scan_service] = lambda: ScanService(
        Settings(storage_dir=tmp_path), ocr_service=OcrService(engine=StubOcrEngine())
    )
    yield tmp_path
    app.dependency_overrides.clear()


def make_image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (640, 480),
    brightness: int = 128,
    blur_radius: float = 0,
    include_edges: bool = True,
) -> bytes:
    """Create a deterministic image fixture with edges for quality tests."""
    output = BytesIO()
    image = Image.new("L", size, color=brightness)
    drawing = ImageDraw.Draw(image)
    if include_edges:
        for x_coordinate in range(0, size[0], 32):
            drawing.line((x_coordinate, 0, x_coordinate, size[1]), fill=0, width=4)
        for y_coordinate in range(0, size[1], 32):
            drawing.line((0, y_coordinate, size[0], y_coordinate), fill=0, width=4)
    if blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    image.convert("RGB").save(output, format=image_format)
    return output.getvalue()


def make_package_label_image_bytes() -> bytes:
    """Create a synthetic package-style label without relying on a real product image."""
    output = BytesIO()
    image = Image.new("RGB", (640, 480), color="white")
    drawing = ImageDraw.Draw(image)
    for index, line in enumerate(
        ["Manufacturer: Example Co.", "Net Quantity: 500 g", "MRP Rs. 99", "Consumer care: 1800-000-000"]
    ):
        drawing.text((30, 40 + index * 80), line, fill="black")
    image.save(output, format="PNG")
    return output.getvalue()


def post_scan(filename: str, content: bytes, content_type: str) -> httpx.Response:
    """Post an image to the ASGI application."""
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/scans",
                files={"image": (filename, content, content_type)},
            )

    return asyncio.run(request())


def test_valid_image_upload_returns_created_metadata(scan_storage: Path) -> None:
    """A valid package image is accepted and described."""
    response = post_scan("package.png", make_image_bytes(), "image/png")

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "received"
    assert body["filename"] == "package.png"
    assert body["content_type"] == "image/png"
    assert body["width"] == 640
    assert body["height"] == 480
    assert body["file_size"] > 0
    assert (scan_storage / "scans" / f"{body['scan_id']}.png").is_file()
    assert (scan_storage / "scans" / "processed" / f"{body['scan_id']}.png").is_file()


def test_unsupported_file_type_is_rejected() -> None:
    """Non-image uploads receive an unsupported-media-type response."""
    response = post_scan("notes.txt", b"not an image", "text/plain")

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_media_type"


def test_corrupted_image_is_rejected() -> None:
    """A file claiming to be an image must still be decodable."""
    response = post_scan("broken.png", b"not a valid PNG", "image/png")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"


def test_image_below_minimum_dimensions_is_rejected() -> None:
    """Images too small for future OCR are rejected."""
    response = post_scan("small.png", make_image_bytes(size=(100, 100)), "image/png")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "image_dimensions_too_small"


def test_scan_id_generation_returns_uuid() -> None:
    """Generated scan identifiers are opaque UUID values."""
    scan_id = generate_scan_id()

    assert str(UUID(scan_id)) == scan_id


def test_scan_response_has_required_structure() -> None:
    """The public response exposes only the defined scan metadata."""
    response = post_scan("package.png", make_image_bytes(), "image/png")

    assert set(response.json()) == {
        "scan_id",
        "status",
        "filename",
        "content_type",
        "file_size",
        "width",
        "height",
        "created_at",
        "quality",
        "ocr",
        "extraction",
        "compliance",
        "provenance",
        "client_scan_id",
        "officer_id",
        "inspector_decision",
    }


def test_scan_response_includes_structured_ocr_evidence() -> None:
    """The API exposes evidence tied to the processed derivative, not a legal decision."""
    response = post_scan("label.png", make_package_label_image_bytes(), "image/png")

    ocr = response.json()["ocr"]
    assert ocr["status"] == "completed"
    assert ocr["source_image_id"].endswith(":processed")
    assert ocr["items"] == [
        {
            "text": "MRP Rs. 99",
            "confidence": 0.94,
            "bbox": [20.0, 30.0, 180.0, 60.0],
            "engine": "paddleocr",
            "source": "processed",
            "source_image_id": ocr["source_image_id"],
            "elements": [],
        }
    ]


def test_sharp_adequately_exposed_image_passes_quality_assessment() -> None:
    """A detailed, normally exposed image passes every engineering check."""
    response = post_scan("sharp.png", make_image_bytes(), "image/png")

    quality = response.json()["quality"]
    assert response.status_code == 201
    assert quality["status"] == "PASS"
    assert all(check["passed"] for check in quality["checks"].values())


def test_obviously_blurry_image_requires_manual_review() -> None:
    """A strongly blurred image exposes a failed sharpness measurement."""
    response = post_scan("blurry.png", make_image_bytes(blur_radius=12), "image/png")

    quality = response.json()["quality"]
    assert quality["status"] == "MANUAL_REVIEW_REQUIRED"
    assert quality["checks"]["sharpness"]["passed"] is False


def test_dark_image_requires_manual_review() -> None:
    """An underexposed image exposes a failed brightness measurement."""
    response = post_scan(
        "dark.png", make_image_bytes(brightness=10, include_edges=False), "image/png"
    )

    quality = response.json()["quality"]
    assert quality["status"] == "MANUAL_REVIEW_REQUIRED"
    assert quality["checks"]["brightness"]["passed"] is False


def test_bright_image_requires_manual_review() -> None:
    """An overexposed image exposes a failed brightness measurement."""
    response = post_scan(
        "bright.png", make_image_bytes(brightness=245, include_edges=False), "image/png"
    )

    quality = response.json()["quality"]
    assert quality["status"] == "MANUAL_REVIEW_REQUIRED"
    assert quality["checks"]["brightness"]["passed"] is False


def test_processed_derivative_does_not_modify_original(scan_storage: Path) -> None:
    """The OCR derivative is separate and leaves original evidence bytes intact."""
    original_content = make_image_bytes()
    response = post_scan("package.png", original_content, "image/png")

    scan_id = response.json()["scan_id"]
    original_path = scan_storage / "scans" / f"{scan_id}.png"
    processed_path = scan_storage / "scans" / "processed" / f"{scan_id}.png"
    assert original_path.read_bytes() == original_content
    assert processed_path.is_file()


def test_edge_scan_provenance() -> None:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            edge_payload = {
                "engine": "google_mlkit",
                "source_width": 1080,
                "source_height": 1920,
                "items": [
                    {
                        "text": "MRP Rs. 250",
                        "confidence": 0.95,
                        "bbox": [100.0, 200.0, 400.0, 250.0]
                    }
                ]
            }
            return await client.post("/api/scans/edge", json=edge_payload)

    res = asyncio.run(request())
    assert res.status_code == 201
    data = res.json()
    assert "provenance" in data
    assert data["provenance"] is not None
    assert len(data["provenance"]["fields"]) > 0
    mrp_field = next(f for f in data["provenance"]["fields"] if f["field_name"] == "mrp")
    assert mrp_field["initial_ocr"]["engine"] == "google_mlkit"
    assert mrp_field["deterministic_extraction"]["status"] == "SUCCESS"
    assert mrp_field["evidence_merge"]["selected_value"] == "250.0"
