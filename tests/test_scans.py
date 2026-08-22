"""Tests for package-image ingestion."""

import asyncio
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from PIL import Image

from app.api.routes.scans import get_scan_service
from app.core.config import Settings
from app.main import app
from app.services.scans import ScanService, generate_scan_id


@pytest.fixture(autouse=True)
def scan_storage(tmp_path: Path) -> Path:
    """Use isolated local storage for every scan test."""
    app.dependency_overrides[get_scan_service] = lambda: ScanService(Settings(storage_dir=tmp_path))
    yield tmp_path
    app.dependency_overrides.clear()


def make_image_bytes(
    image_format: str = "PNG", size: tuple[int, int] = (640, 480)
) -> bytes:
    """Create a valid in-memory image fixture."""
    output = BytesIO()
    Image.new("RGB", size, color="white").save(output, format=image_format)
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
    }
