"""Validation and local storage for package-image scans."""

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.schemas.scans import ScanCreateResponse, ScanStatus

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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def create_scan(self, image: UploadFile) -> ScanCreateResponse:
        """Validate, persist, and describe one uploaded package image."""
        validated_image = await self._validate_image(image)
        scan_id = generate_scan_id()
        self._store_original(scan_id, validated_image)
        created_at = datetime.now(timezone.utc)

        return ScanCreateResponse(
            scan_id=scan_id,
            status=ScanStatus.RECEIVED,
            filename=validated_image.filename,
            content_type=validated_image.content_type,
            file_size=validated_image.file_size,
            width=validated_image.width,
            height=validated_image.height,
            created_at=created_at,
        )

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
        storage_directory = self._settings.storage_dir / "scans"
        storage_directory.mkdir(parents=True, exist_ok=True)
        (storage_directory / f"{scan_id}{image.suffix}").write_bytes(image.content)
