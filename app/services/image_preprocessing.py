"""Conservative preprocessing that preserves the original evidence image."""

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings


@dataclass(frozen=True)
class ProcessedImage:
    """OCR-oriented derivative data ready for isolated storage."""

    content: bytes
    width: int
    height: int


class ImagePreprocessingService:
    """Create a restrained grayscale derivative without changing the original."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def preprocess(self, content: bytes) -> ProcessedImage:
        """Apply EXIF orientation, grayscale, contrast normalization, and optional denoising."""
        oriented_content = self._apply_orientation(content)
        decoded_image = cv2.imdecode(
            np.frombuffer(oriented_content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded_image is None:
            raise ValueError("OpenCV could not decode the image for preprocessing.")

        max_dim = 768
        h, w = decoded_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            decoded_image = cv2.resize(decoded_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        grayscale = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
        normalized = cv2.createCLAHE(
            clipLimit=self._settings.preprocessing_contrast_clip_limit,
            tileGridSize=(8, 8),
        ).apply(grayscale)
        if self._settings.preprocessing_denoise:
            normalized = cv2.fastNlMeansDenoising(normalized, None, h=7, templateWindowSize=7)

        encoded, derivative = cv2.imencode(".png", normalized)
        if not encoded:
            raise ValueError("Could not encode the processed image.")
        height, width = normalized.shape
        return ProcessedImage(content=derivative.tobytes(), width=width, height=height)

    @staticmethod
    def _apply_orientation(content: bytes) -> bytes:
        """Use Pillow's EXIF handling before OpenCV performs pixel operations."""
        try:
            with Image.open(BytesIO(content)) as image:
                oriented_image = ImageOps.exif_transpose(image).convert("RGB")
                output = BytesIO()
                oriented_image.save(output, format="PNG")
                return output.getvalue()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
            raise ValueError("Could not apply image orientation.") from error
