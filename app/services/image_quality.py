"""Deterministic image-quality assessment for the future OCR pipeline."""

import cv2
import numpy as np

from app.core.config import Settings
from app.schemas.scans import (
    BrightnessQualityCheck,
    DimensionsQualityCheck,
    ImageQualityAssessment,
    ImageQualityChecks,
    ImageQualityStatus,
    ReadabilityQualityCheck,
    SharpnessQualityCheck,
)


class ImageQualityService:
    """Measure image properties that affect downstream OCR reliability."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def assess(self, content: bytes, width: int, height: int) -> ImageQualityAssessment:
        """Return explainable, deterministic quality measurements."""
        decoded_image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise ValueError("OpenCV could not decode the image.")

        grayscale = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
        sharpness_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        mean_intensity = float(grayscale.mean())
        dimensions_passed = (
            width >= self._settings.min_image_width
            and height >= self._settings.min_image_height
        )
        sharpness_passed = sharpness_variance >= self._settings.min_sharpness_variance
        brightness_passed = (
            self._settings.min_brightness
            <= mean_intensity
            <= self._settings.max_brightness
        )

        checks = ImageQualityChecks(
            dimensions=DimensionsQualityCheck(
                passed=dimensions_passed,
                width=width,
                height=height,
                minimum_width=self._settings.min_image_width,
                minimum_height=self._settings.min_image_height,
            ),
            sharpness=SharpnessQualityCheck(
                passed=sharpness_passed,
                laplacian_variance=sharpness_variance,
                minimum_variance=self._settings.min_sharpness_variance,
            ),
            brightness=BrightnessQualityCheck(
                passed=brightness_passed,
                mean_intensity=mean_intensity,
                minimum_intensity=self._settings.min_brightness,
                maximum_intensity=self._settings.max_brightness,
            ),
            readability=ReadabilityQualityCheck(passed=True, method="opencv_imdecode"),
        )
        messages = self._build_messages(checks)
        return ImageQualityAssessment(
            status=(
                ImageQualityStatus.PASS
                if not messages
                else ImageQualityStatus.MANUAL_REVIEW_REQUIRED
            ),
            checks=checks,
            messages=messages,
        )

    @staticmethod
    def _build_messages(checks: ImageQualityChecks) -> list[str]:
        """Explain which capture conditions should be improved."""
        messages: list[str] = []
        if not checks.dimensions.passed:
            messages.append("Image dimensions are below the configured minimum.")
        if not checks.sharpness.passed:
            messages.append("Image may be too blurry for reliable downstream OCR.")
        if not checks.brightness.passed:
            messages.append("Image brightness is outside the configured capture range.")
        return messages
