"""Structured field extraction from OCR evidence."""

import re

from app.schemas.scans import (
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    OCRResult,
)


class ExtractionService:
    """Extracts structured facts deterministically from OCR evidence."""

    def __init__(self) -> None:
        self._patterns = {
            "mrp": re.compile(r"MRP\s*:?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE),
            "net_quantity": re.compile(
                r"(?:Net\s*(?:Quantity|Qty|Wt|Weight|Vol)|Weight|Vol|Quantity|Qty)\s*:?\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz)",
                re.IGNORECASE,
            ),
            "consumer_care_phone": re.compile(
                r"(?:Consumer\s*Care|Customer\s*Care|CR\s*Details|Toll\s*Free|Phone)\s*(?::\s*Call|:|Call)?\s*(\d{4}[\-\s]?\d{3}[\-\s]?\d{3,4}|\d{10})",
                re.IGNORECASE,
            ),
            "manufacturing_date": re.compile(
                r"(?:Mfg|Mfd)\s*:?\s*(\d{1,2}[/\-\s]+(?:[a-zA-Z]{3,}|[0-9]{1,2})[/\-\s]+\d{2,4}|\d{1,2}[/\-\s]+\d{2,4})",
                re.IGNORECASE,
            ),
            "packing_date": re.compile(
                r"(?:Packed|Pkd)\s*:?\s*(\d{1,2}[/\-\s]+(?:[a-zA-Z]{3,}|[0-9]{1,2})[/\-\s]+\d{2,4}|\d{1,2}[/\-\s]+\d{2,4})",
                re.IGNORECASE,
            ),
        }

    def extract(self, ocr_result: OCRResult) -> ExtractionResult:
        """Parse OCR items into structured facts."""
        if ocr_result.status == "failed":
            return ExtractionResult(status=ExtractionStatus.FAILED, fields=[])

        fields: list[ExtractedField] = []
        ambiguous = False
        
        # Track found normalized string values per field to detect conflicts
        found_values: dict[str, set[str]] = {key: set() for key in self._patterns}

        for item in ocr_result.items:
            for field_name, pattern in self._patterns.items():
                match = pattern.search(item.text)
                if match:
                    unit = None
                    try:
                        if field_name == "mrp":
                            value = float(match.group(1))
                            normalized_val = f"{value:.2f}"
                        elif field_name == "net_quantity":
                            value = float(match.group(1))
                            unit = match.group(2).lower()
                            normalized_val = f"{value} {unit}"
                        elif field_name == "consumer_care_phone":
                            value = match.group(1).strip()
                            normalized_val = value
                        elif field_name in ("manufacturing_date", "packing_date"):
                            value = match.group(1).strip()
                            normalized_val = value
                        else:
                            value = match.group(1).strip()
                            normalized_val = str(value)
                    except ValueError:
                        ambiguous = True
                        continue

                    found_values[field_name].add(normalized_val)
                    fields.append(
                        ExtractedField(
                            field_name=field_name,
                            value=value,
                            unit=unit,
                            confidence=item.confidence,
                            source_evidence=item,
                        )
                    )

        for values in found_values.values():
            if len(values) > 1:
                ambiguous = True

        status = ExtractionStatus.COMPLETED
        if ambiguous or ocr_result.status == "manual_review_required":
            status = ExtractionStatus.MANUAL_REVIEW_REQUIRED

        return ExtractionResult(status=status, fields=fields)
