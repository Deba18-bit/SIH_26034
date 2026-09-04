"""Structured field extraction from OCR evidence."""

import re

from app.schemas.scans import (
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    OCRResult,
    OCRTextEvidence,
)


class ExtractionService:
    """Extracts structured facts deterministically from OCR evidence."""

    def __init__(self) -> None:
        self._patterns = {
            "mrp": re.compile(
                r"M\.?R\.?P\.?\s*[:=：\.]?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{1,2})?)", 
                re.IGNORECASE
            ),
            "net_quantity": re.compile(
                r"(?:Net\s*(?:Quantity|Qty|Wt|Weight|Vol)|Weight|Vol|Quantity|Qty|Net)\s*[:=：\.]?\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz)",
                re.IGNORECASE,
            ),
            "consumer_care_phone": re.compile(
                r"(?:Consumer\s*Care|Customer\s*Care|CR\s*Details|Toll\s*Free|Phone|Care)\s*(?:[:=：\.]\s*Call|[:=：\.]|Call)?\s*(\d{4}[\-\s]?\d{3}[\-\s]?\d{3,4}|\d{10}|\d{11})",
                re.IGNORECASE,
            ),
            "manufacturing_date": re.compile(
                r"(?:Mfg|Mfd|Mfr|Manufactured)\.?\s*(?:Date)?\s*[:=：\.]?\s*(\d{1,2}[/\-\s]+(?:[a-zA-Z]{3,}|[0-9]{1,2})[/\-\s]+\d{2,4}|\d{1,2}[/\-\s]+\d{2,4})",
                re.IGNORECASE,
            ),
            "packing_date": re.compile(
                r"(?:Packed|Pkd)\.?\s*(?:Date)?\s*[:：\.]?\s*(\d{1,2}[/\-\s]+(?:[a-zA-Z]{3,}|[0-9]{1,2})[/\-\s]+\d{2,4}|\d{1,2}[/\-\s]+\d{2,4})",
                re.IGNORECASE,
            ),
        }

    def _merge_evidence(self, item1: OCRTextEvidence, item2: OCRTextEvidence) -> OCRTextEvidence:
        """Merge two OCRTextEvidence items into one bounding box."""
        left = min(item1.bbox[0], item2.bbox[0])
        top = min(item1.bbox[1], item2.bbox[1])
        right = max(item1.bbox[2], item2.bbox[2])
        bottom = max(item1.bbox[3], item2.bbox[3])
        
        return OCRTextEvidence(
            text=f"{item1.text} {item2.text}",
            confidence=min(item1.confidence, item2.confidence),
            bbox=(left, top, right, bottom),
            engine=item1.engine,
            source=item1.source,
            source_image_id=item1.source_image_id,
            extraction_method=item1.extraction_method,
        )

    def extract(self, ocr_result: OCRResult) -> ExtractionResult:
        """Parse OCR items into structured facts."""
        if ocr_result.status == "failed":
            return ExtractionResult(status=ExtractionStatus.FAILED, fields=[])

        fields: list[ExtractedField] = []
        ambiguous = False
        found_values: dict[str, set[str]] = {key: set() for key in self._patterns}

        def process_match(field_name: str, match: re.Match, source_item: OCRTextEvidence) -> None:
            nonlocal ambiguous
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
                return

            if normalized_val not in found_values[field_name]:
                found_values[field_name].add(normalized_val)
                fields.append(
                    ExtractedField(
                        field_name=field_name,
                        value=value,
                        unit=unit,
                        confidence=source_item.confidence,
                        source_evidence=source_item,
                    )
                )

        # Pass 1: Check single items
        for item in ocr_result.items:
            for field_name, pattern in self._patterns.items():
                match = pattern.search(item.text)
                if match:
                    process_match(field_name, match, item)

        # Pass 2: Check adjacent pairs (lookahead) to merge split key/values
        for i in range(len(ocr_result.items) - 1):
            item = ocr_result.items[i]
            next_item = ocr_result.items[i+1]
            merged = self._merge_evidence(item, next_item)
            for field_name, pattern in self._patterns.items():
                match = pattern.search(merged.text)
                if match:
                    process_match(field_name, match, merged)

        for values in found_values.values():
            if len(values) > 1:
                ambiguous = True

        status = ExtractionStatus.COMPLETED
        if ambiguous or ocr_result.status == "manual_review_required":
            status = ExtractionStatus.MANUAL_REVIEW_REQUIRED

        return ExtractionResult(status=status, fields=fields)
