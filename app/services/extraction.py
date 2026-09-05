"""Structured field extraction from OCR evidence."""

import re
from typing import Any

from app.schemas.scans import (
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    OCRResult,
    OCRTextEvidence,
)


def normalize_date_text(text: str) -> str:
    """Normalize common dot-matrix OCR misrecognitions in numeric date strings."""
    # Fix trailing E, B, S in year (e.g., 2E -> 28, 2B -> 28)
    text = re.sub(r'(\d{1,2}[\./\-])(\d{1,2}[\./\-])(\d{1,2})[EBbSs]', r'\g<1>\g<2>\g<3>8', text)

    def replace_dot_matrix(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace("ü", "0").replace("Ü", "0")
        s = s.replace("O", "0").replace("o", "0")
        s = s.replace("S", "8").replace("s", "8")
        s = s.replace("B", "8")
        return s

    return re.sub(
        r"[üÜOo0-9SsBb]{1,2}[\./\-][üÜOo0-9SsBb]{1,2}[\./\-][üÜOo0-9SsBb]{2,4}",
        replace_dot_matrix,
        text,
    )


def normalize_quantity_text(text: str) -> str:
    """Normalize common OCR misrecognitions for units in metric quantities."""
    text = re.sub(r"(\d+)\s*gm\b", r"\1 g", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*kgs?\b", r"\1 kg", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*mls?\b", r"\1 ml", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*ltrs?\b", r"\1 l", text, flags=re.IGNORECASE)
    return text


def normalize_mrp_text(text: str) -> str:
    """Normalize common dot-matrix OCR misrecognitions for MRP and prices."""
    # Dot matrix 'E' at the start of a 2-4 digit price (e.g. E20 -> 620, E99 -> 699)
    text = re.sub(r'\bE(\d{2,4})\b', r'6\1', text)
    # Colon or semicolon instead of slash-dash: 620: -> 620/-
    text = re.sub(r'(\d+)\s*[:;]\s*(?=[E\d\.]|\()', r'\1/- ', text)
    return text


class ExtractionService:
    """Extracts structured facts deterministically from OCR evidence."""

    def __init__(self) -> None:
        self._patterns = {
            "mrp": re.compile(
                r"M\.?R\.?P\.?\s*(?:₹|Rs\.?)?\s*[:=：\.]?\s*(?:₹|Rs\.?)?\s*(\d+(?:\.\d{1,2})?)(?:\s*/\-|\s*\.\-)?(?:\s*\(.*?\))?", 
                re.IGNORECASE
            ),
            "net_quantity": re.compile(
                r"(?:Net\s*(?:Quantity|Qty|Wt|Weight|Vol|Content)|Weight|Vol|Quantity|Qty|Net)\s*[:=：\.]?\s*(\d+(?:\.\d+)?)\s*(g|gm|kg|ml|l|oz|9)\b",
                re.IGNORECASE,
            ),
            "consumer_care_phone": re.compile(
                r"(?:Consumer\s*Care|Customer\s*Care|CR\s*Details|Toll\s*Free|Phone|Care|Contact)\s*(?:[:=：\.]\s*Call|[:=：\.\n])*\s*(?:[:=：\.]\s*)?(?:\+?91|:?91|0)?[\-\s]?(\d{4}[\-\s]?\d{3}[\-\s]?\d{3,4}|\d{5}[\-\s]?\d{5}|\d{10}|\d{11})",
                re.IGNORECASE,
            ),
            "manufacturing_date": re.compile(
                r"(?:Mfg|Mfd|Mfr|Manufactured)\.?\s*(?:Date)?\s*[:=：\.]?\s*(\d{1,2}[\./\-\s]+(?:[a-zA-Z]{3,}|[0-9]{1,2})[\./\-\s]+\d{2,4}|\d{1,2}[\./\-\s]+\d{2,4})",
                re.IGNORECASE,
            ),
            "packing_date": re.compile(
                r"(?:Packed|Pkd)\.?\s*(?:Date)?\s*[:=：\.]?\s*(\d{1,2}[\./\-\s]+(?:[a-zA-Z]{3,}|[0-9]{1,2})[\./\-\s]+\d{2,4}|\d{1,2}[\./\-\s]+\d{2,4})",
                re.IGNORECASE,
            ),
        }

    def _search_pattern(self, field_name: str, text: str) -> re.Match | None:
        """Search text against field pattern with appropriate domain normalizations."""
        pattern = self._patterns[field_name]
        if field_name == "mrp":
            return pattern.search(normalize_mrp_text(text))
        if field_name in ("manufacturing_date", "packing_date"):
            return pattern.search(normalize_date_text(text))
        if field_name == "net_quantity":
            return pattern.search(normalize_quantity_text(text))
        return pattern.search(text)

    def _are_spatially_adjacent(self, item1: OCRTextEvidence, item2: OCRTextEvidence) -> bool:
        """Check if two OCR detections are geometrically nearby (same line or consecutive lines)."""
        b1, b2 = item1.bbox, item2.bbox
        h1 = b1[3] - b1[1]
        h2 = b2[3] - b2[1]
        max_h = max(h1, h2)
        if max_h <= 0:
            return True

        c_y1 = (b1[1] + b1[3]) / 2.0
        c_y2 = (b2[1] + b2[3]) / 2.0
        dy_center = abs(c_y1 - c_y2)

        dx_gap = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        dy_gap = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))

        # Case A: Same line (horizontal adjacency)
        if dy_center <= max_h * 1.5 and dx_gap <= max_h * 6.0:
            return True

        # Case B: Consecutive lines (vertical key-value pair)
        if dy_gap <= max_h * 2.0 and dx_gap <= max_h * 4.0:
            return True

        return False

    def _merge_evidence(self, item1: OCRTextEvidence, item2: OCRTextEvidence) -> OCRTextEvidence | None:
        """Merge two OCRTextEvidence items into one bounding box if spatially adjacent."""
        if not self._are_spatially_adjacent(item1, item2):
            return None

        left = min(item1.bbox[0], item2.bbox[0])
        top = min(item1.bbox[1], item2.bbox[1])
        right = max(item1.bbox[2], item2.bbox[2])
        bottom = max(item1.bbox[3], item2.bbox[3])

        merged_elements = list(item1.elements or []) + list(item2.elements or [])
        
        return OCRTextEvidence(
            text=f"{item1.text} {item2.text}",
            confidence=min(item1.confidence, item2.confidence),
            bbox=(left, top, right, bottom),
            engine=item1.engine,
            source=item1.source,
            source_image_id=item1.source_image_id,
            elements=merged_elements,
        )

    def _resolve_evidence_bbox(
        self,
        source_item: OCRTextEvidence,
        match: re.Match,
    ) -> OCRTextEvidence:
        """Scope evidence bounding box to exact contributing elements when available."""
        if not source_item.elements:
            return source_item

        match_start = match.start()
        match_end = match.end()

        contributing = []
        current_offset = 0
        for elem in source_item.elements:
            elem_text = elem.text.strip()
            if not elem_text:
                continue
            idx = source_item.text.find(elem_text, current_offset)
            if idx != -1:
                elem_start = idx
                elem_end = idx + len(elem_text)
                current_offset = elem_end
                if max(elem_start, match_start) < min(elem_end, match_end):
                    contributing.append(elem)

        if not contributing:
            return source_item

        left = min(e.bbox[0] for e in contributing)
        top = min(e.bbox[1] for e in contributing)
        right = max(e.bbox[2] for e in contributing)
        bottom = max(e.bbox[3] for e in contributing)
        conf = min(e.confidence for e in contributing)

        return OCRTextEvidence(
            text=match.group(0),
            confidence=conf,
            bbox=(left, top, right, bottom),
            engine=source_item.engine,
            source=source_item.source,
            source_image_id=source_item.source_image_id,
            elements=contributing,
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
                    raw_unit = match.group(2).lower()
                    unit = "g" if raw_unit == "9" else raw_unit
                    normalized_val = f"{value} {unit}"
                elif field_name == "consumer_care_phone":
                    value = match.group(1).strip()
                    normalized_val = value
                elif field_name in ("manufacturing_date", "packing_date"):
                    value = match.group(1).strip(".:;- ")
                    normalized_val = re.sub(r"[\.\-\s]+", "/", value)
                else:
                    value = match.group(1).strip()
                    normalized_val = str(value)
            except ValueError:
                ambiguous = True
                return

            tight_evidence = self._resolve_evidence_bbox(source_item, match)

            if normalized_val not in found_values[field_name]:
                found_values[field_name].add(normalized_val)
                fields.append(
                    ExtractedField(
                        field_name=field_name,
                        value=value,
                        unit=unit,
                        confidence=tight_evidence.confidence,
                        source_evidence=tight_evidence,
                    )
                )

        # Pass 1: Check single items
        for item in ocr_result.items:
            for field_name in self._patterns:
                match = self._search_pattern(field_name, item.text)
                if match:
                    process_match(field_name, match, item)

        # Pass 2: Check adjacent pairs (lookahead) to merge split key/values
        for i in range(len(ocr_result.items) - 1):
            item = ocr_result.items[i]
            next_item = ocr_result.items[i+1]
            merged = self._merge_evidence(item, next_item)
            if merged is None:
                continue
            for field_name in self._patterns:
                match = self._search_pattern(field_name, merged.text)
                if match:
                    process_match(field_name, match, merged)

        # Pass 3: Check two-column table row alignment with 1-to-1 best vertical distance matching
        pass3_candidates: list[dict[str, Any]] = []
        for i, item_left in enumerate(ocr_result.items):
            b1 = item_left.bbox
            h1 = b1[3] - b1[1]
            if h1 <= 0:
                continue
            c_y1 = (b1[1] + b1[3]) / 2.0

            for j, item_right in enumerate(ocr_result.items):
                if i == j:
                    continue
                b2 = item_right.bbox
                # Must be positioned horizontally to the right
                if b2[0] < b1[0]:
                    continue
                h2 = b2[3] - b2[1]
                if h2 <= 0:
                    continue
                max_h = max(h1, h2)
                c_y2 = (b2[1] + b2[3]) / 2.0

                # Check vertical overlap or close vertical center
                has_v_overlap = min(b1[3], b2[3]) > max(b1[1], b2[1])
                dy_center = abs(c_y1 - c_y2)
                if not (has_v_overlap or dy_center <= max_h * 1.5):
                    continue

                # Horizontal gap must be reasonable
                dx_gap = max(0.0, b2[0] - b1[2])
                if dx_gap > max_h * 6.0:
                    continue

                # Split into lines for fine-grained table row alignment
                lines_left = [l.strip() for l in item_left.text.split("\n") if l.strip()]
                lines_right = [l.strip() for l in item_right.text.split("\n") if l.strip()]

                if lines_left and lines_right:
                    line_h1 = h1 / len(lines_left)
                    line_h2 = h2 / len(lines_right)
                    for l_idx, l_text in enumerate(lines_left):
                        y1_c = b1[1] + (l_idx + 0.5) * line_h1
                        for r_idx, r_text in enumerate(lines_right):
                            y2_c = b2[1] + (r_idx + 0.5) * line_h2
                            dy = abs(y1_c - y2_c)
                            # Row-level vertical centers must align (allowing for realistic package curve/tilt)
                            if dy > max(line_h1, line_h2) * 1.25:
                                continue

                            row_text = f"{l_text} {r_text}"
                            for field_name in self._patterns:
                                match = self._search_pattern(field_name, row_text)
                                if match:
                                    y1_t = b1[1] + (l_idx * line_h1)
                                    y1_b = b1[1] + ((l_idx + 1) * line_h1)
                                    y2_t = b2[1] + (r_idx * line_h2)
                                    y2_b = b2[1] + ((r_idx + 1) * line_h2)
                                    row_bbox = (
                                        min(b1[0], b2[0]),
                                        round(min(y1_t, y2_t), 1),
                                        max(b1[2], b2[2]),
                                        round(max(y1_b, y2_b), 1),
                                    )
                                    row_elements = list(item_left.elements or []) + list(item_right.elements or [])
                                    row_item = OCRTextEvidence(
                                        text=row_text,
                                        confidence=min(item_left.confidence, item_right.confidence),
                                        bbox=row_bbox,
                                        engine=item_left.engine,
                                        source=item_left.source,
                                        source_image_id=item_left.source_image_id,
                                        elements=row_elements,
                                    )
                                    pass3_candidates.append({
                                        "left_key": (i, l_idx),
                                        "right_key": (j, r_idx),
                                        "field_name": field_name,
                                        "dy": dy,
                                        "match": match,
                                        "row_item": row_item,
                                    })

        # Sort candidate table matches by vertical alignment distance (closest first)
        pass3_candidates.sort(key=lambda c: c["dy"])
        used_left_by_field: dict[str, set[tuple[int, int]]] = {k: set() for k in self._patterns}
        used_right_by_field: dict[str, set[tuple[int, int]]] = {k: set() for k in self._patterns}

        for cand in pass3_candidates:
            f = cand["field_name"]
            l_key = cand["left_key"]
            r_key = cand["right_key"]
            # 1-to-1 matching: each left label line and right value line can be paired at most once per field
            if l_key in used_left_by_field[f] or r_key in used_right_by_field[f]:
                continue
            used_left_by_field[f].add(l_key)
            used_right_by_field[f].add(r_key)
            process_match(f, cand["match"], cand["row_item"])

        for values in found_values.values():
            if len(values) > 1:
                ambiguous = True

        status = ExtractionStatus.COMPLETED
        if ambiguous or ocr_result.status == "manual_review_required":
            status = ExtractionStatus.MANUAL_REVIEW_REQUIRED

        return ExtractionResult(status=status, fields=fields)
