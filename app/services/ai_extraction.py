"""Selective AI fallback for structured extraction."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from google import genai
from pydantic import BaseModel, Field

from app.schemas.scans import (
    ExtractedField,
    ExtractionResult,
    OCRResult,
    OCRTextEvidence,
)


class AiExtractionProvider(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    async def extract_fields(self, prompt_data: dict[str, Any], image_bytes: bytes | None = None) -> dict[str, Any]:
        """Send prompt data to the LLM and return structured JSON."""
        pass


class FakeAiProvider(AiExtractionProvider):
    """A dummy provider for unit tests."""

    def __init__(self, stub_response: dict[str, Any] | None = None) -> None:
        self.stub_response = stub_response or {"extracted_fields": []}

    async def extract_fields(self, prompt_data: dict[str, Any], image_bytes: bytes | None = None) -> dict[str, Any]:
        return self.stub_response



class GeminiAiProvider(AiExtractionProvider):
    """A real AI provider using Google Gemini."""

    class AiExtractedFieldModel(BaseModel):
        field_name: str
        value: str | float
        unit: str | None = None
        source_ocr_ids: list[int] | None = None
        bbox_1000: list[int] | None = Field(default=None, description="[ymin, xmin, ymax, xmax] normalized to 1000")
        ai_semantic_confidence: float = Field(ge=0.0, le=1.0)

    class AiExtractionResponseModel(BaseModel):
        extracted_fields: list["GeminiAiProvider.AiExtractedFieldModel"]

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("SIH_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API key is not set. Cannot use GeminiAiProvider.")
        self.client = genai.Client(api_key=key)

    async def extract_fields(self, prompt_data: dict[str, Any], image_bytes: bytes | None = None) -> dict[str, Any]:
        prompt = (
            "You are a specialized legal metrology AI extraction assistant.\n"
            "Analyze the provided structured OCR text blocks to extract the missing required fields.\n"
            "If an image is provided, you may visually inspect it to find missing fields (like curved or glossy text that OCR missed).\n"
            "Rules:\n"
            "1. Do not invent or guess any values.\n"
            "2. Map every extracted field to the source_ocr_ids if found in the text blocks.\n"
            "3. If found purely from the image (OCR missed it), provide the bounding box in bbox_1000 format [ymin, xmin, ymax, xmax] (normalized 0-1000).\n"
            "4. ai_semantic_confidence should be between 0.0 and 1.0 based on how sure you are of the semantic meaning.\n"
            "Terminology:\n"
            "- mrp: Maximum Retail Price (e.g. MRP: 450.00)\n"
            "- net_quantity: Weight or Volume (e.g. 500g, 1L, Net Wt 924g)\n"
            "- manufacturing_date: The date of manufacture (e.g. Mfg Date 16/05/2026)\n"
            "- packing_date: The date of packaging (e.g. Pkd Date)\n"
            "- consumer_care_phone: A customer support phone number (e.g. Consumer Care: 9821486487, Toll Free)\n\n"
            f"Input Data:\n{json.dumps(prompt_data, indent=2)}"
        )
        
        contents: list[Any] = [prompt]
        if image_bytes:
            contents.append(genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            
        response = await self.client.aio.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=contents,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.AiExtractionResponseModel,
                temperature=0.1,
            )
        )
        
        return json.loads(response.text)

class AiExtractionService:

    """Manages fallback logic and safe merging of AI results."""

    def __init__(self, provider: AiExtractionProvider | None = None) -> None:
        self._provider = provider or FakeAiProvider()
        self._required_fields = {"mrp", "net_quantity", "consumer_care_phone"}

    def should_fallback(self, extraction: ExtractionResult) -> tuple[bool, set[str]]:
        """Determine if AI fallback is needed and which fields to find."""
        if extraction.status == "manual_review_required":
            # If deterministic was ambiguous, we don't try to resolve it with AI.
            # But if we are just missing fields, we proceed.
            pass

        found_fields = {f.field_name for f in extraction.fields}
        missing_fields = self._required_fields - found_fields
        
        has_date = "manufacturing_date" in found_fields or "packing_date" in found_fields
        if not has_date:
            missing_fields.add("manufacturing_date")
            missing_fields.add("packing_date")

        if not missing_fields:
            return False, set()
            
        return True, missing_fields

    async def enrich(self, ocr: OCRResult, extraction: ExtractionResult, image_bytes: bytes | None = None, source_width: int = 1000, source_height: int = 1000) -> ExtractionResult:
        """Call AI for missing fields and merge safely."""
        needs_fallback, missing_fields = self.should_fallback(extraction)
        
        if not needs_fallback:
            return extraction
            
        # 1. Prepare OCR items with IDs
        ocr_item_map = {i: item for i, item in enumerate(ocr.items)}
        available_blocks = [
            {
                "ocr_id": i,
                "text": item.text,
                "confidence": item.confidence,
                "bbox": item.bbox
            }
            for i, item in ocr_item_map.items()
        ]
        
        prompt_data = {
            "missing_fields_to_find": list(missing_fields),
            "available_ocr_blocks": available_blocks
        }
        
        # 2. Call AI
        try:
            ai_response = await self._provider.extract_fields(prompt_data, image_bytes)
        except Exception:
            # If AI crashes, safely return the deterministic result
            return extraction
            
        # 3. Merge AI fields
        new_fields = list(extraction.fields)
        found_deterministic = {f.field_name for f in extraction.fields}
        
        extracted_fields = ai_response.get("extracted_fields", [])
        for ai_field in extracted_fields:
            field_name = ai_field.get("field_name")
            value = ai_field.get("value")
            source_ids = ai_field.get("source_ocr_ids", [])
            bbox_1000 = ai_field.get("bbox_1000")
            ai_confidence = ai_field.get("ai_semantic_confidence", 0.0)
            
            # Strict Priority: Never override a deterministic field
            if field_name in found_deterministic:
                continue
                
            # Compute synthetic bounding box & confidence
            if source_ids:
                sources = [ocr_item_map[i] for i in source_ids if i in ocr_item_map]
                if not sources:
                    continue
                min_x = min(s.bbox[0] for s in sources)
                min_y = min(s.bbox[1] for s in sources)
                max_x = max(s.bbox[2] for s in sources)
                max_y = max(s.bbox[3] for s in sources)
                min_ocr_conf = min(s.confidence for s in sources)
                final_confidence = min(ai_confidence, min_ocr_conf)
                engine = sources[0].engine
                source = sources[0].source
                evidence_text = " ".join(s.text for s in sources)
            elif bbox_1000 and len(bbox_1000) == 4:
                ymin, xmin, ymax, xmax = bbox_1000
                min_x = (xmin / 1000.0) * source_width
                min_y = (ymin / 1000.0) * source_height
                max_x = (xmax / 1000.0) * source_width
                max_y = (ymax / 1000.0) * source_height
                final_confidence = ai_confidence * 0.9  # slight penalty for vision-only box
                engine = "gemini-3.1-flash-lite"
                source = "vision"
                evidence_text = str(value)
            else:
                continue
                
            synthetic_evidence = OCRTextEvidence(
                text=evidence_text,
                confidence=final_confidence,
                bbox=(min_x, min_y, max_x, max_y),
                engine=engine,
                source=source,
                source_image_id=ocr.source_image_id,
                extraction_method="ai_fallback"
            )
            
            new_fields.append(ExtractedField(
                field_name=field_name,
                value=value,
                unit=ai_field.get("unit"),
                confidence=final_confidence,
                source_evidence=synthetic_evidence
            ))
            
        return ExtractionResult(
            status=extraction.status,
            fields=new_fields
        )
