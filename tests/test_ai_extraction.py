"""Tests for the selective AI fallback mechanism."""

import pytest
from app.schemas.scans import (
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    OCRResult,
    OCRTextEvidence,
    OcrStatus
)
from app.services.ai_extraction import AiExtractionService, FakeAiProvider

def make_evidence(text: str, confidence: float, bbox: tuple = (0, 0, 10, 10)) -> OCRTextEvidence:
    return OCRTextEvidence(
        text=text,
        confidence=confidence,
        bbox=bbox,
        engine="paddleocr",
        source="processed",
        source_image_id="scan:test"
    )

def make_extracted_field(field_name: str, value: str, text: str) -> ExtractedField:
    return ExtractedField(
        field_name=field_name,
        value=value,
        confidence=0.99,
        source_evidence=make_evidence(text, 0.99)
    )

@pytest.mark.anyio
async def test_ai_not_called_when_all_fields_present():
    """AI is NOT called when deterministic extraction already succeeds."""
    provider = FakeAiProvider()
    service = AiExtractionService(provider=provider)
    
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            make_extracted_field("mrp", "150.0", "MRP: Rs. 150"),
            make_extracted_field("net_quantity", "250.0 g", "Quantity: 250g"),
            make_extracted_field("consumer_care_phone", "1800-123", "Call: 1800-123"),
            make_extracted_field("manufacturing_date", "15 OCT 2023", "Mfg: 15 OCT 2023")
        ]
    )
    
    needs_fallback, _ = service.should_fallback(extraction)
    assert needs_fallback is False
    
    result = await service.enrich(OCRResult(status=OcrStatus.COMPLETED, items=[], source_image_id="test", source_width=100, source_height=100), extraction)
    
    # Provider should not be called (it returns empty dict by default, but we wouldn't even reach it)
    assert len(result.fields) == 4

@pytest.mark.anyio
async def test_ai_called_when_field_missing():
    """AI fallback is called when a required field is genuinely missing."""
    stub_response = {
        "extracted_fields": [
            {
                "field_name": "mrp",
                "value": 150.0,
                "source_ocr_ids": [0, 1],
                "ai_semantic_confidence": 0.90
            }
        ]
    }
    provider = FakeAiProvider(stub_response)
    service = AiExtractionService(provider=provider)
    
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            make_extracted_field("net_quantity", "250.0 g", "Quantity: 250g"),
            make_extracted_field("consumer_care_phone", "1800-123", "Call: 1800-123"),
            make_extracted_field("manufacturing_date", "15 OCT 2023", "Mfg: 15 OCT 2023")
        ]
    )
    
    needs_fallback, missing = service.should_fallback(extraction)
    assert needs_fallback is True
    assert "mrp" in missing
    
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            make_evidence("MRP", 0.99, (0, 0, 10, 10)),
            make_evidence("150.00", 0.95, (10, 0, 20, 10))
        ],
        source_image_id="test",
        source_width=100,
        source_height=100
    )
    
    result = await service.enrich(ocr, extraction)
    
    # Now should have 4 fields
    assert len(result.fields) == 4
    
    # Check the newly added AI field
    mrp_field = next(f for f in result.fields if f.field_name == "mrp")
    assert mrp_field.value == 150.0
    assert mrp_field.source_evidence.engine == "paddleocr"
    assert mrp_field.source_evidence.text == "MRP 150.00"
    assert mrp_field.source_evidence.bbox == (0, 0, 20, 10)
    assert mrp_field.confidence == 0.90 # min(0.90, min(0.99, 0.95))

@pytest.mark.anyio
async def test_ai_cannot_override_deterministic():
    """High-confidence deterministic evidence cannot be silently overridden by AI."""
    stub_response = {
        "extracted_fields": [
            {
                "field_name": "mrp",
                "value": 999.0, # Incorrect override attempt
                "source_ocr_ids": [0],
                "ai_semantic_confidence": 0.90
            }
        ]
    }
    provider = FakeAiProvider(stub_response)
    service = AiExtractionService(provider=provider)
    
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            make_extracted_field("mrp", "150.0", "MRP: Rs. 150"), # Missing other fields to trigger fallback
        ]
    )
    
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[make_evidence("999", 0.99)],
        source_image_id="test",
        source_width=100,
        source_height=100
    )
    
    result = await service.enrich(ocr, extraction)
    
    mrp_fields = [f for f in result.fields if f.field_name == "mrp"]
    assert len(mrp_fields) == 1
    assert mrp_fields[0].value == "150.0"

@pytest.mark.anyio
async def test_ai_failure_does_not_crash():
    """If AI fails, the deterministic result remains available."""
    class CrashingProvider(FakeAiProvider):
        async def extract_fields(self, prompt_data):
            raise Exception("API down")
            
    service = AiExtractionService(provider=CrashingProvider())
    extraction = ExtractionResult(status=ExtractionStatus.COMPLETED, fields=[])
    ocr = OCRResult(status=OcrStatus.COMPLETED, items=[make_evidence("MRP", 0.99)], source_image_id="test", source_width=10, source_height=10)
    
    # Should safely catch exception and return original extraction
    result = await service.enrich(ocr, extraction)
    assert len(result.fields) == 0

def test_gemini_provider_init():
    from app.services.ai_extraction import GeminiAiProvider
    import pytest
    import os
    
    with pytest.raises(ValueError):
        # Should raise if no key
        if "SIH_GEMINI_API_KEY" in os.environ:
            del os.environ["SIH_GEMINI_API_KEY"]
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        GeminiAiProvider(api_key=None)
        
    provider = GeminiAiProvider(api_key="test-key")
    assert provider.client is not None


@pytest.mark.anyio
async def test_ai_rejects_nutrition_table_for_net_quantity():
    """Verify that AI results referencing nutrition facts (e.g. 100g) are rejected as net_quantity."""
    stub_response = {
        "extracted_fields": [
            {
                "field_name": "net_quantity",
                "value": "100",
                "unit": "g",
                "source_ocr_ids": [1],
                "ai_semantic_confidence": 0.95
            }
        ]
    }
    provider = FakeAiProvider(stub_response)
    service = AiExtractionService(provider=provider)

    extraction = ExtractionResult(status=ExtractionStatus.COMPLETED, fields=[])
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            make_evidence("Nutritional Information", 0.95), # Item 0
            make_evidence("Approximate Value: 100g", 0.95), # Item 1 (nutrition reference)
        ],
        source_image_id="test",
        source_width=1000,
        source_height=1000
    )

    result = await service.enrich(ocr, extraction)
    net_qty_fields = [f for f in result.fields if f.field_name == "net_quantity"]
    # Nutrition table value must NOT be merged as package net quantity
    assert len(net_qty_fields) == 0

