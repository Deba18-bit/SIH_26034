"""Unit tests for OCR provenance and debugging layer."""

from app.schemas.scans import (
    ComplianceFinding,
    ComplianceResult,
    ComplianceStatus,
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
    OCREngine,
    OCRResult,
    OCRSource,
    OCRTextEvidence,
    OcrStatus,
)
from app.services.provenance import build_scan_provenance, format_terminal_trace


def test_provenance_deterministic_success():
    evidence = OCRTextEvidence(
        text="MRP Rs. 99",
        confidence=0.95,
        bbox=(10.0, 20.0, 100.0, 50.0),
        engine=OCREngine.GOOGLE_MLKIT,
        source=OCRSource.EDGE,
        source_image_id="scan:1:edge"
    )
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[evidence],
        source_image_id="scan:1:edge",
        source_width=1000,
        source_height=1000
    )
    field = ExtractedField(
        field_name="mrp",
        value=99.0,
        confidence=0.95,
        source_evidence=evidence
    )
    det_extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[field]
    )
    compliance = ComplianceResult(
        status=ComplianceStatus.COMPLIANT,
        findings=[
            ComplianceFinding(
                rule_id="LMPC-6-1-e",
                status=ComplianceStatus.COMPLIANT,
                message="MRP verified",
                legal_reference="Rule 6(1)(e)",
                evidence=[field]
            )
        ]
    )

    provenance = build_scan_provenance(
        ocr=ocr,
        deterministic_extraction=det_extraction,
        final_extraction=det_extraction,
        compliance=compliance,
        fallback_triggered=False,
        fallback_reasons=[],
        ai_telemetry={}
    )

    mrp_trace = next(f for f in provenance.fields if f.field_name == "mrp")
    assert mrp_trace.initial_ocr.engine == "google_mlkit"
    assert mrp_trace.initial_ocr.raw_text == "MRP Rs. 99"
    assert mrp_trace.deterministic_extraction.status == "SUCCESS"
    assert mrp_trace.gemini_fallback.triggered is False
    assert mrp_trace.evidence_merge.selected_source == "google_mlkit"
    assert mrp_trace.compliance.status == "COMPLIANT"
    assert mrp_trace.frontend_evidence.displayed_from == "google_mlkit"

    output = format_terminal_trace(provenance)
    assert "OCR PROVENANCE TRACE" in output
    assert "FIELD: MRP" in output
    assert "Initial Detection" in output
    assert "Deterministic Extraction" in output
    assert "Gemini Vision" in output
    assert "Evidence Merge" in output
    assert "Compliance" in output
    assert "Frontend Evidence" in output
    assert "SUMMARY" in output


def test_provenance_gemini_recovery():
    raw_evidence = OCRTextEvidence(
        text="Net Wt",
        confidence=0.8,
        bbox=(50.0, 60.0, 120.0, 80.0),
        engine=OCREngine.GOOGLE_MLKIT,
        source=OCRSource.EDGE,
        source_image_id="scan:2:edge"
    )
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[raw_evidence],
        source_image_id="scan:2:edge",
        source_width=1000,
        source_height=1000
    )
    det_extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[]
    )
    gemini_evidence = OCRTextEvidence(
        text="500g",
        confidence=0.9,
        bbox=(55.0, 62.0, 150.0, 85.0),
        engine=OCREngine.GEMINI_VISION,
        source=OCRSource.VISION,
        source_image_id="scan:2:edge"
    )
    gemini_field = ExtractedField(
        field_name="net_quantity",
        value=500.0,
        unit="g",
        confidence=0.9,
        source_evidence=gemini_evidence
    )
    final_extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[gemini_field]
    )
    compliance = ComplianceResult(
        status=ComplianceStatus.COMPLIANT,
        findings=[
            ComplianceFinding(
                rule_id="LMPC-6-1-c",
                status=ComplianceStatus.COMPLIANT,
                message="Net quantity verified",
                legal_reference="Rule 6(1)(c)",
                evidence=[gemini_field]
            )
        ]
    )

    ai_telemetry = {
        "triggered": True,
        "reasons": ["net_quantity missing after deterministic extraction"],
        "raw_ai_fields": [
            {
                "field_name": "net_quantity",
                "value": "500",
                "unit": "g",
                "ai_semantic_confidence": 0.95
            }
        ],
        "recovered_field_names": ["net_quantity"]
    }

    provenance = build_scan_provenance(
        ocr=ocr,
        deterministic_extraction=det_extraction,
        final_extraction=final_extraction,
        compliance=compliance,
        fallback_triggered=True,
        fallback_reasons=ai_telemetry["reasons"],
        ai_telemetry=ai_telemetry
    )

    net_trace = next(f for f in provenance.fields if f.field_name == "net_quantity")
    assert net_trace.deterministic_extraction.status == "FAILED"
    assert net_trace.gemini_fallback.triggered is True
    assert net_trace.gemini_fallback.value == "500 g"
    assert net_trace.evidence_merge.selected_source == "gemini_vision"
    assert net_trace.frontend_evidence.displayed_from == "gemini_vision"

    assert provenance.summary["recovered_by_gemini"] == 1
    assert provenance.summary["detected_by_initial_ocr"] == 0

    output = format_terminal_trace(provenance)
    assert "Recovered by Gemini       : 1" in output
    assert "Gemini Fallback Triggered : YES" in output
