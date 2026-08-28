"""Tests for the compliance rule engine."""

from app.schemas.scans import ExtractionResult, ExtractionStatus, ExtractedField, OCRTextEvidence, ComplianceStatus, ComplianceFinding
from app.services.compliance import ComplianceService

def make_evidence(field_name: str, value: str | float) -> ExtractedField:
    ocr = OCRTextEvidence(
        text=str(value),
        confidence=0.95,
        bbox=(0, 0, 10, 10),
        engine="paddleocr",
        source="processed",
        source_image_id="scan:test",
        extraction_method="paddleocr_pp_ocr"
    )
    return ExtractedField(
        field_name=field_name,
        value=value,
        confidence=0.95,
        source_evidence=ocr
    )

def test_compliance_all_present():
    service = ComplianceService()
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            make_evidence("mrp", 150.0),
            make_evidence("net_quantity", 500.0),
            make_evidence("consumer_care_phone", "1800-123-456"),
            make_evidence("manufacturing_date", "12/2023")
        ]
    )
    result = service.evaluate(extraction)
    
    assert result.status == ComplianceStatus.COMPLIANT
    assert len(result.findings) == 4
    for finding in result.findings:
        assert finding.status == ComplianceStatus.COMPLIANT
        assert len(finding.evidence) >= 1

def test_compliance_missing_mrp_triggers_manual_review():
    service = ComplianceService()
    # Missing MRP
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            make_evidence("net_quantity", 500.0),
            make_evidence("consumer_care_phone", "1800-123-456"),
            make_evidence("manufacturing_date", "12/2023")
        ]
    )
    result = service.evaluate(extraction)
    
    assert result.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    mrp_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-e")
    assert mrp_finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    assert len(mrp_finding.evidence) == 0

def test_compliance_ambiguous_mrp_triggers_manual_review():
    service = ComplianceService()
    extraction = ExtractionResult(
        status=ExtractionStatus.MANUAL_REVIEW_REQUIRED,
        fields=[
            make_evidence("mrp", 150.0),
            make_evidence("mrp", 200.0), # Conflicting
            make_evidence("net_quantity", 500.0),
            make_evidence("consumer_care_phone", "1800-123-456"),
            make_evidence("manufacturing_date", "12/2023")
        ]
    )
    result = service.evaluate(extraction)
    
    assert result.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    mrp_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-e")
    assert mrp_finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    assert len(mrp_finding.evidence) == 2
    
    # Even though other fields were fine, if extraction status was MANUAL_REVIEW_REQUIRED,
    # the validators propagate it for the fields that are present.
    # Wait, in the validator, if extraction.status == MANUAL_REVIEW_REQUIRED, it flags all present fields as MANUAL_REVIEW_REQUIRED.
    net_qty_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-c")
    assert net_qty_finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED

def test_compliance_failed_extraction_triggers_manual_review():
    service = ComplianceService()
    extraction = ExtractionResult(
        status=ExtractionStatus.FAILED,
        fields=[]
    )
    result = service.evaluate(extraction)
    assert result.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    for finding in result.findings:
        assert finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED

def test_compliance_not_applicable():
    # Create a custom rule and validator to test NOT_APPLICABLE
    from app.rules.definitions import RuleDefinition
    rule = RuleDefinition(
        rule_id="DUMMY-NA",
        title="Dummy NA",
        legal_reference="Dummy",
        applicability="None",
        description="Dummy",
        validation_type="dummy",
        severity="low",
        effective_date=None,
        evidence_requirement="None"
    )
    class NAValidator:
        def evaluate(self, rule, extraction):
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.NOT_APPLICABLE,
                message="Not applicable",
                legal_reference=rule.legal_reference,
                evidence=[]
            )
    
    finding = NAValidator().evaluate(rule, ExtractionResult(status=ExtractionStatus.COMPLETED, fields=[]))
    assert finding.status == ComplianceStatus.NOT_APPLICABLE

def test_compliance_violation():
    # Create a custom rule and validator to test VIOLATION
    from app.rules.definitions import RuleDefinition
    rule = RuleDefinition(
        rule_id="DUMMY-VIOL",
        title="Dummy Viol",
        legal_reference="Dummy",
        applicability="All",
        description="Dummy",
        validation_type="dummy",
        severity="high",
        effective_date=None,
        evidence_requirement="None"
    )
    class ViolValidator:
        def evaluate(self, rule, extraction):
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.VIOLATION,
                message="Explicit violation",
                legal_reference=rule.legal_reference,
                evidence=[]
            )
    
    finding = ViolValidator().evaluate(rule, ExtractionResult(status=ExtractionStatus.COMPLETED, fields=[]))
    assert finding.status == ComplianceStatus.VIOLATION

def test_integration_ocr_to_compliance():
    from app.services.extraction import ExtractionService
    from app.schemas.scans import OCRResult, OcrStatus
    
    ocr = OCRResult(
        status=OcrStatus.COMPLETED,
        items=[
            OCRTextEvidence(
                text="MRP Rs. 150.00",
                confidence=0.98,
                bbox=(0, 0, 10, 10),
                engine="paddleocr",
                source="processed",
                source_image_id="scan:test",
                extraction_method="paddleocr_pp_ocr"
            )
        ],
        source_image_id="scan:test",
        source_width=100,
        source_height=100
    )
    
    extraction_service = ExtractionService()
    extraction = extraction_service.extract(ocr)
    
    compliance_service = ComplianceService()
    compliance = compliance_service.evaluate(extraction)
    
    # We only provided MRP, so others should be missing (MANUAL_REVIEW_REQUIRED)
    assert compliance.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    
    mrp_finding = next(f for f in compliance.findings if f.rule_id == "LMPC-6-1-e")
    assert mrp_finding.status == ComplianceStatus.COMPLIANT
    assert len(mrp_finding.evidence) == 1
    assert mrp_finding.evidence[0].field_name == "mrp"
    assert mrp_finding.evidence[0].value == 150.0
    
    net_qty_finding = next(f for f in compliance.findings if f.rule_id == "LMPC-6-1-c")
    assert net_qty_finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED

def test_low_confidence_mrp_triggers_only_mrp_manual_review():
    service = ComplianceService()
    mrp_field = make_evidence("mrp", 150.0)
    mrp_field.confidence = 0.45  # Low confidence
    
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            mrp_field,
            make_evidence("net_quantity", 500.0),
            make_evidence("consumer_care_phone", "1800-123-456"),
            make_evidence("manufacturing_date", "12/2023")
        ]
    )
    result = service.evaluate(extraction)
    
    assert result.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    
    mrp_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-e")
    assert mrp_finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    
    net_qty_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-c")
    assert net_qty_finding.status == ComplianceStatus.COMPLIANT

def test_high_confidence_evidence_remains_compliant_when_another_is_low():
    service = ComplianceService()
    date_field = make_evidence("manufacturing_date", "12/2023")
    date_field.confidence = 0.40  # Low confidence
    
    extraction = ExtractionResult(
        status=ExtractionStatus.COMPLETED,
        fields=[
            make_evidence("mrp", 150.0),
            make_evidence("net_quantity", 500.0),
            make_evidence("consumer_care_phone", "1800-123-456"),
            date_field
        ]
    )
    result = service.evaluate(extraction)
    
    # Overall status should still be manual review because one rule failed (date)
    assert result.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    
    date_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-d")
    assert date_finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED
    
    mrp_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-e")
    assert mrp_finding.status == ComplianceStatus.COMPLIANT
    
    net_qty_finding = next(f for f in result.findings if f.rule_id == "LMPC-6-1-c")
    assert net_qty_finding.status == ComplianceStatus.COMPLIANT

