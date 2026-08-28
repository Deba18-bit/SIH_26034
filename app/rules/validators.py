"""Rule evaluation logic."""

from app.rules.definitions import RuleDefinition
from app.schemas.scans import ExtractionResult, ComplianceFinding, ComplianceStatus, ExtractionStatus

class PresenceValidator:
    """Checks for the unambiguous presence of a required extracted field."""
    
    def __init__(self, required_field_name: str) -> None:
        self.required_field_name = required_field_name
        
    def evaluate(self, rule: RuleDefinition, extraction: ExtractionResult) -> ComplianceFinding:
        # Filter fields matching the required field name
        matching_fields = [f for f in extraction.fields if f.field_name == self.required_field_name]
        
        # If OCR extraction itself flagged manual review overall and this field was involved/missing,
        # or if we have multiple conflicting extractions for this field (handled by ExtractionService)
        # Actually, if extraction is FAILED, manual review required.
        if extraction.status == ExtractionStatus.FAILED:
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.MANUAL_REVIEW_REQUIRED,
                message="Extraction failed, cannot verify rule.",
                legal_reference=rule.legal_reference,
                evidence=[]
            )
            
        if not matching_fields:
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.MANUAL_REVIEW_REQUIRED,
                message=f"Missing evidence for {self.required_field_name}. Cannot automatically establish violation without human review.",
                legal_reference=rule.legal_reference,
                evidence=[]
            )
            
        # Check if the field extraction itself is ambiguous (e.g. multiple distinct values)
        # We can see this if matching_fields > 1 and they don't have exactly the same value
        distinct_values = {f.value for f in matching_fields}
        is_low_confidence = any(f.confidence < 0.50 for f in matching_fields)
        if len(distinct_values) > 1 or extraction.status == ExtractionStatus.MANUAL_REVIEW_REQUIRED or is_low_confidence:
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.MANUAL_REVIEW_REQUIRED,
                message=f"Ambiguous or low-confidence evidence for {self.required_field_name}.",
                legal_reference=rule.legal_reference,
                evidence=matching_fields
            )
            
        # If there's exactly 1 distinct value, it's compliant (presence is satisfied)
        return ComplianceFinding(
            rule_id=rule.rule_id,
            status=ComplianceStatus.COMPLIANT,
            message=f"Successfully verified presence of {self.required_field_name}.",
            legal_reference=rule.legal_reference,
            evidence=matching_fields
        )

class EitherPresenceValidator:
    """Checks for the unambiguous presence of at least one of multiple fields."""
    
    def __init__(self, field_names: list[str], display_name: str) -> None:
        self.field_names = field_names
        self.display_name = display_name
        
    def evaluate(self, rule: RuleDefinition, extraction: ExtractionResult) -> ComplianceFinding:
        if extraction.status == ExtractionStatus.FAILED:
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.MANUAL_REVIEW_REQUIRED,
                message="Extraction failed, cannot verify rule.",
                legal_reference=rule.legal_reference,
                evidence=[]
            )

        matching_fields = [f for f in extraction.fields if f.field_name in self.field_names]
        
        if not matching_fields:
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.MANUAL_REVIEW_REQUIRED,
                message=f"Missing evidence for {self.display_name}. Cannot automatically establish violation without human review.",
                legal_reference=rule.legal_reference,
                evidence=[]
            )
            
        # If extraction is flagged for manual review, propagate it.
        is_low_confidence = any(f.confidence < 0.50 for f in matching_fields)
        if extraction.status == ExtractionStatus.MANUAL_REVIEW_REQUIRED or is_low_confidence:
            return ComplianceFinding(
                rule_id=rule.rule_id,
                status=ComplianceStatus.MANUAL_REVIEW_REQUIRED,
                message=f"Ambiguous or low-confidence evidence for {self.display_name}.",
                legal_reference=rule.legal_reference,
                evidence=matching_fields
            )
            
        return ComplianceFinding(
            rule_id=rule.rule_id,
            status=ComplianceStatus.COMPLIANT,
            message=f"Successfully verified presence of {self.display_name}.",
            legal_reference=rule.legal_reference,
            evidence=matching_fields
        )
