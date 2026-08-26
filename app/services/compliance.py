"""Service for evaluating compliance rules against extracted facts."""

from app.schemas.scans import ExtractionResult, ComplianceResult, ComplianceStatus
from app.rules.registry import RULE_REGISTRY

class ComplianceService:
    """Evaluates rules against extracted package data."""
    
    def evaluate(self, extraction: ExtractionResult) -> ComplianceResult:
        """Run all registered rules against the provided extraction."""
        findings = []
        overall_status = ComplianceStatus.COMPLIANT
        
        for rule_def, validator in RULE_REGISTRY:
            finding = validator.evaluate(rule_def, extraction)
            findings.append(finding)
            
            # Aggregate status
            if finding.status == ComplianceStatus.VIOLATION:
                overall_status = ComplianceStatus.VIOLATION
            elif finding.status == ComplianceStatus.MANUAL_REVIEW_REQUIRED and overall_status == ComplianceStatus.COMPLIANT:
                overall_status = ComplianceStatus.MANUAL_REVIEW_REQUIRED
                
        return ComplianceResult(
            status=overall_status,
            findings=findings
        )
