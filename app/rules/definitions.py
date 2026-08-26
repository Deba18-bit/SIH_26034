"""Definitions for the compliance rule framework."""

from dataclasses import dataclass
from typing import Protocol

from app.schemas.scans import ExtractionResult, ComplianceFinding

@dataclass(frozen=True)
class RuleDefinition:
    """A verified legal metrology rule."""
    
    rule_id: str
    title: str
    legal_reference: str
    applicability: str
    description: str
    validation_type: str
    severity: str
    effective_date: str | None
    evidence_requirement: str


class RuleValidator(Protocol):
    """Protocol for rule evaluation."""
    
    def evaluate(self, rule: RuleDefinition, extraction: ExtractionResult) -> ComplianceFinding:
        """Evaluate a rule against extracted evidence."""
        ...
