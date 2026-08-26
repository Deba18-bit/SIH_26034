"""Registry of implemented compliance rules."""

from app.rules.definitions import RuleDefinition, RuleValidator
from app.rules.validators import PresenceValidator, EitherPresenceValidator

# Note: The following rules represent an initial verified presence-check subset, 
# not the complete Legal Metrology compliance system.

# Rule 6(1)(e): MRP
MRP_RULE = RuleDefinition(
    rule_id="LMPC-6-1-e",
    title="Maximum Retail Price Declaration",
    legal_reference="Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(e)",
    applicability="All packaged commodities intended for retail sale",
    description="Requires declaration of the retail sale price of the package.",
    validation_type="presence_check",
    severity="high",
    effective_date="2011-04-01",
    evidence_requirement="Explicit mention of MRP with numerical price."
)

# Rule 6(1)(c): Net Quantity
NET_QTY_RULE = RuleDefinition(
    rule_id="LMPC-6-1-c",
    title="Net Quantity Declaration",
    legal_reference="Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c)",
    applicability="All packaged commodities intended for retail sale",
    description="Requires declaration of net quantity in terms of standard unit of weight or measure.",
    validation_type="presence_check",
    severity="high",
    effective_date="2011-04-01",
    evidence_requirement="Explicit mention of net quantity with unit."
)

# Rule 6(1)(g): Consumer Care
CONSUMER_CARE_RULE = RuleDefinition(
    rule_id="LMPC-6-1-g",
    title="Consumer Care Information",
    legal_reference="Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(g)",
    applicability="All packaged commodities intended for retail sale",
    description="Requires name, address, telephone number, e-mail address for consumer complaints.",
    validation_type="presence_check",
    severity="high",
    effective_date="2011-04-01",
    evidence_requirement="Explicit mention of consumer care phone number or contact details."
)

# Rule 6(1)(d): Manufacture or Packing Date
DATE_RULE = RuleDefinition(
    rule_id="LMPC-6-1-d",
    title="Manufacture or Packing Date",
    legal_reference="Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d)",
    applicability="All packaged commodities intended for retail sale",
    description="Requires declaration of the month and year in which the commodity is manufactured or pre-packed or imported.",
    validation_type="presence_check",
    severity="high",
    effective_date="2011-04-01",
    evidence_requirement="Explicit mention of manufacturing or packing date."
)

# Note: Additional complex legal rules (Unit Sale Price, font-size compliance, manufacturer-address parsing, 
# product classification) are explicitly deferred for later phases to preserve deterministic safety.

RULE_REGISTRY: list[tuple[RuleDefinition, RuleValidator]] = [
    (MRP_RULE, PresenceValidator("mrp")),
    (NET_QTY_RULE, PresenceValidator("net_quantity")),
    (CONSUMER_CARE_RULE, PresenceValidator("consumer_care_phone")),
    (DATE_RULE, EitherPresenceValidator(["manufacturing_date", "packing_date"], "manufacture or packing date")),
]
