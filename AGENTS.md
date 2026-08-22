# SIH 26034 — AI-Powered Legal Metrology Compliance System

## 1. Project Goal

Build a software system for SIH Problem Statement 26034:

"Software System to check compliance of Packaged Commodities under Legal Metrology (Packaged Commodities) Rules, 2011 by scanning products, images and labels."

The system should analyze package images and product information, extract relevant declarations, validate them against an approved machine-readable compliance rule set, identify possible non-compliances, and generate evidence-backed compliance reports.

The system must distinguish between:
- Automatically verified
- Automatically detected but requiring human verification
- Not detectable reliably from available evidence

Do not claim that an AI prediction is a legally authoritative determination.

---

## 2. Core Architecture

The planned pipeline is:

Package Image
→ Image Quality Check
→ Image Preprocessing
→ OCR
→ OCR Text + Bounding Boxes
→ Information Extraction
→ Product/Package Context
→ Compliance Rule Engine
→ Compliance Findings
→ Evidence
→ Report/API
→ Frontend

Keep these stages modular.

Do not combine the entire pipeline into one large function.

---

## 3. Technology Stack

### Backend
- Python
- FastAPI
- Pydantic
- PostgreSQL

### AI / Computer Vision
- OCR
- OpenCV
- NLP / information extraction
- LLM only where deterministic methods are insufficient

### Frontend
- React
- TypeScript
- Tailwind CSS

### Testing
- pytest
- API tests
- unit tests for compliance rules
- extraction tests

Use open-source components where practical.

---

## 4. Legal Compliance Rules

This is a legal/compliance-oriented application.

NEVER invent, guess, or hallucinate a Legal Metrology requirement.

Legal requirements must come from:
- Official Department of Consumer Affairs sources
- Legal Metrology (Packaged Commodities) Rules, 2011
- Official notified amendments applicable to the project

The approved compliance rules must be stored separately from application logic.

Do not hard-code legal requirements directly inside FastAPI routes.

Every compliance rule should have:
- rule_id
- legal_reference
- requirement
- applicability
- validation_type
- severity
- effective date where applicable
- evidence requirement

Draft amendments must not be treated as current law.

---

## 5. Compliance Engine

The rule engine must be deterministic wherever possible.

Prefer:

OCR / extraction
→ normalized structured data
→ deterministic rule evaluation

over:

Image
→ LLM
→ "compliant"

LLMs may assist with:
- semantic extraction
- classification
- ambiguous text interpretation
- explanations

LLMs must NOT be the sole authority for legal compliance decisions.

Every violation should contain evidence whenever possible.

Example evidence:

{
  "text": "MRP ₹99",
  "confidence": 0.97,
  "bbox": [x1, y1, x2, y2]
}

---

## 6. Canonical Data Model

Extract package information into structured data.

Potential fields include:

- product name
- generic/common name
- manufacturer
- packer
- importer
- address
- country of origin
- net quantity
- MRP
- unit sale price
- manufacture/packing date
- best-before/use-by where applicable
- consumer-care details
- dimensions where applicable

Do not assume every field is applicable to every commodity.

Applicability must be determined by the compliance rules.

---

## 7. Evidence and Explainability

Every important extracted field should retain, where possible:

- normalized value
- original OCR text
- confidence
- bounding box
- source image
- extraction method

Every compliance finding should explain:

1. What was checked
2. Which rule was applied
3. What evidence was found
4. Why the system considers it compliant/non-compliant
5. Whether human verification is recommended

---

## 8. API Design

Keep API routes thin.

Business logic belongs in services.

Use a structure similar to:

app/
├── api/
├── services/
├── schemas/
├── models/
├── core/
└── main.py

Do not place OCR, extraction, CV, or compliance logic directly inside API route handlers.

---

## 9. Development Strategy

Build incrementally.

Milestones:

1. Project setup
2. Image upload and validation
3. OCR pipeline
4. Structured information extraction
5. Compliance rule engine
6. Evidence generation
7. Computer vision checks
8. Database/history
9. Frontend integration
10. PDF/report generation
11. Testing
12. Demo hardening

Do NOT implement the entire application in one step.

Complete and test each milestone before moving to the next.

---

## 10. Coding Standards

- Prefer clear, maintainable Python.
- Use type hints.
- Use Pydantic schemas for API contracts.
- Use environment variables for secrets/configuration.
- Never commit API keys or credentials.
- Avoid unnecessary dependencies.
- Avoid premature abstraction.
- Keep functions focused.
- Add tests for important business logic.
- Do not rewrite working code without a reason.
- Preserve existing behavior when modifying a module.

---

## 11. Git Discipline

Make small, meaningful commits.

Suggested commit style:

feat: add package image upload
feat: add OCR service
feat: add declaration extraction
feat: add compliance rule engine
test: add MRP compliance tests
fix: improve OCR preprocessing

Do not make unrelated changes in the same commit.

---

## 12. AI Agent Behavior

Before making significant architectural changes:

1. Inspect the existing repository.
2. Explain the proposed change briefly.
3. Identify affected files.
4. Implement the smallest reasonable change.
5. Run relevant tests.
6. Report what changed and whether tests passed.

Do not create unnecessary files.

Do not install large dependencies without explaining why they are needed.

Do not replace working implementations merely because another approach is available.

---

## 13. Current Priority

The immediate goal is NOT to build the entire application.

The first milestone is:

Package Image
→ Image Validation
→ OCR
→ Text + Bounding Boxes
→ Structured JSON

Only after this works reliably should the project proceed to compliance-rule implementation.

---

## 14. Human Review

This system is an AI-assisted compliance screening tool.

When image quality, OCR confidence, legal applicability, or visual measurement is uncertain, return:

`MANUAL_REVIEW_REQUIRED`

Do not manufacture certainty.

---

## 15. Important Project Principle

Build a credible working system rather than a collection of AI buzzwords.

Prioritize:

Correctness
→ Explainability
→ Reliability
→ Testability
→ Demo quality
→ Advanced AI features