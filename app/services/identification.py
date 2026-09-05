"""Deterministic-first, evidence-based Brand and Product Identification Service.

Evaluates OCR evidence without invoking LLMs (0 tokens by default), strictly
distinguishing between consumer-facing Brand, Commodity/Product Name, and
Corporate Manufacturer.
"""

from enum import Enum
import re
from typing import Any
from pydantic import BaseModel, Field


class IdentificationStatus(str, Enum):
    """Classification status of the package identification."""
    IDENTIFIED = "IDENTIFIED"
    PARTIALLY_IDENTIFIED = "PARTIALLY_IDENTIFIED"
    UNIDENTIFIED = "UNIDENTIFIED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class IdentificationMethod(str, Enum):
    """Method utilized to determine package identity."""
    DETERMINISTIC = "DETERMINISTIC"
    AI_ASSISTED = "AI_ASSISTED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class IdentificationResult(BaseModel):
    """Structured result of package brand and commodity identification."""
    brand_name: str | None = None
    product_name: str | None = None
    manufacturer_name: str | None = None
    identification_status: IdentificationStatus = IdentificationStatus.UNIDENTIFIED
    identification_method: IdentificationMethod = IdentificationMethod.DETERMINISTIC
    confidence: float = 0.0
    identification_evidence: list[str] = Field(default_factory=list)


# Known Indian & Global Packaged Brands
KNOWN_BRANDS = [
    "NAKPRO", "AMUL", "TATA", "NESTLE", "BRITANNIA", "PARLE", "HALDIRAM", "FORTUNE",
    "AASHIRVAAD", "DABUR", "PATANJALI", "MARICO", "CADBURY", "HORLICKS", "BOOST",
    "OPTIMUM NUTRITION", "MUSCLETECH", "MYPROTEIN", "AS-IT-IS", "BIGMUSCLES", "HIMALAYA",
    "MOTHER DAIRY", "KWALITY WALLS", "SAFFOLA", "SUGAR FREE", "SUNFEAST", "MAGGI",
    "LAYS", "KURKURE", "BINGO", "HALDIRAMS", "BIKANO", "EVEREST", "MDH", "CATCH",
    "MTR", "REAL", "TROPICANA", "FROOTI", "MAAZA", "THUMS UP", "COCA-COLA", "PEPSI",
    "SPRITE", "LIMCA", "MIRINDA", "BISLERI", "KINLEY", "AQUAFINA", "DETTOL", "LIFEBUOY",
    "SURF EXCEL", "ARIEL", "TIDE", "COLGATE", "PEPSODENT", "CLOSEUP", "SENSODYNE",
    "DOVE", "LUX", "PEARS", "NIVEA", "HEAD & SHOULDERS", "PANTENE", "SUNSILK", "CLINIC PLUS"
]

# Standard Commodity / Product Categories
KNOWN_COMMODITIES = [
    "Creatine Monohydrate", "Micronized Creatine", "Creatine",
    "Whey Protein Concentrate", "Whey Protein Isolate", "Whey Protein",
    "Plant Protein", "Peanut Butter",
    "Iodized Salt", "Wheat Flour", "Chakki Atta", "Atta", "Basmati Rice", "Rice",
    "Refined Sunflower Oil", "Mustard Oil", "Soyabean Oil", "Refined Oil",
    "Full Cream Milk", "Toned Milk", "Pasteurized Butter", "Table Butter", "Butter", "Paneer",
    "Green Tea", "Instant Coffee", "Tea", "Coffee",
    "Digestive Biscuits", "Cookies", "Biscuits", "Instant Noodles", "Noodles",
    "Potato Chips", "Chips", "Tomato Ketchup", "Ketchup", "Chilli Sauce",
    "Toothpaste", "Washing Powder", "Detergent Bar", "Bath Soap"
]

GENERIC_EMAIL_DOMAINS = {
    "GMAIL", "YAHOO", "HOTMAIL", "OUTLOOK", "SUPPORT", "INFO", "CARE",
    "HELP", "CONTACT", "FEEDBACK", "RED झालIFF", "ICLOUD"
}

GENERIC_WEB_DOMAINS = {
    "GOOGLE", "FACEBOOK", "INSTAGRAM", "TWITTER", "AMAZON", "FLIPKART",
    "APPLE", "YOUTUBE", "LINKEDIN", "WHATSAPP", "PLAY"
}


class IdentificationService:
    """Service to deterministically extract Brand, Product, and Manufacturer."""

    @classmethod
    def identify(
        cls,
        ocr_text: str | Any,
        ocr_items: list[Any] | None = None
    ) -> IdentificationResult:
        """Run deterministic identification cascade in priority order."""
        if hasattr(ocr_text, "items"):
            ocr_items = ocr_text.items
            text = " ".join([it.text for it in ocr_items if hasattr(it, "text") and it.text])
        elif isinstance(ocr_text, str):
            text = ocr_text
        else:
            text = str(ocr_text or "")
        ocr_text = text
        ocr_items = ocr_items or []

        evidence: list[str] = []
        brand: str | None = None
        product: str | None = None
        manufacturer: str | None = None

        # -------------------------------------------------------------
        # A. Priority 1: Corporate Website Domain
        # -------------------------------------------------------------
        web_match = re.search(
            r"(?:https?://)?(?:www\.)([a-zA-Z0-9\-]+)\.(?:com|in|co|org|net|coop)",
            ocr_text,
            re.IGNORECASE
        )
        if web_match:
            dom = web_match.group(1).upper()
            if dom not in GENERIC_WEB_DOMAINS and len(dom) >= 3:
                brand = dom
                evidence.append(f"Website domain '{web_match.group(0)}' indicates Brand '{dom}'")

        # -------------------------------------------------------------
        # B. Priority 2: Mandatory Consumer Care Email (Rule 6(1)(g))
        # -------------------------------------------------------------
        email_match = re.search(
            r"[\w\.-]+@([a-zA-Z0-9\-]+)\.(?:com|in|co|org|net|coop)",
            ocr_text,
            re.IGNORECASE
        )
        if email_match:
            dom = email_match.group(1).upper()
            if dom not in GENERIC_EMAIL_DOMAINS and len(dom) >= 3:
                if not brand:
                    brand = dom
                    evidence.append(f"Consumer care email '{email_match.group(0)}' indicates Brand '{dom}'")
                elif brand == dom:
                    evidence.append(f"Consumer care email '{email_match.group(0)}' reinforces Brand '{dom}'")

        # -------------------------------------------------------------
        # C. Priority 3: Manufacturer / Packer Evidence (Rule 6(1)(a))
        # -------------------------------------------------------------
        mfg_match = re.search(
            r"(?:Mfg\.?\s*(?:&|and)?\s*Mkt\.?\s*by|Manufactured\s*(?:&|and)?\s*Marketed\s*by|Manufactured\s*by|Packed\s*by|Packaged\s*by|Marketed\s*by)\s*[:=：\-]?\s*([A-Za-z0-9\s\.\-&]{3,60}?)(?:\s*(?:Pvt\.?|Private|Ltd\.?|Limited|LLP|Inc\.?|Corp\.?)|,\s*|\n|Plot|Phase|Sector|Sy\.?\s*No|No\.|At\b)",
            ocr_text,
            re.IGNORECASE
        )
        if mfg_match:
            raw_mfg = mfg_match.group(1).strip()
            # Clean leading honorifics
            clean_mfg = re.sub(r"^(?:The|M/s\.?)\s+", "", raw_mfg, flags=re.IGNORECASE).strip()
            if len(clean_mfg) >= 3:
                # Retain full legal name with Ltd / Pvt Ltd if in snippet
                match_end = mfg_match.end(0)
                full_snippet = ocr_text[mfg_match.start(0):min(match_end + 20, len(ocr_text))]
                if re.search(r"\b(?:Pvt\.?\s*Ltd\.?|Private\s*Limited|Limited|Ltd\.?|LLP)\b", full_snippet, re.IGNORECASE):
                    corp_type = re.search(r"\b(?:Pvt\.?\s*Ltd\.?|Private\s*Limited|Limited|Ltd\.?|LLP)\b", full_snippet, re.IGNORECASE).group(0)
                    clean_mfg = f"{clean_mfg} {corp_type}"
                manufacturer = clean_mfg
                evidence.append(f"Manufacturer detected: '{clean_mfg}' (Rule 6(1)(a))")

                # If brand is still missing, check if manufacturer name explicitly matches a known brand
                if not brand:
                    first_token = clean_mfg.split()[0].upper()
                    for kb in KNOWN_BRANDS:
                        if kb == first_token or re.search(r"\b" + re.escape(kb) + r"\b", clean_mfg, re.IGNORECASE):
                            brand = kb
                            evidence.append(f"Brand '{kb}' derived from manufacturer name '{clean_mfg}'")
                            break

        # -------------------------------------------------------------
        # D. Priority 4: Explicit Brand Detection from Dictionary
        # -------------------------------------------------------------
        if not brand:
            for kb in KNOWN_BRANDS:
                if re.search(r"\b" + re.escape(kb) + r"\b", ocr_text, re.IGNORECASE):
                    brand = kb
                    evidence.append(f"Brand '{kb}' detected directly in OCR text")
                    break

        # -------------------------------------------------------------
        # E. Priority 5: Product / Commodity Name Detection (Rule 6(1)(b))
        # -------------------------------------------------------------
        for comm in KNOWN_COMMODITIES:
            if re.search(r"\b" + re.escape(comm) + r"\b", ocr_text, re.IGNORECASE):
                product = comm
                evidence.append(f"Commodity '{comm}' detected directly in OCR text")
                break

        # -------------------------------------------------------------
        # F. Supporting Signal: OCR Visual Prominence (Largest BBox Line)
        # -------------------------------------------------------------
        if not brand and ocr_items:
            # Find largest non-numeric text item that could be a brand header
            candidate = cls._find_prominent_text_candidate(ocr_items)
            if candidate:
                evidence.append(f"Visual prominence candidate: '{candidate}'")
                # If candidate matches any known brand or looks like a title
                for kb in KNOWN_BRANDS:
                    if kb in candidate.upper():
                        brand = kb
                        evidence.append(f"Brand '{kb}' verified from visually prominent block")
                        break

        # -------------------------------------------------------------
        # Status & Confidence Determination
        # -------------------------------------------------------------
        if brand and product:
            status = IdentificationStatus.IDENTIFIED
            confidence = 0.95 if len(evidence) >= 2 else 0.85
        elif brand and not product:
            status = IdentificationStatus.PARTIALLY_IDENTIFIED
            confidence = 0.75
        elif not brand and (product or manufacturer):
            status = IdentificationStatus.PARTIALLY_IDENTIFIED
            confidence = 0.65
        else:
            status = IdentificationStatus.UNIDENTIFIED
            confidence = 0.0

        return IdentificationResult(
            brand_name=brand,
            product_name=product,
            manufacturer_name=manufacturer,
            identification_status=status,
            identification_method=IdentificationMethod.DETERMINISTIC,
            confidence=confidence,
            identification_evidence=evidence
        )

    @classmethod
    def _find_prominent_text_candidate(cls, ocr_items: list[Any]) -> str | None:
        """Find the most visually prominent text block from bounding box heights."""
        best_text: str | None = None
        max_height = 0.0

        for item in ocr_items:
            text = getattr(item, "text", "") or ""
            text_clean = text.strip()
            # Ignore standard numerical / meta lines
            if re.search(r"\b(?:MRP|Mfg|Exp|Batch|Net|Qty|Rs|Phone|Email|Date|Lic)\b", text_clean, re.IGNORECASE):
                continue
            if re.search(r"^\d+[\./\-]", text_clean) or len(text_clean) < 3 or len(text_clean) > 35:
                continue

            bbox = getattr(item, "bbox", None)
            if bbox and len(bbox) == 4:
                height = bbox[3] - bbox[1]
                if height > max_height:
                    max_height = height
                    best_text = text_clean

        return best_text
