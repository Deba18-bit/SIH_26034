"""Pydantic schemas for scan history and dashboard statistics."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class ScanSummaryItem(BaseModel):
    """Compact summary of a scan for tabular history listings."""

    scan_id: str
    client_scan_id: str | None = None
    officer_id: str = "OFFICER-DEFAULT"
    created_at: datetime
    status: str
    sync_source: str = "realtime"
    width: int
    height: int
    mrp: float | None = None
    net_quantity: str | None = None
    manufacturing_date: str | None = None
    consumer_care: str | None = None
    product_name: str | None = None
    brand_name: str | None = None
    manufacturer_name: str | None = None
    identification_status: str = "UNIDENTIFIED"
    identification_method: str = "DETERMINISTIC"
    identification_confidence: float = 0.0
    identification_evidence: list[str] = Field(default_factory=list)
    ocr_items_count: int = 0
    compliant_rules_count: int = 0
    review_rules_count: int = 0
    violation_rules_count: int = 0
    inspector_decision: str | None = None
    inspector_notes: str | None = None
    inspector_decided_at: datetime | None = None
    inspector_id: str | None = None


class ScanHistoryResponse(BaseModel):
    """Paginated list of historical inspection scans."""

    items: list[ScanSummaryItem]
    total: int
    limit: int
    offset: int


class ScanStatsResponse(BaseModel):
    """Aggregate statistics across all recorded inspections."""

    total_scans: int
    compliant_count: int
    manual_review_count: int
    violation_count: int
    pending_fallback_count: int
    offline_synced_count: int
