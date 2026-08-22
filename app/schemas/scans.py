"""Pydantic schemas for package-image scans."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ScanStatus(str, Enum):
    """Lifecycle state of an ingested scan."""

    RECEIVED = "received"


class ScanCreateResponse(BaseModel):
    """Public metadata returned after a package image is ingested."""

    scan_id: str
    status: ScanStatus
    filename: str
    content_type: str
    file_size: int
    width: int
    height: int
    created_at: datetime
