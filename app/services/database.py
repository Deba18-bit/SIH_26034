"""Database service for storing, indexing, and querying inspection scans.

Supports PostgreSQL with native JSONB (primary) and SQLite (fallback).
"""

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.schemas.history import ScanHistoryResponse, ScanStatsResponse, ScanSummaryItem
from app.schemas.scans import ScanCreateResponse, ScanStatus
from app.services.identification import IdentificationService

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


class DatabaseService:
    """Persistent database service supporting PostgreSQL and SQLite fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._sqlite_path = self._settings.storage_dir / "metri_check.db"
        self._is_postgres = False
        self._check_connection()
        self._init_db()

    def _check_connection(self) -> None:
        """Probe PostgreSQL availability; fallback to SQLite if unreachable."""
        if HAS_PSYCOPG and self._settings.database_url:
            try:
                with psycopg.connect(self._settings.database_url, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                self._is_postgres = True
                logger.info("Connected to PostgreSQL at %s", self._settings.database_url)
                return
            except Exception as e:
                logger.warning("PostgreSQL unreachable (%s). Falling back to SQLite at %s", e, self._sqlite_path)
        self._is_postgres = False

    def _get_pg_connection(self) -> Any:
        """Create a PostgreSQL connection with dict_row factory."""
        return psycopg.connect(self._settings.database_url, row_factory=dict_row)

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        """Create an SQLite connection with WAL mode enabled."""
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._sqlite_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        """Initialize tables and indexes on the active database."""
        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS scans (
                                scan_id TEXT PRIMARY KEY,
                                client_scan_id TEXT,
                                officer_id TEXT NOT NULL DEFAULT 'OFFICER-DEFAULT',
                                created_at TIMESTAMPTZ NOT NULL,
                                updated_at TIMESTAMPTZ NOT NULL,
                                status TEXT NOT NULL,
                                sync_source TEXT NOT NULL DEFAULT 'realtime',
                                image_path TEXT,
                                width INTEGER NOT NULL,
                                height INTEGER NOT NULL,
                                mrp REAL,
                                net_quantity TEXT,
                                manufacturing_date TEXT,
                                consumer_care TEXT,
                                product_name TEXT,
                                brand_name TEXT,
                                ocr_items_count INTEGER NOT NULL DEFAULT 0,
                                compliant_rules_count INTEGER NOT NULL DEFAULT 0,
                                review_rules_count INTEGER NOT NULL DEFAULT 0,
                                violation_rules_count INTEGER NOT NULL DEFAULT 0,
                                ocr_data JSONB NOT NULL,
                                extraction_data JSONB NOT NULL,
                                compliance_data JSONB NOT NULL,
                                provenance_data JSONB
                            );
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS product_name TEXT;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS brand_name TEXT;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS manufacturer_name TEXT;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS identification_status TEXT NOT NULL DEFAULT 'UNIDENTIFIED';
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS identification_method TEXT NOT NULL DEFAULT 'DETERMINISTIC';
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS identification_confidence REAL NOT NULL DEFAULT 0.0;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS identification_evidence JSONB DEFAULT '[]'::jsonb;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS inspector_decision TEXT;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS inspector_notes TEXT;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS inspector_decided_at TIMESTAMPTZ;
                            ALTER TABLE scans ADD COLUMN IF NOT EXISTS inspector_id TEXT;
                            CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans (created_at DESC);
                            CREATE INDEX IF NOT EXISTS idx_scans_status ON scans (status);
                            CREATE INDEX IF NOT EXISTS idx_scans_client_id ON scans (client_scan_id);
                            CREATE INDEX IF NOT EXISTS idx_scans_officer_id ON scans (officer_id);
                            CREATE INDEX IF NOT EXISTS idx_scans_brand ON scans (brand_name);
                            CREATE INDEX IF NOT EXISTS idx_scans_id_status ON scans (identification_status);
                            CREATE INDEX IF NOT EXISTS idx_scans_manufacturer ON scans (manufacturer_name);
                            CREATE INDEX IF NOT EXISTS idx_scans_inspector_decision ON scans (inspector_decision);
                        """)
                    conn.commit()
                self._backfill_legacy_states_pg()
                return
            except Exception as e:
                logger.error("Failed to initialize PostgreSQL: %s. Reverting to SQLite.", e)
                self._is_postgres = False

        # SQLite fallback table creation
        with self._get_sqlite_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY,
                    client_scan_id TEXT,
                    officer_id TEXT NOT NULL DEFAULT 'OFFICER-DEFAULT',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sync_source TEXT NOT NULL DEFAULT 'realtime',
                    image_path TEXT,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    mrp REAL,
                    net_quantity TEXT,
                    manufacturing_date TEXT,
                    consumer_care TEXT,
                    product_name TEXT,
                    brand_name TEXT,
                    manufacturer_name TEXT,
                    identification_status TEXT NOT NULL DEFAULT 'UNIDENTIFIED',
                    identification_method TEXT NOT NULL DEFAULT 'DETERMINISTIC',
                    identification_confidence REAL NOT NULL DEFAULT 0.0,
                    identification_evidence TEXT NOT NULL DEFAULT '[]',
                    ocr_items_count INTEGER NOT NULL DEFAULT 0,
                    compliant_rules_count INTEGER NOT NULL DEFAULT 0,
                    review_rules_count INTEGER NOT NULL DEFAULT 0,
                    violation_rules_count INTEGER NOT NULL DEFAULT 0,
                    ocr_data TEXT NOT NULL,
                    extraction_data TEXT NOT NULL,
                    compliance_data TEXT NOT NULL,
                    provenance_data TEXT
                )
            """)
            for col, col_def in [
                ("product_name", "TEXT"),
                ("brand_name", "TEXT"),
                ("manufacturer_name", "TEXT"),
                ("identification_status", "TEXT NOT NULL DEFAULT 'UNIDENTIFIED'"),
                ("identification_method", "TEXT NOT NULL DEFAULT 'DETERMINISTIC'"),
                ("identification_confidence", "REAL NOT NULL DEFAULT 0.0"),
                ("identification_evidence", "TEXT NOT NULL DEFAULT '[]'"),
                ("inspector_decision", "TEXT"),
                ("inspector_notes", "TEXT"),
                ("inspector_decided_at", "TEXT"),
                ("inspector_id", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE scans ADD COLUMN {col} {col_def}")
                except Exception:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans (created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_status ON scans (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_client_id ON scans (client_scan_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_officer_id ON scans (officer_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_id_status ON scans (identification_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_inspector_decision ON scans (inspector_decision)")
            conn.commit()

        self._backfill_legacy_states_sqlite()

    def _extract_summary_fields(self, scan_response: ScanCreateResponse) -> dict[str, Any]:
        """Extract quick lookup fields from extraction and compliance results."""
        mrp: float | None = None
        net_quantity: str | None = None
        manufacturing_date: str | None = None
        consumer_care: str | None = None

        for field in scan_response.extraction.fields:
            if field.field_name == "mrp":
                try:
                    mrp = float(field.value)
                except (ValueError, TypeError):
                    pass
            elif field.field_name == "net_quantity":
                val = str(field.value)
                if field.unit:
                    val += f" {field.unit}"
                net_quantity = val
            elif field.field_name in ("manufacturing_date", "packing_date"):
                if not manufacturing_date:
                    manufacturing_date = str(field.value)
            elif field.field_name in ("consumer_care_phone", "consumer_care_email"):
                if not consumer_care:
                    consumer_care = str(field.value)

        # Deterministic Brand, Product, and Manufacturer Identification (0 LLM tokens)
        combined_text = " ".join([it.text for it in scan_response.ocr.items])
        id_result = IdentificationService.identify(combined_text, scan_response.ocr.items)

        compliant_count = 0
        review_count = 0
        violation_count = 0

        for finding in scan_response.compliance.findings:
            st = (finding.status.value if hasattr(finding.status, "value") else str(finding.status)).lower()
            if st == "compliant":
                compliant_count += 1
            elif st in ("violation", "non_compliant"):
                violation_count += 1
            else:
                review_count += 1

        return {
            "mrp": mrp,
            "net_quantity": net_quantity,
            "manufacturing_date": manufacturing_date,
            "consumer_care": consumer_care,
            "product_name": id_result.product_name,
            "brand_name": id_result.brand_name,
            "manufacturer_name": id_result.manufacturer_name,
            "identification_status": id_result.identification_status.value,
            "identification_method": id_result.identification_method.value,
            "identification_confidence": id_result.confidence,
            "identification_evidence": id_result.identification_evidence,
            "compliant_rules_count": compliant_count,
            "review_rules_count": review_count,
            "violation_rules_count": violation_count,
        }

    def _backfill_legacy_states_pg(self) -> None:
        """Backfill existing JSON state files into PostgreSQL."""
        states_dir = self._settings.storage_dir / "scans" / "states"
        if not states_dir.exists():
            return
        with self._get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT scan_id FROM scans")
                existing_ids = {row["scan_id"] for row in cur.fetchall()}
                for state_file in states_dir.glob("*.json"):
                    scan_id = state_file.stem
                    if scan_id in existing_ids:
                        continue
                    try:
                        data = json.loads(state_file.read_text("utf-8"))
                        scan_response = ScanCreateResponse.model_validate(data)
                        self._insert_pg(cur, scan_response, sync_source="legacy_file")
                    except Exception:
                        continue
            conn.commit()

    def _backfill_legacy_states_sqlite(self) -> None:
        """Backfill existing JSON state files into SQLite."""
        states_dir = self._settings.storage_dir / "scans" / "states"
        if not states_dir.exists():
            return
        with self._get_sqlite_connection() as conn:
            existing_ids = {row["scan_id"] for row in conn.execute("SELECT scan_id FROM scans").fetchall()}
            for state_file in states_dir.glob("*.json"):
                scan_id = state_file.stem
                if scan_id in existing_ids:
                    continue
                try:
                    data = json.loads(state_file.read_text("utf-8"))
                    scan_response = ScanCreateResponse.model_validate(data)
                    self._insert_sqlite(conn, scan_response, sync_source="legacy_file")
                except Exception:
                    continue
            conn.commit()

    def _insert_pg(
        self,
        cur: Any,
        scan_response: ScanCreateResponse,
        client_scan_id: str | None = None,
        officer_id: str = "OFFICER-DEFAULT",
        sync_source: str = "realtime",
        image_path: str | None = None,
    ) -> None:
        summary = self._extract_summary_fields(scan_response)
        now = datetime.now(timezone.utc)
        created_at = scan_response.created_at or now

        if scan_response.status == ScanStatus.PENDING_FALLBACK:
            status_str = "pending_fallback"
        elif hasattr(scan_response.compliance, "status") and scan_response.compliance.status:
            comp_st = scan_response.compliance.status
            status_str = comp_st.value if hasattr(comp_st, "value") else str(comp_st)
        else:
            status_str = scan_response.status.value if hasattr(scan_response.status, "value") else str(scan_response.status)

        insp_dec = scan_response.inspector_decision
        insp_decision_val = insp_dec.decision.value if insp_dec else None
        insp_notes_val = insp_dec.notes if insp_dec else None
        insp_decided_at_val = insp_dec.decided_at if insp_dec else None
        insp_officer_id_val = insp_dec.officer_id if insp_dec else None

        cur.execute(
            """
            INSERT INTO scans (
                scan_id, client_scan_id, officer_id, created_at, updated_at,
                status, sync_source, image_path, width, height,
                mrp, net_quantity, manufacturing_date, consumer_care,
                product_name, brand_name, manufacturer_name,
                identification_status, identification_method, identification_confidence, identification_evidence,
                ocr_items_count, compliant_rules_count, review_rules_count, violation_rules_count,
                ocr_data, extraction_data, compliance_data, provenance_data,
                inspector_decision, inspector_notes, inspector_decided_at, inspector_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (scan_id) DO UPDATE SET
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at,
                mrp = EXCLUDED.mrp,
                net_quantity = EXCLUDED.net_quantity,
                manufacturing_date = EXCLUDED.manufacturing_date,
                consumer_care = EXCLUDED.consumer_care,
                product_name = EXCLUDED.product_name,
                brand_name = EXCLUDED.brand_name,
                manufacturer_name = EXCLUDED.manufacturer_name,
                identification_status = EXCLUDED.identification_status,
                identification_method = EXCLUDED.identification_method,
                identification_confidence = EXCLUDED.identification_confidence,
                identification_evidence = EXCLUDED.identification_evidence,
                compliant_rules_count = EXCLUDED.compliant_rules_count,
                review_rules_count = EXCLUDED.review_rules_count,
                violation_rules_count = EXCLUDED.violation_rules_count,
                ocr_data = EXCLUDED.ocr_data,
                extraction_data = EXCLUDED.extraction_data,
                compliance_data = EXCLUDED.compliance_data,
                provenance_data = EXCLUDED.provenance_data,
                inspector_decision = COALESCE(EXCLUDED.inspector_decision, scans.inspector_decision),
                inspector_notes = COALESCE(EXCLUDED.inspector_notes, scans.inspector_notes),
                inspector_decided_at = COALESCE(EXCLUDED.inspector_decided_at, scans.inspector_decided_at),
                inspector_id = COALESCE(EXCLUDED.inspector_id, scans.inspector_id)
            """,
            (
                scan_response.scan_id,
                client_scan_id or scan_response.client_scan_id,
                officer_id or scan_response.officer_id or "OFFICER-DEFAULT",
                created_at,
                now,
                status_str.lower(),
                sync_source,
                image_path,
                scan_response.width,
                scan_response.height,
                summary["mrp"],
                summary["net_quantity"],
                summary["manufacturing_date"],
                summary["consumer_care"],
                summary["product_name"],
                summary["brand_name"],
                summary["manufacturer_name"],
                summary["identification_status"],
                summary["identification_method"],
                summary["identification_confidence"],
                Jsonb(summary["identification_evidence"]),
                len(scan_response.ocr.items),
                summary["compliant_rules_count"],
                summary["review_rules_count"],
                summary["violation_rules_count"],
                Jsonb(scan_response.ocr.model_dump()),
                Jsonb(scan_response.extraction.model_dump()),
                Jsonb(scan_response.compliance.model_dump()),
                Jsonb(scan_response.provenance.model_dump()) if scan_response.provenance else None,
                insp_decision_val,
                insp_notes_val,
                insp_decided_at_val,
                insp_officer_id_val,
            ),
        )

    def _insert_sqlite(
        self,
        conn: sqlite3.Connection,
        scan_response: ScanCreateResponse,
        client_scan_id: str | None = None,
        officer_id: str = "OFFICER-DEFAULT",
        sync_source: str = "realtime",
        image_path: str | None = None,
    ) -> None:
        summary = self._extract_summary_fields(scan_response)
        now_iso = datetime.now(timezone.utc).isoformat()
        created_iso = scan_response.created_at.isoformat() if scan_response.created_at else now_iso

        if scan_response.status == ScanStatus.PENDING_FALLBACK:
            status_str = "pending_fallback"
        elif hasattr(scan_response.compliance, "status") and scan_response.compliance.status:
            comp_st = scan_response.compliance.status
            status_str = comp_st.value if hasattr(comp_st, "value") else str(comp_st)
        else:
            status_str = scan_response.status.value if hasattr(scan_response.status, "value") else str(scan_response.status)

        insp_dec = scan_response.inspector_decision
        insp_decision_val = insp_dec.decision.value if insp_dec else None
        insp_notes_val = insp_dec.notes if insp_dec else None
        insp_decided_at_val = insp_dec.decided_at.isoformat() if insp_dec else None
        insp_officer_id_val = insp_dec.officer_id if insp_dec else None

        conn.execute(
            """
            INSERT OR REPLACE INTO scans (
                scan_id, client_scan_id, officer_id, created_at, updated_at,
                status, sync_source, image_path, width, height,
                mrp, net_quantity, manufacturing_date, consumer_care,
                product_name, brand_name, manufacturer_name,
                identification_status, identification_method, identification_confidence, identification_evidence,
                ocr_items_count, compliant_rules_count, review_rules_count, violation_rules_count,
                ocr_data, extraction_data, compliance_data, provenance_data,
                inspector_decision, inspector_notes, inspector_decided_at, inspector_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_response.scan_id,
                client_scan_id or scan_response.client_scan_id,
                officer_id or scan_response.officer_id or "OFFICER-DEFAULT",
                created_iso,
                now_iso,
                status_str.lower(),
                sync_source,
                image_path,
                scan_response.width,
                scan_response.height,
                summary["mrp"],
                summary["net_quantity"],
                summary["manufacturing_date"],
                summary["consumer_care"],
                summary["product_name"],
                summary["brand_name"],
                summary["manufacturer_name"],
                summary["identification_status"],
                summary["identification_method"],
                summary["identification_confidence"],
                json.dumps(summary["identification_evidence"]),
                len(scan_response.ocr.items),
                summary["compliant_rules_count"],
                summary["review_rules_count"],
                summary["violation_rules_count"],
                scan_response.ocr.model_dump_json(),
                scan_response.extraction.model_dump_json(),
                scan_response.compliance.model_dump_json(),
                scan_response.provenance.model_dump_json() if scan_response.provenance else None,
                insp_decision_val,
                insp_notes_val,
                insp_decided_at_val,
                insp_officer_id_val,
            ),
        )

    def save_scan(
        self,
        scan_response: ScanCreateResponse,
        client_scan_id: str | None = None,
        officer_id: str = "OFFICER-DEFAULT",
        sync_source: str = "realtime",
        image_path: str | None = None,
    ) -> None:
        """Persist or update an inspection scan in the database."""
        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        self._insert_pg(cur, scan_response, client_scan_id, officer_id, sync_source, image_path)
                    conn.commit()
                return
            except Exception as e:
                logger.warning("Postgres write failed (%s). Writing to SQLite.", e)

        with self._get_sqlite_connection() as conn:
            self._insert_sqlite(conn, scan_response, client_scan_id, officer_id, sync_source, image_path)
            conn.commit()

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        """Retrieve a scan by its primary scan_id."""
        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT * FROM scans WHERE scan_id = %s", (scan_id,))
                        row = cur.fetchone()
                        return dict(row) if row else None
            except Exception:
                pass

        with self._get_sqlite_connection() as conn:
            row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
            return self._sqlite_row_to_dict(row) if row else None

    def get_scan_by_client_id(self, client_scan_id: str) -> dict[str, Any] | None:
        """Retrieve a scan by client UUID to ensure idempotent sync."""
        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT * FROM scans WHERE client_scan_id = %s", (client_scan_id,))
                        row = cur.fetchone()
                        return dict(row) if row else None
            except Exception:
                pass

        with self._get_sqlite_connection() as conn:
            row = conn.execute("SELECT * FROM scans WHERE client_scan_id = ?", (client_scan_id,)).fetchone()
            return self._sqlite_row_to_dict(row) if row else None

    def _row_to_scan_summary_item(self, row: dict[str, Any] | Any) -> ScanSummaryItem:
        """Convert a database row into a structured ScanSummaryItem."""
        if not isinstance(row, dict):
            try:
                row = dict(row)
            except Exception:
                row = self._sqlite_row_to_dict(row)

        created_at = row.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at_dt = datetime.now(timezone.utc)
        elif isinstance(created_at, datetime):
            created_at_dt = created_at
        else:
            created_at_dt = datetime.now(timezone.utc)

        raw_evidence = row.get("identification_evidence")
        if isinstance(raw_evidence, str):
            try:
                evidence = json.loads(raw_evidence)
            except Exception:
                evidence = []
        elif isinstance(raw_evidence, list):
            evidence = raw_evidence
        else:
            evidence = []

        mrp_val = row.get("mrp")
        try:
            mrp_float = float(mrp_val) if mrp_val is not None else None
        except (ValueError, TypeError):
            mrp_float = None

        conf_val = row.get("identification_confidence")
        try:
            conf_float = float(conf_val) if conf_val is not None else 0.0
        except (ValueError, TypeError):
            conf_float = 0.0

        return ScanSummaryItem(
            scan_id=str(row.get("scan_id", "")),
            client_scan_id=row.get("client_scan_id"),
            officer_id=row.get("officer_id") or "OFFICER-DEFAULT",
            created_at=created_at_dt,
            status=str(row.get("status", "received")),
            sync_source=str(row.get("sync_source") or "realtime"),
            width=int(row.get("width") or 0),
            height=int(row.get("height") or 0),
            mrp=mrp_float,
            net_quantity=row.get("net_quantity"),
            manufacturing_date=row.get("manufacturing_date"),
            consumer_care=row.get("consumer_care"),
            product_name=row.get("product_name"),
            brand_name=row.get("brand_name"),
            manufacturer_name=row.get("manufacturer_name"),
            identification_status=str(row.get("identification_status") or "UNIDENTIFIED"),
            identification_method=str(row.get("identification_method") or "DETERMINISTIC"),
            identification_confidence=conf_float,
            identification_evidence=evidence,
            ocr_items_count=int(row.get("ocr_items_count") or 0),
            compliant_rules_count=int(row.get("compliant_rules_count") or 0),
            review_rules_count=int(row.get("review_rules_count") or 0),
            violation_rules_count=int(row.get("violation_rules_count") or 0),
            inspector_decision=row.get("inspector_decision"),
            inspector_notes=row.get("inspector_notes"),
            inspector_decided_at=(
                datetime.fromisoformat(row["inspector_decided_at"].replace("Z", "+00:00"))
                if isinstance(row.get("inspector_decided_at"), str)
                else row.get("inspector_decided_at")
                if isinstance(row.get("inspector_decided_at"), datetime)
                else None
            ),
            inspector_id=row.get("inspector_id"),
        )

    def get_scan_summary(self, scan_id: str) -> ScanSummaryItem | None:
        """Retrieve a scan as a ScanSummaryItem."""
        row = self.get_scan(scan_id)
        if not row:
            return None
        return self._row_to_scan_summary_item(row)

    def list_scans(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        officer_id: str | None = None,
        search: str | None = None,
    ) -> ScanHistoryResponse:
        """Query paginated inspection history with optional filters."""
        if self._is_postgres:
            try:
                query = "SELECT * FROM scans WHERE 1=1"
                count_query = "SELECT COUNT(*) FROM scans WHERE 1=1"
                params: list[Any] = []

                if status:
                    query += " AND LOWER(status) = LOWER(%s)"
                    count_query += " AND LOWER(status) = LOWER(%s)"
                    params.append(status)

                if officer_id:
                    query += " AND officer_id = %s"
                    count_query += " AND officer_id = %s"
                    params.append(officer_id)

                if search:
                    search_param = f"%{search}%"
                    query += " AND (scan_id ILIKE %s OR client_scan_id ILIKE %s OR brand_name ILIKE %s OR product_name ILIKE %s OR manufacturer_name ILIKE %s OR manufacturing_date ILIKE %s OR consumer_care ILIKE %s)"
                    count_query += " AND (scan_id ILIKE %s OR client_scan_id ILIKE %s OR brand_name ILIKE %s OR product_name ILIKE %s OR manufacturer_name ILIKE %s OR manufacturing_date ILIKE %s OR consumer_care ILIKE %s)"
                    params.extend([search_param, search_param, search_param, search_param, search_param, search_param, search_param])

                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(count_query, params)
                        total = cur.fetchone()["count"]

                        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                        paged_params = list(params) + [limit, offset]
                        cur.execute(query, paged_params)
                        rows = cur.fetchall()

                        items = [self._row_to_scan_summary_item(row) for row in rows]
                return ScanHistoryResponse(items=items, total=total, limit=limit, offset=offset)
            except Exception as e:
                logger.warning("PostgreSQL query failed (%s). Falling back to SQLite.", e)

        # SQLite fallback listing
        query = "SELECT * FROM scans WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM scans WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND LOWER(status) = LOWER(?)"
            count_query += " AND LOWER(status) = LOWER(?)"
            params.append(status)

        if officer_id:
            query += " AND officer_id = ?"
            count_query += " AND officer_id = ?"
            params.append(officer_id)

        if search:
            search_param = f"%{search}%"
            query += " AND (scan_id LIKE ? OR client_scan_id LIKE ? OR brand_name LIKE ? OR product_name LIKE ? OR manufacturer_name LIKE ? OR manufacturing_date LIKE ? OR consumer_care LIKE ?)"
            count_query += " AND (scan_id LIKE ? OR client_scan_id LIKE ? OR brand_name LIKE ? OR product_name LIKE ? OR manufacturer_name LIKE ? OR manufacturing_date LIKE ? OR consumer_care LIKE ?)"
            params.extend([search_param, search_param, search_param, search_param, search_param, search_param, search_param])

        with self._get_sqlite_connection() as conn:
            total = conn.execute(count_query, params).fetchone()[0]

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            paged_params = list(params) + [limit, offset]
            rows = conn.execute(query, paged_params).fetchall()

            items = [self._row_to_scan_summary_item(row) for row in rows]

        return ScanHistoryResponse(items=items, total=total, limit=limit, offset=offset)

    def update_scan_identification(
        self,
        scan_id: str,
        brand_name: str | None,
        product_name: str | None,
        manufacturer_name: str | None,
        identification_status: str,
        identification_method: str,
        identification_confidence: float,
        identification_evidence: list[str],
    ) -> dict[str, Any] | None:
        """Update identification fields after manual AI fallback without altering compliance findings."""
        now = datetime.now(timezone.utc)
        evidence_json = json.dumps(identification_evidence)

        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE scans SET
                                brand_name = %s,
                                product_name = %s,
                                manufacturer_name = %s,
                                identification_status = %s,
                                identification_method = %s,
                                identification_confidence = %s,
                                identification_evidence = %s,
                                updated_at = %s
                            WHERE scan_id = %s
                            RETURNING *
                            """,
                            (
                                brand_name,
                                product_name,
                                manufacturer_name,
                                identification_status,
                                identification_method,
                                identification_confidence,
                                Jsonb(identification_evidence),
                                now,
                                scan_id,
                            ),
                        )
                        row = cur.fetchone()
                        conn.commit()
                        return dict(row) if row else None
            except Exception as e:
                logger.warning("Postgres update_scan_identification failed: %s", e)

        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                UPDATE scans SET
                    brand_name = ?,
                    product_name = ?,
                    manufacturer_name = ?,
                    identification_status = ?,
                    identification_method = ?,
                    identification_confidence = ?,
                    identification_evidence = ?,
                    updated_at = ?
                WHERE scan_id = ?
                """,
                (
                    brand_name,
                    product_name,
                    manufacturer_name,
                    identification_status,
                    identification_method,
                    identification_confidence,
                    evidence_json,
                    now.isoformat(),
                    scan_id,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
            return self._sqlite_row_to_dict(row) if row else None

    def apply_inspector_decision(
        self,
        scan_id: str,
        decision: str,
        officer_id: str = "OFFICER-DEFAULT",
        notes: str | None = None,
    ) -> ScanSummaryItem:
        """Apply an authoritative enforcement decision (compliant/violation) by an inspector."""
        normalized_decision = decision.lower()
        if normalized_decision not in ("compliant", "violation"):
            raise ValueError(f"Invalid decision '{decision}'. Must be 'compliant' or 'violation'.")

        now = datetime.now(timezone.utc)

        # 1. Update State File on Disk (if exists)
        state_file = self._settings.storage_dir / "scans" / "states" / f"{scan_id}.json"
        compliance_dict = None
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text("utf-8"))
                data["status"] = normalized_decision
                if "compliance" in data and isinstance(data["compliance"], dict):
                    data["compliance"]["status"] = normalized_decision
                    for finding in data["compliance"].get("findings", []):
                        cur_status = str(finding.get("status", "")).lower()
                        if cur_status in ("manual_review_required", "review"):
                            finding["status"] = normalized_decision
                            finding_note = f"[Inspector Confirmed {normalized_decision.capitalize()}]: {notes}" if notes else f"[Inspector Confirmed {normalized_decision.capitalize()}]"
                            finding["message"] = f"{finding_note} - {finding.get('message', '')}"
                data["inspector_decision"] = {
                    "decision": normalized_decision,
                    "officer_id": officer_id,
                    "notes": notes,
                    "decided_at": now.isoformat(),
                }
                state_file.write_text(json.dumps(data, indent=2), "utf-8")
                compliance_dict = data.get("compliance")
            except Exception as e:
                logger.warning("Could not update scan state file %s: %s", state_file, e)

        # 2. Get current counts to adjust
        current_row = self.get_scan(scan_id)
        if not current_row:
            raise ValueError(f"Scan {scan_id} not found.")

        compliant_count = int(current_row.get("compliant_rules_count") or 0)
        review_count = int(current_row.get("review_rules_count") or 0)
        violation_count = int(current_row.get("violation_rules_count") or 0)

        if normalized_decision == "compliant":
            compliant_count += review_count
            review_count = 0
        else:
            violation_count += review_count
            review_count = 0

        # If compliance_dict wasn't read from state_file, update existing compliance_data
        if not compliance_dict and current_row.get("compliance_data"):
            comp_data = current_row["compliance_data"]
            if isinstance(comp_data, str):
                try:
                    comp_data = json.loads(comp_data)
                except Exception:
                    comp_data = {}
            if isinstance(comp_data, dict):
                comp_data["status"] = normalized_decision
                for finding in comp_data.get("findings", []):
                    cur_status = str(finding.get("status", "")).lower()
                    if cur_status in ("manual_review_required", "review"):
                        finding["status"] = normalized_decision
                        finding_note = f"[Inspector Confirmed {normalized_decision.capitalize()}]: {notes}" if notes else f"[Inspector Confirmed {normalized_decision.capitalize()}]"
                        finding["message"] = f"{finding_note} - {finding.get('message', '')}"
                compliance_dict = comp_data

        # 3. Update Database (Postgres or SQLite)
        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE scans SET
                                status = %s,
                                compliant_rules_count = %s,
                                review_rules_count = %s,
                                violation_rules_count = %s,
                                inspector_decision = %s,
                                inspector_notes = %s,
                                inspector_decided_at = %s,
                                inspector_id = %s,
                                compliance_data = COALESCE(%s, compliance_data),
                                updated_at = %s
                            WHERE scan_id = %s
                            RETURNING *
                            """,
                            (
                                normalized_decision,
                                compliant_count,
                                review_count,
                                violation_count,
                                normalized_decision,
                                notes,
                                now,
                                officer_id,
                                Jsonb(compliance_dict) if compliance_dict else None,
                                now,
                                scan_id,
                            ),
                        )
                        row = cur.fetchone()
                        conn.commit()
                        if row:
                            return self._row_to_scan_summary_item(row)
            except Exception as e:
                logger.warning("Postgres apply_inspector_decision failed: %s", e)

        # Fallback to SQLite
        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                UPDATE scans SET
                    status = ?,
                    compliant_rules_count = ?,
                    review_rules_count = ?,
                    violation_rules_count = ?,
                    inspector_decision = ?,
                    inspector_notes = ?,
                    inspector_decided_at = ?,
                    inspector_id = ?,
                    compliance_data = COALESCE(?, compliance_data),
                    updated_at = ?
                WHERE scan_id = ?
                """,
                (
                    normalized_decision,
                    compliant_count,
                    review_count,
                    violation_count,
                    normalized_decision,
                    notes,
                    now.isoformat(),
                    officer_id,
                    json.dumps(compliance_dict) if compliance_dict else None,
                    now.isoformat(),
                    scan_id,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
            if not row:
                raise ValueError(f"Scan {scan_id} not found in database.")
            return self._row_to_scan_summary_item(row)

    def get_stats(self) -> ScanStatsResponse:
        """Compute aggregate metrics across all recorded scans."""
        if self._is_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) AS total FROM scans")
                        total = cur.fetchone()["total"]
                        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE LOWER(status) = 'compliant'")
                        compliant = cur.fetchone()["c"]
                        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE LOWER(status) IN ('manual_review_required', 'review')")
                        review = cur.fetchone()["c"]
                        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE LOWER(status) IN ('violation', 'non_compliant')")
                        violation = cur.fetchone()["c"]
                        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE LOWER(status) = 'pending_fallback'")
                        pending = cur.fetchone()["c"]
                        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE sync_source = 'offline_sync'")
                        offline = cur.fetchone()["c"]

                return ScanStatsResponse(
                    total_scans=total,
                    compliant_count=compliant,
                    manual_review_count=review,
                    violation_count=violation,
                    pending_fallback_count=pending,
                    offline_synced_count=offline,
                )
            except Exception as e:
                logger.warning("Postgres stats query failed (%s). Falling back to SQLite.", e)

        with self._get_sqlite_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            compliant = conn.execute("SELECT COUNT(*) FROM scans WHERE LOWER(status) = 'compliant'").fetchone()[0]
            review = conn.execute("SELECT COUNT(*) FROM scans WHERE LOWER(status) IN ('manual_review_required', 'review')").fetchone()[0]
            violation = conn.execute("SELECT COUNT(*) FROM scans WHERE LOWER(status) IN ('violation', 'non_compliant')").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM scans WHERE LOWER(status) = 'pending_fallback'").fetchone()[0]
            offline = conn.execute("SELECT COUNT(*) FROM scans WHERE sync_source = 'offline_sync'").fetchone()[0]

        return ScanStatsResponse(
            total_scans=total,
            compliant_count=compliant,
            manual_review_count=review,
            violation_count=violation,
            pending_fallback_count=pending,
            offline_synced_count=offline,
        )

    def _sqlite_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert an SQLite row into a structured dictionary with parsed JSON fields."""
        d = dict(row)
        for json_key in ("ocr_data", "extraction_data", "compliance_data", "provenance_data"):
            if d.get(json_key) and isinstance(d[json_key], str):
                try:
                    d[json_key] = json.loads(d[json_key])
                except Exception:
                    pass
        return d
