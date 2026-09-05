"""OCR Provenance and Debugging Tracking Service."""

from typing import Any

from app.schemas.scans import (
    ComplianceFinding,
    ComplianceResult,
    ComplianceTrace,
    DeterministicExtractionTrace,
    EvidenceMergeTrace,
    ExtractedField,
    ExtractionResult,
    FieldProvenance,
    FrontendEvidenceTrace,
    GeminiFallbackTrace,
    InitialOcrTrace,
    OCRResult,
    ScanProvenance,
)


FIELD_RULE_MAP = {
    "mrp": "LMPC-6-1-e",
    "net_quantity": "LMPC-6-1-c",
    "consumer_care_phone": "LMPC-6-1-g",
    "manufacturing_date": "LMPC-6-1-d",
    "packing_date": "LMPC-6-1-d",
}

FIELD_KEYWORDS = {
    "mrp": ["mrp", "m.r.p", "rs.", "rs", "₹", "price"],
    "net_quantity": ["net qty", "net quantity", "net wt", "net weight", "net vol", "qty", "quantity"],
    "consumer_care_phone": ["customer care", "consumer care", "toll free", "phone", "contact"],
    "manufacturing_date": ["mfg date", "mfd date", "mfg", "mfd", "date of mfg"],
    "packing_date": ["pkd date", "pkd", "packed", "date of pkg"],
}


def _find_initial_ocr_item(field_name: str, ocr: OCRResult, raw_ai_item: dict[str, Any] | None) -> tuple[str, str | None, float | None, list[float] | None]:
    """Find the initial OCR evidence that matches this field."""
    import re

    # 1. If Gemini mapped to source OCR ids, use those exact items
    if raw_ai_item and raw_ai_item.get("source_ocr_ids"):
        source_ids = raw_ai_item["source_ocr_ids"]
        valid_items = [ocr.items[i] for i in source_ids if i < len(ocr.items)]
        if valid_items:
            min_x = min(item.bbox[0] for item in valid_items)
            min_y = min(item.bbox[1] for item in valid_items)
            max_x = max(item.bbox[2] for item in valid_items)
            max_y = max(item.bbox[3] for item in valid_items)
            conf = min(item.confidence for item in valid_items)
            text = " ".join(item.text for item in valid_items)
            eng = valid_items[0].engine.value if hasattr(valid_items[0].engine, "value") else str(valid_items[0].engine)
            return eng, text, round(conf, 3), [round(min_x, 1), round(min_y, 1), round(max_x, 1), round(max_y, 1)]

    # 2. Check if any OCR block contains keywords for this field
    keywords = FIELD_KEYWORDS.get(field_name, [field_name])
    for item in ocr.items:
        text_lower = item.text.lower()
        matched = False
        for kw in keywords:
            if kw.isalnum():
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    matched = True
                    break
            else:
                if kw in text_lower:
                    matched = True
                    break
        if matched:
            eng = item.engine.value if hasattr(item.engine, "value") else str(item.engine)
            b = [round(x, 1) for x in item.bbox]
            return eng, item.text, round(item.confidence, 3), b

    # 3. Fallback: engine with no match
    default_engine = ocr.items[0].engine.value if (ocr.items and hasattr(ocr.items[0].engine, "value")) else (str(ocr.items[0].engine) if ocr.items else "unknown")
    return default_engine, None, None, None


def _find_compliance_for_field(field_name: str, compliance: ComplianceResult) -> tuple[str, str | None, str | None]:
    """Find the compliance finding associated with a field."""
    for finding in compliance.findings:
        for ev in finding.evidence:
            if ev.field_name == field_name:
                status_str = finding.status.value.upper() if hasattr(finding.status, "value") else str(finding.status).upper()
                return status_str, finding.rule_id, finding.message

    expected_rule = FIELD_RULE_MAP.get(field_name)
    if expected_rule:
        for finding in compliance.findings:
            if finding.rule_id == expected_rule:
                status_str = finding.status.value.upper() if hasattr(finding.status, "value") else str(finding.status).upper()
                return status_str, finding.rule_id, finding.message

    return "NOT_EVALUATED", None, None


def build_field_provenance(
    field_name: str,
    ocr: OCRResult,
    deterministic_field: ExtractedField | None,
    final_field: ExtractedField | None,
    compliance: ComplianceResult,
    fallback_triggered: bool,
    fallback_reasons: list[str],
    ai_telemetry: dict[str, Any],
) -> FieldProvenance:
    """Construct complete provenance record for a single field."""
    raw_ai_fields = ai_telemetry.get("raw_ai_fields", [])
    raw_ai_item = next((f for f in raw_ai_fields if f.get("field_name") == field_name), None)
    recovered_names = set(ai_telemetry.get("recovered_field_names", []))

    # 1. Initial OCR Trace
    if deterministic_field:
        init_engine = deterministic_field.source_evidence.engine.value if hasattr(deterministic_field.source_evidence.engine, "value") else str(deterministic_field.source_evidence.engine)
        init_text = deterministic_field.source_evidence.text
        init_conf = round(deterministic_field.source_evidence.confidence, 3)
        init_bbox = [round(x, 1) for x in deterministic_field.source_evidence.bbox]
    else:
        init_engine, init_text, init_conf, init_bbox = _find_initial_ocr_item(field_name, ocr, raw_ai_item)

    initial_ocr = InitialOcrTrace(
        engine=init_engine,
        raw_text=init_text,
        confidence=init_conf,
        bbox=init_bbox,
    )

    # 2. Deterministic Extraction Trace
    if deterministic_field:
        val_str = f"{deterministic_field.value}{(' ' + deterministic_field.unit) if deterministic_field.unit else ''}"
        deterministic = DeterministicExtractionTrace(
            status="SUCCESS",
            value=val_str,
            reason=None,
        )
    else:
        deterministic = DeterministicExtractionTrace(
            status="FAILED",
            value=None,
            reason="OCR text did not match expected pattern or field was missing",
        )

    # 3. Gemini Fallback Trace
    if deterministic_field:
        gemini_trace = GeminiFallbackTrace(
            triggered=False,
            reason="Field already successfully extracted",
            value=None,
            confidence=None,
            bbox=None,
        )
    else:
        if fallback_triggered:
            if raw_ai_item:
                ai_val = f"{raw_ai_item.get('value')}{(' ' + raw_ai_item.get('unit')) if raw_ai_item.get('unit') else ''}"
                ai_conf = raw_ai_item.get("ai_semantic_confidence")
                if final_field and field_name in recovered_names:
                    fb_bbox = [round(x, 1) for x in final_field.source_evidence.bbox]
                else:
                    fb_bbox = None
                gemini_trace = GeminiFallbackTrace(
                    triggered=True,
                    reason=f"{field_name} missing after deterministic extraction",
                    value=ai_val,
                    confidence=round(ai_conf, 3) if ai_conf is not None else None,
                    bbox=fb_bbox,
                )
            else:
                gemini_trace = GeminiFallbackTrace(
                    triggered=True,
                    reason=f"{field_name} missing after deterministic extraction",
                    value=None,
                    confidence=None,
                    bbox=None,
                )
        else:
            gemini_trace = GeminiFallbackTrace(
                triggered=False,
                reason="Fallback not yet triggered (awaiting fallback image)",
                value=None,
                confidence=None,
                bbox=None,
            )

    # 4. Evidence Merge Trace
    if final_field:
        if field_name in recovered_names:
            merge_source = "gemini_vision"
        else:
            merge_source = final_field.source_evidence.engine.value if hasattr(final_field.source_evidence.engine, "value") else str(final_field.source_evidence.engine)

        merge_val = f"{final_field.value}{(' ' + final_field.unit) if final_field.unit else ''}"
        merge_bbox = [round(x, 1) for x in final_field.source_evidence.bbox]
    elif deterministic_field:
        merge_source = deterministic_field.source_evidence.engine.value if hasattr(deterministic_field.source_evidence.engine, "value") else str(deterministic_field.source_evidence.engine)
        merge_val = f"{deterministic_field.value}{(' ' + deterministic_field.unit) if deterministic_field.unit else ''}"
        merge_bbox = [round(x, 1) for x in deterministic_field.source_evidence.bbox]
    else:
        merge_source = "none"
        merge_val = "Not extracted"
        merge_bbox = [0.0, 0.0, 0.0, 0.0]

    evidence_merge = EvidenceMergeTrace(
        selected_source=merge_source,
        selected_value=merge_val,
        selected_bbox=merge_bbox,
    )

    # 5. Compliance Trace
    comp_status, comp_rule, comp_msg = _find_compliance_for_field(field_name, compliance)
    compliance_trace = ComplianceTrace(
        status=comp_status,
        rule_id=comp_rule,
        message=comp_msg,
    )

    # 6. Frontend Evidence Trace
    frontend_evidence = FrontendEvidenceTrace(
        displayed_from=merge_source,
        displayed_bbox=merge_bbox,
    )

    # Determine 5-stage lifecycle state
    if comp_status != "NOT_EVALUATED":
        lifecycle_stage = "VERIFIED"
    elif merge_source != "none":
        lifecycle_stage = "MERGED"
    elif gemini_trace.triggered and gemini_trace.value:
        lifecycle_stage = "RECOVERED"
    elif deterministic.status == "SUCCESS":
        lifecycle_stage = "EXTRACTED"
    else:
        lifecycle_stage = "DETECTED"

    return FieldProvenance(
        field_name=field_name,
        lifecycle_stage=lifecycle_stage,
        initial_ocr=initial_ocr,
        deterministic_extraction=deterministic,
        gemini_fallback=gemini_trace,
        evidence_merge=evidence_merge,
        compliance=compliance_trace,
        frontend_evidence=frontend_evidence,
    )


def build_scan_provenance(
    ocr: OCRResult,
    deterministic_extraction: ExtractionResult,
    final_extraction: ExtractionResult,
    compliance: ComplianceResult,
    fallback_triggered: bool,
    fallback_reasons: list[str],
    ai_telemetry: dict[str, Any] | None = None,
) -> ScanProvenance:
    """Build full scan provenance for all extracted and required fields."""
    telemetry = ai_telemetry or {}
    det_map = {f.field_name: f for f in deterministic_extraction.fields}
    final_map = {f.field_name: f for f in final_extraction.fields}

    candidate_names: list[str] = []
    for f in final_extraction.fields:
        if f.field_name not in candidate_names:
            candidate_names.append(f.field_name)
    for f in deterministic_extraction.fields:
        if f.field_name not in candidate_names:
            candidate_names.append(f.field_name)

    for req in ["mrp", "net_quantity", "consumer_care_phone", "manufacturing_date"]:
        if req not in candidate_names:
            candidate_names.append(req)

    field_traces: list[FieldProvenance] = []
    recovered_names = set(telemetry.get("recovered_field_names", []))
    mlkit_det_count = 0
    gemini_rec_count = 0
    failed_count = 0

    sources_count: dict[str, int] = {}
    gemini_boxes = 0
    final_boxes = 0

    for name in candidate_names:
        det_f = det_map.get(name)
        fin_f = final_map.get(name)
        trace = build_field_provenance(
            field_name=name,
            ocr=ocr,
            deterministic_field=det_f,
            final_field=fin_f,
            compliance=compliance,
            fallback_triggered=fallback_triggered,
            fallback_reasons=fallback_reasons,
            ai_telemetry=telemetry,
        )
        field_traces.append(trace)

        if trace.evidence_merge.selected_source != "none":
            final_boxes += 1
            src = trace.evidence_merge.selected_source
            sources_count[src] = sources_count.get(src, 0) + 1
            if src == "gemini_vision":
                gemini_rec_count += 1
                gemini_boxes += 1
            else:
                mlkit_det_count += 1
        else:
            failed_count += 1

    engine_name = ocr.items[0].engine.value if (ocr.items and hasattr(ocr.items[0].engine, "value")) else "OCR"
    if "mlkit" in str(engine_name).lower():
        engine_label = "ML Kit"
    elif "paddle" in str(engine_name).lower():
        engine_label = "PaddleOCR"
    else:
        engine_label = str(engine_name)

    summary = {
        "total_fields": len([f for f in field_traces if f.evidence_merge.selected_source != "none"]),
        "detected_by_initial_ocr": mlkit_det_count,
        "recovered_by_gemini": gemini_rec_count,
        "failed_completely": failed_count,
        "engine_label": engine_label,
        "raw_ocr_boxes_received": len(ocr.items),
        "gemini_boxes_received": gemini_boxes,
        "final_boxes_displayed": final_boxes,
        "sources_breakdown": sources_count,
    }

    return ScanProvenance(
        fields=field_traces,
        fallback_triggered=fallback_triggered,
        fallback_reasons=fallback_reasons,
        summary=summary,
    )


def format_terminal_trace(provenance: ScanProvenance) -> str:
    """Format provenance into a clean, human-readable ASCII debug trace."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"{'OCR PROVENANCE TRACE':^60}")
    lines.append("=" * 60)

    for field in provenance.fields:
        lines.append("")
        lines.append(f"FIELD: {field.field_name.upper()}")
        lines.append(f"Lifecycle Stage : [{field.lifecycle_stage}]")
        lines.append("-" * 60)

        # [1] Initial Detection
        lines.append("[1] Initial Detection")
        lines.append(f"  Engine       : {field.initial_ocr.engine}")
        lines.append(f"  Raw Text     : {repr(field.initial_ocr.raw_text) if field.initial_ocr.raw_text is not None else 'None'}")
        lines.append(f"  Confidence   : {field.initial_ocr.confidence if field.initial_ocr.confidence is not None else 'N/A'}")
        lines.append(f"  BBox         : {field.initial_ocr.bbox if field.initial_ocr.bbox is not None else 'None'}")
        lines.append("")

        # [2] Deterministic Extraction
        lines.append("[2] Deterministic Extraction")
        lines.append(f"  Status       : {field.deterministic_extraction.status}")
        if field.deterministic_extraction.status == "SUCCESS":
            lines.append(f"  Value        : {field.deterministic_extraction.value}")
        else:
            lines.append(f"  Reason       : {field.deterministic_extraction.reason}")
        lines.append("")

        # [3] Gemini Vision
        lines.append("[3] Gemini Vision")
        lines.append(f"  Triggered    : {'YES' if field.gemini_fallback.triggered else 'NO'}")
        lines.append(f"  Reason       : {field.gemini_fallback.reason or 'N/A'}")
        if field.gemini_fallback.triggered and field.gemini_fallback.value:
            lines.append(f"  Value        : {field.gemini_fallback.value}")
            if field.gemini_fallback.confidence is not None:
                lines.append(f"  Confidence   : {field.gemini_fallback.confidence}")
            lines.append(f"  BBox         : {field.gemini_fallback.bbox}")
        lines.append("")

        # [4] Evidence Merge
        lines.append("[4] Evidence Merge")
        lines.append(f"  Selected     : {field.evidence_merge.selected_source}")
        lines.append(f"  Final Value  : {field.evidence_merge.selected_value}")
        lines.append(f"  Final BBox   : {field.evidence_merge.selected_bbox}")
        lines.append("")

        # [5] Compliance
        lines.append("[5] Compliance")
        lines.append(f"  Status       : {field.compliance.status}")
        if field.compliance.rule_id:
            lines.append(f"  Rule         : {field.compliance.rule_id}")
        lines.append(f"  Message      : {field.compliance.message or 'N/A'}")
        lines.append("  Disclaimer   : AI decision support screening. Authoritative enforcement decision belongs to the Legal Metrology Officer.")
        lines.append("")

        # [DISPLAY] Frontend Evidence
        lines.append("[DISPLAY] Frontend Evidence")
        lines.append(f"  Displayed From : {field.frontend_evidence.displayed_from}")
        lines.append(f"  Displayed BBox  : {field.frontend_evidence.displayed_bbox}")
        lines.append("=" * 60)

    # Summary Section
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(f"Total Fields              : {provenance.summary.get('total_fields', 0)}")
    engine_label = provenance.summary.get('engine_label', 'Initial OCR')
    lines.append(f"Detected by {engine_label:<13} : {provenance.summary.get('detected_by_initial_ocr', 0)}")
    lines.append(f"Recovered by Gemini       : {provenance.summary.get('recovered_by_gemini', 0)}")
    lines.append(f"Failed Completely         : {provenance.summary.get('failed_completely', 0)}")
    lines.append("")
    lines.append(f"Gemini Fallback Triggered : {'YES' if provenance.fallback_triggered else 'NO'}")
    if provenance.fallback_reasons:
        lines.append("Fallback Reason           :")
        for r in provenance.fallback_reasons:
            lines.append(f"  - {r}")
    else:
        lines.append("Fallback Reason           : None (All required fields satisfied)")
    lines.append("")

    lines.append("Final Evidence Sources:")
    sources = provenance.summary.get('sources_breakdown', {})
    if sources:
        for src, count in sources.items():
            lines.append(f"  {src:<15} : {count}")
    else:
        lines.append("  None")
    lines.append("")

    lines.append("Bounding Boxes:")
    lines.append(f"  {engine_label} boxes received : {provenance.summary.get('raw_ocr_boxes_received', 0)}")
    lines.append(f"  Gemini boxes received   : {provenance.summary.get('gemini_boxes_received', 0)}")
    lines.append(f"  Final boxes displayed   : {provenance.summary.get('final_boxes_displayed', 0)}")
    lines.append("=" * 60)
    lines.append("")

    return chr(10).join(lines)
