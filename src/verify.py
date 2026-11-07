"""
Step 6 - Verification engine.

Rule-based audit checks over the extracted fields. Each rule returns a
finding with a severity. These are plain, auditable business rules — exactly
what an auditor can read and trust.
"""
import re
from datetime import date

from . import config as C

TAX_RATE = 0.18          # matches the synthetic generator; make configurable
TOTAL_TOL = 0.01         # 1% tolerance on total reconciliation

SEVERITY = {
    "total_mismatch": "HIGH",
    "missing_signature": "HIGH",
    "duplicate_invoice": "HIGH",
    "date_out_of_range": "MEDIUM",
    "unreadable_field": "LOW",
}
RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "OK": 0}


def _to_date(s):
    if not s:
        return None
    raw = re.sub(r"[./]", "-", str(s).strip())        # OCR reads - as . or /
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)  # ISO: YYYY-MM-DD
    if not m:
        m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", raw)  # DD-MM-YYYY
        if m2:
            d, mo, y = m2.groups()
            m = None
            try:
                return date(int(y), int(mo), int(d))
            except ValueError:
                return None
        return None
    y, mo, d = m.groups()
    try:
        return date(int(y), int(mo), int(d))
    except ValueError:
        return None


def verify_document(fields):
    """Return list of findings for one document (duplicates handled at batch
    level). Each finding: {rule, severity, detail}."""
    findings = []

    # 1. Total reconciliation.
    # Prefer the robust Subtotal + Tax anchor (single well-read tokens);
    # fall back to summing line items + assumed tax when those are missing.
    total = fields.get("total")
    subtotal, tax = fields.get("subtotal"), fields.get("tax")
    items = fields.get("line_items") or []
    expected = basis = None
    if subtotal is not None and tax is not None:
        expected, basis = round(subtotal + tax, 2), "subtotal + tax"
    elif items:
        expected, basis = round(sum(items) * (1 + TAX_RATE), 2), \
            f"items + {int(TAX_RATE*100)}% tax"

    if total is None or expected is None:
        findings.append({"rule": "unreadable_field", "severity": "LOW",
                         "detail": "Total or its components could not be read."})
    elif abs(total - expected) > max(1.0, TOTAL_TOL * expected):
        findings.append({"rule": "total_mismatch", "severity": "HIGH",
                         "detail": f"Printed total {total:,.2f} vs expected "
                                   f"{expected:,.2f} ({basis})."})

    # 2. Missing signature
    if not fields.get("has_signature"):
        findings.append({"rule": "missing_signature", "severity": "HIGH",
                         "detail": "No authorised signature detected."})

    # 3. Date within audit period
    d = _to_date(fields.get("date"))
    if d is None:
        findings.append({"rule": "unreadable_field", "severity": "LOW",
                         "detail": "Invoice date could not be parsed."})
    elif not (date.fromisoformat(C.AUDIT_START) <= d <= date.fromisoformat(C.AUDIT_END)):
        findings.append({"rule": "date_out_of_range", "severity": "MEDIUM",
                         "detail": f"Date {d} outside audit period "
                                   f"{C.AUDIT_START} to {C.AUDIT_END}."})
    return findings


def verify_batch(records):
    """records: list of {doc, fields}. Adds duplicate-invoice detection across
    the batch. Returns list of {doc, fields, findings, severity}."""
    seen = {}
    for r in records:
        inv = (r["fields"].get("invoice_no") or "").strip()
        seen.setdefault(inv, []).append(r["doc"])

    out = []
    for r in records:
        findings = verify_document(r["fields"])
        inv = (r["fields"].get("invoice_no") or "").strip()
        if inv and len(seen[inv]) > 1:
            others = [d for d in seen[inv] if d != r["doc"]]
            findings.append({"rule": "duplicate_invoice", "severity": "HIGH",
                             "detail": f"Invoice no {inv} also on: {', '.join(others)}."})
        sev = max((f["severity"] for f in findings), key=lambda s: RANK[s],
                  default="OK")
        out.append({"doc": r["doc"], "fields": r["fields"],
                    "findings": findings, "severity": sev})
    out.sort(key=lambda x: RANK[x["severity"]], reverse=True)
    return out