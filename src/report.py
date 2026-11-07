"""Step 8 - Export an audit-findings report as PDF (reportlab)."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from . import config as C

_SEV_COLOR = {"HIGH": colors.HexColor("#c0392b"),
              "MEDIUM": colors.HexColor("#e67e22"),
              "LOW": colors.HexColor("#f1c40f"),
              "OK": colors.HexColor("#27ae60")}


def build_report(results, out_path=None):
    out_path = str(out_path or C.DATA / "audit_report.pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    el = [Paragraph("AuditLens AI - Audit Findings Report", styles["Title"]),
          Spacer(1, 6)]
    n_flagged = sum(1 for r in results if r["severity"] != "OK")
    el.append(Paragraph(f"{len(results)} documents reviewed &middot; "
                        f"{n_flagged} flagged for attention.", styles["Normal"]))
    el.append(Spacer(1, 12))

    rows = [["Document", "Severity", "Invoice No", "Findings"]]
    for r in results:
        issues = "; ".join(f"{f['rule']}" for f in r["findings"]) or "clean"
        rows.append([r["doc"], r["severity"],
                     r["fields"].get("invoice_no") or "-", issues])
    t = Table(rows, colWidths=[70, 60, 90, 250])
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5b")),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTSIZE", (0, 0), (-1, -1), 8),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "TOP")]
    for i, r in enumerate(results, start=1):
        style.append(("TEXTCOLOR", (1, i), (1, i), _SEV_COLOR[r["severity"]]))
    t.setStyle(TableStyle(style))
    el.append(t)
    doc.build(el)
    return out_path
