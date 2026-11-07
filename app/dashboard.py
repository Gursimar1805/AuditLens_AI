"""
Step 9 - Reviewer dashboard (Streamlit).

Upload document images, run the pipeline, and review findings ranked by
severity with the original document alongside each flag.

Run from the project root:  streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config as C
from src.extract import extract_fields
from src.ocr import ocr_tokens
from src.verify import verify_batch
from src.report import build_report

st.set_page_config(page_title="AuditLens AI", layout="wide")
st.title("AuditLens AI — Audit Evidence Review")
st.caption("Upload invoices → model extracts fields → rules flag issues → you review exceptions.")

SEV_BADGE = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟡 LOW", "OK": "🟢 OK"}

src_mode = st.radio("Document source", ["Upload images", "Use sample batch"],
                    horizontal=True)

uploads = None
if src_mode == "Upload images":
    uploads = st.file_uploader("Invoice images (PNG/JPG)", type=["png", "jpg", "jpeg"],
                               accept_multiple_files=True)

if st.button("Run review", type="primary"):
    records, imgs = [], {}
    if src_mode == "Use sample batch":
        files = sorted(C.IMAGES.glob("*.png"))[:30]
        for fp in files:
            records.append({"doc": fp.stem, "fields": extract_fields(ocr_tokens(fp))})
            imgs[fp.stem] = str(fp)
    else:
        if not uploads:
            st.warning("Upload at least one image."); st.stop()
        tmp = C.DATA / "uploads"; tmp.mkdir(exist_ok=True)
        for u in uploads:
            fp = tmp / u.name
            fp.write_bytes(u.getbuffer())
            records.append({"doc": u.name, "fields": extract_fields(ocr_tokens(fp))})
            imgs[u.name] = str(fp)

    results = verify_batch(records)
    st.session_state["results"] = results
    st.session_state["imgs"] = imgs

if "results" in st.session_state:
    results = st.session_state["results"]
    imgs = st.session_state["imgs"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(results))
    c2.metric("Flagged", sum(1 for r in results if r["severity"] != "OK"))
    c3.metric("High severity", sum(1 for r in results if r["severity"] == "HIGH"))

    st.subheader("Findings (ranked by severity)")
    df = pd.DataFrame([{
        "Document": r["doc"],
        "Severity": SEV_BADGE[r["severity"]],
        "Invoice No": r["fields"].get("invoice_no"),
        "Date": r["fields"].get("date"),
        "Total": r["fields"].get("total"),
        "Issues": ", ".join(f["rule"] for f in r["findings"]) or "clean",
    } for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Inspect a document")
    pick = st.selectbox("Document", [r["doc"] for r in results])
    rec = next(r for r in results if r["doc"] == pick)
    left, right = st.columns([1, 1])
    with left:
        if pick in imgs:
            st.image(imgs[pick], caption=pick, use_container_width=True)
    with right:
        st.write("**Extracted fields**")
        st.json(rec["fields"])
        st.write("**Findings**")
        if rec["findings"]:
            for f in rec["findings"]:
                st.error(f"[{f['severity']}] {f['rule']} — {f['detail']}")
        else:
            st.success("No issues found.")

    if st.button("Export PDF report"):
        path = build_report(results)
        st.success(f"Saved: {path}")
        with open(path, "rb") as fh:
            st.download_button("Download report", fh, file_name="audit_report.pdf")
