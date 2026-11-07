"""
Optional FastAPI backend — exposes the pipeline as an HTTP endpoint so the
extraction/verification can be queried programmatically.

Run:  uvicorn app.api:app --reload
Then POST an image to /review  (multipart form field 'file').
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tempfile

from fastapi import FastAPI, UploadFile, File

from src.extract import extract_fields
from src.ocr import ocr_tokens
from src.verify import verify_document

app = FastAPI(title="AuditLens AI")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review")
async def review(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    fields = extract_fields(ocr_tokens(path))
    findings = verify_document(fields)
    severity = max((f["severity"] for f in findings),
                   key=lambda s: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(s, 0),
                   default="OK")
    return {"file": file.filename, "severity": severity,
            "fields": fields, "findings": findings}
