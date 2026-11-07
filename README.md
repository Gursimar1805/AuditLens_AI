# AuditLens AI — Intelligent Document Review for Audit Evidence

A document-intelligence tool that reads invoices, extracts key fields with a
model **you train yourself**, and flags audit inconsistencies for a reviewer.

**No LLM API anywhere.** The "AI" is a small PyTorch classifier trained from
scratch on your own synthetic data, plus Tesseract OCR for reading images.

---

## What it does (the pipeline)

```
Invoice image
   │
   ▼  (1) OCR            Tesseract reads words + their boxes
   ▼  (2) Features       each word → geometry + text + neighbour-keyword vector
   ▼  (3) Model          your trained MLP tags each word: TOTAL / DATE / ... 
   ▼  (4) Assemble       tagged words → structured record (invoice no, total…)
   ▼  (5) Verify         rule engine checks totals, signature, dates, duplicates
   ▼  (6) Rank + review  findings sorted HIGH→LOW in a dashboard, export PDF
```

The model classes are: `OTHER, VENDOR, INVOICE_NO, DATE, TOTAL, LINE_ITEM,
SIGNATURE`.

---

## Setup

**1. Install Tesseract (the OCR engine)**
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`

**2. Python environment**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
You have a local GPU — PyTorch will use it automatically for training. CPU
also works (the model is tiny).

---

## Build it step by step

Run every command from the project root.

### Step 1 — Generate labelled synthetic data
No real financial data. Renders invoices as images and records each word's box
and field label. ~20% get planted errors (wrong total, missing signature,
out-of-period date, duplicate invoice number).
```bash
python -m src.generate_data --n 300
```
→ `data/images/*.png`, `data/labels/*.json`, `data/batch_meta.json`

### Step 2 & 3 — Features + the model
`src/features.py` turns a word into a numeric vector (where it sits + what it
looks like + nearby keywords). `src/model.py` is the PyTorch MLP. Nothing to
run yet — they're used by training.

### Step 4 — Train your model
```bash
python -m src.train --epochs 40
```
Prints a per-class report and saves `models/token_classifier.pt` +
`models/scaler.joblib`.

### Step 5 — Run the full pipeline (OCR → extract → verify)
```bash
python -m src.pipeline --mode images     # real OCR on the images
python -m src.pipeline --mode labels     # skip OCR, use clean tokens (fast)
```

### Step 6 — Measure accuracy (your headline metric)
```bash
python -m src.pipeline --eval --mode images
```
Reports flagging precision / recall / F1 against the planted errors.

### Step 7 — Reviewer dashboard
```bash
streamlit run app/dashboard.py
```
Upload images or use the sample batch; review findings ranked by severity with
the document alongside each flag; export a PDF report.

### Step 8 — (optional) API backend
```bash
uvicorn app.api:app --reload      # POST an image to /review
```

---

## Results (on 300 synthetic invoices)

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Field-tagging model (test tokens) | 1.00 | 1.00 | 1.00 |
| Verification logic (clean fields) | 1.00 | 1.00 | 1.00 |
| **End-to-end with real OCR** | **1.00** | **0.90** | **0.94** |

Per-rule end-to-end: total-mismatch, missing-signature and date-range all
≈1.00 P/R. Duplicate-invoice recall ≈0.75 — the only weak spot, because OCR
sometimes misreads an invoice number so two copies no longer match.

---

## How this maps to your defence slides

| Slide | Where it lives |
|---|---|
| Proposed Methodology | the 6-step pipeline above |
| System Architecture | `src/` modules + `app/` |
| Technology Stack | Tesseract, PyTorch, scikit-learn, Streamlit, FastAPI, SQLite |
| Verification Logic | `src/verify.py` |
| Expected Outcomes | the results table |
| Data & Ethics | `src/generate_data.py` — synthetic only |

---

## Honest limitations (good viva answers / future scope)

- Trained and tested on **synthetic** invoices of one layout. Real-world,
  multi-layout documents will score lower.
- Field tagging uses position + text patterns; it is **layout-sensitive**.
  Future: fine-tune **LayoutLMv3** for layout-independent extraction.
- Duplicate detection needs **fuzzy matching** of invoice numbers to survive
  OCR errors.
- Tax rate for total reconciliation is assumed 18% — make it configurable or
  extract the printed rate.

---

## Project layout
```
auditlens_ai/
├── src/
│   ├── config.py        paths, classes, audit period
│   ├── generate_data.py Step 1 — synthetic labelled invoices
│   ├── features.py      Step 2 — token → feature vector
│   ├── dataset.py       load features + labels
│   ├── model.py         Step 3 — PyTorch MLP
│   ├── train.py         Step 4 — train + evaluate + save
│   ├── ocr.py           Step 5a — Tesseract OCR
│   ├── extract.py       Step 5b — tag tokens → structured fields
│   ├── verify.py        Step 6 — rule-based audit checks + severity
│   ├── pipeline.py      orchestrator + evaluation
│   ├── report.py        Step 8 — PDF report
│   └── db.py            SQLite persistence
├── app/
│   ├── dashboard.py     Streamlit reviewer UI
│   └── api.py           FastAPI backend
├── requirements.txt
└── README.md
```
