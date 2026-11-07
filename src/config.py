"""Central configuration for AuditLens AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
MODELS = ROOT / "models"
for d in (DATA, IMAGES, LABELS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# Field classes the model learns to tag on each OCR/word token.
CLASSES = ["OTHER", "VENDOR", "INVOICE_NO", "DATE", "TOTAL", "LINE_ITEM", "SIGNATURE"]

# Page canvas size (pixels). Used to normalise geometric features.
PAGE_W, PAGE_H = 1000, 1400

# The audit period. Dates outside this window are flagged by the verifier.
AUDIT_START = "2026-01-01"
AUDIT_END = "2026-03-31"

SEED = 42

# Saved artefacts produced by train.py
MODEL_PATH = MODELS / "token_classifier.pt"
SCALER_PATH = MODELS / "scaler.joblib"
META_PATH = MODELS / "model_meta.json"
DB_PATH = DATA / "auditlens.db"
