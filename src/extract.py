"""
Step 5b - Extraction (hybrid: trained model + robust keyword fallback).

The trained model tags each word. On clean data that is enough, but real OCR
(different Tesseract builds, noisy scans) can mislabel words, so for every key
field we also read it by its printed label - the value sitting to the right of
"Invoice No", "Date", "Subtotal", "Tax", "Total". We take the model's answer
when it has one and fall back to the label-based reading otherwise, so fields
no longer come back empty on real documents.
"""
import json
import re

import joblib
import numpy as np
import torch

from . import config as C
from .features import token_features
from .model import TokenClassifier, get_device

_model = None
_scaler = None
_device = None

INV_RE = re.compile(r"[A-Za-z]{2,}[-\s]?\d{3,}")
DATE_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def _load():
    global _model, _scaler, _device
    if _model is None:
        _device = get_device()
        _model = TokenClassifier().to(_device)
        _model.load_state_dict(torch.load(C.MODEL_PATH, map_location=_device))
        _model.eval()
        _scaler = joblib.load(C.SCALER_PATH)


def parse_amount(s):
    s = re.sub(r"[^\d.]", "", str(s))
    if s.count(".") > 1:                      # "35.157.87" -> keep last as decimal
        head, _, tail = s.rpartition(".")
        s = head.replace(".", "") + "." + tail
    try:
        return float(s)
    except ValueError:
        return None


def predict(tokens):
    _load()
    if not tokens:
        return []
    X = np.array([token_features(tokens, i) for i in range(len(tokens))], "float32")
    X = _scaler.transform(X)
    with torch.no_grad():
        prob = torch.softmax(_model(torch.tensor(X).to(_device)), 1).cpu().numpy()
    idx = prob.argmax(1)
    return [(C.CLASSES[j], float(prob[k, j])) for k, j in enumerate(idx)]


# ---------- label-anchored readers (robust to OCR noise) ----------

def _row_right(tokens, label_tok, y_tol=18):
    x1 = label_tok["bbox"][2]
    yc = (label_tok["bbox"][1] + label_tok["bbox"][3]) / 2
    row = [u for u in tokens if u is not label_tok and u["bbox"][0] >= x1 - 4
           and abs((u["bbox"][1] + u["bbox"][3]) / 2 - yc) < y_tol]
    row.sort(key=lambda u: u["bbox"][0])
    return row


def _match(tok_text, keyword, exact):
    tt = tok_text.lower().strip(": ")
    return tt == keyword if exact else keyword in tok_text.lower()


def amount_right_of(tokens, keyword, exact=False):
    """Rightmost numeric token on the label's row = the amount column.
    Rightmost beats '(18%)' and survives a decimal read as a comma."""
    for t in tokens:
        if _match(t["text"], keyword, exact):
            nums = [u for u in _row_right(tokens, t)
                    if any(c.isdigit() for c in u["text"])
                    and parse_amount(u["text"]) is not None]
            if nums:
                return parse_amount(nums[-1]["text"])
    return None


def text_right_of(tokens, keyword, pattern=None, exact=False):
    for t in tokens:
        if _match(t["text"], keyword, exact):
            for u in _row_right(tokens, t):
                if pattern is None or pattern.search(u["text"]):
                    return u["text"]
    return None


def _first_match(tokens, pattern):
    for t in tokens:
        if pattern.search(t["text"]):
            return t["text"]
    return None


def extract_fields(tokens):
    preds = predict(tokens)
    buckets = {c: [] for c in C.CLASSES}
    for tok, (lab, conf) in zip(tokens, preds):
        buckets[lab].append((tok, conf))

    def best(label):
        items = buckets[label]
        return max(items, key=lambda x: x[1])[0]["text"] if items else None

    # vendor: model's VENDOR tokens, else the topmost line of text on the page
    vend = sorted(buckets["VENDOR"], key=lambda x: x[0]["bbox"][1])
    vendor = " ".join(t["text"] for t, _ in vend) or None
    if not vendor:
        alpha = [t for t in tokens if any(c.isalpha() for c in t["text"])]
        if alpha:
            ty = min(t["bbox"][1] for t in alpha)
            row = sorted([t for t in alpha if abs(t["bbox"][1] - ty) < 16],
                         key=lambda t: t["bbox"][0])
            vendor = " ".join(t["text"] for t in row) or None

    # invoice no / date / total: model first, then label-anchored fallback
    invoice_no = best("INVOICE_NO") or text_right_of(tokens, "invoice", INV_RE) \
        or _first_match(tokens, INV_RE)
    date = best("DATE") or text_right_of(tokens, "date", DATE_RE) \
        or _first_match(tokens, DATE_RE)
    total = (parse_amount(best("TOTAL")) if best("TOTAL") else None)
    if total is None:
        total = amount_right_of(tokens, "total", exact=True)

    line_items = [v for v in (parse_amount(t["text"]) for t, _ in buckets["LINE_ITEM"])
                  if v is not None]

    has_sig = len(buckets["SIGNATURE"]) > 0
    if not has_sig:
        cap = next((t for t in tokens if "signatory" in t["text"].lower()), None)
        if cap:
            cy = cap["bbox"][3]
            has_sig = any(t["text"].isalpha() and t["bbox"][1] > cy
                          and t["bbox"][0] < C.PAGE_W * 0.4 for t in tokens)

    return {
        "vendor": vendor,
        "invoice_no": invoice_no,
        "date": date,
        "total": total,
        "line_items": line_items,
        "subtotal": amount_right_of(tokens, "subtotal"),
        "tax": amount_right_of(tokens, "tax"),
        "has_signature": has_sig,
    }


def extract_from_label_file(fp):
    tokens = json.loads(fp.read_text())["tokens"]
    return extract_fields(tokens)