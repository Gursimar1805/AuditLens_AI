"""
Step 2 - Feature engineering.

Turn one text token (with its bounding box) into a numeric feature vector.
Features combine WHERE the token sits on the page (geometry) with WHAT it
looks like (regex/text flags) and its NEIGHBOURHOOD (nearest keyword to the
left on the same row). This is what lets a simple classifier learn which
tokens are totals, dates, invoice numbers, etc. — no deep OCR model needed.
"""
import re

from . import config as C

# Keywords that commonly sit just left of a field's value.
KEYWORDS = {
    "invoice": "kw_invoice", "inv": "kw_invoice", "no": "kw_invoice",
    "date": "kw_date", "dated": "kw_date",
    "total": "kw_total", "amount": "kw_total", "balance": "kw_total",
    "signatory": "kw_sign", "authorised": "kw_sign", "signature": "kw_sign",
}
KW_FEATURES = ["kw_invoice", "kw_date", "kw_total", "kw_sign"]

AMOUNT_RE = re.compile(r"^[₹$]?\d{1,3}(,\d{3})*(\.\d{1,2})?$|^\d+\.\d{1,2}$")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
INV_RE = re.compile(r"[A-Za-z]{2,}-?\d{3,}")

FEATURE_NAMES = [
    "x_center", "y_center", "width", "height", "x_left", "y_top",
    "n_chars", "n_digits", "digit_ratio",
    "is_amount", "is_date", "is_inv", "is_alpha", "has_currency", "has_colon",
] + KW_FEATURES
N_FEATURES = len(FEATURE_NAMES)


def _row_neighbours(tokens, i, y_tol=12):
    """Keywords appearing to the left of token i on roughly the same row."""
    x0, y0, x1, y1 = tokens[i]["bbox"]
    yc = (y0 + y1) / 2
    flags = {k: 0 for k in KW_FEATURES}
    for j, t in enumerate(tokens):
        if j == i:
            continue
        a0, b0, a1, b1 = t["bbox"]
        if abs((b0 + b1) / 2 - yc) < y_tol and a1 <= x0 + 5:
            key = KEYWORDS.get(t["text"].lower().strip(":"))
            if key:
                flags[key] = 1
    return flags


def token_features(tokens, i):
    t = tokens[i]
    x0, y0, x1, y1 = t["bbox"]
    s = t["text"]
    digits = sum(ch.isdigit() for ch in s)
    nb = _row_neighbours(tokens, i)
    feats = {
        "x_center": (x0 + x1) / 2 / C.PAGE_W,
        "y_center": (y0 + y1) / 2 / C.PAGE_H,
        "width": (x1 - x0) / C.PAGE_W,
        "height": (y1 - y0) / C.PAGE_H,
        "x_left": x0 / C.PAGE_W,
        "y_top": y0 / C.PAGE_H,
        "n_chars": min(len(s), 30) / 30,
        "n_digits": min(digits, 15) / 15,
        "digit_ratio": digits / max(len(s), 1),
        "is_amount": 1.0 if AMOUNT_RE.match(s) else 0.0,
        "is_date": 1.0 if DATE_RE.search(s) else 0.0,
        "is_inv": 1.0 if INV_RE.fullmatch(s) else 0.0,
        "is_alpha": 1.0 if s.isalpha() else 0.0,
        "has_currency": 1.0 if any(ch in s for ch in "₹$") else 0.0,
        "has_colon": 1.0 if ":" in s else 0.0,
    }
    feats.update(nb)
    return [feats[k] for k in FEATURE_NAMES]
