"""
Orchestrator - ties OCR + extraction + verification together, and provides
an evaluation routine that measures flagging precision/recall against the
planted errors (your headline metric for the defence).

Run end-to-end on the synthetic images:
  python -m src.pipeline --mode images      # uses real Tesseract OCR
  python -m src.pipeline --mode labels      # uses clean tokens (no OCR)
  python -m src.pipeline --eval             # precision/recall vs planted errors
"""
import argparse
import json

from . import config as C
from . import db
from .extract import extract_fields, extract_from_label_file
from .ocr import ocr_tokens
from .verify import verify_batch


def process_image(image_path):
    return extract_fields(ocr_tokens(image_path))


def run_batch(mode="labels", limit=None):
    """mode='labels' uses clean generator tokens; 'images' uses OCR."""
    records = []
    files = sorted(C.LABELS.glob("*.json"))
    if limit:
        files = files[:limit]
    for fp in files:
        doc = fp.stem
        if mode == "images":
            fields = process_image(C.IMAGES / f"{doc}.png")
        else:
            fields = extract_from_label_file(fp)
        records.append({"doc": doc, "fields": fields})
    results = verify_batch(records)
    db.save_results(results)
    return results


def evaluate(mode="labels", limit=None):
    """Compare predicted error flags to the planted-error ground truth."""
    meta = json.loads((C.DATA / "batch_meta.json").read_text())
    truth = {m["doc"]: set(m["planted_errors"]) for m in meta}
    # Duplicate invoice is a batch property: every doc sharing a reused invoice
    # number is a true positive, not just the second copy that was planted.
    inv_docs = {}
    for m in meta:
        inv_docs.setdefault(m["invoice_no"], []).append(m["doc"])
    for docs in inv_docs.values():
        if len(docs) > 1:
            for d in docs:
                truth[d].add("duplicate_invoice")
    results = run_batch(mode=mode, limit=limit)
    rules = ["total_mismatch", "missing_signature", "date_out_of_range",
             "duplicate_invoice"]
    tp = fp = fn = 0
    per_rule = {r: [0, 0, 0] for r in rules}  # tp, fp, fn
    for res in results:
        pred = {f["rule"] for f in res["findings"]} & set(rules)
        gt = truth.get(res["doc"], set()) & set(rules)
        for r in rules:
            if r in pred and r in gt: per_rule[r][0] += 1; tp += 1
            elif r in pred and r not in gt: per_rule[r][1] += 1; fp += 1
            elif r not in pred and r in gt: per_rule[r][2] += 1; fn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"\n=== Flagging evaluation (mode={mode}) ===")
    print(f"Overall  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  "
          f"(tp={tp} fp={fp} fn={fn})")
    print("\nPer rule (tp / fp / fn):")
    for r in rules:
        t, f, n = per_rule[r]
        p = t / (t + f) if t + f else 0
        rc = t / (t + n) if t + n else 0
        print(f"  {r:20s} tp={t:3d} fp={f:3d} fn={n:3d}  P={p:.2f} R={rc:.2f}")
    return {"precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["labels", "images"], default="labels")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.eval:
        evaluate(mode=a.mode, limit=a.limit)
    else:
        res = run_batch(mode=a.mode, limit=a.limit)
        flagged = sum(1 for r in res if r["severity"] != "OK")
        print(f"Processed {len(res)} docs; {flagged} flagged. Saved to DB.")
