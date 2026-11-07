"""
Step 1 - Synthetic data generation.

We render fake invoices as PNG images with PIL and, crucially, record the
bounding box + field label of every word we draw. That gives us a fully
labelled token dataset WITHOUT any manual annotation and WITHOUT real
financial data. Some invoices get deliberately planted errors so the
verification engine (Step 5) has something to catch.

Run:  python -m src.generate_data --n 300
"""
import argparse
import json
import random
from datetime import date, timedelta

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from . import config as C

fake = Faker()


def _font(size):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rand_date(in_period=True):
    if in_period:
        start = date.fromisoformat(C.AUDIT_START)
        end = date.fromisoformat(C.AUDIT_END)
    else:  # out of audit period -> planted error
        start = date(2025, 6, 1)
        end = date(2025, 12, 20)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


class TokenCanvas:
    """Draws text and remembers each word's bbox + label."""

    def __init__(self):
        self.img = Image.new("RGB", (C.PAGE_W, C.PAGE_H), "white")
        self.draw = ImageDraw.Draw(self.img)
        self.tokens = []  # list of {text, bbox:[x0,y0,x1,y1], label}

    def text(self, x, y, s, font, label="OTHER"):
        """Draw a string; store each word as a separately-boxed token."""
        cx = x
        for word in str(s).split():
            w = self.draw.textlength(word + " ", font=font)
            ww = self.draw.textlength(word, font=font)
            asc, desc = font.getmetrics()
            h = asc + desc
            self.draw.text((cx, y), word, fill="black", font=font)
            self.tokens.append(
                {"text": word, "bbox": [cx, y, cx + ww, y + h], "label": label}
            )
            cx += w
        return cx


def build_invoice(dup_pool, force_errors=None):
    """Return (canvas, meta). dup_pool: list of prior invoice numbers for
    duplicate-error injection. force_errors: optional set to force."""
    c = TokenCanvas()
    f_big, f_hd, f_md, f_sm = _font(34), _font(22), _font(18), _font(16)

    errors = set()
    roll = lambda p: random.random() < p
    inv_no = f"INV-{random.randint(10000, 99999)}"

    # ---- planted errors ----
    if force_errors is None:
        if roll(0.20) and dup_pool:
            inv_no = random.choice(dup_pool)       # duplicate invoice number
            errors.add("duplicate_invoice")
        date_bad = roll(0.20)                       # date outside audit window
        miss_sig = roll(0.20)                       # missing signature
        total_bad = roll(0.25)                      # total != sum(line items)
    else:
        date_bad = "date_out_of_range" in force_errors
        miss_sig = "missing_signature" in force_errors
        total_bad = "total_mismatch" in force_errors

    inv_date = _rand_date(in_period=not date_bad)
    if date_bad:
        errors.add("date_out_of_range")

    vendor = fake.company()

    # ---- header ----
    c.text(60, 50, vendor, f_big, "VENDOR")
    c.text(60, 100, fake.street_address(), f_sm, "OTHER")
    c.text(60, 125, f"{fake.city()}, {fake.state()}", f_sm, "OTHER")

    c.text(680, 55, "Invoice No:", f_md, "OTHER")
    c.text(820, 55, inv_no, f_md, "INVOICE_NO")
    c.text(680, 85, "Date:", f_md, "OTHER")
    c.text(820, 85, inv_date, f_md, "DATE")

    c.text(60, 200, "Bill To:", f_md, "OTHER")
    c.text(60, 225, fake.name(), f_sm, "OTHER")

    # ---- line items table ----
    c.text(60, 300, "Description", f_md, "OTHER")
    c.text(500, 300, "Qty", f_md, "OTHER")
    c.text(620, 300, "Unit Price", f_md, "OTHER")
    c.text(820, 300, "Amount", f_md, "OTHER")

    y = 340
    line_amounts = []
    for _ in range(random.randint(3, 6)):
        qty = random.randint(1, 9)
        unit = round(random.uniform(50, 2000), 2)
        amt = round(qty * unit, 2)
        line_amounts.append(amt)
        c.text(60, y, fake.bs().title()[:28], f_sm, "OTHER")
        c.text(500, y, str(qty), f_sm, "OTHER")
        c.text(620, y, f"{unit:,.2f}", f_sm, "OTHER")
        c.text(820, y, f"{amt:,.2f}", f_sm, "LINE_ITEM")
        y += 34

    subtotal = round(sum(line_amounts), 2)
    tax = round(subtotal * 0.18, 2)
    true_total = round(subtotal + tax, 2)
    printed_total = true_total
    if total_bad:                                   # corrupt the printed total
        printed_total = round(true_total * random.uniform(1.05, 1.4), 2)
        errors.add("total_mismatch")

    y += 20
    c.text(620, y, "Subtotal:", f_md, "OTHER")
    c.text(820, y, f"{subtotal:,.2f}", f_md, "OTHER")
    c.text(620, y + 34, "Tax (18%):", f_md, "OTHER")
    c.text(820, y + 34, f"{tax:,.2f}", f_md, "OTHER")
    c.text(620, y + 72, "Total:", f_hd, "OTHER")
    c.text(820, y + 72, f"{printed_total:,.2f}", f_hd, "TOTAL")

    # ---- signature ----
    if not miss_sig:
        c.text(60, C.PAGE_H - 120, "Authorised Signatory", f_sm, "OTHER")
        c.text(60, C.PAGE_H - 90, fake.first_name(), _font(26), "SIGNATURE")
    else:
        c.text(60, C.PAGE_H - 120, "Authorised Signatory", f_sm, "OTHER")
        errors.add("missing_signature")

    meta = {
        "invoice_no": inv_no,
        "date": inv_date,
        "vendor": vendor,
        "printed_total": printed_total,
        "line_items": line_amounts,
        "computed_total": true_total,
        "has_signature": not miss_sig,
        "planted_errors": sorted(errors),
    }
    return c, meta


def main(n):
    random.seed(C.SEED)
    Faker.seed(C.SEED)
    batch, dup_pool = [], []
    for i in range(n):
        c, meta = build_invoice(dup_pool)
        if "duplicate_invoice" not in meta["planted_errors"]:
            dup_pool.append(meta["invoice_no"])
        stem = f"doc_{i:04d}"
        c.img.save(C.IMAGES / f"{stem}.png")
        (C.LABELS / f"{stem}.json").write_text(
            json.dumps({"tokens": c.tokens, "meta": meta}, indent=2)
        )
        meta["doc"] = stem
        batch.append(meta)
    (C.DATA / "batch_meta.json").write_text(json.dumps(batch, indent=2))
    n_err = sum(1 for m in batch if m["planted_errors"])
    print(f"Generated {n} invoices -> {C.IMAGES}")
    print(f"{n_err} contain at least one planted error.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of invoices")
    main(ap.parse_args().n)
