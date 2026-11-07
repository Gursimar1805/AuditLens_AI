"""
Step 5a - OCR (inference time only).

Reads words + bounding boxes from a document image with Tesseract. The
synthetic images are low-resolution (~85 DPI), which Tesseract reads poorly,
so we UPSCALE the image before OCR (big accuracy gain, fixes digit confusion
like 3 vs 8) and then scale the returned boxes back to the original
coordinates so they still line up with the model's features.

Requires the tesseract binary installed:
  Windows: install from github.com/UB-Mannheim/tesseract, then set the path below.
  Ubuntu:  sudo apt install tesseract-ocr
"""
import os
from PIL import Image
import pytesseract
from pytesseract import Output

# If tesseract is on your PATH you can delete this block; otherwise point it at
# your install (default Windows location shown).
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_SCALE = 3          # upscale factor before OCR; 3 works well for these images


def ocr_tokens(image_path, min_conf=30, scale=OCR_SCALE):
    """Return [{text, bbox:[x0,y0,x1,y1]}] in ORIGINAL image coordinates."""
    img = Image.open(image_path).convert("RGB")
    if scale and scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    s = scale or 1
    tokens = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            if float(data["conf"][i]) < min_conf:
                continue
        except ValueError:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        tokens.append({"text": text,
                       "bbox": [x / s, y / s, (x + w) / s, (y + h) / s]})
    return tokens