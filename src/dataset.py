"""Load labelled tokens from data/labels/*.json into feature matrix X, y."""
import json

import numpy as np

from . import config as C
from .features import token_features


def load_xy():
    X, y = [], []
    files = sorted(C.LABELS.glob("*.json"))
    if not files:
        raise FileNotFoundError("No labels found. Run: python -m src.generate_data")
    cls_idx = {c: i for i, c in enumerate(C.CLASSES)}
    for fp in files:
        tokens = json.loads(fp.read_text())["tokens"]
        for i, t in enumerate(tokens):
            X.append(token_features(tokens, i))
            y.append(cls_idx[t["label"]])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="int64")
