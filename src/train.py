"""
Step 4 - Train and evaluate your model.

Loads the labelled tokens, standardises features, trains the MLP with a
class-weighted loss (fields are rare vs OTHER), prints a per-class report,
and saves the weights + scaler for inference.

Run:  python -m src.train --epochs 40
"""
import argparse
import json

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from . import config as C
from .dataset import load_xy
from .model import TokenClassifier, get_device


def main(epochs=40, lr=1e-3, batch=256):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = get_device()
    print("Device:", device)

    X, y = load_xy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=C.SEED, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # class weights (inverse frequency) so rare fields aren't ignored
    counts = np.bincount(ytr, minlength=len(C.CLASSES))
    weights = torch.tensor(
        (counts.sum() / (len(C.CLASSES) * np.maximum(counts, 1))), dtype=torch.float32
    ).to(device)

    Xtr_t = torch.tensor(Xtr).to(device)
    ytr_t = torch.tensor(ytr).to(device)

    model = TokenClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    n = len(Xtr_t)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {total / n:.4f}")

    # ---- evaluate ----
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte).to(device)).argmax(1).cpu().numpy()
    print("\n=== Test set per-class report ===")
    print(classification_report(yte, pred, target_names=C.CLASSES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, pred))

    # ---- save artefacts ----
    torch.save(model.state_dict(), C.MODEL_PATH)
    joblib.dump(scaler, C.SCALER_PATH)
    C.META_PATH.write_text(json.dumps({"classes": C.CLASSES}, indent=2))
    print(f"\nSaved model -> {C.MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    main(ap.parse_args().epochs, ap.parse_args().lr)
