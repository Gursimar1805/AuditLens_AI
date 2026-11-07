"""
Step 3 - The model you build yourself.

A small Multi-Layer Perceptron (MLP) token classifier in PyTorch. It takes
one token's feature vector and predicts its field class. No pretrained LLM,
no external API — you train these weights from scratch on your own data.
Runs on CPU; uses your GPU automatically if available.
"""
import torch
import torch.nn as nn

from .features import N_FEATURES
from . import config as C


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TokenClassifier(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_classes=len(C.CLASSES), hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
