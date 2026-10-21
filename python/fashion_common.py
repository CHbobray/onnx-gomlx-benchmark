"""Shared pieces for the Python side: data loading, the CNN definition, and paths.

The model takes raw pixel values (0-255, float32, shape [N, 1, 28, 28]) and does
its own normalization in forward(). Baking preprocessing into the network means
the exported ONNX file is self-contained: the Go program only has to convert
bytes to float32, so Python and Go cannot drift apart on preprocessing.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

TRAIN_IMAGES = DATA_DIR / "train-images-idx3-ubyte.gz"
TRAIN_LABELS = DATA_DIR / "train-labels-idx1-ubyte.gz"
TEST_IMAGES = DATA_DIR / "t10k-images-idx3-ubyte.gz"
TEST_LABELS = DATA_DIR / "t10k-labels-idx1-ubyte.gz"

PYTORCH_WEIGHTS = MODELS_DIR / "fashion_cnn.pt"
ONNX_MODEL = MODELS_DIR / "fashion_cnn.onnx"

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)

# Fashion-MNIST training-set statistics, on the 0-255 pixel scale.
PIXEL_MEAN = 0.2860 * 255.0
PIXEL_STD = 0.3530 * 255.0

IDX_IMAGES_MAGIC = 2051
IDX_LABELS_MAGIC = 2049


def read_idx_images(path: Path) -> np.ndarray:
    """Read a gzipped IDX image file into a uint8 array of shape [N, 28, 28]."""
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != IDX_IMAGES_MAGIC:
            raise ValueError(f"{path}: not an IDX image file (magic {magic})")
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    return pixels.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    """Read a gzipped IDX label file into a uint8 array of shape [N]."""
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != IDX_LABELS_MAGIC:
            raise ValueError(f"{path}: not an IDX label file (magic {magic})")
        labels = np.frombuffer(stream.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"{path}: header says {count} labels, found {labels.size}")
    return labels


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert uint8 [N, 28, 28] images to the model's float32 [N, 1, 28, 28] input."""
    return torch.from_numpy(images.astype(np.float32)).unsqueeze(1)


class FashionCNN(nn.Module):
    """Two conv blocks followed by a small classifier head; ~0.2M parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, raw_pixels: torch.Tensor) -> torch.Tensor:
        normalized = (raw_pixels - PIXEL_MEAN) / PIXEL_STD
        return self.classifier(self.features(normalized))
