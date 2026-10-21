"""Train the Fashion-MNIST CNN, save its weights, and export it to ONNX.

Usage (from the repository root):
    python python/train_and_export.py --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import fashion_common as fc

log = logging.getLogger("train_and_export")

ONNX_OPSET = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5, help="training epochs (1-50)")
    parser.add_argument("--batch-size", type=int, default=128, help="training batch size")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 50:
        parser.error("--epochs must be between 1 and 50")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def train(model: nn.Module, loader: DataLoader, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        mean_loss = total_loss / len(loader.dataset)
        log.info(f"epoch {epoch}/{epochs}  loss={mean_loss:.4f}  ({time.perf_counter() - started:.1f}s)")


def test_accuracy(model: nn.Module, images: torch.Tensor, labels: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        predicted = model(images).argmax(dim=1).numpy()
    return float((predicted == labels).mean())


def export_onnx(model: nn.Module) -> None:
    """Export with a dynamic batch dimension so any batch size can be used later."""
    model.eval()
    example = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        str(fc.ONNX_MODEL),
        input_names=["pixels"],
        output_names=["logits"],
        dynamic_axes={"pixels": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    exported = onnx.load(str(fc.ONNX_MODEL))
    removed = strip_default_maxpool_dilations(exported)
    onnx.checker.check_model(exported)
    onnx.save(exported, str(fc.ONNX_MODEL))
    log.info(f"removed default dilations=[1, 1] from {removed} MaxPool node(s) for GoMLX compatibility")
    ops = sorted({node.op_type for node in exported.graph.node})
    log.info(f"exported {fc.ONNX_MODEL.name} (opset {ONNX_OPSET}); ops used: {', '.join(ops)}")


def strip_default_maxpool_dilations(model_proto: onnx.ModelProto) -> int:
    """Drop MaxPool 'dilations' attributes that equal the ONNX default of all ones.

    PyTorch always writes the attribute, but onnx-gomlx rejects any MaxPool that
    carries it, even with default values. Removing a default-valued attribute
    leaves the model's meaning unchanged. Returns the number of nodes edited.
    """
    removed = 0
    for node in model_proto.graph.node:
        if node.op_type != "MaxPool":
            continue
        for attribute in list(node.attribute):
            if attribute.name == "dilations" and all(d == 1 for d in attribute.ints):
                node.attribute.remove(attribute)
                removed += 1
    return removed


def main() -> None:
    args = parse_args()
    fc.setup_logging(log, "training_log.txt")
    torch.manual_seed(args.seed)
    fc.MODELS_DIR.mkdir(exist_ok=True)

    train_images = fc.images_to_tensor(fc.read_idx_images(fc.TRAIN_IMAGES))
    train_labels = torch.from_numpy(fc.read_idx_labels(fc.TRAIN_LABELS).astype(np.int64))
    loader = DataLoader(TensorDataset(train_images, train_labels),
                        batch_size=args.batch_size, shuffle=True)

    model = fc.FashionCNN()
    log.info(f"training FashionCNN ({sum(p.numel() for p in model.parameters()):,} parameters) "
          f"on {len(train_labels):,} images")
    train(model, loader, args.epochs)

    test_images = fc.images_to_tensor(fc.read_idx_images(fc.TEST_IMAGES))
    test_labels = fc.read_idx_labels(fc.TEST_LABELS)
    log.info(f"test accuracy: {test_accuracy(model, test_images, test_labels):.4f}")

    torch.save(model.state_dict(), fc.PYTORCH_WEIGHTS)
    log.info(f"saved weights to {fc.PYTORCH_WEIGHTS.relative_to(fc.REPO_ROOT)}")
    export_onnx(model)


if __name__ == "__main__":
    main()
