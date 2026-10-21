"""Unit tests for the Python side. Run from the repository root:  pytest python"""

from __future__ import annotations

import gzip
import struct

import numpy as np
import onnx
import pytest
import torch
from onnx import helper

import fashion_common as fc
from predict_and_time import summarize
from train_and_export import strip_default_maxpool_dilations


def write_idx_images(path, images: np.ndarray) -> None:
    count, rows, cols = images.shape
    with gzip.open(path, "wb") as out:
        out.write(struct.pack(">IIII", fc.IDX_IMAGES_MAGIC, count, rows, cols))
        out.write(images.astype(np.uint8).tobytes())


def test_idx_round_trip(tmp_path):
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    path = tmp_path / "images.gz"
    write_idx_images(path, images)
    assert np.array_equal(fc.read_idx_images(path), images)


def test_idx_rejects_label_file_as_images(tmp_path):
    path = tmp_path / "labels.gz"
    with gzip.open(path, "wb") as out:
        out.write(struct.pack(">II", fc.IDX_LABELS_MAGIC, 20) + bytes(range(20)))
    assert fc.read_idx_labels(path).tolist() == list(range(20))
    with pytest.raises(ValueError, match="not an IDX image"):
        fc.read_idx_images(path)


def test_model_output_shape_and_normalization():
    model = fc.FashionCNN().eval()
    pixels = fc.images_to_tensor(np.full((4, 28, 28), 255, dtype=np.uint8))
    assert pixels.shape == (4, 1, 28, 28)
    with torch.no_grad():
        assert model(pixels).shape == (4, len(fc.CLASS_NAMES))


def test_strip_only_default_dilations():
    default = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], dilations=[1, 1])
    dilated = helper.make_node("MaxPool", ["y"], ["z"], kernel_shape=[2, 2], dilations=[2, 2])
    graph = helper.make_graph([default, dilated], "g", [], [])
    model = helper.make_model(graph)

    assert strip_default_maxpool_dilations(model) == 1
    names = [[a.name for a in node.attribute] for node in model.graph.node]
    assert "dilations" not in names[0]
    assert "dilations" in names[1]  # non-default dilation must be preserved


def test_summarize_statistics():
    stats = summarize([0.04, 0.01, 0.03, 0.02], images=1000)
    assert stats["mean_ms"] == pytest.approx(25)
    assert stats["median_ms"] == pytest.approx(25)
    assert stats["min_ms"] == pytest.approx(10)
    assert stats["images_per_second"] == pytest.approx(40_000)


@pytest.mark.skipif(not fc.ONNX_MODEL.exists() or not fc.PYTORCH_WEIGHTS.exists(),
                    reason="run train_and_export.py first")
def test_exported_onnx_matches_pytorch():
    import onnxruntime as ort

    model = fc.FashionCNN()
    model.load_state_dict(torch.load(fc.PYTORCH_WEIGHTS, weights_only=True))
    model.eval()
    pixels = fc.images_to_tensor(fc.read_idx_images(fc.TEST_IMAGES)[:32]).numpy()
    with torch.no_grad():
        expected = model(torch.from_numpy(pixels)).numpy()
    session = ort.InferenceSession(str(fc.ONNX_MODEL), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"pixels": pixels})[0]
    assert np.abs(actual - expected).max() < 1e-4
    onnx.checker.check_model(onnx.load(str(fc.ONNX_MODEL)))
