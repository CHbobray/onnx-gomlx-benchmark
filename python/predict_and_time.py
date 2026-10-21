"""Predict the 10,000-image test set in Python and time it with a Monte Carlo benchmark.

Two Python engines are measured:
  * pytorch      - the original trained model (models/fashion_cnn.pt)
  * onnxruntime  - the exported ONNX file, as an extra reference point

Outputs (in results/):
  python_predictions.csv  - one row per test image: label, predicted class, 10 logits
  python_timing.json      - timing statistics for every engine

Usage (from the repository root):
    python python/predict_and_time.py --runs 30 --batch-size 1000 --latency 500
"""

from __future__ import annotations

import argparse
import logging
import csv
import json
import platform
import statistics
import time
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch

import fashion_common as fc

log = logging.getLogger("predict_and_time")

PREDICTIONS_CSV = fc.RESULTS_DIR / "python_predictions.csv"
TIMING_JSON = fc.RESULTS_DIR / "python_timing.json"

Predictor = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30, help="timed Monte Carlo runs (1-1000)")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warm-up runs")
    parser.add_argument("--batch-size", type=int, default=1000, help="images per inference call")
    parser.add_argument("--latency", type=int, default=500, help="single-image requests to time")
    args = parser.parse_args()
    if not 1 <= args.runs <= 1000:
        parser.error("--runs must be between 1 and 1000")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if not 1 <= args.latency <= 10_000:
        parser.error("--latency must be between 1 and 10000")
    if not 1 <= args.batch_size <= 10_000:
        parser.error("--batch-size must be between 1 and 10000")
    return args


def make_pytorch_predictor() -> Predictor:
    model = fc.FashionCNN()
    model.load_state_dict(torch.load(fc.PYTORCH_WEIGHTS, weights_only=True))
    model.eval()

    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(batch)).numpy()

    return predict


def make_onnxruntime_predictor() -> Predictor:
    session = ort.InferenceSession(str(fc.ONNX_MODEL), providers=["CPUExecutionProvider"])

    def predict(batch: np.ndarray) -> np.ndarray:
        return session.run(["logits"], {"pixels": batch})[0]

    return predict


def predict_all(predict: Predictor, pixels: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the whole test set through `predict`, batch by batch."""
    batches = [predict(pixels[start:start + batch_size])
               for start in range(0, len(pixels), batch_size)]
    return np.concatenate(batches)


def time_engine(predict: Predictor, pixels: np.ndarray, args: argparse.Namespace) -> dict:
    """Monte Carlo benchmark: repeat the full test-set pass and summarize the run times."""
    for _ in range(args.warmup):
        predict_all(predict, pixels, args.batch_size)

    seconds = []
    for _ in range(args.runs):
        started = time.perf_counter()
        predict_all(predict, pixels, args.batch_size)
        seconds.append(time.perf_counter() - started)
    return summarize(seconds, len(pixels))


def time_latency(predict: Predictor, pixels: np.ndarray, samples: int) -> dict:
    """Time single-image calls one by one, as a server answering one request would."""
    warmup = 5
    for i in range(warmup):
        predict(pixels[i:i + 1])
    seconds = []
    for i in range(samples):
        image = pixels[i % len(pixels):i % len(pixels) + 1]
        started = time.perf_counter()
        predict(image)
        seconds.append(time.perf_counter() - started)
    return summarize(seconds, 1)


def summarize(seconds: list[float], images: int) -> dict:
    ordered = sorted(seconds)
    mean = statistics.fmean(ordered)
    return {
        "runs": len(ordered),
        "images_per_run": images,
        "mean_ms": mean * 1e3,
        "std_ms": (statistics.stdev(ordered) if len(ordered) > 1 else 0.0) * 1e3,
        "min_ms": ordered[0] * 1e3,
        "median_ms": statistics.median(ordered) * 1e3,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1e3,
        "max_ms": ordered[-1] * 1e3,
        "images_per_second": images / mean,
    }


def write_predictions(labels: np.ndarray, logits: np.ndarray) -> None:
    fc.RESULTS_DIR.mkdir(exist_ok=True)
    with PREDICTIONS_CSV.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["index", "label", "predicted"] + [f"logit_{k}" for k in range(logits.shape[1])])
        for index, (label, row) in enumerate(zip(labels, logits)):
            writer.writerow([index, int(label), int(row.argmax())] + [f"{v:.6f}" for v in row])


def peak_memory_mb() -> float:
    """Peak resident memory of this process, in MB, on Windows, Linux, or macOS."""
    if platform.system() == "Windows":
        import psutil  # the resource module does not exist on Windows
        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is reported in bytes on macOS and in KiB on Linux.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def main() -> None:
    args = parse_args()
    fc.setup_logging(log, "python_run_log.txt")
    labels = fc.read_idx_labels(fc.TEST_LABELS)
    pixels = fc.images_to_tensor(fc.read_idx_images(fc.TEST_IMAGES)).numpy()
    log.info(f"test set: {len(labels):,} images, batch size {args.batch_size}, "
          f"{args.runs} timed runs after {args.warmup} warm-ups")

    engines = {"pytorch": make_pytorch_predictor(), "onnxruntime": make_onnxruntime_predictor()}

    logits = predict_all(engines["pytorch"], pixels, args.batch_size)
    accuracy = float((logits.argmax(axis=1) == labels).mean())
    write_predictions(labels, logits)
    log.info(f"PyTorch accuracy {accuracy:.4f}; predictions written to "
          f"{PREDICTIONS_CSV.relative_to(fc.REPO_ROOT)}")

    ort_logits = predict_all(engines["onnxruntime"], pixels, args.batch_size)
    log.info(f"ONNX Runtime vs PyTorch: max |logit diff| = {np.abs(ort_logits - logits).max():.2e}, "
          f"class agreement = {(ort_logits.argmax(1) == logits.argmax(1)).mean():.4f}")

    timing = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "onnxruntime": ort.__version__,
        "batch_size": args.batch_size,
        "accuracy": accuracy,
        "engines": {},
    }
    for name, predict in engines.items():
        throughput = time_engine(predict, pixels, args)
        latency = time_latency(predict, pixels, args.latency)
        timing["engines"][name] = {"throughput": throughput, "latency": latency}
        log.info(f"{name:12s} full test set: mean {throughput['mean_ms']:8.1f} ms "
              f"(std {throughput['std_ms']:.1f}, {throughput['images_per_second']:,.0f} images/s); "
              f"single image: mean {latency['mean_ms']:.3f} ms, p95 {latency['p95_ms']:.3f} ms")
    timing["peak_memory_mb"] = peak_memory_mb()
    log.info(f"peak resident memory: {timing['peak_memory_mb']:.0f} MB")

    TIMING_JSON.write_text(json.dumps(timing, indent=2))
    log.info(f"timing written to {TIMING_JSON.relative_to(fc.REPO_ROOT)}")


if __name__ == "__main__":
    main()
