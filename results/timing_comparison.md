**Throughput** - full 10000-image test set per run

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Images/s | vs Go | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 100.4 | 1.854 | 100.2 | 104.4 | 99585 | 146.1x faster | 30 |
| Python + PyTorch | 463.6 | 58.1 | 448.9 | 616.4 | 21568 | 31.6x faster | 30 |
| Go + GoMLX (pure-Go backend) | 14670.7 | 5765.0 | 17392.7 | 22995.3 | 682 | 1.0x | 30 |

**Latency** - one image per call (a single user request)

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Requests/s | vs Go | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 0.076 | 0.005 | 0.075 | 0.085 | 13155 | 32.6x faster | 500 |
| Python + PyTorch | 0.218 | 0.443 | 0.180 | 0.334 | 4593 | 11.4x faster | 500 |
| Go + GoMLX (pure-Go backend) | 2.481 | 2.161 | 2.053 | 6.668 | 403 | 1.0x | 500 |
