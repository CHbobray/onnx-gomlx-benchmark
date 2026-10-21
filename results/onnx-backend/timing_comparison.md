**Throughput** - full 10000-image test set per run

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Images/s | vs Go | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 179.6 | 14.2 | 175.2 | 207.8 | 55684 | 1.2x faster | 30 |
| Go + GoMLX (ONNX Runtime backend) | 222.8 | 30.5 | 219.0 | 277.8 | 44889 | 1.0x | 30 |
| Python + PyTorch | 1506.0 | 79.9 | 1515.7 | 1611.6 | 6640 | 6.8x slower | 30 |

**Latency** - one image per call (a single user request)

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Requests/s | vs Go | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 0.050 | 0.012 | 0.047 | 0.070 | 20078 | 1.6x faster | 500 |
| Go + GoMLX (ONNX Runtime backend) | 0.078 | 0.120 | 0.060 | 0.145 | 12860 | 1.0x | 500 |
| Python + PyTorch | 0.183 | 0.044 | 0.166 | 0.260 | 5454 | 2.4x slower | 500 |
