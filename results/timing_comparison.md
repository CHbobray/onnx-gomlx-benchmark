**Throughput** - full 10000-image test set per run

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Images/s | vs Go | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 179.6 | 14.2 | 175.2 | 207.8 | 55684 | 166.5x faster | 30 |
| Python + PyTorch | 1506.0 | 79.9 | 1515.7 | 1611.6 | 6640 | 19.8x faster | 30 |
| Go + GoMLX (pure-Go backend) | 29893.0 | 367.8 | 29894.6 | 30498.8 | 335 | 1.0x | 30 |

**Latency** - one image per call (a single user request)

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Requests/s | vs Go | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 0.050 | 0.012 | 0.047 | 0.070 | 20078 | 74.0x faster | 500 |
| Python + PyTorch | 0.183 | 0.044 | 0.166 | 0.260 | 5454 | 20.1x faster | 500 |
| Go + GoMLX (pure-Go backend) | 3.688 | 0.828 | 3.451 | 5.144 | 271 | 1.0x | 500 |
