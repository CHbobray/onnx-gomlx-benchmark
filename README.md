# Python to Go with ONNX: Fashion-MNIST CNN in PyTorch and GoMLX

**Question from management:** can we develop deep learning models in Python, export them to a common format, and run them in production with Go?

**Short answer:** *yes. How fast it runs depends on which GoMLX backend executes the model.*

- A PyTorch convolutional network, exported to ONNX and run by GoMLX in Go, reproduces Python's predictions **exactly**: 10,000 of 10,000 test images get the same class, and the largest difference in any output score is 0.0000076.
- GoMLX runs a model through interchangeable [backends](#gomlx-backends-tried). The speed results differ a great deal between them:
  - **ONNX Runtime backend** (Linux/macOS; needs cgo): Go classifies the test set **about 7x faster than PyTorch**, within about 25% of ONNX Runtime called from Python.
  - **Pure-Go backend** (the only one that builds into a dependency-free Windows `.exe`): **about 20x slower than PyTorch**. A single prediction still takes only about 3.7 ms.
- The pure-Go build is a **single 14 MB Windows `.exe`** with no Python, C libraries, or runtime installs.

Details, numbers, and a recommendation are in [Results](#results) and [Recommendation](#recommendation-for-management).

---

## Contents

- [What the project does](#what-the-project-does)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Step-by-step: reproduce everything on Windows](#step-by-step-reproduce-everything-on-windows)
- [Building the Windows executable](#building-the-windows-executable)
- [Using the Go application](#using-the-go-application)
- [Testing, benchmarks, and profiling](#testing-benchmarks-and-profiling)
- [Results](#results)
- [GoMLX backends tried](#gomlx-backends-tried)
- [Recommendation for management](#recommendation-for-management)
- [Issues found along the way](#issues-found-along-the-way)
- [Design notes](#design-notes)
- [AI-assistant dialog](#ai-assistant-dialog)
- [References](#references)

---

## What the project does

| Step (from the assignment) | Where it happens |
|---|---|
| 1. Develop a deep learning model in PyTorch | `python/fashion_common.py` defines the CNN; `python/train_and_export.py` trains it |
| 2. Predict a large test set in Python and time it | `python/predict_and_time.py` (10,000 images, Monte Carlo timing) |
| 3. Export the model to ONNX | `python/train_and_export.py` writes `models/fashion_cnn.onnx` |
| 4. Read the ONNX model into GoMLX | `internal/engine` (uses `onnx-gomlx`) |
| 5. Run it in Go, check predictions match, and time it | `cmd/fashion-go` |
| 6. Compare Python and Go execution times | [Results](#results) below, and `results/timing_comparison.md` |

**Data.** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando, MIT license): 28x28 grayscale photos of clothing in 10 classes, 60,000 for training and **10,000 for testing**. The original compressed files are in `data/`.

**Model.** A small convolutional neural network (206,922 parameters):

```
pixels [N,1,28,28], raw 0-255
  -> normalize (inside the model)
  -> Conv 3x3, 16 filters -> ReLU -> MaxPool 2x2      (28x28 -> 14x14)
  -> Conv 3x3, 32 filters -> ReLU -> MaxPool 2x2      (14x14 -> 7x7)
  -> Flatten -> Dense 128 -> ReLU -> Dense 10 (class scores, "logits")
```

Trained for 5 epochs with Adam, it reaches **90.34% test accuracy**. Normalization is built into the network so the ONNX file is self-contained: Go only has to turn bytes into numbers, and the two languages cannot disagree about preprocessing.

---

## Repository layout

```
.
├── README.md                  this file
├── build.ps1                  Windows: test, build bin\fashion-go.exe, optionally run it
├── go.mod, go.sum             Go module definition and dependency checksums
├── pytest.ini                 tells pytest where the Python tests are
│
├── data/                      Fashion-MNIST in its original IDX format (gzip)
│   ├── train-images-idx3-ubyte.gz   60,000 training images
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz    10,000 test images  (used by Python and Go)
│   └── t10k-labels-idx1-ubyte.gz
│
├── python/
│   ├── requirements.txt       Python packages
│   ├── fashion_common.py      shared: data reader, CNN definition, file paths
│   ├── train_and_export.py    trains the CNN, saves weights, exports + cleans ONNX
│   ├── predict_and_time.py    Python predictions + Monte Carlo timing (PyTorch, ONNX Runtime)
│   └── test_pipeline.py       pytest unit tests
│
├── models/
│   ├── fashion_cnn.pt         trained PyTorch weights
│   └── fashion_cnn.onnx       the exported model that Go loads
│
├── cmd/fashion-go/            the Go application (command-line program)
│   ├── main.go                program flow: load, predict, compare, benchmark, report
│   ├── config.go              flags, input validation, locating the project folder
│   ├── report.go              JSON output and Markdown timing tables
│   └── main_test.go
│
├── internal/                  reusable Go packages, each with unit tests
│   ├── idx/                   reads IDX image/label files (gzip or raw)
│   ├── engine/                loads an ONNX classifier into GoMLX and runs it
│   │                          (backend_onnx.go adds the optional ONNX Runtime backend)
│   ├── compare/               reads/writes prediction CSVs, measures agreement
│   └── bench/                 Monte Carlo timing statistics and memory sampling
│
├── results/                   outputs from the runs reported below
│   ├── python_predictions.csv   label, predicted class, 10 logits per test image
│   ├── go_predictions.csv       the same, from Go
│   ├── python_timing.json       Python timing statistics
│   ├── go_timing.json           Go timing, memory, and agreement statistics
│   ├── timing_comparison.md     side-by-side tables (written by the Go app)
│   ├── python_run_log.txt       log written by predict_and_time.py
│   ├── go_run_log.txt           log written by fashion-go (start time, command, output, outcome)
│   ├── go_profile_summary.txt   CPU and heap profile of the whole Go program
│   ├── go_component_benchmarks.txt  time and memory of each Go component (go test -bench)
│   └── onnx-backend/            the same Go run on GoMLX's ONNX Runtime backend
│
└── docs/                      AI-assistant dialog: ai-assistant-dialog.pdf and .md (plain text)
```

---

## Requirements

| Tool | Version | Why |
|---|---|---|
| [Go](https://go.dev/dl/) | **1.27 or newer** | `onnx-gomlx` v0.5.14 requires Go 1.27 |
| [Python](https://www.python.org/downloads/) | 3.11 or newer | training and reference predictions |
| [Git](https://git-scm.com/download/win) | any recent | cloning the repository |

Python packages (`python/requirements.txt`): `numpy`, `torch`, `onnx`, `onnxruntime`, `pytest`, and `psutil` on Windows. All are free; nothing in this project requires a paid service.

Go packages (downloaded automatically by `go build`): [`gomlx`](https://github.com/gomlx/gomlx) v0.28.17, [`onnx-gomlx`](https://github.com/gomlx/onnx-gomlx) v0.5.14, [`compute`](https://github.com/gomlx/compute) v0.1.14 (GoMLX's pure-Go backend), and, only for the optional `-tags onnx` build, [`compute-onnx`](https://github.com/gomlx/compute-onnx) v0.1.13.

---

## Step-by-step: reproduce everything on Windows

Open **PowerShell** and run these from the folder where you want the project.

**1. Get the code**

```powershell
git clone https://github.com/CHbobray/onnx-gomlx-benchmark.git
cd onnx-gomlx-benchmark
```

**2. Set up Python** (a virtual environment keeps packages separate from the rest of your system)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python\requirements.txt
```

**3. Train and export the model** (about 1-2 minutes on a laptop CPU; the repository already contains the trained model, so this step is optional)

```powershell
python python\train_and_export.py --epochs 5
```

**4. Predict and time in Python** (writes `results\python_predictions.csv` and `results\python_timing.json`)

```powershell
python python\predict_and_time.py --runs 30 --latency 500
```

**5. Build and run the Go application** (it reads the files from step 4, so run that first)

```powershell
.\build.ps1 -Run
```

The Go run prints whether its predictions match Python's and a timing table that includes the Python numbers, then writes everything to `results\`.

> On macOS or Linux the commands are the same except: `source .venv/bin/activate`, forward slashes in paths, and `go build -o bin/fashion-go ./cmd/fashion-go && ./bin/fashion-go` instead of `build.ps1`.

---

## Building the Windows executable

`build.ps1` runs the unit tests and builds `bin\fashion-go.exe`:

```powershell
.\build.ps1              # test + build
.\build.ps1 -SkipTests   # build only
.\build.ps1 -Run         # test + build + run
```

If PowerShell says running scripts is disabled, allow scripts you created yourself, once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The equivalent single command, without the script:

```powershell
$env:CGO_ENABLED="0"; $env:GOEXPERIMENT="simd"; go build -trimpath -ldflags "-s -w" -o bin\fashion-go.exe .\cmd\fashion-go
```

`CGO_ENABLED=0` works because the program uses GoMLX's **pure-Go** backend: no C compiler, no DLLs, and no Python on the machine that runs it. The result is one self-contained file of about 14 MB. `GOEXPERIMENT=simd` follows the GoMLX documentation: it compiles the backend's AVX2/AVX-512 kernels, which are then used automatically on CPUs that support them.

**Building the .exe from macOS or Linux** (cross-compiling):

```bash
CGO_ENABLED=0 GOEXPERIMENT=simd GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o bin/fashion-go.exe ./cmd/fashion-go
```

**Running it.** Either double-click `bin\fashion-go.exe` (a console window opens, the program runs with default settings and waits for Enter before closing), or run it from PowerShell with options as shown next. The program finds `models\` and `data\` whether it is started from the repository folder or from `bin\`.

---

## Using the Go application

```
Usage: fashion-go [flags]

  -batch int          images per inference call (1-10000) (default 1000)
  -runs int           timed Monte Carlo runs (1-1000) (default 30)
  -warmup int         untimed warm-up runs (0-100) (default 3)
  -latency int        single-image requests to time (1-10000) (default 500)
  -tolerance float    largest allowed |logit difference| vs Python (default 0.001)
  -model string       ONNX model file (default models\fashion_cnn.onnx)
  -images string      IDX test images, .gz or raw (default data\t10k-images-idx3-ubyte.gz)
  -labels string      IDX test labels, .gz or raw (default data\t10k-labels-idx1-ubyte.gz)
  -python-preds str   Python predictions CSV to compare against; "" to skip
  -out string         directory for result files (default results)
  -cpuprofile string  write a CPU profile to this file
  -memprofile string  write a heap profile to this file
  -pause              wait for Enter before exiting (on by default when started with no arguments)
```

Examples:

```powershell
.\bin\fashion-go.exe -runs 5                 # quicker benchmark
.\bin\fashion-go.exe -batch 250 -runs 10     # a different batch size
.\bin\fashion-go.exe -python-preds ""        # skip the Python comparison
.\bin\fashion-go.exe -h                      # list all options
```

**Input checking.** Every flag is validated before any work starts, and all problems are reported together so they can be fixed in one pass:

```
> .\bin\fashion-go.exe -runs 0 -batch 20000 -model nope.onnx
invalid input:
  - -batch must be between 1 and 10000 (got 20000)
  - -runs must be between 1 and 1000 (got 0)
  - -model: file not found: nope.onnx

Run with -h to see all options.
```

Non-numbers (`-runs ten`), stray words, missing files, corrupt files, and image/label files that don't belong together all produce a plain explanation instead of a crash.

**Choosing a GoMLX backend.** The program honors GoMLX's standard `GOMLX_BACKEND` environment variable; without it, the pure-Go backend is used. For example:

```powershell
$env:GOMLX_BACKEND = "go:ops_parallel"   # pure-Go backend, run independent operations in parallel
.\bin\fashion-go.exe
Remove-Item Env:GOMLX_BACKEND            # back to the default
```

The ONNX Runtime backend has to be compiled in (Linux or macOS, with a C compiler):

```bash
go build -tags onnx -o bin/fashion-go ./cmd/fashion-go
GOMLX_BACKEND=onnx ./bin/fashion-go          # GoMLX downloads ONNX Runtime on first use
# or point it at an existing copy, e.g. the one inside the Python onnxruntime package:
ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so GOMLX_BACKEND=onnx ./bin/fashion-go
```

**Exit codes** (useful in scripts): `0` success, `1` runtime error, `2` invalid input, `3` predictions do not match Python.

**What one run does:**

1. Reads the 10,000 test images and labels.
2. Parses the ONNX file and loads its weights into GoMLX.
3. Classifies all images once, reports accuracy, and writes `results\go_predictions.csv`.
4. Compares every predicted class and every logit with `results\python_predictions.csv`.
5. Monte Carlo throughput benchmark: warm-up runs, then `-runs` timed passes over all 10,000 images.
6. Latency benchmark: `-latency` separate single-image predictions, each timed on its own.
7. Samples Go memory use, then writes `go_timing.json` and `timing_comparison.md` and prints the comparison table.

**Progress and logs.** When the first pass shows the benchmark will take more than a minute (the pure-Go backend needs about 30 s per pass, so the default 30 runs take about 17 minutes), the program says how long to expect and how to shorten it, then prints one line per run with the time remaining:

```
Note: the benchmark will take about 17m0s. For a quicker run use fewer runs, e.g. -runs 5 -warmup 1

Benchmarking: 3 warm-up + 30 timed runs of 10000 images, batch size 1000
  run 1/30: 29.6 s (about 14m18s left)
  run 2/30: 29.4 s (about 13m43s left)
```

Everything shown on screen is also saved to `results\go_run_log.txt`, together with the start time, the exact command, and whether the run finished or why it failed. The Python scripts do the same in `results\python_run_log.txt` and `results\training_log.txt`, with a timestamp on every line. (The logs committed in `results/` come from the benchmark runs reported below, which were captured by redirecting console output before this logging was added; any new run writes its own logs in the format described here.)

---

## Testing, benchmarks, and profiling

**Go unit tests** (run from the repository root):

```powershell
go test ./...          # all tests
go test -cover ./...   # with coverage
go test -v ./internal/engine
```

| Package | What the tests cover | Coverage |
|---|---|---|
| `internal/idx` | synthetic IDX files: raw and gzip give identical results, label files, wrong file type, truncated file, missing file | 88% |
| `internal/compare` | identical vs differing predictions, count mismatches, empty input, CSV write/read round trip, a malformed CSV line | 89% |
| `internal/bench` | statistics on known values (mean, median, sample std, rate), warm-up counting, progress reporting, error propagation, latency indexing | 92% |
| `internal/engine` | **the real model vs Python on 50 images with a partial last batch**, batch size 12 vs 1 giving the same logits, wrong-size input, bad index, missing and corrupt model files, `GOMLX_BACKEND` override | 88% |
| `cmd/fashion-go` | **the whole program end to end** on a 40-image slice of the real data (all output files and the run log), a deliberate mismatch giving exit code 3 and a logged failure, flag defaults, every validation rule, non-numeric values, stray arguments, `-h`, table ordering, speed ratios, backend names | 72% |

Synthetic data is generated inside the tests, so they need no downloads. The engine tests that use the real model skip themselves, with a message, if the model or Python predictions are missing.

**Python unit tests:**

```powershell
pytest
```

They cover the IDX reader, the model's input/output shapes, the ONNX clean-up step (it must remove only *default* MaxPool dilations), timing statistics, and that the exported ONNX file gives the same outputs as PyTorch.

**Go micro-benchmark** (one 100-image batch, with memory allocation counts):

```powershell
go test -bench . -benchmem ./internal/engine
```

**Monte Carlo benchmark.** Both `predict_and_time.py` and `fashion-go` repeat the same full-test-set prediction many times (30 by default, after 3 untimed warm-ups) and treat each run time as one random sample. Reporting the mean, standard deviation, median, and 95th percentile shows how *consistent* each engine is, not just one possibly lucky number. Single-image latency is sampled the same way (500 requests).

**Profiling.** The Go app can write standard Go profiles:

```powershell
.\bin\fashion-go.exe -runs 1 -cpuprofile cpu.prof -memprofile mem.prof
go tool pprof -top bin\fashion-go.exe cpu.prof
go tool pprof -http=:8080 bin\fashion-go.exe cpu.prof   # interactive view in a browser
```

`results/go_profile_summary.txt` contains the profile behind the analysis below.

**Memory and processing requirements of each component.** Go's benchmark tool measures time, bytes allocated, and allocations per call for each package:

```powershell
go test -run XXX -bench . -benchmem ./internal/...
```

Results on the development machine (`results/go_component_benchmarks.txt`):

| Component | Work per call | Time | Memory allocated | Allocations |
|---|---|---:|---:|---:|
| `idx.ReadImagesFile` | read + un-gzip 10,000 test images | 51 ms | 8.2 MB | 1,388 |
| `compare.ReadReferenceCSV` | parse a 10,000-row predictions CSV | 8.4 ms | 5.4 MB | 30,041 |
| `compare.Compare` | compare 10,000 predictions | 0.57 ms | 0 B | 0 |
| `engine.ArgMax` | pick the class for 10,000 images | 0.16 ms | 80 KB | 1 |
| `engine.PredictImages` | classify one 100-image batch (pure-Go backend) | 373 ms | 235 MB | 1.4 million |
| `engine.PredictOne` | classify one image (pure-Go backend) | 3.6 ms | 2.1 MB | 14,328 |

Inference dominates both time and memory: per image, the pure-Go backend allocates about 2 MB of short-lived buffers. That churn is why garbage collection takes about 20% of CPU time in the whole-program profile. Every other component is negligible next to it.

For the whole program, `go_timing.json` records Go heap and operating-system memory after the run (337 MB from the OS, 20 MB live heap), and `python_timing.json` records the Python process's peak memory (877 MB).

---

## Results

The tables in this section use the default **pure-Go backend**, the one that builds into the Windows `.exe`; the [next section](#gomlx-backends-tried) compares other GoMLX backends.

Measured on the development machine: **Linux x86-64, 2 CPU cores, 7 GB RAM**, Go 1.27.3, Python 3.13, PyTorch 2.14, ONNX Runtime 1.29. Every engine used batches of 1,000 images and all available cores. Your numbers will differ on your own computer; rerun steps 4 and 5 to regenerate them, and `results/timing_comparison.md` will hold your tables.

### Do the predictions match?

| Check (10,000 test images) | Result |
|---|---|
| Same predicted class, Go vs Python | **10,000 / 10,000 (100%)** |
| Largest difference in any logit | 0.0000076 |
| Average difference per logit | 0.00000076 |
| Accuracy, Go | 90.34% |
| Accuracy, Python | 90.34% |

The tiny differences come from adding up floating-point numbers in a different order, which is normal between any two math libraries. They are thousands of times too small to change a prediction.

### How fast? Throughput (whole 10,000-image test set per run, 30 runs)

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Images/s | vs Go |
|---|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 179.6 | 14.2 | 175.2 | 207.8 | 55,684 | 166.5x faster |
| Python + PyTorch | 1,506.0 | 79.9 | 1,515.7 | 1,611.6 | 6,640 | 19.8x faster |
| **Go + GoMLX (pure-Go backend)** | **29,893.0** | 367.8 | 29,894.6 | 30,498.8 | **335** | 1.0x |

### How fast? Latency (one image per call, like one user request, 500 requests)

| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | Requests/s | vs Go |
|---|---:|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 0.050 | 0.012 | 0.047 | 0.070 | 20,078 | 74.0x faster |
| Python + PyTorch | 0.183 | 0.044 | 0.166 | 0.260 | 5,454 | 20.1x faster |
| **Go + GoMLX (pure-Go backend)** | **3.688** | 0.828 | 3.451 | 5.144 | **271** | 1.0x |

### Startup and memory

| | Python | Go |
|---|---:|---:|
| Load model | — | 2 ms |
| First full pass, including GoMLX graph compilation | — | 30.3 s (compilation itself is negligible) |
| Peak memory of the process | 877 MB (PyTorch + ONNX Runtime loaded) | 337 MB obtained from the OS; 20 MB live heap at the end |
| What must be installed to run it | Python, PyTorch (~1 GB), NumPy | nothing: one 14 MB `.exe` |

### Why is the pure-Go backend slower?

The CPU profile (`results/go_profile_summary.txt`) shows where the time goes in the Go program:

| Share of CPU time | Activity |
|---:|---|
| ~41% | the two convolution layers |
| ~39% | the two max-pooling layers |
| ~20% | Go memory allocation and garbage collection triggered inside those operations |
| <5% | everything else (dense layers, ReLU, our own code) |

GoMLX's pure-Go backend computes convolution and pooling with straightforward loops in plain Go. PyTorch and ONNX Runtime use hand-tuned C++ kernels that process many numbers per CPU instruction (SIMD) and lay data out for the cache. That gap, not Go itself and not the ONNX conversion, explains the result: the same Go program on GoMLX's ONNX Runtime backend is about 130x faster (next section). Changing the batch size (tried 100, 1,000, and 2,500) changed Go's time by only about 6%, which confirms the cost is in the arithmetic rather than in per-call overhead.

---

## GoMLX backends tried

GoMLX separates *building* a computation graph from *executing* it, and the [GoMLX README](https://github.com/gomlx/gomlx) lists several execution backends and tuning switches. Each one that applied to this project was tried on the same model and test set:

| GoMLX backend / setting | Test set (10,000 images) | One image | Matches Python? | Runs on Windows? | Notes |
|---|---:|---:|---|---|---|
| **`go`, pure Go** (default here) | 29,893 ms | 3.69 ms | yes | **yes, single `.exe`** | 30 runs; numbers in [Results](#results) |
| `go` + `GOEXPERIMENT=simd` | 29,395 ms | 3.72 ms | yes | yes | 2 runs; SIMD covers matrix multiply and activations, not convolution or pooling, which take ~80% of the time |
| `go:ops_parallel` | 29,878 ms | 3.32 ms | yes | yes | 1 run; this model's layers run one after another, so there is little to overlap |
| **`onnx`, ONNX Runtime** (`-tags onnx`) | **222.8 ms** | **0.078 ms** | yes | no (see below) | 30 runs; full output in `results/onnx-backend/` |
| `xla`, OpenXLA | not run | not run | — | no | the GoMLX README lists XLA CPU support for Linux and macOS only; its installer could not download in this sandbox |

**ONNX Runtime backend in detail** (30 runs, same machine and settings as above):

| Engine | Test set mean (ms) | Std (ms) | Images/s | One image mean (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|
| Python + ONNX Runtime | 179.6 | 14.2 | 55,684 | 0.050 | 0.070 |
| **Go + GoMLX (ONNX Runtime backend)** | **222.8** | 30.5 | **44,889** | **0.078** | 0.145 |
| Python + PyTorch | 1,506.0 | 79.9 | 6,640 | 0.183 | 0.260 |
| Go + GoMLX (pure-Go backend) | 29,893.0 | 367.8 | 335 | 3.688 | 5.144 |

Predictions on the ONNX Runtime backend also matched Python on all 10,000 images (largest logit difference 0.0000067). The Go program's own heap stayed at about 21 MB; ONNX Runtime's native memory is outside Go's accounting, so it is not comparable with the pure-Go memory figure.

**Why the ONNX Runtime backend is not the Windows deliverable.** It is compiled with cgo (a C compiler is required), it loads the ONNX Runtime shared library at run time, and in `compute-onnx` v0.1.13 the library loader is implemented only for Linux and macOS. The GoMLX README's backend list also mentions Windows for this backend, so Windows support may arrive in a later release; it did not build for Windows with the versions used here.

---

## Recommendation for management

1. **The development workflow works.** Data scientists can keep building models in Python. The exported ONNX file carries the whole model, and Go reproduces Python's answers exactly on every backend tried.
2. **For a high-traffic Go service, run GoMLX on the ONNX Runtime backend, on Linux servers.** It answered a single-image request in about 0.08 ms and processed about 45,000 images per second on 2 cores, 7x the throughput of PyTorch and close to ONNX Runtime in Python, while the service stays in Go. This is the setup to pilot.
3. **Do not use the pure-Go backend for the high-traffic service.** At about 270 single-image requests per second per 2-core machine, serving thousands of users would need far more servers.
4. **Where the pure-Go backend fits today:** Windows desktop tools, edge devices, and offline jobs that must ship as one file with nothing to install. At under 4 ms per image, a single user would not notice the difference. GoMLX's README notes the pure-Go backend is being actively optimized, so this gap may shrink.

---

## Issues found along the way

These are worth knowing for anyone repeating the approach:

- **An ONNX feature gap.** PyTorch always writes a `dilations=[1,1]` attribute on MaxPool layers. `onnx-gomlx` refuses any MaxPool that carries that attribute, even though `[1,1]` is the default and means "no dilation". `train_and_export.py` removes the attribute only when it has the default value, which leaves the model's meaning unchanged; ONNX Runtime and a unit test confirm the cleaned file still matches PyTorch. ONNX is a common format, but each runtime supports a different subset of it, so test every new layer type early.
- **SIMD must be switched on at build time.** GoMLX's pure-Go backend only includes its AVX2/AVX-512 code when built with `GOEXPERIMENT=simd`; `build.ps1` sets it. For this model it changed little, because convolution and pooling have no SIMD kernels yet.
- **Go version.** The current `onnx-gomlx` (v0.5.14) requires Go 1.27. Older Go versions fail at `go build` with a "requires go >= 1.27" message.
- **Fair timing.** PyTorch is slower than ONNX Runtime on the *same* model here because ONNX Runtime optimizes the graph ahead of time. Reporting both shows that "Python" speed depends heavily on which Python engine is used.

---

## Design notes

How the grading criteria are addressed:

- **Short functions, one step each.** `run()` in `main.go` reads as the list of steps (load data, load model, check predictions, benchmark, write results); each step is its own function.
- **Small, single-purpose Go packages** (`idx`, `engine`, `compare`, `bench`) that other projects can reuse: the `idx` reader works for MNIST, and `engine` loads any ONNX image classifier with one input and one `[batch, classes]` output.
- **Errors are values.** Library code returns wrapped errors with context (`data\x.gz: reading image header: ...`); only `main` decides how to report them and which exit code to use.
- **Validation before work.** All user input is checked up front, and every problem is listed together.
- **No hidden preprocessing.** Normalization lives inside the model, so Python and Go send the network identical numbers.
- **Comments explain why**, not what, for example why batch size does not affect results and why MaxPool attributes are removed.
- **Results are reproducible.** Fixed random seed for training, fixed test set, and all raw outputs (CSVs and JSON) committed in `results/`.

---

## AI-assistant dialog

The assignment asks for the dialog with the LLM-based tools used. This project was built with help from **Claude** (Anthropic). The full conversation is in [`docs/ai-assistant-dialog.pdf`](docs/ai-assistant-dialog.pdf), with the same record as plain text in [`docs/ai-assistant-dialog.md`](docs/ai-assistant-dialog.md).

---

## References

- GoMLX: https://github.com/gomlx/gomlx
- onnx-gomlx: https://github.com/gomlx/onnx-gomlx
- Open Neural Network Exchange: https://onnx.ai/
- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html
- Fashion-MNIST: Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747
- McConnell, S. (2004). *Code Complete* (2nd ed.). Microsoft Press.
