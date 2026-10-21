# Python to Go with ONNX: Fashion-MNIST CNN in PyTorch and GoMLX

**Question from management:** can we develop deep learning models in Python, export them to a common format, and run them in production with Go?

**Short answer:** *yes for correctness, not yet for speed with the pure-Go engine.*

- A PyTorch convolutional network, exported to ONNX and run by GoMLX in Go, reproduces Python's predictions **exactly**: 10,000 of 10,000 test images get the same class, and the largest difference in any output score is 0.0000076.
- The pure-Go GoMLX engine is **about 20x slower than PyTorch** and **about 170x slower than ONNX Runtime** on this model. A single prediction still takes only about 3.7 ms, so it is usable for light traffic, but it is not the fast-response option for thousands of users.
- The Go program is a **single 14 MB Windows `.exe`** with no Python, C libraries, or runtime installs, and it used about 340 MB of memory versus about 880 MB for the Python process.

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
│   ├── compare/               reads/writes prediction CSVs, measures agreement
│   └── bench/                 Monte Carlo timing statistics and memory sampling
│
├── results/                   outputs from the runs reported below
│   ├── python_predictions.csv   label, predicted class, 10 logits per test image
│   ├── go_predictions.csv       the same, from Go
│   ├── python_timing.json       Python timing statistics
│   ├── go_timing.json           Go timing, memory, and agreement statistics
│   ├── timing_comparison.md     side-by-side tables (written by the Go app)
│   ├── python_run_log.txt       console output of the Python run
│   ├── go_run_log.txt           console output of the Go run
│   └── go_profile_summary.txt   CPU and heap profile of the Go app
│
└── docs/                      AI-assistant dialog (see below)
```

---

## Requirements

| Tool | Version | Why |
|---|---|---|
| [Go](https://go.dev/dl/) | **1.27 or newer** | `onnx-gomlx` v0.5.14 requires Go 1.27 |
| [Python](https://www.python.org/downloads/) | 3.11 or newer | training and reference predictions |
| [Git](https://git-scm.com/download/win) | any recent | cloning the repository |

Python packages (`python/requirements.txt`): `numpy`, `torch`, `onnx`, `onnxruntime`, `pytest`, and `psutil` on Windows. All are free; nothing in this project requires a paid service.

Go packages (downloaded automatically by `go build`): [`gomlx`](https://github.com/gomlx/gomlx) v0.28.17, [`onnx-gomlx`](https://github.com/gomlx/onnx-gomlx) v0.5.14, and [`compute`](https://github.com/gomlx/compute) v0.1.14 (GoMLX's backends).

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
$env:CGO_ENABLED="0"; go build -trimpath -ldflags "-s -w" -o bin\fashion-go.exe .\cmd\fashion-go
```

`CGO_ENABLED=0` works because the program uses GoMLX's **pure-Go** backend: no C compiler, no DLLs, and no Python on the machine that runs it. The result is one self-contained file of about 14 MB.

**Building the .exe from macOS or Linux** (cross-compiling):

```bash
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o bin/fashion-go.exe ./cmd/fashion-go
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

**Exit codes** (useful in scripts): `0` success, `1` runtime error, `2` invalid input, `3` predictions do not match Python.

**What one run does:**

1. Reads the 10,000 test images and labels.
2. Parses the ONNX file and loads its weights into GoMLX.
3. Classifies all images once, reports accuracy, and writes `results\go_predictions.csv`.
4. Compares every predicted class and every logit with `results\python_predictions.csv`.
5. Monte Carlo throughput benchmark: warm-up runs, then `-runs` timed passes over all 10,000 images.
6. Latency benchmark: `-latency` separate single-image predictions, each timed on its own.
7. Samples Go memory use, then writes `go_timing.json` and `timing_comparison.md` and prints the comparison table.

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
| `internal/compare` | identical vs differing predictions, count mismatches, CSV write/read round trip, a malformed CSV line | 88% |
| `internal/bench` | statistics on known values (mean, median, sample std, rate), warm-up counting, error propagation, latency indexing | 92% |
| `internal/engine` | **the real model vs Python on 50 images with a partial last batch**, batch size 12 vs 1 giving the same logits, wrong-size input, bad index, missing and corrupt model files | 88% |
| `cmd/fashion-go` | flag defaults, every validation rule, non-numeric values, stray arguments, `-h`, table ordering and speed ratios | 32% (the rest is the full run, exercised end to end) |

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

---

## Results

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

### Why is Go slower here?

The CPU profile (`results/go_profile_summary.txt`) shows where the time goes in the Go program:

| Share of CPU time | Activity |
|---:|---|
| ~41% | the two convolution layers |
| ~39% | the two max-pooling layers |
| ~20% | Go memory allocation and garbage collection triggered inside those operations |
| <5% | everything else (dense layers, ReLU, our own code) |

GoMLX's pure-Go backend computes convolution and pooling with straightforward loops in plain Go. PyTorch and ONNX Runtime use hand-tuned C++ kernels that process many numbers per CPU instruction (SIMD) and lay data out for the cache. That gap, not Go itself and not the ONNX conversion, explains the result. Changing the batch size (tried 100, 1,000, and 2,500) changed Go's time by only about 6%, which confirms the cost is in the arithmetic rather than in per-call overhead.

---

## Recommendation for management

1. **The development workflow works.** Data scientists can keep building models in Python. The exported ONNX file carries the whole model, and Go reproduces Python's answers exactly. That part of the proposal is proven.
2. **Do not use the pure-Go GoMLX backend for the high-traffic service.** At about 270 single-image requests per second per machine (on 2 cores), serving thousands of users with fast responses would need many more servers than the Python alternatives.
3. **To keep Go in production and be fast, use a native engine behind Go.** Two options to evaluate next:
   - GoMLX's **XLA backend** (Google's optimizing compiler, also used by TensorFlow and JAX). This needs only a small change in `internal/engine` (a different backend import and name) but downloads native libraries at first run; it could not be tested in this project's sandbox.
   - **ONNX Runtime called from Go** (for example `github.com/yalue/onnxruntime_go`). The ONNX Runtime row above shows the speed that becomes available.
4. **Where pure-Go GoMLX is a good fit today:** low-volume or offline jobs, edge devices, and tools that must ship as a single file with no installs. At under 4 ms per image, a single user would not notice the difference.

---

## Issues found along the way

These are worth knowing for anyone repeating the approach:

- **An ONNX feature gap.** PyTorch always writes a `dilations=[1,1]` attribute on MaxPool layers. `onnx-gomlx` refuses any MaxPool that carries that attribute, even though `[1,1]` is the default and means "no dilation". `train_and_export.py` removes the attribute only when it has the default value, which leaves the model's meaning unchanged; ONNX Runtime and a unit test confirm the cleaned file still matches PyTorch. ONNX is a common format, but each runtime supports a different subset of it, so test every new layer type early.
- **Go version.** The current `onnx-gomlx` (v0.5.14) requires Go 1.27. Older Go versions fail at `go build` with a "requires go >= 1.27" message.
- **Fair timing.** PyTorch is slower than ONNX Runtime on the *same* model here because ONNX Runtime optimizes the graph ahead of time. Reporting both shows that "Python" speed depends heavily on which Python engine is used.

---

## Design notes

How the grading criteria are addressed:

- **Small, single-purpose Go packages** (`idx`, `engine`, `compare`, `bench`) that other projects can reuse: the `idx` reader works for MNIST, and `engine` loads any ONNX image classifier with one input and one `[batch, classes]` output.
- **Errors are values.** Library code returns wrapped errors with context (`data\x.gz: reading image header: ...`); only `main` decides how to report them and which exit code to use.
- **Validation before work.** All user input is checked up front, and every problem is listed together.
- **No hidden preprocessing.** Normalization lives inside the model, so Python and Go send the network identical numbers.
- **Comments explain why**, not what, for example why batch size does not affect results and why MaxPool attributes are removed.
- **Results are reproducible.** Fixed random seed for training, fixed test set, and all raw outputs (CSVs and JSON) committed in `results/`.

---

## AI-assistant dialog

The assignment asks for the dialog with the LLM-based tools used. This project was built with help from **Claude** (Anthropic). The conversation is saved in [`docs/`](docs/) as a PDF.

---

## References

- GoMLX: https://github.com/gomlx/gomlx
- onnx-gomlx: https://github.com/gomlx/onnx-gomlx
- Open Neural Network Exchange: https://onnx.ai/
- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html
- Fashion-MNIST: Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747
- McConnell, S. (2004). *Code Complete* (2nd ed.). Microsoft Press.
