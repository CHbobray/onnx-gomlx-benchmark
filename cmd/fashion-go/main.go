// Command fashion-go classifies the Fashion-MNIST test set with an ONNX model
// executed by GoMLX, verifies that its predictions match the Python model's,
// and runs a Monte Carlo benchmark of the inference time.
//
// Run with -h for the list of flags. Exit codes: 0 success, 1 runtime error,
// 2 invalid input, 3 predictions do not match Python.
package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/CHbobray/onnx-gomlx-benchmark/internal/bench"
	"github.com/CHbobray/onnx-gomlx-benchmark/internal/compare"
	"github.com/CHbobray/onnx-gomlx-benchmark/internal/engine"
	"github.com/CHbobray/onnx-gomlx-benchmark/internal/idx"
)

const (
	exitOK       = 0
	exitError    = 1
	exitBadInput = 2
	exitMismatch = 3
)

// latencyWarmup untimed single-image calls let GoMLX compile the batch-1 graph first.
const latencyWarmup = 5

// longRunWarning: benchmarks expected to take longer than this get a time
// estimate and a hint on how to shorten them, so nobody waits in silence.
const longRunWarning = time.Minute

var errMismatch = errors.New("Go predictions do not match Python predictions")

// out receives all progress messages: the console, plus the run log file
// once run() has opened it.
var out io.Writer = os.Stdout

func main() {
	cfg, err := parseConfig(os.Args[1:], findProjectRoot(), os.Stderr)
	code := exitOK
	switch {
	case errors.Is(err, flag.ErrHelp):
		// Usage was already printed.
	case err != nil:
		fmt.Fprintf(os.Stderr, "%v\n\nRun with -h to see all options.\n", err)
		code = exitBadInput
	default:
		code = exitCode(run(cfg))
	}
	if cfg.Pause {
		fmt.Print("\nPress Enter to exit...")
		bufio.NewReader(os.Stdin).ReadString('\n')
	}
	os.Exit(code)
}

func exitCode(err error) int {
	if err == nil {
		return exitOK
	}
	fmt.Fprintln(os.Stderr, "error:", err)
	if errors.Is(err, errMismatch) {
		return exitMismatch
	}
	return exitError
}

// run performs the whole job, one step per helper function. Any error it
// returns is also recorded at the end of the run log.
func run(cfg config) (runErr error) {
	if err := os.MkdirAll(cfg.ResultsDir, 0o755); err != nil {
		return fmt.Errorf("creating results directory: %w", err)
	}
	closeLog, err := openRunLog(filepath.Join(cfg.ResultsDir, "go_run_log.txt"))
	if err != nil {
		return err
	}
	defer func() { closeLog(runErr) }()

	if cfg.CPUProfile != "" {
		stopProfile, err := startCPUProfile(cfg.CPUProfile)
		if err != nil {
			return err
		}
		defer stopProfile()
	}

	images, labels, err := loadTestSet(cfg)
	if err != nil {
		return err
	}
	classifier, timing, err := loadClassifier(cfg)
	if err != nil {
		return err
	}
	defer classifier.Close()

	matched, err := checkPredictions(cfg, classifier, images, labels, &timing)
	if err != nil {
		return err
	}
	if err := runBenchmarks(cfg, classifier, images, &timing); err != nil {
		return err
	}
	if err := writeResults(cfg, timing); err != nil {
		return err
	}
	if cfg.MemProfile != "" {
		if err := writeHeapProfile(cfg.MemProfile); err != nil {
			return err
		}
	}
	if !matched {
		return errMismatch
	}
	return nil
}

// openRunLog copies everything printed to out into a log file as well,
// starting with a timestamp and the exact command line. The returned
// function records how the run ended and closes the file.
func openRunLog(path string) (closeLog func(runErr error), err error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("creating run log: %w", err)
	}
	started := time.Now()
	out = io.MultiWriter(os.Stdout, file)
	fmt.Fprintf(file, "fashion-go run started %s\ncommand: %q\n\n", started.Format(time.RFC3339), os.Args)
	return func(runErr error) {
		out = os.Stdout
		outcome := "finished successfully"
		if runErr != nil {
			outcome = "failed: " + runErr.Error()
		}
		fmt.Fprintf(file, "\nrun %s after %s\n", outcome, time.Since(started).Round(time.Second))
		file.Close()
	}, nil
}

func loadTestSet(cfg config) (*idx.Images, []uint8, error) {
	images, err := idx.ReadImagesFile(cfg.ImagesPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading images: %w", err)
	}
	labels, err := idx.ReadLabelsFile(cfg.LabelsPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading labels: %w", err)
	}
	if len(labels) != images.Count {
		return nil, nil, fmt.Errorf("%d images but %d labels: are -images and -labels from the same set?",
			images.Count, len(labels))
	}
	fmt.Fprintf(out, "Loaded %d test images (%dx%d) from %s\n", images.Count, images.Rows, images.Cols, cfg.ImagesPath)
	return images, labels, nil
}

// loadClassifier loads the ONNX model and starts the timing record.
func loadClassifier(cfg config) (*engine.Classifier, goTiming, error) {
	started := time.Now()
	classifier, err := engine.Load(cfg.ModelPath)
	if err != nil {
		return nil, goTiming{}, err
	}
	timing := goTiming{
		Platform:  platformName(),
		GoVersion: runtime.Version(),
		Backend:   engine.BackendConfig(),
		CPUs:      runtime.NumCPU(),
		BatchSize: cfg.BatchSize,
		LoadMs:    msSince(started),
	}
	fmt.Fprintf(out, "Loaded ONNX model %s into GoMLX (backend %q) in %.0f ms\n",
		cfg.ModelPath, timing.Backend, timing.LoadMs)
	return classifier, timing, nil
}

// checkPredictions classifies the test set once, saves the predictions, and
// compares them with Python's. It reports whether they match.
func checkPredictions(cfg config, classifier *engine.Classifier, images *idx.Images,
	labels []uint8, timing *goTiming) (bool, error) {

	started := time.Now()
	logits, err := classifier.PredictImages(images, cfg.BatchSize)
	if err != nil {
		return false, err
	}
	timing.FirstRunMs = msSince(started)
	classes := engine.ArgMax(logits, classifier.NumClasses)
	timing.Accuracy = accuracyOf(classes, labels)
	fmt.Fprintf(out, "First pass (includes graph compilation): %.0f ms, accuracy %.4f\n",
		timing.FirstRunMs, timing.Accuracy)

	predictionsPath := filepath.Join(cfg.ResultsDir, "go_predictions.csv")
	if err := compare.WritePredictionsCSV(predictionsPath, labels, logits, classes); err != nil {
		return false, fmt.Errorf("writing predictions: %w", err)
	}
	return compareWithPython(cfg, logits, classes, timing)
}

// compareWithPython checks Go output against the Python CSV, if one was given.
// It returns true when they match (or when there is nothing to compare).
func compareWithPython(cfg config, logits []float32, classes []int, timing *goTiming) (bool, error) {
	if cfg.PythonPredsPath == "" {
		fmt.Fprintln(out, "\nSkipping comparison with Python (-python-preds is empty)")
		return true, nil
	}
	references, err := compare.ReadReferenceCSV(cfg.PythonPredsPath)
	if err != nil {
		return false, fmt.Errorf("reading Python predictions: %w", err)
	}
	report, err := compare.Compare(references, logits, classes)
	if err != nil {
		return false, fmt.Errorf("comparing with Python: %w", err)
	}
	printComparison(out, report, cfg.Tolerance)
	timing.Comparison = newComparison(report)
	return report.Matches(cfg.Tolerance), nil
}

// runBenchmarks measures batched throughput (Monte Carlo) and single-image latency.
func runBenchmarks(cfg config, classifier *engine.Classifier, images *idx.Images, timing *goTiming) error {
	firstRun := time.Duration(timing.FirstRunMs * float64(time.Millisecond))
	warnIfLong(cfg, firstRun)

	fmt.Fprintf(out, "\nBenchmarking: %d warm-up + %d timed runs of %d images, batch size %d\n",
		cfg.Warmup, cfg.Runs, images.Count, cfg.BatchSize)
	var err error
	timing.Stats, err = bench.MonteCarlo(cfg.Runs, cfg.Warmup, images.Count, func() error {
		_, err := classifier.PredictImages(images, cfg.BatchSize)
		return err
	}, reportRun(firstRun))
	if err != nil {
		return fmt.Errorf("benchmark: %w", err)
	}

	fmt.Fprintf(out, "Measuring single-image latency over %d requests...\n", cfg.LatencySamples)
	timing.Latency, err = bench.Latency(cfg.LatencySamples, latencyWarmup, func(i int) error {
		_, err := classifier.PredictOne(images, i%images.Count)
		return err
	})
	if err != nil {
		return fmt.Errorf("latency benchmark: %w", err)
	}
	timing.MemoryAfterRun = bench.SampleMemory()
	return nil
}

// warnIfLong estimates the benchmark duration from the first pass and, when
// it is long, tells the user how long to expect and how to shorten it.
func warnIfLong(cfg config, firstRun time.Duration) {
	estimate := time.Duration(cfg.Warmup+cfg.Runs) * firstRun
	if estimate < longRunWarning {
		return
	}
	fmt.Fprintf(out, "\nNote: the benchmark will take about %s. For a quicker run use fewer runs, e.g. -runs 5 -warmup 1\n",
		estimate.Round(time.Minute))
}

// reportRun prints one progress line per timed run when runs are slow enough
// that the user would otherwise see nothing for a long time.
func reportRun(firstRun time.Duration) bench.Progress {
	if firstRun < time.Second {
		return nil
	}
	return func(run, runs int, took time.Duration) {
		remaining := time.Duration(runs-run) * took
		fmt.Fprintf(out, "  run %d/%d: %.1f s (about %s left)\n", run, runs, took.Seconds(), remaining.Round(time.Second))
	}
}

func writeResults(cfg config, timing goTiming) error {
	timingPath := filepath.Join(cfg.ResultsDir, "go_timing.json")
	if err := writeJSON(timingPath, timing); err != nil {
		return fmt.Errorf("writing timing: %w", err)
	}

	python, err := readPythonTiming(filepath.Join(cfg.ResultsDir, "python_timing.json"))
	if err != nil {
		fmt.Fprintln(out, "(no Python timing found; run python/predict_and_time.py to include it)")
		python = nil
	}
	table := timingTables(collectEngines(timing, python), timing.Backend)
	fmt.Fprintf(out, "\nInference timing on %s (batch size %d):\n\n%s", platformName(), cfg.BatchSize, table)
	memory := timing.MemoryAfterRun
	fmt.Fprintf(out, "\nGo memory: heap in use %.1f MB, obtained from OS %.1f MB, %d GCs\n",
		memory.HeapInUseMB, memory.SysMB, memory.NumGC)

	tablePath := filepath.Join(cfg.ResultsDir, "timing_comparison.md")
	if err := os.WriteFile(tablePath, []byte(table), 0o644); err != nil {
		return fmt.Errorf("writing timing table: %w", err)
	}
	fmt.Fprintf(out, "Results and run log written to %s\n", cfg.ResultsDir)
	return nil
}

func accuracyOf(classes []int, labels []uint8) float64 {
	correct := 0
	for i, class := range classes {
		if class == int(labels[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(classes))
}

func msSince(started time.Time) float64 {
	return float64(time.Since(started)) / float64(time.Millisecond)
}

func startCPUProfile(path string) (stop func(), err error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("creating CPU profile: %w", err)
	}
	if err := pprof.StartCPUProfile(file); err != nil {
		file.Close()
		return nil, fmt.Errorf("starting CPU profile: %w", err)
	}
	return func() {
		pprof.StopCPUProfile()
		file.Close()
	}, nil
}

func writeHeapProfile(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating heap profile: %w", err)
	}
	defer file.Close()
	runtime.GC() // report live objects, not garbage awaiting collection
	return pprof.WriteHeapProfile(file)
}
