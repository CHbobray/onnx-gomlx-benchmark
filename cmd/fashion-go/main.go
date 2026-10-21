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

var errMismatch = errors.New("Go predictions do not match Python predictions")

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
	switch {
	case err == nil:
		return exitOK
	case errors.Is(err, errMismatch):
		fmt.Fprintln(os.Stderr, "error:", err)
		return exitMismatch
	default:
		fmt.Fprintln(os.Stderr, "error:", err)
		return exitError
	}
}

func run(cfg config) error {
	if cfg.CPUProfile != "" {
		stop, err := startCPUProfile(cfg.CPUProfile)
		if err != nil {
			return err
		}
		defer stop()
	}
	if err := os.MkdirAll(cfg.ResultsDir, 0o755); err != nil {
		return fmt.Errorf("creating results directory: %w", err)
	}

	images, labels, err := loadTestSet(cfg)
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d test images (%dx%d) from %s\n", images.Count, images.Rows, images.Cols, cfg.ImagesPath)

	loadStarted := time.Now()
	classifier, err := engine.Load(cfg.ModelPath)
	if err != nil {
		return err
	}
	defer classifier.Close()
	loadMs := msSince(loadStarted)
	fmt.Printf("Loaded ONNX model %s into GoMLX in %.0f ms\n", cfg.ModelPath, loadMs)

	firstStarted := time.Now()
	logits, err := classifier.PredictImages(images, cfg.BatchSize)
	if err != nil {
		return err
	}
	firstRunMs := msSince(firstStarted)
	classes := engine.ArgMax(logits, classifier.NumClasses)
	accuracy := accuracyOf(classes, labels)
	fmt.Printf("First pass (includes graph compilation): %.0f ms, accuracy %.4f\n", firstRunMs, accuracy)

	predictionsPath := filepath.Join(cfg.ResultsDir, "go_predictions.csv")
	if err := compare.WritePredictionsCSV(predictionsPath, labels, logits, classes); err != nil {
		return fmt.Errorf("writing predictions: %w", err)
	}

	timing := goTiming{
		Platform: platformName(), GoVersion: runtime.Version(), Backend: engine.BackendConfig,
		CPUs: runtime.NumCPU(), BatchSize: cfg.BatchSize, Accuracy: accuracy,
		LoadMs: loadMs, FirstRunMs: firstRunMs,
	}
	matched, err := compareWithPython(cfg, logits, classes, &timing)
	if err != nil {
		return err
	}

	fmt.Printf("\nBenchmarking: %d warm-up + %d timed runs of %d images, batch size %d...\n",
		cfg.Warmup, cfg.Runs, images.Count, cfg.BatchSize)
	timing.Stats, err = bench.MonteCarlo(cfg.Runs, cfg.Warmup, images.Count, func() error {
		_, err := classifier.PredictImages(images, cfg.BatchSize)
		return err
	})
	if err != nil {
		return fmt.Errorf("benchmark: %w", err)
	}

	fmt.Printf("Measuring single-image latency over %d requests...\n", cfg.LatencySamples)
	timing.Latency, err = bench.Latency(cfg.LatencySamples, latencyWarmup, func(i int) error {
		_, err := classifier.PredictOne(images, i%images.Count)
		return err
	})
	if err != nil {
		return fmt.Errorf("latency benchmark: %w", err)
	}
	timing.MemoryAfterRun = bench.SampleMemory()

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
	return images, labels, nil
}

// compareWithPython checks Go output against the Python CSV, if one was given.
// It returns true when they match (or when there is nothing to compare).
func compareWithPython(cfg config, logits []float32, classes []int, timing *goTiming) (bool, error) {
	if cfg.PythonPredsPath == "" {
		fmt.Println("\nSkipping comparison with Python (-python-preds is empty)")
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
	printComparison(os.Stdout, report, cfg.Tolerance)
	timing.Comparison = newComparison(report)
	return report.Matches(cfg.Tolerance), nil
}

func writeResults(cfg config, timing goTiming) error {
	timingPath := filepath.Join(cfg.ResultsDir, "go_timing.json")
	if err := writeJSON(timingPath, timing); err != nil {
		return fmt.Errorf("writing timing: %w", err)
	}

	python, err := readPythonTiming(filepath.Join(cfg.ResultsDir, "python_timing.json"))
	if err != nil {
		fmt.Println("(no Python timing found; run python/predict_and_time.py to include it)")
		python = nil
	}
	table := timingTables(collectEngines(timing, python))
	fmt.Printf("\nInference timing on %s (batch size %d):\n\n%s", platformName(), cfg.BatchSize, table)
	m := timing.MemoryAfterRun
	fmt.Printf("\nGo memory: heap in use %.1f MB, obtained from OS %.1f MB, %d GCs\n", m.HeapInUseMB, m.SysMB, m.NumGC)

	tablePath := filepath.Join(cfg.ResultsDir, "timing_comparison.md")
	if err := os.WriteFile(tablePath, []byte(table), 0o644); err != nil {
		return fmt.Errorf("writing timing table: %w", err)
	}
	fmt.Printf("Results written to %s\n", cfg.ResultsDir)
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
	return func() { pprof.StopCPUProfile(); file.Close() }, nil
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
