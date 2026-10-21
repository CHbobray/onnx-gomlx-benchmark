package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/CHbobray/onnx-gomlx-benchmark/internal/bench"
	"github.com/CHbobray/onnx-gomlx-benchmark/internal/compare"
)

// engineTiming is the pair of benchmarks recorded for every engine,
// in both results/go_timing.json and results/python_timing.json.
type engineTiming struct {
	Throughput bench.Stats `json:"throughput"` // full test set, batched
	Latency    bench.Stats `json:"latency"`    // one image per call
}

// goTiming is written to results/go_timing.json.
type goTiming struct {
	Platform       string       `json:"platform"`
	GoVersion      string       `json:"go"`
	Backend        string       `json:"gomlx_backend"`
	CPUs           int          `json:"cpus"`
	BatchSize      int          `json:"batch_size"`
	Accuracy       float64      `json:"accuracy"`
	LoadMs         float64      `json:"model_load_ms"`
	FirstRunMs     float64      `json:"first_run_ms"` // includes graph compilation
	Stats          bench.Stats  `json:"throughput"`
	Latency        bench.Stats  `json:"latency"`
	MemoryAfterRun bench.Memory `json:"memory"`
	Comparison     *comparison  `json:"comparison_with_python,omitempty"`
}

type comparison struct {
	ClassAgreement   float64 `json:"class_agreement"`
	Disagreements    int     `json:"disagreements"`
	MaxAbsLogitDiff  float64 `json:"max_abs_logit_diff"`
	MeanAbsLogitDiff float64 `json:"mean_abs_logit_diff"`
}

// pythonTiming mirrors the parts of results/python_timing.json we use.
type pythonTiming struct {
	Platform  string                  `json:"platform"`
	BatchSize int                     `json:"batch_size"`
	Engines   map[string]engineTiming `json:"engines"`
}

// Display names for the engines, in the order they appear in tables.
var engineOrder = []struct{ key, name string }{
	{"gomlx", "Go + GoMLX (pure-Go backend)"},
	{"pytorch", "Python + PyTorch"},
	{"onnxruntime", "Python + ONNX Runtime"},
}

func newComparison(report compare.Report) *comparison {
	return &comparison{
		ClassAgreement:   report.ClassAgreement,
		Disagreements:    len(report.DisagreeIndices),
		MaxAbsLogitDiff:  report.MaxAbsLogitDiff,
		MeanAbsLogitDiff: report.MeanAbsLogitDiff,
	}
}

func platformName() string { return runtime.GOOS + " " + runtime.GOARCH }

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func readPythonTiming(path string) (*pythonTiming, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var timing pythonTiming
	if err := json.Unmarshal(data, &timing); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &timing, nil
}

func printComparison(out io.Writer, report compare.Report, tolerance float64) {
	fmt.Fprintf(out, "\nGo vs Python predictions on %d images\n", report.Images)
	fmt.Fprintf(out, "  predicted-class agreement : %.4f (%d differ)\n", report.ClassAgreement, len(report.DisagreeIndices))
	fmt.Fprintf(out, "  max |logit difference|    : %.2e (tolerance %.0e)\n", report.MaxAbsLogitDiff, tolerance)
	fmt.Fprintf(out, "  mean |logit difference|   : %.2e\n", report.MeanAbsLogitDiff)
	fmt.Fprintf(out, "  accuracy  Go %.4f   Python %.4f\n", report.GoAccuracy, report.RefAccuracy)
	if report.Matches(tolerance) {
		fmt.Fprintln(out, "  RESULT: MATCH - Go reproduces the Python model's predictions")
	} else {
		fmt.Fprintln(out, "  RESULT: MISMATCH - see results/go_predictions.csv")
	}
}

// collectEngines gathers Go's timing and, when available, Python's, keyed by engine.
func collectEngines(goResult goTiming, python *pythonTiming) map[string]engineTiming {
	engines := map[string]engineTiming{
		"gomlx": {Throughput: goResult.Stats, Latency: goResult.Latency},
	}
	if python != nil {
		for key, timing := range python.Engines {
			engines[key] = timing
		}
	}
	return engines
}

// timingTables renders Markdown tables for throughput and single-image latency.
func timingTables(engines map[string]engineTiming) string {
	var out strings.Builder
	images := engines["gomlx"].Throughput.ImagesPerRun
	fmt.Fprintf(&out, "**Throughput** - full %d-image test set per run\n\n", images)
	writeTable(&out, engines, func(t engineTiming) bench.Stats { return t.Throughput }, "Images/s")
	fmt.Fprintf(&out, "\n**Latency** - one image per call (a single user request)\n\n")
	writeTable(&out, engines, func(t engineTiming) bench.Stats { return t.Latency }, "Requests/s")
	return out.String()
}

func writeTable(out *strings.Builder, engines map[string]engineTiming,
	pick func(engineTiming) bench.Stats, rateLabel string) {

	goMean := pick(engines["gomlx"]).MeanMs
	fmt.Fprintf(out, "| Engine | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | %s | vs Go | Runs |\n", rateLabel)
	fmt.Fprintf(out, "|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, engine := range orderedEngines(engines, pick) {
		s := pick(engines[engine.key])
		fmt.Fprintf(out, "| %s | %s | %s | %s | %s | %.0f | %s | %d |\n",
			engine.name, formatMs(s.MeanMs), formatMs(s.StdMs), formatMs(s.MedianMs), formatMs(s.P95Ms),
			s.ImagesPerSecond, speedRatio(goMean, s.MeanMs), s.Runs)
	}
}

// orderedEngines lists the engines present, fastest first.
func orderedEngines(engines map[string]engineTiming, pick func(engineTiming) bench.Stats) []struct{ key, name string } {
	var present []struct{ key, name string }
	for _, engine := range engineOrder {
		if _, ok := engines[engine.key]; ok {
			present = append(present, engine)
		}
	}
	slices.SortStableFunc(present, func(a, b struct{ key, name string }) int {
		return cmpFloat(pick(engines[a.key]).MeanMs, pick(engines[b.key]).MeanMs)
	})
	return present
}

// speedRatio describes how an engine's mean time compares with Go's.
func speedRatio(goMean, mean float64) string {
	switch {
	case mean <= 0 || goMean == mean:
		return "1.0x"
	case mean < goMean:
		return fmt.Sprintf("%.1fx faster", goMean/mean)
	default:
		return fmt.Sprintf("%.1fx slower", mean/goMean)
	}
}

// formatMs keeps small latencies readable without padding large ones.
func formatMs(ms float64) string {
	if ms < 10 {
		return fmt.Sprintf("%.3f", ms)
	}
	return fmt.Sprintf("%.1f", ms)
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}
