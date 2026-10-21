package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Limits on user-supplied numbers, chosen to keep runs between a second and a few minutes.
const (
	maxBatchSize = 10_000
	maxRuns      = 1_000
	maxWarmup    = 100
	maxLatency   = 10_000
)

// Relative locations of project files, from the repository root.
var (
	defaultModel       = filepath.Join("models", "fashion_cnn.onnx")
	defaultImages      = filepath.Join("data", "t10k-images-idx3-ubyte.gz")
	defaultLabels      = filepath.Join("data", "t10k-labels-idx1-ubyte.gz")
	defaultPythonPreds = filepath.Join("results", "python_predictions.csv")
	defaultResultsDir  = "results"
)

// config holds every setting for one run of the program.
type config struct {
	ModelPath       string
	ImagesPath      string
	LabelsPath      string
	PythonPredsPath string
	ResultsDir      string
	BatchSize       int
	Runs            int
	Warmup          int
	LatencySamples  int
	Tolerance       float64
	CPUProfile      string
	MemProfile      string
	Pause           bool
}

// parseConfig reads command-line flags. Relative default paths are resolved
// against root, the directory that holds the project's models/ and data/ folders.
func parseConfig(args []string, root string, usageOut io.Writer) (config, error) {
	cfg := config{}
	flags := flag.NewFlagSet("fashion-go", flag.ContinueOnError)
	flags.SetOutput(usageOut)
	flags.StringVar(&cfg.ModelPath, "model", filepath.Join(root, defaultModel), "ONNX model file")
	flags.StringVar(&cfg.ImagesPath, "images", filepath.Join(root, defaultImages), "IDX test images (.gz or raw)")
	flags.StringVar(&cfg.LabelsPath, "labels", filepath.Join(root, defaultLabels), "IDX test labels (.gz or raw)")
	flags.StringVar(&cfg.PythonPredsPath, "python-preds", filepath.Join(root, defaultPythonPreds),
		"Python predictions CSV to compare against (set to \"\" to skip)")
	flags.StringVar(&cfg.ResultsDir, "out", filepath.Join(root, defaultResultsDir), "directory for result files")
	flags.IntVar(&cfg.BatchSize, "batch", 1000, fmt.Sprintf("images per inference call (1-%d)", maxBatchSize))
	flags.IntVar(&cfg.Runs, "runs", 30, fmt.Sprintf("timed Monte Carlo runs (1-%d)", maxRuns))
	flags.IntVar(&cfg.Warmup, "warmup", 3, fmt.Sprintf("untimed warm-up runs (0-%d)", maxWarmup))
	flags.IntVar(&cfg.LatencySamples, "latency", 500, fmt.Sprintf("single-image requests to time (1-%d)", maxLatency))
	flags.Float64Var(&cfg.Tolerance, "tolerance", 1e-3, "largest allowed |logit difference| vs Python")
	flags.StringVar(&cfg.CPUProfile, "cpuprofile", "", "write a CPU profile to this file")
	flags.StringVar(&cfg.MemProfile, "memprofile", "", "write a heap profile to this file")
	flags.BoolVar(&cfg.Pause, "pause", len(args) == 0,
		"wait for Enter before exiting (default on when started with no arguments, e.g. by double-click)")
	flags.Usage = func() {
		fmt.Fprintf(usageOut, "Usage: fashion-go [flags]\n\n"+
			"Classifies the Fashion-MNIST test set with an ONNX model run by GoMLX,\n"+
			"checks the predictions against Python's, and benchmarks the run time.\n\nFlags:\n")
		flags.PrintDefaults()
	}

	if err := flags.Parse(args); err != nil {
		return cfg, err
	}
	if flags.NArg() > 0 {
		return cfg, fmt.Errorf("unexpected argument %q (all options are flags, e.g. -runs 10)", flags.Arg(0))
	}
	return cfg, cfg.validate()
}

// validate checks numeric ranges and that every input file exists,
// collecting all problems so the user can fix them in one pass.
func (cfg config) validate() error {
	var problems []string
	if cfg.BatchSize < 1 || cfg.BatchSize > maxBatchSize {
		problems = append(problems, fmt.Sprintf("-batch must be between 1 and %d (got %d)", maxBatchSize, cfg.BatchSize))
	}
	if cfg.Runs < 1 || cfg.Runs > maxRuns {
		problems = append(problems, fmt.Sprintf("-runs must be between 1 and %d (got %d)", maxRuns, cfg.Runs))
	}
	if cfg.Warmup < 0 || cfg.Warmup > maxWarmup {
		problems = append(problems, fmt.Sprintf("-warmup must be between 0 and %d (got %d)", maxWarmup, cfg.Warmup))
	}
	if cfg.LatencySamples < 1 || cfg.LatencySamples > maxLatency {
		problems = append(problems, fmt.Sprintf("-latency must be between 1 and %d (got %d)", maxLatency, cfg.LatencySamples))
	}
	if cfg.Tolerance <= 0 {
		problems = append(problems, fmt.Sprintf("-tolerance must be positive (got %g)", cfg.Tolerance))
	}

	requiredFiles := map[string]string{"-model": cfg.ModelPath, "-images": cfg.ImagesPath, "-labels": cfg.LabelsPath}
	if cfg.PythonPredsPath != "" {
		requiredFiles["-python-preds"] = cfg.PythonPredsPath
	}
	for _, name := range []string{"-model", "-images", "-labels", "-python-preds"} {
		path, wanted := requiredFiles[name]
		if wanted && !isRegularFile(path) {
			problems = append(problems, fmt.Sprintf("%s: file not found: %s", name, path))
		}
	}

	if len(problems) == 0 {
		return nil
	}
	return errors.New("invalid input:\n  - " + strings.Join(problems, "\n  - "))
}

func isRegularFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

// findProjectRoot returns the first candidate directory containing the default
// model file: the working directory, the executable's directory, or its parent.
// This lets the .exe work when double-clicked from the repository or its bin/ folder.
func findProjectRoot() string {
	candidates := []string{"."}
	if executable, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(executable)
		candidates = append(candidates, exeDir, filepath.Dir(exeDir))
	}
	for _, dir := range candidates {
		if isRegularFile(filepath.Join(dir, defaultModel)) {
			return dir
		}
	}
	return "."
}
