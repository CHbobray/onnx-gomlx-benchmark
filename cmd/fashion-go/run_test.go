package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CHbobray/onnx-gomlx-benchmark/internal/idx"
)

// Real project files, relative to this package directory.
var (
	repoModel       = filepath.Join("..", "..", defaultModel)
	repoImages      = filepath.Join("..", "..", defaultImages)
	repoPythonPreds = filepath.Join("..", "..", defaultPythonPreds)
)

const sampleImages = 40

// smallTestSet writes the first sampleImages real test images, their labels,
// and the matching rows of Python's predictions to dir, so the whole program
// can run end to end in a fraction of a second.
func smallTestSet(t *testing.T, dir string) config {
	t.Helper()
	for _, path := range []string{repoModel, repoImages, repoPythonPreds} {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("skipping: %s not found", path)
		}
	}
	images, err := idx.ReadImagesFile(repoImages)
	if err != nil {
		t.Fatal(err)
	}
	pixels := images.Pixels[:sampleImages*images.PixelsPerImage()]
	imagesPath := filepath.Join(dir, "images.idx")
	writeIDX(t, imagesPath, []uint32{idx.ImagesMagic, sampleImages, uint32(images.Rows), uint32(images.Cols)}, pixels)

	predsPath := filepath.Join(dir, "python_predictions.csv")
	labels := copyFirstLines(t, repoPythonPreds, predsPath, sampleImages+1)
	labelsPath := filepath.Join(dir, "labels.idx")
	writeIDX(t, labelsPath, []uint32{idx.LabelsMagic, sampleImages}, labels)

	return config{
		ModelPath: repoModel, ImagesPath: imagesPath, LabelsPath: labelsPath,
		PythonPredsPath: predsPath, ResultsDir: filepath.Join(dir, "results"),
		BatchSize: 16, Runs: 2, Warmup: 1, LatencySamples: 3, Tolerance: 1e-3,
	}
}

func writeIDX(t *testing.T, path string, header []uint32, data []uint8) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if err := binary.Write(file, binary.BigEndian, header); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(data); err != nil {
		t.Fatal(err)
	}
}

// copyFirstLines copies the header plus the first rows of a predictions CSV
// and returns the true labels (second column) of the copied rows.
func copyFirstLines(t *testing.T, from, to string, lines int) []uint8 {
	t.Helper()
	source, err := os.Open(from)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	var kept []string
	var labels []uint8
	scanner := bufio.NewScanner(source)
	for len(kept) < lines && scanner.Scan() {
		line := scanner.Text()
		if len(kept) > 0 {
			fields := strings.Split(line, ",")
			labels = append(labels, uint8(fields[1][0]-'0'))
		}
		kept = append(kept, line)
	}
	if err := os.WriteFile(to, []byte(strings.Join(kept, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return labels
}

func TestRunEndToEnd(t *testing.T) {
	cfg := smallTestSet(t, t.TempDir())
	if err := run(cfg); err != nil {
		t.Fatalf("run failed: %v", err)
	}
	for _, name := range []string{"go_predictions.csv", "go_timing.json", "timing_comparison.md", "go_run_log.txt"} {
		if _, err := os.Stat(filepath.Join(cfg.ResultsDir, name)); err != nil {
			t.Errorf("expected output %s: %v", name, err)
		}
	}
	log, _ := os.ReadFile(filepath.Join(cfg.ResultsDir, "go_run_log.txt"))
	if !strings.Contains(string(log), "RESULT: MATCH") || !strings.Contains(string(log), "finished successfully") {
		t.Errorf("run log is missing the result or outcome:\n%s", log)
	}
}

func TestRunReportsMismatch(t *testing.T) {
	dir := t.TempDir()
	cfg := smallTestSet(t, dir)
	// In every Python row, turn the first negative logit into a large positive one
	// ("-5.1" becomes "95.1"), so the logits no longer match Go's.
	data, err := os.ReadFile(cfg.PythonPredsPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	for i := 1; i < len(lines); i++ {
		lines[i] = strings.Replace(lines[i], ",-", ",9", 1)
	}
	os.WriteFile(cfg.PythonPredsPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644)

	if err := run(cfg); !errors.Is(err, errMismatch) {
		t.Errorf("expected errMismatch, got %v", err)
	}
	log, _ := os.ReadFile(filepath.Join(cfg.ResultsDir, "go_run_log.txt"))
	if !strings.Contains(string(log), "run failed") {
		t.Errorf("run log should record the failure:\n%s", log)
	}
}
