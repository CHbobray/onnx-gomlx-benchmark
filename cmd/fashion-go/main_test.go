package main

import (
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CHbobray/onnx-gomlx-benchmark/internal/bench"
)

// fakeProject creates a directory with empty placeholder input files, enough
// for configuration validation (which only checks that files exist).
func fakeProject(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, rel := range []string{defaultModel, defaultImages, defaultLabels, defaultPythonPreds} {
		path := filepath.Join(root, rel)
		os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestDefaultsAreValid(t *testing.T) {
	cfg, err := parseConfig([]string{"-runs", "5"}, fakeProject(t), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Runs != 5 || cfg.BatchSize != 1000 || cfg.Pause {
		t.Errorf("unexpected config: %+v", cfg)
	}
}

func TestNoArgumentsTurnsOnPause(t *testing.T) {
	cfg, err := parseConfig(nil, fakeProject(t), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Pause {
		t.Error("expected -pause to default on when no arguments are given")
	}
}

func TestInvalidInputsAreAllReported(t *testing.T) {
	args := []string{"-runs", "0", "-batch", "20000", "-latency", "-3", "-model", "nope.onnx"}
	_, err := parseConfig(args, fakeProject(t), io.Discard)
	if err == nil {
		t.Fatal("expected validation errors")
	}
	for _, want := range []string{"-runs", "-batch", "-latency", "-model: file not found"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %q:\n%v", want, err)
		}
	}
}

func TestNonNumericFlagIsRejected(t *testing.T) {
	if _, err := parseConfig([]string{"-runs", "ten"}, fakeProject(t), io.Discard); err == nil {
		t.Error("expected a parse error for -runs ten")
	}
}

func TestStrayArgumentIsRejected(t *testing.T) {
	_, err := parseConfig([]string{"30"}, fakeProject(t), io.Discard)
	if err == nil || !strings.Contains(err.Error(), "unexpected argument") {
		t.Errorf("expected an unexpected-argument error, got %v", err)
	}
}

func TestHelpFlag(t *testing.T) {
	_, err := parseConfig([]string{"-h"}, fakeProject(t), io.Discard)
	if !errors.Is(err, flag.ErrHelp) {
		t.Errorf("expected flag.ErrHelp, got %v", err)
	}
}

func TestEmptyPythonPredsSkipsThatFile(t *testing.T) {
	root := fakeProject(t)
	os.Remove(filepath.Join(root, defaultPythonPreds))
	if _, err := parseConfig([]string{"-python-preds", ""}, root, io.Discard); err != nil {
		t.Errorf("empty -python-preds should be allowed: %v", err)
	}
}

func TestTimingTablesOrderAndRatios(t *testing.T) {
	engines := map[string]engineTiming{
		"gomlx":   {Throughput: bench.Stats{MeanMs: 100, Runs: 3}, Latency: bench.Stats{MeanMs: 2, Runs: 3}},
		"pytorch": {Throughput: bench.Stats{MeanMs: 50, Runs: 3}, Latency: bench.Stats{MeanMs: 4, Runs: 3}},
	}
	tables := timingTables(engines)
	throughput, latency, _ := strings.Cut(tables, "**Latency**")
	if strings.Index(throughput, "PyTorch") > strings.Index(throughput, "GoMLX") {
		t.Error("faster engine should be listed first")
	}
	if !strings.Contains(throughput, "2.0x faster") || !strings.Contains(latency, "2.0x slower") {
		t.Errorf("unexpected ratios:\n%s", tables)
	}
}
