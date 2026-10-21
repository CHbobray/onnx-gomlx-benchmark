// Package bench runs Monte Carlo timing benchmarks and samples memory use.
//
// "Monte Carlo" here means repeating the same workload many times and
// treating each run time as a random sample, so we can report the spread
// (standard deviation, percentiles) instead of a single noisy number.
package bench

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"time"
)

// Stats summarizes a sample of run durations, in milliseconds.
type Stats struct {
	Runs            int     `json:"runs"`
	ImagesPerRun    int     `json:"images_per_run"`
	MeanMs          float64 `json:"mean_ms"`
	StdMs           float64 `json:"std_ms"`
	MinMs           float64 `json:"min_ms"`
	MedianMs        float64 `json:"median_ms"`
	P95Ms           float64 `json:"p95_ms"`
	MaxMs           float64 `json:"max_ms"`
	ImagesPerSecond float64 `json:"images_per_second"`
}

// Progress, if not nil, is called after each timed run with the 1-based run
// number, the total number of runs, and how long that run took. It is called
// outside the timed region, so reporting does not affect the measurements.
type Progress func(run, runs int, took time.Duration)

// MonteCarlo calls workload warmup times untimed, then runs times timed,
// and returns statistics over the timed runs. It stops at the first error.
func MonteCarlo(runs, warmup, imagesPerRun int, workload func() error, progress Progress) (Stats, error) {
	if runs < 1 {
		return Stats{}, fmt.Errorf("runs must be at least 1, got %d", runs)
	}
	for range warmup {
		if err := workload(); err != nil {
			return Stats{}, fmt.Errorf("warm-up: %w", err)
		}
	}
	durations := make([]time.Duration, runs)
	for run := range runs {
		started := time.Now()
		if err := workload(); err != nil {
			return Stats{}, fmt.Errorf("run %d: %w", run+1, err)
		}
		durations[run] = time.Since(started)
		if progress != nil {
			progress(run+1, runs, durations[run])
		}
	}
	return Summarize(durations, imagesPerRun), nil
}

// Latency times each of samples single calls to request(i) separately, after
// warmup untimed calls, and summarizes the per-call durations. This models a
// server answering one user request at a time.
func Latency(samples, warmup int, request func(i int) error) (Stats, error) {
	if samples < 1 {
		return Stats{}, fmt.Errorf("samples must be at least 1, got %d", samples)
	}
	for i := range warmup {
		if err := request(i); err != nil {
			return Stats{}, fmt.Errorf("warm-up: %w", err)
		}
	}
	durations := make([]time.Duration, samples)
	for i := range samples {
		started := time.Now()
		if err := request(i); err != nil {
			return Stats{}, fmt.Errorf("request %d: %w", i, err)
		}
		durations[i] = time.Since(started)
	}
	return Summarize(durations, 1), nil
}

// Summarize computes Stats for a set of durations.
func Summarize(durations []time.Duration, imagesPerRun int) Stats {
	ms := make([]float64, len(durations))
	for i, d := range durations {
		ms[i] = float64(d) / float64(time.Millisecond)
	}
	slices.Sort(ms)

	var sum float64
	for _, v := range ms {
		sum += v
	}
	mean := sum / float64(len(ms))

	var squares float64
	for _, v := range ms {
		squares += (v - mean) * (v - mean)
	}
	std := 0.0
	if len(ms) > 1 {
		std = math.Sqrt(squares / float64(len(ms)-1)) // sample standard deviation
	}

	return Stats{
		Runs:            len(ms),
		ImagesPerRun:    imagesPerRun,
		MeanMs:          mean,
		StdMs:           std,
		MinMs:           ms[0],
		MedianMs:        median(ms),
		P95Ms:           ms[min(len(ms)-1, int(0.95*float64(len(ms))))],
		MaxMs:           ms[len(ms)-1],
		ImagesPerSecond: float64(imagesPerRun) / (mean / 1000),
	}
}

func median(sorted []float64) float64 {
	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[middle]
	}
	return (sorted[middle-1] + sorted[middle]) / 2
}

// Memory is a snapshot of Go heap usage, in megabytes.
type Memory struct {
	HeapInUseMB  float64 `json:"heap_in_use_mb"`
	TotalAllocMB float64 `json:"total_alloc_mb"`
	SysMB        float64 `json:"sys_mb"` // memory obtained from the OS
	NumGC        uint32  `json:"num_gc"`
}

// SampleMemory reads the Go runtime's memory statistics.
func SampleMemory() Memory {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	const mb = 1 << 20
	return Memory{
		HeapInUseMB:  float64(stats.HeapInuse) / mb,
		TotalAllocMB: float64(stats.TotalAlloc) / mb,
		SysMB:        float64(stats.Sys) / mb,
		NumGC:        stats.NumGC,
	}
}
