package bench

import (
	"errors"
	"math"
	"testing"
	"time"
)

func TestSummarizeKnownValues(t *testing.T) {
	durations := []time.Duration{40, 10, 30, 20} // shuffled on purpose
	for i := range durations {
		durations[i] *= time.Millisecond
	}
	stats := Summarize(durations, 1000)

	checks := map[string][2]float64{
		"mean":   {stats.MeanMs, 25},
		"median": {stats.MedianMs, 25},
		"min":    {stats.MinMs, 10},
		"max":    {stats.MaxMs, 40},
		"std":    {stats.StdMs, math.Sqrt(500.0 / 3)},
		"ips":    {stats.ImagesPerSecond, 40000},
	}
	for name, pair := range checks {
		if math.Abs(pair[0]-pair[1]) > 1e-9 {
			t.Errorf("%s: got %v, want %v", name, pair[0], pair[1])
		}
	}
}

func TestMonteCarloCountsRuns(t *testing.T) {
	calls := 0
	var reported []int
	stats, err := MonteCarlo(5, 2, 10, func() error { calls++; return nil },
		func(run, runs int, _ time.Duration) { reported = append(reported, run) })
	if err != nil {
		t.Fatal(err)
	}
	if calls != 7 || stats.Runs != 5 {
		t.Errorf("got %d calls and %d runs, want 7 and 5", calls, stats.Runs)
	}
	if len(reported) != 5 || reported[4] != 5 {
		t.Errorf("progress should report timed runs 1-5 only, got %v", reported)
	}
}

func TestMonteCarloStopsOnError(t *testing.T) {
	failure := errors.New("boom")
	_, err := MonteCarlo(3, 0, 1, func() error { return failure }, nil)
	if !errors.Is(err, failure) {
		t.Errorf("expected wrapped workload error, got %v", err)
	}
}

func TestMonteCarloRejectsZeroRuns(t *testing.T) {
	if _, err := MonteCarlo(0, 0, 1, func() error { return nil }, nil); err == nil {
		t.Error("expected an error for zero runs")
	}
}

func TestLatencyPassesSampleIndex(t *testing.T) {
	var seen []int
	stats, err := Latency(3, 1, func(i int) error { seen = append(seen, i); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 4 || seen[3] != 2 || stats.Runs != 3 || stats.ImagesPerRun != 1 {
		t.Errorf("unexpected calls %v / stats %+v", seen, stats)
	}
}

func TestSampleMemoryIsPositive(t *testing.T) {
	if memory := SampleMemory(); memory.SysMB <= 0 {
		t.Errorf("expected positive system memory, got %+v", memory)
	}
}
