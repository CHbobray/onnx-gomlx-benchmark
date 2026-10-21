package compare

import (
	"path/filepath"
	"strings"
	"testing"
)

func syntheticReferences() []Reference {
	return []Reference{
		{Label: 1, Predicted: 1, Logits: []float32{0.1, 2.0, -1.0}},
		{Label: 0, Predicted: 2, Logits: []float32{0.5, 0.0, 1.5}},
	}
}

func TestCompareIdentical(t *testing.T) {
	refs := syntheticReferences()
	logits := append(append([]float32{}, refs[0].Logits...), refs[1].Logits...)
	report, err := Compare(refs, logits, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Matches(1e-6) || report.ClassAgreement != 1 {
		t.Errorf("identical inputs should match: %+v", report)
	}
	if report.GoAccuracy != 0.5 || report.RefAccuracy != 0.5 {
		t.Errorf("accuracy: got go=%v ref=%v, want 0.5", report.GoAccuracy, report.RefAccuracy)
	}
}

func TestCompareDetectsDisagreement(t *testing.T) {
	refs := syntheticReferences()
	logits := []float32{0.1, 2.0, -1.0, 3.0, 0.0, 1.5} // second image now predicts class 0
	report, err := Compare(refs, logits, []int{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if report.Matches(1e-3) {
		t.Error("expected a mismatch")
	}
	if len(report.DisagreeIndices) != 1 || report.DisagreeIndices[0] != 1 {
		t.Errorf("disagreements: got %v, want [1]", report.DisagreeIndices)
	}
	if diff := report.MaxAbsLogitDiff; diff < 2.49 || diff > 2.51 {
		t.Errorf("max diff: got %v, want 2.5", diff)
	}
}

func TestCompareLengthMismatch(t *testing.T) {
	if _, err := Compare(syntheticReferences(), []float32{1, 2, 3}, []int{0}); err == nil {
		t.Error("expected an error when counts differ")
	}
}

func TestCSVRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "preds.csv")
	labels := []uint8{1, 0}
	logits := []float32{0.1, 2.0, -1.0, 0.5, 0.0, 1.5}
	if err := WritePredictionsCSV(path, labels, logits, []int{1, 2}); err != nil {
		t.Fatal(err)
	}
	refs, err := ReadReferenceCSV(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 2 || refs[1].Predicted != 2 || refs[1].Logits[2] != 1.5 {
		t.Errorf("round trip lost data: %+v", refs)
	}
}

func TestReadReferenceRejectsBadNumber(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.csv")
	if err := WritePredictionsCSV(path, []uint8{1}, []float32{1, 2}, []int{1}); err != nil {
		t.Fatal(err)
	}
	// Corrupt the file by appending a row with a non-numeric logit.
	appendLine(t, path, "1,1,1,abc,2\n")
	if _, err := ReadReferenceCSV(path); err == nil || !strings.Contains(err.Error(), "line 3") {
		t.Errorf("expected a line-3 parse error, got %v", err)
	}
}

func TestRejectsEmptyPredictions(t *testing.T) {
	if _, err := Compare(nil, nil, nil); err == nil {
		t.Error("Compare should reject empty input")
	}
	path := filepath.Join(t.TempDir(), "empty.csv")
	if err := WritePredictionsCSV(path, nil, nil, nil); err == nil {
		t.Error("WritePredictionsCSV should reject empty input")
	}
}

// syntheticPredictions builds n images' worth of predictions with 10 classes.
func syntheticPredictions(n int) ([]Reference, []float32, []int) {
	const numClasses = 10
	references := make([]Reference, n)
	logits := make([]float32, n*numClasses)
	classes := make([]int, n)
	for i := range n {
		class := i % numClasses
		row := make([]float32, numClasses)
		row[class] = 1
		copy(logits[i*numClasses:], row)
		references[i] = Reference{Label: class, Predicted: class, Logits: row}
		classes[i] = class
	}
	return references, logits, classes
}

// BenchmarkCompare measures comparing 10,000 predictions, the size of the test set.
func BenchmarkCompare(b *testing.B) {
	references, logits, classes := syntheticPredictions(10_000)
	for b.Loop() {
		if _, err := Compare(references, logits, classes); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkReadReferenceCSV measures parsing a 10,000-row predictions file.
func BenchmarkReadReferenceCSV(b *testing.B) {
	references, logits, classes := syntheticPredictions(10_000)
	labels := make([]uint8, len(references))
	path := filepath.Join(b.TempDir(), "preds.csv")
	if err := WritePredictionsCSV(path, labels, logits, classes); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := ReadReferenceCSV(path); err != nil {
			b.Fatal(err)
		}
	}
}
