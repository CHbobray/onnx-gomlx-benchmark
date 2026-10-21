package engine

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/CHbobray/onnx-gomlx-benchmark/internal/compare"
	"github.com/CHbobray/onnx-gomlx-benchmark/internal/idx"
)

// Paths from this package directory to the repository's model and data.
var (
	modelPath       = filepath.Join("..", "..", "models", "fashion_cnn.onnx")
	testImagesPath  = filepath.Join("..", "..", "data", "t10k-images-idx3-ubyte.gz")
	pythonPredsPath = filepath.Join("..", "..", "results", "python_predictions.csv")
)

func TestArgMax(t *testing.T) {
	values := []float32{
		0.1, 0.9, 0.3,
		5, -1, 4,
		-2, -3, -1,
	}
	got := ArgMax(values, 3)
	want := []int{1, 0, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ArgMax: got %v, want %v", got, want)
		}
	}
}

func requireFiles(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("skipping: %s not found (run the Python scripts first)", path)
		}
	}
}

func loadClassifier(t *testing.T) *Classifier {
	t.Helper()
	classifier, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(classifier.Close)
	return classifier
}

// firstImages returns a copy of images holding only the first n.
func firstImages(images *idx.Images, n int) *idx.Images {
	subset := *images
	subset.Count = n
	subset.Pixels = images.Pixels[:n*images.PixelsPerImage()]
	return &subset
}

// TestMatchesPythonOnSample checks the real model against Python's predictions
// for the first images of the test set, with an uneven batch size so the last
// batch is partial.
func TestMatchesPythonOnSample(t *testing.T) {
	requireFiles(t, modelPath, testImagesPath, pythonPredsPath)
	const sampleSize, batchSize = 50, 16

	images, err := idx.ReadImagesFile(testImagesPath)
	if err != nil {
		t.Fatal(err)
	}
	references, err := compare.ReadReferenceCSV(pythonPredsPath)
	if err != nil {
		t.Fatal(err)
	}

	classifier := loadClassifier(t)
	logits, err := classifier.PredictImages(firstImages(images, sampleSize), batchSize)
	if err != nil {
		t.Fatal(err)
	}
	report, err := compare.Compare(references[:sampleSize], logits, ArgMax(logits, classifier.NumClasses))
	if err != nil {
		t.Fatal(err)
	}
	if !report.Matches(1e-3) {
		t.Errorf("Go disagrees with Python: %+v", report)
	}
}

// TestBatchSizeDoesNotChangeResults: batching is an implementation detail and
// must not alter any logit beyond float rounding.
func TestBatchSizeDoesNotChangeResults(t *testing.T) {
	requireFiles(t, modelPath, testImagesPath)
	images, err := idx.ReadImagesFile(testImagesPath)
	if err != nil {
		t.Fatal(err)
	}
	sample := firstImages(images, 12)
	classifier := loadClassifier(t)

	whole, err := classifier.PredictImages(sample, 12)
	if err != nil {
		t.Fatal(err)
	}
	single, err := classifier.PredictOne(sample, 7)
	if err != nil {
		t.Fatal(err)
	}
	for k, value := range single {
		if diff := math.Abs(float64(value - whole[7*classifier.NumClasses+k])); diff > 1e-4 {
			t.Errorf("logit %d differs by %g between batch sizes 12 and 1", k, diff)
		}
	}
}

func TestPredictRejectsWrongSize(t *testing.T) {
	requireFiles(t, modelPath)
	classifier := loadClassifier(t)
	if _, err := classifier.Predict(make([]float32, 10), 1, 1, 28, 28); err == nil {
		t.Error("expected an error for a mis-sized input")
	}
}

func TestPredictOneRejectsBadIndex(t *testing.T) {
	requireFiles(t, modelPath)
	classifier := loadClassifier(t)
	images := &idx.Images{Count: 1, Rows: 28, Cols: 28, Pixels: make([]uint8, 784)}
	if _, err := classifier.PredictOne(images, 1); err == nil {
		t.Error("expected an out-of-range error")
	}
}

func TestLoadRejectsMissingOrBadFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "missing.onnx")); err == nil {
		t.Error("expected an error for a missing model")
	}
	bad := filepath.Join(t.TempDir(), "bad.onnx")
	os.WriteFile(bad, []byte("this is not a protobuf"), 0o644)
	if _, err := Load(bad); err == nil {
		t.Error("expected an error for a corrupt model")
	}
}

// BenchmarkPredictBatch measures one 100-image batch; run with
// go test -bench . -benchmem ./internal/engine
func BenchmarkPredictBatch(b *testing.B) {
	images, err := idx.ReadImagesFile(testImagesPath)
	if err != nil {
		b.Skip(err)
	}
	classifier, err := Load(modelPath)
	if err != nil {
		b.Skip(err)
	}
	defer classifier.Close()
	sample := firstImages(images, 100)
	for b.Loop() {
		if _, err := classifier.PredictImages(sample, 100); err != nil {
			b.Fatal(err)
		}
	}
}
