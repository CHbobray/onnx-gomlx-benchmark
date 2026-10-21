// Package compare checks Go predictions against the reference predictions
// that the Python script wrote to CSV, and writes Go predictions in the same format.
package compare

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
)

// Reference is one row of results/python_predictions.csv.
type Reference struct {
	Label     int
	Predicted int
	Logits    []float32
}

// Report summarizes how closely two sets of predictions agree.
type Report struct {
	Images           int
	ClassAgreement   float64 // fraction of images with the same predicted class
	DisagreeIndices  []int   // images whose predicted class differs
	MaxAbsLogitDiff  float64
	MeanAbsLogitDiff float64
	GoAccuracy       float64 // Go predictions vs true labels
	RefAccuracy      float64 // Python predictions vs true labels
}

// Matches reports whether every predicted class agrees and logits are within tolerance.
func (r Report) Matches(tolerance float64) bool {
	return len(r.DisagreeIndices) == 0 && r.MaxAbsLogitDiff <= tolerance
}

// ReadReferenceCSV parses the Python predictions file.
// Columns: index, label, predicted, logit_0 ... logit_{k-1}.
func ReadReferenceCSV(path string) ([]Reference, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	references, err := parseReferences(csv.NewReader(file))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return references, nil
}

func parseReferences(reader *csv.Reader) ([]Reference, error) {
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	const leadingColumns = 3
	numClasses := len(header) - leadingColumns
	if numClasses < 1 {
		return nil, fmt.Errorf("header has %d columns, expected index,label,predicted,logit_...", len(header))
	}

	var references []Reference
	for line := 2; ; line++ {
		record, err := reader.Read()
		if err == io.EOF {
			return references, nil
		}
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		reference, err := parseRecord(record, numClasses)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		references = append(references, reference)
	}
}

func parseRecord(record []string, numClasses int) (Reference, error) {
	label, err := strconv.Atoi(record[1])
	if err != nil {
		return Reference{}, fmt.Errorf("label: %w", err)
	}
	predicted, err := strconv.Atoi(record[2])
	if err != nil {
		return Reference{}, fmt.Errorf("predicted: %w", err)
	}
	logits := make([]float32, numClasses)
	for k := range logits {
		value, err := strconv.ParseFloat(record[3+k], 32)
		if err != nil {
			return Reference{}, fmt.Errorf("logit_%d: %w", k, err)
		}
		logits[k] = float32(value)
	}
	return Reference{Label: label, Predicted: predicted, Logits: logits}, nil
}

// Compare measures agreement between Go logits/classes and the Python references.
func Compare(references []Reference, goLogits []float32, goClasses []int) (Report, error) {
	if len(goClasses) == 0 {
		return Report{}, fmt.Errorf("no Go predictions to compare")
	}
	if len(references) != len(goClasses) {
		return Report{}, fmt.Errorf("python has %d predictions, Go has %d", len(references), len(goClasses))
	}
	numClasses := len(goLogits) / len(goClasses)
	report := Report{Images: len(references)}
	var sumAbsDiff float64
	var goCorrect, refCorrect int

	for i, reference := range references {
		if len(reference.Logits) != numClasses {
			return Report{}, fmt.Errorf("image %d: python has %d logits, Go has %d", i, len(reference.Logits), numClasses)
		}
		for k, want := range reference.Logits {
			diff := math.Abs(float64(goLogits[i*numClasses+k] - want))
			sumAbsDiff += diff
			report.MaxAbsLogitDiff = math.Max(report.MaxAbsLogitDiff, diff)
		}
		if goClasses[i] != reference.Predicted {
			report.DisagreeIndices = append(report.DisagreeIndices, i)
		}
		if goClasses[i] == reference.Label {
			goCorrect++
		}
		if reference.Predicted == reference.Label {
			refCorrect++
		}
	}
	n := float64(report.Images)
	report.ClassAgreement = 1 - float64(len(report.DisagreeIndices))/n
	report.MeanAbsLogitDiff = sumAbsDiff / (n * float64(numClasses))
	report.GoAccuracy = float64(goCorrect) / n
	report.RefAccuracy = float64(refCorrect) / n
	return report, nil
}

// WritePredictionsCSV writes Go predictions in the same layout as the Python file.
func WritePredictionsCSV(path string, labels []uint8, logits []float32, classes []int) error {
	if len(classes) == 0 || len(logits)%len(classes) != 0 {
		return fmt.Errorf("got %d logits for %d images: cannot split into rows", len(logits), len(classes))
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	numClasses := len(logits) / len(classes)

	header := []string{"index", "label", "predicted"}
	for k := range numClasses {
		header = append(header, fmt.Sprintf("logit_%d", k))
	}
	writer.Write(header)
	for i, class := range classes {
		row := []string{strconv.Itoa(i), strconv.Itoa(int(labels[i])), strconv.Itoa(class)}
		for _, value := range logits[i*numClasses : (i+1)*numClasses] {
			row = append(row, strconv.FormatFloat(float64(value), 'f', 6, 32))
		}
		writer.Write(row)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}
