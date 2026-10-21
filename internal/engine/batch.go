package engine

import (
	"fmt"

	"github.com/CHbobray/onnx-gomlx-benchmark/internal/idx"
)

// PredictImages classifies every image in images, batchSize images at a time,
// and returns len(images)*NumClasses logits. Raw 0-255 pixel values are passed
// straight through because the model normalizes its own input.
func (c *Classifier) PredictImages(images *idx.Images, batchSize int) ([]float32, error) {
	if batchSize < 1 {
		return nil, fmt.Errorf("batch size must be positive, got %d", batchSize)
	}
	pixelsPerImage := images.PixelsPerImage()
	logits := make([]float32, 0, images.Count*c.NumClasses)
	batchPixels := make([]float32, batchSize*pixelsPerImage)

	for start := 0; start < images.Count; start += batchSize {
		count := min(batchSize, images.Count-start)
		source := images.Pixels[start*pixelsPerImage : (start+count)*pixelsPerImage]
		pixels := batchPixels[:len(source)]
		for i, value := range source {
			pixels[i] = float32(value)
		}
		batchLogits, err := c.Predict(pixels, count, 1, images.Rows, images.Cols)
		if err != nil {
			return nil, fmt.Errorf("batch starting at image %d: %w", start, err)
		}
		logits = append(logits, batchLogits...)
	}
	return logits, nil
}

// PredictOne classifies the single image at index, returning NumClasses logits.
func (c *Classifier) PredictOne(images *idx.Images, index int) ([]float32, error) {
	if index < 0 || index >= images.Count {
		return nil, fmt.Errorf("image index %d out of range [0, %d)", index, images.Count)
	}
	n := images.PixelsPerImage()
	pixels := make([]float32, n)
	for i, value := range images.Pixels[index*n : (index+1)*n] {
		pixels[i] = float32(value)
	}
	return c.Predict(pixels, 1, 1, images.Rows, images.Cols)
}

// ArgMax returns the index of the largest value in each row of a row-major
// [rows, cols] matrix: the predicted class for each image.
func ArgMax(values []float32, cols int) []int {
	rows := len(values) / cols
	best := make([]int, rows)
	for row := range rows {
		rowValues := values[row*cols : (row+1)*cols]
		for col, value := range rowValues {
			if value > rowValues[best[row]] {
				best[row] = col
			}
		}
	}
	return best
}
