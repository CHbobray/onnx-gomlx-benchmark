// Package idx reads the IDX file format used by MNIST and Fashion-MNIST.
//
// An IDX file is a big-endian header (magic number plus dimension sizes)
// followed by raw uint8 values. Files may be gzip-compressed; the reader
// detects compression from the first two bytes, so callers need not care.
package idx

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Magic numbers identifying uint8 image (3-D) and label (1-D) files.
const (
	ImagesMagic uint32 = 2051
	LabelsMagic uint32 = 2049
)

var gzipSignature = []byte{0x1f, 0x8b}

// Images holds a stack of grayscale images in row-major order.
type Images struct {
	Count, Rows, Cols int
	Pixels            []uint8 // length Count*Rows*Cols
}

// PixelsPerImage is the number of pixels in one image.
func (im *Images) PixelsPerImage() int { return im.Rows * im.Cols }

// ReadImagesFile opens path (gzipped or raw) and parses an IDX image file.
func ReadImagesFile(path string) (*Images, error) {
	var images *Images
	err := withReader(path, func(r io.Reader) error {
		var parseErr error
		images, parseErr = ReadImages(r)
		return parseErr
	})
	return images, err
}

// ReadLabelsFile opens path (gzipped or raw) and parses an IDX label file.
func ReadLabelsFile(path string) ([]uint8, error) {
	var labels []uint8
	err := withReader(path, func(r io.Reader) error {
		var parseErr error
		labels, parseErr = ReadLabels(r)
		return parseErr
	})
	return labels, err
}

// ReadImages parses an uncompressed IDX image stream.
func ReadImages(r io.Reader) (*Images, error) {
	var header [4]uint32 // magic, count, rows, cols
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("reading image header: %w", err)
	}
	if header[0] != ImagesMagic {
		return nil, fmt.Errorf("not an IDX image file: magic %d, want %d", header[0], ImagesMagic)
	}
	images := &Images{Count: int(header[1]), Rows: int(header[2]), Cols: int(header[3])}
	images.Pixels = make([]uint8, images.Count*images.PixelsPerImage())
	if _, err := io.ReadFull(r, images.Pixels); err != nil {
		return nil, fmt.Errorf("reading %d images of %dx%d: %w", images.Count, images.Rows, images.Cols, err)
	}
	return images, nil
}

// ReadLabels parses an uncompressed IDX label stream.
func ReadLabels(r io.Reader) ([]uint8, error) {
	var header [2]uint32 // magic, count
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("reading label header: %w", err)
	}
	if header[0] != LabelsMagic {
		return nil, fmt.Errorf("not an IDX label file: magic %d, want %d", header[0], LabelsMagic)
	}
	labels := make([]uint8, header[1])
	if _, err := io.ReadFull(r, labels); err != nil {
		return nil, fmt.Errorf("reading %d labels: %w", len(labels), err)
	}
	return labels, nil
}

// withReader opens path, transparently un-gzips it, and passes the stream to parse.
func withReader(path string, parse func(io.Reader) error) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	buffered := bufio.NewReader(file)
	signature, err := buffered.Peek(len(gzipSignature))
	if err != nil {
		return fmt.Errorf("%s: file too short: %w", path, err)
	}
	var stream io.Reader = buffered
	if signature[0] == gzipSignature[0] && signature[1] == gzipSignature[1] {
		unzipped, err := gzip.NewReader(buffered)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		defer unzipped.Close()
		stream = unzipped
	}
	if err := parse(stream); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
