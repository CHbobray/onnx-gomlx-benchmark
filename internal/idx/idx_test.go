package idx

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// encodeImages builds a synthetic IDX image file in memory.
func encodeImages(t *testing.T, count, rows, cols int) []byte {
	t.Helper()
	var buf bytes.Buffer
	header := []uint32{ImagesMagic, uint32(count), uint32(rows), uint32(cols)}
	if err := binary.Write(&buf, binary.BigEndian, header); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < count*rows*cols; i++ {
		buf.WriteByte(uint8(i % 256))
	}
	return buf.Bytes()
}

func encodeLabels(t *testing.T, labels []uint8) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.BigEndian, []uint32{LabelsMagic, uint32(len(labels))}); err != nil {
		t.Fatal(err)
	}
	buf.Write(labels)
	return buf.Bytes()
}

func gzipBytes(t *testing.T, raw []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)
	if _, err := writer.Write(raw); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func writeTemp(t *testing.T, name string, contents []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadImagesRawAndGzipAgree(t *testing.T) {
	raw := encodeImages(t, 3, 4, 5)
	for name, contents := range map[string][]byte{"raw.idx": raw, "zipped.idx.gz": gzipBytes(t, raw)} {
		images, err := ReadImagesFile(writeTemp(t, name, contents))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if images.Count != 3 || images.Rows != 4 || images.Cols != 5 {
			t.Errorf("%s: got %dx%dx%d, want 3x4x5", name, images.Count, images.Rows, images.Cols)
		}
		if len(images.Pixels) != 60 || images.Pixels[59] != 59 {
			t.Errorf("%s: pixel data not read correctly", name)
		}
	}
}

func TestReadLabels(t *testing.T) {
	want := []uint8{9, 0, 3, 7}
	got, err := ReadLabelsFile(writeTemp(t, "labels.gz", gzipBytes(t, encodeLabels(t, want))))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestRejectsWrongMagic(t *testing.T) {
	labelFile := writeTemp(t, "labels", encodeLabels(t, make([]uint8, 20)))
	if _, err := ReadImagesFile(labelFile); err == nil || !strings.Contains(err.Error(), "not an IDX image") {
		t.Errorf("expected wrong-magic error, got %v", err)
	}
}

func TestRejectsTruncatedData(t *testing.T) {
	raw := encodeImages(t, 2, 3, 3)
	if _, err := ReadImages(bytes.NewReader(raw[:len(raw)-1])); err == nil {
		t.Error("expected an error for truncated pixel data")
	}
}

func TestMissingFile(t *testing.T) {
	if _, err := ReadLabelsFile(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Error("expected an error for a missing file")
	}
}

// BenchmarkReadImagesFile measures reading and un-gzipping the real 10,000-image
// test file (skipped if the data folder is missing).
func BenchmarkReadImagesFile(b *testing.B) {
	path := filepath.Join("..", "..", "data", "t10k-images-idx3-ubyte.gz")
	if _, err := os.Stat(path); err != nil {
		b.Skip(err)
	}
	for b.Loop() {
		if _, err := ReadImagesFile(path); err != nil {
			b.Fatal(err)
		}
	}
}
