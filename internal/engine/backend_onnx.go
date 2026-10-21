//go:build onnx

// Optional: with `go build -tags onnx`, the program also contains GoMLX's
// ONNX Runtime backend. Select it at run time with GOMLX_BACKEND=onnx.
// It needs cgo and the ONNX Runtime shared library (point
// ONNXRUNTIME_SHARED_LIBRARY_PATH at it, or let GoMLX download it).
// In compute-onnx v0.1.13 the native loader is POSIX-only, so this build
// works on Linux and macOS but not on Windows.

package engine

import _ "github.com/gomlx/compute-onnx" // registers the "onnx" backend
