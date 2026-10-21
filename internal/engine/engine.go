// Package engine loads an ONNX image classifier into GoMLX and runs inference.
//
// It uses GoMLX's pure-Go compute backend, so the resulting program needs no
// C libraries, Python, or ONNX Runtime and cross-compiles to a single Windows .exe.
package engine

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend" // registers the pure-Go "go" backend
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

// BackendConfig selects GoMLX's pure-Go backend.
const BackendConfig = "go"

// Classifier is an ONNX model compiled to a GoMLX computation graph.
//
// The model must take one float32 input shaped [batch, channels, height, width]
// and return one float32 output shaped [batch, classes].
type Classifier struct {
	onnxModel  onnx.Model
	exec       *model.Exec
	inputName  string
	outputName string
	NumClasses int
}

// Load parses the ONNX file at path and prepares it for execution.
func Load(path string) (*Classifier, error) {
	onnxModel, err := parser.ParseFile(path)
	if err != nil {
		return nil, fmt.Errorf("parsing ONNX model %s: %w", path, err)
	}
	inputNames, _ := onnxModel.Inputs()
	outputNames, outputShapes := onnxModel.Outputs()
	if len(inputNames) != 1 || len(outputNames) != 1 {
		return nil, fmt.Errorf("%s: expected 1 input and 1 output, found %d and %d",
			path, len(inputNames), len(outputNames))
	}
	if outputShapes[0].Rank() != 2 {
		return nil, fmt.Errorf("%s: expected output shaped [batch, classes], got %s", path, outputShapes[0])
	}

	backend, err := compute.NewWithConfig(BackendConfig)
	if err != nil {
		return nil, fmt.Errorf("starting GoMLX backend %q: %w", BackendConfig, err)
	}
	store := model.NewStore()
	if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
		return nil, fmt.Errorf("loading ONNX weights into GoMLX: %w", err)
	}

	classifier := &Classifier{
		onnxModel:  onnxModel,
		inputName:  inputNames[0],
		outputName: outputNames[0],
		NumClasses: outputShapes[0].Dimensions[1],
	}
	classifier.exec, err = model.NewExec(backend, store, classifier.buildGraph)
	if err != nil {
		return nil, fmt.Errorf("building GoMLX executor: %w", err)
	}
	return classifier, nil
}

// buildGraph converts the ONNX graph into GoMLX ops. GoMLX calls it (and
// compiles the result) once per distinct input shape, i.e. per batch size.
func (c *Classifier) buildGraph(scope *model.Scope, pixels *graph.Node) *graph.Node {
	outputs := c.onnxModel.CallGraph(scope, pixels.Graph(),
		map[string]*graph.Node{c.inputName: pixels}, c.outputName)
	return outputs[0]
}

// Predict runs one batch. pixels holds batch*channels*height*width values in
// row-major order; the returned slice holds batch*NumClasses logits.
func (c *Classifier) Predict(pixels []float32, batch, channels, height, width int) ([]float32, error) {
	if want := batch * channels * height * width; len(pixels) != want {
		return nil, fmt.Errorf("got %d pixel values, want %d for a %dx%dx%dx%d batch",
			len(pixels), want, batch, channels, height, width)
	}
	input := tensors.FromFlatDataAndDimensions(pixels, batch, channels, height, width)
	defer input.FinalizeAll()

	output, err := c.exec.Call1(input)
	if err != nil {
		return nil, fmt.Errorf("executing model: %w", err)
	}
	defer output.FinalizeAll()
	return tensors.CopyFlatData[float32](output)
}

// Close releases the compiled graphs and model resources.
func (c *Classifier) Close() {
	c.exec.Finalize()
	c.onnxModel.Close()
}
