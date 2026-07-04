// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/safetensors"
)

func TestModelMerge_WriteLinearMergedTensorChunksGood(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.layers.0.mlp.down_proj.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{
		{Name: name, Shape: []int{5}, Data: []float32{0, 2, 4, 6, 8}},
	})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{
		{Name: name, Shape: []int{5}, Data: []float32{10, 12, 14, 16, 18}},
	})
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}
	outPath := core.PathJoin(t.TempDir(), "out.bin")
	created := core.Create(outPath)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)

	err = writeLinearChunks(context.Background(), file, []safetensors.TensorRef{
		leftIndex.Tensors[name],
		rightIndex.Tensors[name],
	}, []float64{0.25, 0.75}, 2)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("writeLinearChunks() error = %v", err)
	}

	read := core.ReadFile(outPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	values, err := safetensors.DecodeFloatData("F32", read.Value.([]byte), 5)
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}
	assertFloat32Values(t, values, []float32{7.5, 9.5, 11.5, 13.5, 15.5})
}

func TestModelMerge_WriteSLERPMergedTensorChunksGood(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.embed_tokens.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{
		{Name: name, Shape: []int{2}, Data: []float32{1, 0}},
	})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{
		{Name: name, Shape: []int{2}, Data: []float32{0, 1}},
	})
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}
	outPath := core.PathJoin(t.TempDir(), "out.bin")
	created := core.Create(outPath)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)

	err = writeSLERPChunks(context.Background(), file, []safetensors.TensorRef{
		leftIndex.Tensors[name],
		rightIndex.Tensors[name],
	}, 0.5, 1)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("writeSLERPChunks() error = %v", err)
	}

	read := core.ReadFile(outPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	values, err := safetensors.DecodeFloatData("F32", read.Value.([]byte), 2)
	if err != nil {
		t.Fatalf("decode output: %v", err)
	}
	want := float32(math.Sqrt(0.5))
	assertFloat32Values(t, values, []float32{want, want})
}

func TestModelMerge_SafetensorChunkHelpersGood(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "source.safetensors")
	name := "model.embed_tokens.weight"
	writeTestSafetensorsF32(t, path, []safetensorTestTensor{
		{Name: name, Shape: []int{5}, Data: []float32{0, 2, 4, 6, 8}},
	})
	index, err := safetensors.IndexFiles([]string{path})
	if err != nil {
		t.Fatalf("index source: %v", err)
	}
	ref := index.Tensors[name]
	chunk, err := safetensors.ReadRefFloat32Chunk(ref, 1, 2)
	if err != nil {
		t.Fatalf("read chunk: %v", err)
	}
	assertFloat32Values(t, chunk, []float32{2, 4})

	outPath := core.PathJoin(t.TempDir(), "copy.bin")
	created := core.Create(outPath)
	if !created.OK {
		t.Fatalf("create output: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	err = safetensors.WriteRefFloat32Chunks(context.Background(), file, ref, 2)
	if closeErr := file.Close(); closeErr != nil {
		t.Fatalf("close output: %v", closeErr)
	}
	if err != nil {
		t.Fatalf("write copy chunks: %v", err)
	}
	read := core.ReadFile(outPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	values, err := safetensors.DecodeFloatData("F32", read.Value.([]byte), 5)
	if err != nil {
		t.Fatalf("decode copy: %v", err)
	}
	assertFloat32Values(t, values, []float32{0, 2, 4, 6, 8})
}

func TestModelMerge_ReadMergeTensorValuesGood(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.norm.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{1, 2}}})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{3, 4}}})
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}

	values, complete, err := readTensorValues([]safetensors.Index{leftIndex, rightIndex}, name)
	if err != nil {
		t.Fatalf("readTensorValues() error = %v", err)
	}
	if !complete || len(values) != 2 {
		t.Fatalf("values len/complete = %d/%v, want 2/true", len(values), complete)
	}
	assertFloat32Values(t, values[0], []float32{1, 2})
	assertFloat32Values(t, values[1], []float32{3, 4})
}

func TestModelMerge_ChunkHelperErrorsBad(t *testing.T) {
	if _, err := safetensors.DTypeByteSize("F16"); err != nil {
		t.Fatalf("F16 byte size: %v", err)
	}
	if _, err := safetensors.DTypeByteSize("BF16"); err != nil {
		t.Fatalf("BF16 byte size: %v", err)
	}
	if _, err := safetensors.DTypeByteSize("F64"); err != nil {
		t.Fatalf("F64 byte size: %v", err)
	}
	if _, err := safetensors.DTypeByteSize("I32"); err == nil {
		t.Fatal("expected unsupported dtype error")
	}
	if err := writeLinearChunks(context.Background(), nil, nil, nil, 2); err == nil {
		t.Fatal("expected no tensors error")
	}
	if err := writeLinearChunks(context.Background(), nil, []safetensors.TensorRef{{Elements: 1}}, nil, 2); err == nil {
		t.Fatal("expected weight/source mismatch error")
	}
	if _, err := safetensors.ReadRefFloat32Chunk(safetensors.TensorRef{DType: "F32", Elements: 1}, 1, 1); err == nil {
		t.Fatal("expected chunk bounds error")
	}
	if err := resultError(core.Ok("ok")); err != nil {
		t.Fatalf("resultError(ok) = %v", err)
	}
	if err := resultError(core.Result{Value: "bad", OK: false}); err == nil {
		t.Fatal("expected non-error core result failure")
	}
}

// TestModelMerge_SLERPChunkedWeights_Good covers the single-call
// slerpChunkedWeights wrapper. Orthogonal unit vectors at t = 0.5 produce
// equal interpolation weights of sin(theta/2)/sin(theta).
func TestModelMerge_SLERPChunkedWeightsGood(t *testing.T) {
	leftPath := core.PathJoin(t.TempDir(), "left.safetensors")
	rightPath := core.PathJoin(t.TempDir(), "right.safetensors")
	name := "model.embed_tokens.weight"
	writeTestSafetensorsF32(t, leftPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{1, 0}}})
	writeTestSafetensorsF32(t, rightPath, []safetensorTestTensor{{Name: name, Shape: []int{2}, Data: []float32{0, 1}}})
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		t.Fatalf("index left: %v", err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		t.Fatalf("index right: %v", err)
	}
	refs := []safetensors.TensorRef{leftIndex.Tensors[name], rightIndex.Tensors[name]}

	weights, err := slerpChunkedWeights(context.Background(), refs, 0.5, 4)
	if err != nil {
		t.Fatalf("slerpChunkedWeights() error = %v", err)
	}
	if len(weights) != 2 {
		t.Fatalf("weights len = %d, want 2", len(weights))
	}
	// Orthogonal vectors, theta = 90deg, t = 0.5 -> both weights sin(45)/sin(90).
	want := math.Sin(math.Pi/4) / math.Sin(math.Pi/2)
	if math.Abs(weights[0]-want) > 1e-9 || math.Abs(weights[1]-want) > 1e-9 {
		t.Fatalf("weights = %v, want both %f", weights, want)
	}

	// Mismatched element counts are rejected.
	if _, err := slerpChunkedWeights(context.Background(), []safetensors.TensorRef{{Elements: 1}, {Elements: 2}}, 0.5, 4); err == nil {
		t.Fatal("slerpChunkedWeights(length mismatch) error = nil")
	}
	// Exactly two refs are required.
	if _, err := slerpChunkedWeights(context.Background(), refs[:1], 0.5, 4); err == nil {
		t.Fatal("slerpChunkedWeights(one ref) error = nil")
	}
}
