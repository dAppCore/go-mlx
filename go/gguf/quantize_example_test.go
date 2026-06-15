// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// ExampleValidationSummary renders a one-line summary of GGUF validation
// findings; tensor-scoped issues print as code:tensor.
func ExampleValidationSummary() {
	summary := ValidationSummary([]ValidationIssue{
		{Severity: GGUFValidationError, Code: "shape_mismatch", Tensor: "blk.0.attn_q.weight"},
		{Severity: GGUFValidationWarning, Code: "missing_alignment"},
	})
	core.Println(summary)
	core.Println(ValidationSummary(nil))
	// Output:
	// shape_mismatch:blk.0.attn_q.weight, missing_alignment
	// unknown validation failure
}

// ExampleQuantizeModelPack shows the end-to-end GGUF quantisation flow: take a
// dense-safetensors model pack, quantise it to a K-quant GGUF, then read the
// result back with ReadInfo to confirm the on-disk ggml type. No weights are
// materialised into MLX — quantise writes, ReadInfo parses the header only.
//
// The payload-level decode round-trips (that the bytes dequantise back within
// tolerance) live in quantize_kquant_test.go; here we demonstrate the public
// caller's view: format in, validated GGUF out.
func ExampleQuantizeModelPack() {
	source, cleanup := writeExampleDenseSafetensorsPack()
	defer cleanup()

	outDir, outCleanup := exampleTempDir()
	defer outCleanup()
	output := core.PathJoin(outDir, "quantized")

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: exampleSourcePack(source),
		OutputPath: output,
		Format:     QuantizeQ4_K_M,
	})
	if err != nil {
		core.Println(err.Error())
		return
	}

	// RequestedFormat keeps the caller's label (q4_k_m); Format is the encoder
	// it resolved to (q4_k). The generated GGUF passed metadata validation.
	core.Println(result.RequestedFormat, result.Format, result.Info.Valid())
	core.Println(result.TensorCount, result.QuantizedTensors)
	// Output:
	// q4_k_m q4_k true
	// 2 2
}

// ExampleReadInfo_quantizedTensorType demonstrates inspecting the ggml quant
// type a tensor was written as — the same read path the engine uses to decide
// how to dequantise on load. Here a synthetic pack is quantised to Q6_K and the
// 2-D weight's decoded type / bit width / block size are reported.
func ExampleReadInfo_quantizedTensorType() {
	source, cleanup := writeExampleDenseSafetensorsPack()
	defer cleanup()

	outDir, outCleanup := exampleTempDir()
	defer outCleanup()
	output := core.PathJoin(outDir, "quantized")

	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: exampleSourcePack(source),
		OutputPath: output,
		Format:     QuantizeQ6_K,
	}); err != nil {
		core.Println(err.Error())
		return
	}

	info, err := ReadInfo(output)
	if err != nil {
		core.Println(err.Error())
		return
	}
	weight := exampleFindTensor(info.Tensors, "model.layers.0.self_attn.q_proj.weight")
	if weight == nil {
		core.Println("weight tensor missing")
		return
	}
	core.Println(weight.TypeName, weight.Bits, weight.BlockSize)
	// Output: q6_k 6 256
}

// --- T-free Example helpers (mirror the *testing.T helpers in
// quantize_test.go so Example functions, which take no *testing.T, can build
// the same synthetic fixtures). ---

// writeExampleDenseSafetensorsPack writes a minimal dense-safetensors model
// pack (config.json + tokenizer.json + one F32 model.safetensors) whose tensors
// are a whole number of 256-element K-quant blocks, then returns its directory
// and a cleanup func. T-free sibling of writeDenseSafetensorsPack.
func writeExampleDenseSafetensorsPack() (string, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-gguf-quant-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	cleanup := func() { core.RemoveAll(dir) }

	failClean := func(v any) {
		cleanup()
		panic(v)
	}

	config := `{
		"model_type": "qwen3",
		"vocab_size": 151936,
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960
	}`
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(config), 0o644); !r.OK {
		failClean(r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(`{"version":"1.0"}`), 0o644); !r.OK {
		failClean(r.Value)
	}

	// One 2-D weight (256 cols x 2 rows = 2 blocks) plus a 1-D norm the
	// quantiser passes through — the same shape the round-trip tests use.
	const block = 256
	tensors := []exampleTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{block, 2}, Data: exampleAscendingFloat32s(block * 2)},
		{Name: "model.norm.weight", Shape: []int{block}, Data: exampleAscendingFloat32s(block)},
	}
	if err := writeExampleSafetensorsF32(core.PathJoin(dir, "model.safetensors"), tensors); err != nil {
		failClean(err)
	}
	return dir, cleanup
}

type exampleTensor struct {
	Name  string
	Shape []int
	Data  []float32
}

// writeExampleSafetensorsF32 emits an F32 safetensors file (the standard
// [uint64 header-len][JSON header][raw data] layout). T-free sibling of
// writeTestSafetensorsF32.
func writeExampleSafetensorsF32(path string, tensors []exampleTensor) error {
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		buf := make([]byte, len(tensor.Data)*4)
		for i, value := range tensor.Data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(value))
		}
		data = append(data, buf...)
		header[tensor.Name] = entry{DType: "F32", Shape: tensor.Shape, DataOffsets: []int{start, len(data)}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		return encoded.Value.(error)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if r := core.WriteFile(path, out, 0o644); !r.OK {
		return r.Value.(error)
	}
	return nil
}

// exampleAscendingFloat32s mirrors ascendingFloat32s: a deterministic small-
// magnitude ramp the quantisers handle cleanly.
func exampleAscendingFloat32s(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%17-8) / 4
	}
	return out
}

// exampleSourcePack builds the dense-safetensors ModelPack pointer the
// quantiser consumes. T-free sibling of sourcePackFromDir.
func exampleSourcePack(dir string) mp.ModelPack {
	return mp.ModelPack{
		Root:        dir,
		Path:        dir,
		Format:      mp.ModelPackFormatSafetensors,
		WeightFiles: []string{core.PathJoin(dir, "model.safetensors")},
	}
}

func exampleFindTensor(tensors []TensorInfo, name string) *TensorInfo {
	for i := range tensors {
		if tensors[i].Name == name {
			return &tensors[i]
		}
	}
	return nil
}

func exampleTempDir() (string, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-gguf-out-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	return dir, func() { core.RemoveAll(dir) }
}
