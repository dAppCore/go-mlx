// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestQuantizeModelPackToGGUF_Q8RoundTrip_Good(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
		{Name: "model.norm.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	output := core.PathJoin(t.TempDir(), "out-q8")

	result, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{
		ModelPath:  source,
		OutputPath: output,
		Format:     GGUFQuantizeQ8_0,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPackToGGUF() error = %v", err)
	}
	if result.RequestedFormat != GGUFQuantizeQ8_0 || result.Format != GGUFQuantizeQ8_0 {
		t.Fatalf("formats = requested:%q used:%q", result.RequestedFormat, result.Format)
	}
	if result.TensorCount != 2 || result.QuantizedTensors != 2 {
		t.Fatalf("tensor counts = %+v", result)
	}
	if result.WeightPath != core.PathJoin(output, "model.gguf") {
		t.Fatalf("WeightPath = %q", result.WeightPath)
	}

	info, err := ReadGGUFInfo(output)
	if err != nil {
		t.Fatalf("ReadGGUFInfo(output) error = %v", err)
	}
	if !info.Valid() {
		t.Fatalf("GGUF validation issues = %+v", info.ValidationIssues)
	}
	if info.Architecture != "qwen3" || info.HiddenSize != 2048 || info.NumLayers != 28 || info.ContextLength != 40960 {
		t.Fatalf("metadata = %+v", info)
	}
	if info.QuantType != "q8_0" || info.QuantBits != 8 || info.TensorCount != 2 {
		t.Fatalf("quant info = %+v", info)
	}
	if info.Tensors[0].TypeName != "q8_0" || info.Tensors[0].BlockSize != 32 {
		t.Fatalf("first tensor = %+v", info.Tensors[0])
	}

	pack, err := InspectModelPack(output)
	if err != nil {
		t.Fatalf("InspectModelPack(output) error = %v", err)
	}
	if !pack.Valid() || pack.Format != ModelPackFormatGGUF || pack.QuantType != "q8_0" {
		t.Fatalf("pack = %+v", pack)
	}
	if stat := core.Stat(core.PathJoin(output, "tokenizer.json")); !stat.OK {
		t.Fatalf("tokenizer.json was not preserved: %v", stat.Value)
	}
}

func TestQuantizeModelPackToGGUF_Q4KMFallsBackToQ4_0_Good(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "gemma3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	output := core.PathJoin(t.TempDir(), "out-q4")

	result, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{
		ModelPath:  source,
		OutputPath: output,
		Format:     GGUFQuantizeQ4_K_M,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPackToGGUF() error = %v", err)
	}
	if result.RequestedFormat != GGUFQuantizeQ4_K_M || result.Format != GGUFQuantizeQ4_0 {
		t.Fatalf("formats = requested:%q used:%q", result.RequestedFormat, result.Format)
	}
	if len(result.Notes) == 0 {
		t.Fatal("expected note explaining q4_k_m fallback")
	}
	info, err := ReadGGUFInfo(output)
	if err != nil {
		t.Fatalf("ReadGGUFInfo(output) error = %v", err)
	}
	if info.QuantType != "q4_0" || info.QuantBits != 4 || info.QuantGroup != 32 {
		t.Fatalf("quant info = %+v", info)
	}
}

func TestQuantizeModelPackToGGUF_RejectsNonSafetensors_Bad(t *testing.T) {
	source := t.TempDir()
	writeModelPackFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(source, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestGGUF(t, core.PathJoin(source, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ8_0, Dims: []uint64{32, 2}}},
	)

	_, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{
		ModelPath:  source,
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     GGUFQuantizeQ8_0,
	})
	if err == nil {
		t.Fatal("expected non-safetensors source error")
	}
	if !core.Contains(err.Error(), "safetensors") {
		t.Fatalf("error = %v, want safetensors context", err)
	}
}

func TestQuantizeModelPackToGGUF_InvalidShape_Ugly(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{31, 1}, Data: ascendingFloat32s(31)},
	})

	_, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{
		ModelPath:  source,
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     GGUFQuantizeQ8_0,
	})
	if err == nil {
		t.Fatal("expected block-alignment error")
	}
	if !core.Contains(err.Error(), "block") {
		t.Fatalf("error = %v, want block alignment context", err)
	}
}

type safetensorTestTensor struct {
	Name  string
	Shape []int
	Data  []float32
}

func writeDenseSafetensorsPack(t *testing.T, modelType string, tensors []safetensorTestTensor) string {
	t.Helper()
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), core.Sprintf(`{
		"model_type": %q,
		"vocab_size": 151936,
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960
	}`, modelType))
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestSafetensorsF32(t, core.PathJoin(dir, "model.safetensors"), tensors)
	return dir
}

func writeTestSafetensorsF32(t *testing.T, path string, tensors []safetensorTestTensor) {
	t.Helper()
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
		header[tensor.Name] = entry{
			DType:       "F32",
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

func ascendingFloat32s(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%17-8) / 4
	}
	return out
}
