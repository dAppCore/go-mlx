// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/safetensors"
)

func TestCPUSplitFFNExecutor_QwenDenseGood(t *testing.T) {
	source := writeCPUSplitFFNTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	got, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{1, 2, 3, 4},
	})

	if err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	if !equalSplitFloat32Slices(got.Hidden, []float32{1, 2, 3, 4}) {
		t.Fatalf("ForwardFFN hidden = %v, want residual passthrough", got.Hidden)
	}
}

func TestCPUSplitFFNExecutor_QwenDenseBiasGood(t *testing.T) {
	source := writeCPUSplitFFNBiasTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	got, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{10, 20},
	})

	if err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	want := []float32{10 + cpuSplitSiLU(1)*2 + 0.5, 19.5}
	if !approxSplitFloat32Slices(got.Hidden, want, 1e-5) {
		t.Fatalf("ForwardFFN hidden = %v, want %v", got.Hidden, want)
	}
}

func TestCPUSplitFFNExecutor_QwenLanguageModelAliasGood(t *testing.T) {
	source := writeCPUSplitFFNAliasTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	got, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{1, 2},
	})

	if err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	if !equalSplitFloat32Slices(got.Hidden, []float32{1, 2}) {
		t.Fatalf("ForwardFFN hidden = %v, want residual passthrough through aliases", got.Hidden)
	}
}

func TestCPUSplitFFNExecutor_QwenJANGPackedGood(t *testing.T) {
	source := writeCPUSplitFFNJANGPackedTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	got, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{1, 1},
	})

	if err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	norm := float32(1 / math.Sqrt(1+1e-6))
	activated := cpuSplitSiLU(norm) * (2 * norm)
	want := []float32{1 + activated, 1 + activated}
	if !approxSplitFloat32Slices(got.Hidden, want, 1e-5) {
		t.Fatalf("ForwardFFN hidden = %v, want %v", got.Hidden, want)
	}
}

func TestCPUSplitFFNExecutor_QwenPackedConfigQuantizationGood(t *testing.T) {
	source := writeCPUSplitFFNPackedConfigQuantizationTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	got, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{1, 1},
	})

	if err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	norm := float32(1 / math.Sqrt(1+1e-6))
	activated := cpuSplitSiLU(norm) * (2 * norm)
	want := []float32{1 + activated, 1 + activated}
	if !approxSplitFloat32Slices(got.Hidden, want, 1e-5) {
		t.Fatalf("ForwardFFN hidden = %v, want %v", got.Hidden, want)
	}
}

func TestCPUSplitFFNExecutor_QwenJANGPackedStaysPackedGood(t *testing.T) {
	source := writeCPUSplitFFNJANGPackedTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	layer, err := executor.layer(context.Background(), 0)

	if err != nil {
		t.Fatalf("layer: %v", err)
	}
	if len(layer.gate) != 0 || len(layer.up) != 0 || len(layer.down) != 0 {
		t.Fatalf("packed FFN expanded dense matrices: gate=%d up=%d down=%d", len(layer.gate), len(layer.up), len(layer.down))
	}
}

func TestCPUSplitFFNExecutor_QwenJANGPackedMemoryReportGood(t *testing.T) {
	source := writeCPUSplitFFNJANGPackedTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source)
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}
	if _, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{1, 1},
	}); err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}

	report := executor.MemoryReport()

	if report.LoadedLayers != 1 || report.PackedProjections != 3 || report.DenseProjections != 0 {
		t.Fatalf("MemoryReport placement = %+v, want one packed layer", report)
	}
	if report.PackedProjectionBytes != 3 || report.PackedSidecarBytes != 24 {
		t.Fatalf("MemoryReport packed bytes = %+v, want 3 packed + 24 sidecar bytes", report)
	}
	if report.ResidentBytes != 35 || report.DenseEquivalentBytes != 56 || report.SavedBytes != 21 {
		t.Fatalf("MemoryReport bytes = %+v, want resident=35 dense=56 saved=21", report)
	}
}

func TestCPUSplitFFNExecutor_QwenJANGPackedMemoryReportCacheDisabledGood(t *testing.T) {
	source := writeCPUSplitFFNJANGPackedTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source, WithCPUSplitFFNMaxCachedLayers(-1))
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	if _, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
		Layer:  0,
		Hidden: []float32{1, 1},
	}); err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	report := executor.MemoryReport()

	if !report.CacheDisabled || report.LoadedLayers != 0 || report.ResidentBytes != 0 {
		t.Fatalf("MemoryReport current cache = %+v, want disabled with no resident layers", report)
	}
	if report.LayerLoads != 1 || report.PeakResidentBytes != 35 {
		t.Fatalf("MemoryReport load counters = %+v, want one transient 35 byte layer", report)
	}
}

func TestCPUSplitFFNExecutor_QwenJANGPackedMemoryReportCacheEvictionGood(t *testing.T) {
	source := writeCPUSplitFFNTwoLayerJANGPackedTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source, WithCPUSplitFFNMaxCachedLayers(1))
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	for layer := range 2 {
		if _, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{
			Layer:  layer,
			Hidden: []float32{1, 1},
		}); err != nil {
			t.Fatalf("ForwardFFN(%d): %v", layer, err)
		}
	}
	report := executor.MemoryReport()

	if report.LoadedLayers != 1 || report.ResidentBytes != 35 || report.PeakResidentBytes != 35 {
		t.Fatalf("MemoryReport cache bytes = %+v, want one resident packed layer", report)
	}
	if report.LayerLoads != 2 || report.EvictedLayers != 1 {
		t.Fatalf("MemoryReport cache counters = %+v, want two loads and one eviction", report)
	}
}

func TestCPUSplitFFNExecutor_QwenJANGPackedMemoryEstimateGood(t *testing.T) {
	source := writeCPUSplitFFNTwoLayerJANGPackedTestPack(t)
	executor, err := LoadCPUSplitFFNExecutor(context.Background(), source, WithCPUSplitFFNMaxCachedLayers(1))
	if err != nil {
		t.Fatalf("LoadCPUSplitFFNExecutor: %v", err)
	}

	estimate, err := executor.EstimateMemoryReport(context.Background())

	if err != nil {
		t.Fatalf("EstimateMemoryReport: %v", err)
	}
	if !estimate.Estimated || estimate.TotalLayers != 2 || estimate.LoadedLayers != 1 {
		t.Fatalf("estimate shape = %+v, want estimated two-layer one-resident report", estimate)
	}
	if estimate.LayerLoads != 2 || estimate.EvictedLayers != 1 || estimate.PeakResidentBytes != 35 {
		t.Fatalf("estimate cache = %+v, want two loads, one eviction, 35 peak bytes", estimate)
	}
	if estimate.ResidentBytes != 35 || estimate.DenseEquivalentBytes != 56 || estimate.SavedBytes != 21 {
		t.Fatalf("estimate bytes = %+v, want resident=35 dense=56 saved=21", estimate)
	}
	if live := executor.MemoryReport(); live.LayerLoads != 0 || live.LoadedLayers != 0 || live.ResidentBytes != 0 {
		t.Fatalf("EstimateMemoryReport mutated live report = %+v", live)
	}
}

func TestEstimateCPUSplitFFNMemory_QwenJANGPackedGood(t *testing.T) {
	source := writeCPUSplitFFNTwoLayerJANGPackedTestPack(t)

	estimate, err := EstimateCPUSplitFFNMemory(context.Background(), source, WithCPUSplitFFNMaxCachedLayers(1))

	if err != nil {
		t.Fatalf("EstimateCPUSplitFFNMemory: %v", err)
	}
	if !estimate.Estimated || estimate.TotalLayers != 2 || estimate.LoadedLayers != 1 || estimate.LayerLoads != 2 || estimate.EvictedLayers != 1 {
		t.Fatalf("EstimateCPUSplitFFNMemory = %+v, want two-layer one-resident estimate", estimate)
	}
	if estimate.ResidentBytes != 35 || estimate.PeakResidentBytes != 35 || estimate.SavedBytes != 21 {
		t.Fatalf("EstimateCPUSplitFFNMemory bytes = %+v, want resident=35 peak=35 saved=21", estimate)
	}
}

func TestSplitExecutor_LoadSplitExecutor_GoodCPUFFNOptionMakesPlacementReady(t *testing.T) {
	source := writeCPUSplitFFNTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}

	executor, err := LoadSplitExecutor(context.Background(), slicePath, WithCPUSplitFFNExecutor())

	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}
	if !executor.Placement().Ready {
		t.Fatalf("placement = %+v, want ready with CPU FFN executor", executor.Placement())
	}
}

func writeCPUSplitFFNBiasTestPack(t *testing.T) string {
	t.Helper()
	return writeCPUSplitFFNPack(t, "", map[string]cpuSplitF32Tensor{
		"model.layers.0.post_attention_layernorm.weight": {
			Shape:  []int64{2},
			Values: []float32{0, 0},
		},
		"model.layers.0.mlp.gate_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{0, 0, 0, 0},
		},
		"model.layers.0.mlp.gate_proj.bias": {
			Shape:  []int64{2},
			Values: []float32{1, 0},
		},
		"model.layers.0.mlp.up_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{0, 0, 0, 0},
		},
		"model.layers.0.mlp.up_proj.bias": {
			Shape:  []int64{2},
			Values: []float32{2, 0},
		},
		"model.layers.0.mlp.down_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
		"model.layers.0.mlp.down_proj.bias": {
			Shape:  []int64{2},
			Values: []float32{0.5, -0.5},
		},
	})
}

func writeCPUSplitFFNAliasTestPack(t *testing.T) string {
	t.Helper()
	return writeCPUSplitFFNPack(t, "language_model.", map[string]cpuSplitF32Tensor{})
}

func writeCPUSplitFFNTestPack(t *testing.T) string {
	t.Helper()
	return writeCPUSplitFFNPack(t, "", map[string]cpuSplitF32Tensor{})
}

func writeCPUSplitFFNJANGPackedTestPack(t *testing.T) string {
	t.Helper()
	return writeCPUSplitFFNPackedTestPack(t, `"rms_norm_eps": 0.000001`, `{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"quantization": {"method": "affine+mxtq", "group_size": 4, "bits_default": 2}
	}`)
}

func writeCPUSplitFFNTwoLayerJANGPackedTestPack(t *testing.T) string {
	t.Helper()
	return writeCPUSplitFFNPackedLayerCountTestPack(t, 2, `"rms_norm_eps": 0.000001`, `{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"quantization": {"method": "affine+mxtq", "group_size": 4, "bits_default": 2}
	}`)
}

func writeCPUSplitFFNPackedConfigQuantizationTestPack(t *testing.T) string {
	t.Helper()
	return writeCPUSplitFFNPackedTestPack(t, `"rms_norm_eps": 0.000001,
		"quantization": {"method": "affine+mxtq", "group_size": 4, "bits_default": 2}`, "")
}

func writeCPUSplitFFNPackedTestPack(t *testing.T, configExtra string, jangConfig string) string {
	t.Helper()
	return writeCPUSplitFFNPackedLayerCountTestPack(t, 1, configExtra, jangConfig)
}

func writeCPUSplitFFNPackedLayerCountTestPack(t *testing.T, layers int, configExtra string, jangConfig string) string {
	t.Helper()
	dir := t.TempDir()
	config := `{
		"model_type": "qwen2",
		"vocab_size": 8,
		"hidden_size": 2,
		"intermediate_size": 2,
		"num_hidden_layers": ` + core.Sprintf("%d", layers) + `,
		"max_position_embeddings": 32`
	if core.Trim(configExtra) != "" {
		config += ",\n\t\t" + configExtra
	}
	config += "\n\t}"
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), config)
	if core.Trim(jangConfig) != "" {
		writeModelPackFile(t, core.PathJoin(dir, "jang_config.json"), jangConfig)
	}
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), `{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{"chat_template":"{{ messages }}"}`)
	tensors := map[string]cpuSplitRawTensor{}
	for layer := range layers {
		prefix := core.Sprintf("model.layers.%d", layer)
		tensors[prefix+".post_attention_layernorm.weight"] = cpuSplitRawF32Tensor([]int64{2}, []float32{1, 1})
		tensors[prefix+".mlp.gate_proj.weight"] = cpuSplitRawU8Tensor([]int64{1}, packCPUSplitJANGValues(t, []uint8{1, 0, 0, 1}, 2))
		tensors[prefix+".mlp.gate_proj.weight.scales"] = cpuSplitRawF32Tensor([]int64{1}, []float32{1})
		tensors[prefix+".mlp.gate_proj.weight.biases"] = cpuSplitRawF32Tensor([]int64{1}, []float32{0})
		tensors[prefix+".mlp.up_proj.weight"] = cpuSplitRawU8Tensor([]int64{1}, packCPUSplitJANGValues(t, []uint8{2, 0, 0, 2}, 2))
		tensors[prefix+".mlp.up_proj.weight.scales"] = cpuSplitRawF32Tensor([]int64{1}, []float32{1})
		tensors[prefix+".mlp.up_proj.weight.biases"] = cpuSplitRawF32Tensor([]int64{1}, []float32{0})
		tensors[prefix+".mlp.down_proj.weight"] = cpuSplitRawU8Tensor([]int64{1}, packCPUSplitJANGValues(t, []uint8{1, 0, 0, 1}, 2))
		tensors[prefix+".mlp.down_proj.weight.scales"] = cpuSplitRawF32Tensor([]int64{1}, []float32{1})
		tensors[prefix+".mlp.down_proj.weight.biases"] = cpuSplitRawF32Tensor([]int64{1}, []float32{0})
	}
	writeCPUSplitRawSafetensors(t, core.PathJoin(dir, "model.safetensors"), tensors)
	return dir
}

func writeCPUSplitFFNPack(t *testing.T, prefix string, overrides map[string]cpuSplitF32Tensor) string {
	t.Helper()
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen2",
		"vocab_size": 8,
		"hidden_size": 2,
		"intermediate_size": 2,
		"num_hidden_layers": 1,
		"max_position_embeddings": 32
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), `{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer_config.json"), `{"chat_template":"{{ messages }}"}`)
	tensors := map[string]cpuSplitF32Tensor{
		prefix + "model.embed_tokens.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
		prefix + "model.layers.0.input_layernorm.weight": {
			Shape:  []int64{2},
			Values: []float32{1, 1},
		},
		prefix + "model.layers.0.self_attn.q_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
		prefix + "model.layers.0.post_attention_layernorm.weight": {
			Shape:  []int64{2},
			Values: []float32{0, 0},
		},
		prefix + "model.layers.0.mlp.gate_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
		prefix + "model.layers.0.mlp.up_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
		prefix + "model.layers.0.mlp.down_proj.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
		prefix + "lm_head.weight": {
			Shape:  []int64{2, 2},
			Values: []float32{1, 0, 0, 1},
		},
	}
	for name, tensor := range overrides {
		tensors[prefix+name] = tensor
	}
	writeCPUSplitF32Safetensors(t, core.PathJoin(dir, "model.safetensors"), tensors)
	return dir
}

type cpuSplitF32Tensor struct {
	Shape  []int64
	Values []float32
}

type cpuSplitRawTensor struct {
	DType string
	Shape []int64
	Raw   []byte
}

func cpuSplitRawF32Tensor(shape []int64, values []float32) cpuSplitRawTensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return cpuSplitRawTensor{DType: "F32", Shape: append([]int64(nil), shape...), Raw: raw}
}

func cpuSplitRawU8Tensor(shape []int64, values []byte) cpuSplitRawTensor {
	return cpuSplitRawTensor{DType: "U8", Shape: append([]int64(nil), shape...), Raw: append([]byte(nil), values...)}
}

func writeCPUSplitRawSafetensors(t *testing.T, path string, tensors map[string]cpuSplitRawTensor) {
	t.Helper()
	header := map[string]safetensors.HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		tensor := tensors[name]
		header[name] = safetensors.HeaderEntry{
			DType:       tensor.DType,
			Shape:       append([]int64(nil), tensor.Shape...),
			DataOffsets: []int64{offset, offset + int64(len(tensor.Raw))},
		}
		payload = append(payload, tensor.Raw...)
		offset += int64(len(tensor.Raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}

func packCPUSplitJANGValues(t *testing.T, values []uint8, bits int) []byte {
	t.Helper()
	packed := make([]byte, (len(values)*bits+7)/8)
	maxValue := uint8((1 << bits) - 1)
	for i, value := range values {
		if value > maxValue {
			t.Fatalf("value %d exceeds %d-bit max", value, bits)
		}
		bitOffset := i * bits
		byteIndex := bitOffset / 8
		shift := bitOffset % 8
		packed[byteIndex] |= value << shift
		if shift+bits > 8 {
			packed[byteIndex+1] |= value >> (8 - shift)
		}
	}
	return packed
}

func writeCPUSplitF32Safetensors(t *testing.T, path string, tensors map[string]cpuSplitF32Tensor) {
	t.Helper()
	header := map[string]safetensors.HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		tensor := tensors[name]
		raw := make([]byte, len(tensor.Values)*4)
		for i, value := range tensor.Values {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
		}
		header[name] = safetensors.HeaderEntry{
			DType:       "F32",
			Shape:       append([]int64(nil), tensor.Shape...),
			DataOffsets: []int64{offset, offset + int64(len(raw))},
		}
		payload = append(payload, raw...)
		offset += int64(len(raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}

func approxSplitFloat32Slices(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		delta := a[i] - b[i]
		if delta < 0 {
			delta = -delta
		}
		if delta > tolerance {
			return false
		}
	}
	return true
}
