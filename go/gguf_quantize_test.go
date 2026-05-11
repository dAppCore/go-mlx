// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/gguf"
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

	info, err := gguf.ReadInfo(output)
	if err != nil {
		t.Fatalf("gguf.ReadInfo(output) error = %v", err)
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
	if !pack.Valid() || pack.Format != mp.ModelPackFormatGGUF || pack.QuantType != "q8_0" {
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
	info, err := gguf.ReadInfo(output)
	if err != nil {
		t.Fatalf("gguf.ReadInfo(output) error = %v", err)
	}
	if info.QuantType != "q4_0" || info.QuantBits != 4 || info.QuantGroup != 32 {
		t.Fatalf("quant info = %+v", info)
	}
}

func TestGGUFQuantize_WriteStreamedGGUF_Good(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.k_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := indexSafetensorFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	tensors, refs, err := buildStreamingGGUFQuantizedTensors(index, GGUFQuantizeQ8_0)
	if err != nil {
		t.Fatalf("build streaming tensors: %v", err)
	}
	if len(tensors) != 1 || len(refs) != 1 {
		t.Fatalf("stream tensor counts = %d/%d, want 1/1", len(tensors), len(refs))
	}

	output := core.PathJoin(t.TempDir(), "streamed.gguf")
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, GGUFQuantizeQ8_0, nil)
	if err := writeQuantizedGGUFStream(context.Background(), output, metadata, tensors, refs, GGUFQuantizeQ8_0, 32); err != nil {
		t.Fatalf("writeQuantizedGGUFStream() error = %v", err)
	}

	info, err := gguf.ReadInfo(output)
	if err != nil {
		t.Fatalf("gguf.ReadInfo() error = %v", err)
	}
	if !info.Valid() || info.TensorCount != 1 || info.Tensors[0].TypeName != "q8_0" {
		t.Fatalf("streamed info = %+v", info)
	}
}

func TestGGUFQuantize_WriteBufferedGGUF_Good(t *testing.T) {
	output := core.PathJoin(t.TempDir(), "buffered.gguf")
	values := ascendingFloat32s(32)
	data := quantizeQ8_0(values)
	tensors := []ggufQuantizedTensor{{
		Name:  "model.norm.weight",
		Type:  gguf.TensorTypeQ8_0,
		Shape: []uint64{32},
		Data:  data,
	}}
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, GGUFQuantizeQ8_0, nil)
	if err := writeQuantizedGGUF(output, metadata, tensors); err != nil {
		t.Fatalf("writeQuantizedGGUF() error = %v", err)
	}
	info, err := gguf.ReadInfo(output)
	if err != nil {
		t.Fatalf("gguf.ReadInfo() error = %v", err)
	}
	if !info.Valid() || info.TensorCount != 1 || info.Tensors[0].TypeName != "q8_0" {
		t.Fatalf("buffered info = %+v", info)
	}
	if got := ggufQuantizedTensorDataSize(ggufQuantizedTensor{Size: 12, Data: data}); got != 12 {
		t.Fatalf("ggufQuantizedTensorDataSize(Size) = %d, want 12", got)
	}
}

func TestGGUFQuantize_StreamErrorPaths_Bad(t *testing.T) {
	if _, _, err := buildStreamingGGUFQuantizedTensors(safetensorIndex{
		Names: []string{"bad.weight"},
		Tensors: map[string]safetensorTensorRef{
			"bad.weight": {Name: "bad.weight", DType: "I32", Shape: []uint64{32}, Elements: 32},
		},
	}, GGUFQuantizeQ8_0); err == nil {
		t.Fatal("expected unsupported dtype error")
	}
	if _, _, err := buildStreamingGGUFQuantizedTensors(safetensorIndex{
		Names: []string{"bad.weight"},
		Tensors: map[string]safetensorTensorRef{
			"bad.weight": {Name: "bad.weight", DType: "F32", Shape: []uint64{32}, Elements: 31},
		},
	}, GGUFQuantizeQ8_0); err == nil {
		t.Fatal("expected block alignment error")
	}
	if err := writeQuantizedGGUFStream(context.Background(), core.PathJoin(t.TempDir(), "bad.gguf"), nil, []ggufQuantizedTensor{{}}, nil, GGUFQuantizeQ8_0, 32); err == nil {
		t.Fatal("expected tensor/ref alignment error")
	}
	if _, err := quantizeGGUFValues("q5_0", ascendingFloat32s(32)); err == nil {
		t.Fatal("expected unsupported stream quantization format")
	}
}

func TestQuantizeModelPackToGGUF_RejectsNonSafetensors_Bad(t *testing.T) {
	source := t.TempDir()
	writeModelPackFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(source, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestGGUF(t, core.PathJoin(source, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: gguf.TensorTypeQ8_0, Dims: []uint64{32, 2}}},
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

func TestResolveGGUFQuantizeFormat_Bad(t *testing.T) {
	cases := []struct {
		input     GGUFQuantizeFormat
		requested GGUFQuantizeFormat
		used      GGUFQuantizeFormat
		notes     int
	}{
		{input: "", requested: GGUFQuantizeQ8_0, used: GGUFQuantizeQ8_0},
		{input: "Q4-K-M", requested: GGUFQuantizeQ4_K_M, used: GGUFQuantizeQ4_0, notes: 1},
		{input: " q4_0 ", requested: GGUFQuantizeQ4_0, used: GGUFQuantizeQ4_0},
	}
	for _, tc := range cases {
		requested, used, notes, err := resolveGGUFQuantizeFormat(tc.input)
		if err != nil {
			t.Fatalf("resolveGGUFQuantizeFormat(%q): %v", tc.input, err)
		}
		if requested != tc.requested || used != tc.used || len(notes) != tc.notes {
			t.Fatalf("resolveGGUFQuantizeFormat(%q) = requested:%q used:%q notes:%d", tc.input, requested, used, len(notes))
		}
	}
	if _, _, _, err := resolveGGUFQuantizeFormat("q2_k"); err == nil {
		t.Fatal("expected unsupported quant format error")
	}
}

func TestSafetensorDecodeFloatData_Good(t *testing.T) {
	f32 := make([]byte, 8)
	binary.LittleEndian.PutUint32(f32[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(f32[4:8], math.Float32bits(-2.25))
	got, err := decodeSafetensorFloatData("F32", f32, 2)
	if err != nil {
		t.Fatalf("decode F32: %v", err)
	}
	if got[0] != 1.5 || got[1] != -2.25 {
		t.Fatalf("F32 values = %+v", got)
	}

	f16 := make([]byte, 4)
	binary.LittleEndian.PutUint16(f16[0:2], float32ToFloat16(1.5))
	binary.LittleEndian.PutUint16(f16[2:4], float32ToFloat16(-2))
	got, err = decodeSafetensorFloatData("F16", f16, 2)
	if err != nil {
		t.Fatalf("decode F16: %v", err)
	}
	if got[0] != 1.5 || got[1] != -2 {
		t.Fatalf("F16 values = %+v", got)
	}

	bf16 := make([]byte, 4)
	binary.LittleEndian.PutUint16(bf16[0:2], uint16(math.Float32bits(3.5)>>16))
	binary.LittleEndian.PutUint16(bf16[2:4], uint16(math.Float32bits(-4)>>16))
	got, err = decodeSafetensorFloatData("BF16", bf16, 2)
	if err != nil {
		t.Fatalf("decode BF16: %v", err)
	}
	if got[0] != 3.5 || got[1] != -4 {
		t.Fatalf("BF16 values = %+v", got)
	}

	f64 := make([]byte, 16)
	binary.LittleEndian.PutUint64(f64[0:8], math.Float64bits(6.25))
	binary.LittleEndian.PutUint64(f64[8:16], math.Float64bits(-7.5))
	got, err = decodeSafetensorFloatData("F64", f64, 2)
	if err != nil {
		t.Fatalf("decode F64: %v", err)
	}
	if got[0] != 6.25 || got[1] != -7.5 {
		t.Fatalf("F64 values = %+v", got)
	}
}

func TestSafetensorDecodeFloatData_Bad(t *testing.T) {
	cases := []struct {
		dtype string
		raw   []byte
	}{
		{dtype: "F32", raw: []byte{1}},
		{dtype: "F16", raw: []byte{1}},
		{dtype: "BF16", raw: []byte{1}},
		{dtype: "F64", raw: []byte{1}},
		{dtype: "I32", raw: []byte{1, 2, 3, 4}},
	}
	for _, tc := range cases {
		if _, err := decodeSafetensorFloatData(tc.dtype, tc.raw, 1); err == nil {
			t.Fatalf("decodeSafetensorFloatData(%s) expected error", tc.dtype)
		}
	}
}

func TestReadDenseSafetensors_Malformed_Ugly(t *testing.T) {
	dir := t.TempDir()
	small := core.PathJoin(dir, "small.safetensors")
	if result := core.WriteFile(small, []byte{1, 2, 3}, 0o644); !result.OK {
		t.Fatalf("write small: %v", result.Value)
	}
	if _, err := readDenseSafetensors(small); err == nil {
		t.Fatal("expected small safetensors error")
	}

	badHeaderLen := core.PathJoin(dir, "bad-header-len.safetensors")
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data[:8], 99)
	if result := core.WriteFile(badHeaderLen, data, 0o644); !result.OK {
		t.Fatalf("write bad header length: %v", result.Value)
	}
	if _, err := readDenseSafetensors(badHeaderLen); err == nil {
		t.Fatal("expected bad header length error")
	}

	badJSON := core.PathJoin(dir, "bad-json.safetensors")
	data = make([]byte, 8+1)
	binary.LittleEndian.PutUint64(data[:8], 1)
	data[8] = '{'
	if result := core.WriteFile(badJSON, data, 0o644); !result.OK {
		t.Fatalf("write bad json: %v", result.Value)
	}
	if _, err := readDenseSafetensors(badJSON); err == nil {
		t.Fatal("expected bad JSON error")
	}
}

func TestDecodeDenseSafetensor_InvalidEntries_Bad(t *testing.T) {
	payload := make([]byte, 16)
	cases := []safetensorHeaderEntry{
		{DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0}},
		{DType: "F32", Shape: []int64{1}, DataOffsets: []int64{2, 1}},
		{DType: "F32", Shape: []int64{0}, DataOffsets: []int64{0, 4}},
		{DType: "I32", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
	}
	for index, entry := range cases {
		if _, err := decodeDenseSafetensor("model.safetensors", core.Sprintf("bad_%d", index), entry, payload); err == nil {
			t.Fatalf("decodeDenseSafetensor(%d) expected error", index)
		}
	}
}

func TestLoadDenseSafetensors_DuplicateTensor_Bad(t *testing.T) {
	dir := t.TempDir()
	first := core.PathJoin(dir, "a.safetensors")
	second := core.PathJoin(dir, "b.safetensors")
	tensors := []safetensorTestTensor{{Name: "dup.weight", Shape: []int{32}, Data: ascendingFloat32s(32)}}
	writeTestSafetensorsF32(t, first, tensors)
	writeTestSafetensorsF32(t, second, tensors)

	_, err := loadDenseSafetensors([]string{first, second})
	if err == nil || !core.Contains(err.Error(), "duplicate tensor") {
		t.Fatalf("loadDenseSafetensors duplicate error = %v", err)
	}
	if _, err := loadDenseSafetensors(nil); err == nil {
		t.Fatal("expected no files error")
	}
}

func TestQuantizeGGUFTensor_Helpers_Good(t *testing.T) {
	values := ascendingFloat32s(32)
	q8, err := quantizeGGUFTensor(denseSafetensor{Name: "q8.weight", Shape: []uint64{32}, Data: values}, GGUFQuantizeQ8_0)
	if err != nil {
		t.Fatalf("quantize q8: %v", err)
	}
	if q8.Type != gguf.TensorTypeQ8_0 || len(q8.Data) != 34 {
		t.Fatalf("q8 tensor = %+v len=%d", q8, len(q8.Data))
	}
	q4, err := quantizeGGUFTensor(denseSafetensor{Name: "q4.weight", Shape: []uint64{32}, Data: values}, GGUFQuantizeQ4_0)
	if err != nil {
		t.Fatalf("quantize q4: %v", err)
	}
	if q4.Type != gguf.TensorTypeQ4_0 || len(q4.Data) != 18 {
		t.Fatalf("q4 tensor = %+v len=%d", q4, len(q4.Data))
	}

	if got := maxAbsFloat32([]float32{-1, 0.5, 2}); got != 2 {
		t.Fatalf("maxAbsFloat32() = %f, want 2", got)
	}
	if got := alignPadding(33, 32); got != 31 {
		t.Fatalf("alignPadding(33,32) = %d, want 31", got)
	}
	if got := alignPadding(33, 0); got != 0 {
		t.Fatalf("alignPadding(33,0) = %d, want 0", got)
	}
	if got := clampInt(-1, 0, 4); got != 0 {
		t.Fatalf("clampInt low = %d, want 0", got)
	}
	if got := clampInt(9, 0, 4); got != 4 {
		t.Fatalf("clampInt high = %d, want 4", got)
	}
	if got := appendUint16LE(nil, 0x1234); len(got) != 2 || got[0] != 0x34 || got[1] != 0x12 {
		t.Fatalf("appendUint16LE = %v", got)
	}
}

func TestQuantizeGGUFTensor_ErrorPaths_Bad(t *testing.T) {
	if _, err := quantizeGGUFTensor(denseSafetensor{Name: "bad", Shape: []uint64{32}, Data: ascendingFloat32s(32)}, "q5_0"); err == nil {
		t.Fatal("expected unsupported resolved format error")
	}
	if _, err := quantizeGGUFTensor(denseSafetensor{Name: "bad", Shape: []uint64{32}, Data: ascendingFloat32s(31)}, GGUFQuantizeQ8_0); err == nil {
		t.Fatal("expected data block size error")
	}
	if _, err := quantizeGGUFTensor(denseSafetensor{Name: "bad", Shape: []uint64{31}, Data: ascendingFloat32s(32)}, GGUFQuantizeQ8_0); err == nil {
		t.Fatal("expected shape block size error")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := quantizeGGUFTensors(cancelled, []denseSafetensor{{Name: "x", Shape: []uint64{32}, Data: ascendingFloat32s(32)}}, GGUFQuantizeQ8_0); err != context.Canceled {
		t.Fatalf("quantizeGGUFTensors(cancelled) = %v, want context.Canceled", err)
	}
}

func TestGGUFQuantizeMetadata_LabelsAndDenseFloats_Ugly(t *testing.T) {
	source := mp.ModelPack{Architecture: "qwen3", VocabSize: 10, HiddenSize: 20, NumLayers: 2, ContextLength: 128}
	metadata := ggufQuantizeMetadata(source, GGUFQuantizeQ4_0, map[string]string{"z": "last", "a": "first"})
	if len(metadata) != 11 {
		t.Fatalf("metadata entries = %d, want 11", len(metadata))
	}
	if metadata[len(metadata)-2].Key != "go_mlx.label.a" || metadata[len(metadata)-1].Key != "go_mlx.label.z" {
		t.Fatalf("labels were not sorted: %+v", metadata[len(metadata)-2:])
	}

	floatCases := []float32{0, 1, -2, float32(math.Inf(1)), float32(math.NaN())}
	for _, value := range floatCases {
		half := float32ToFloat16(value)
		roundTrip := float16ToFloat32(half)
		if math.IsNaN(float64(value)) {
			if !math.IsNaN(float64(roundTrip)) {
				t.Fatalf("NaN roundtrip = %v", roundTrip)
			}
			continue
		}
		if math.IsInf(float64(value), 0) {
			if !math.IsInf(float64(roundTrip), 0) {
				t.Fatalf("Inf roundtrip = %v", roundTrip)
			}
			continue
		}
		if value != 0 && roundTrip == 0 {
			t.Fatalf("float16 roundtrip of %v underflowed unexpectedly", value)
		}
	}
}

func TestQuantizeModelPackToGGUF_ValidationErrors_Bad(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := QuantizeModelPackToGGUF(cancelled, QuantizeGGUFOptions{}); err != context.Canceled {
		t.Fatalf("QuantizeModelPackToGGUF(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{}); err == nil {
		t.Fatal("expected source path validation error")
	}
	if _, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{ModelPath: t.TempDir()}); err == nil {
		t.Fatal("expected output path validation error")
	}
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	if _, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{ModelPath: source, OutputPath: core.PathJoin(t.TempDir(), "model.gguf")}); err == nil {
		t.Fatal("expected output directory validation error")
	}
	if _, err := QuantizeModelPackToGGUF(context.Background(), QuantizeGGUFOptions{ModelPath: source, OutputPath: source}); err == nil {
		t.Fatal("expected same path validation error")
	}
	occupied := core.PathJoin(t.TempDir(), "occupied")
	if result := core.MkdirAll(occupied, 0o755); !result.OK {
		t.Fatalf("mkdir occupied: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(occupied, "existing.gguf"), []byte("x"), 0o644); !result.OK {
		t.Fatalf("write occupied: %v", result.Value)
	}
	if err := ensureEmptyGGUFQuantizeDestination(occupied); err == nil {
		t.Fatal("expected occupied destination error")
	}
	if err := ensureEmptyGGUFQuantizeDestination(core.PathJoin(t.TempDir(), "missing")); err != nil {
		t.Fatalf("missing destination should be allowed: %v", err)
	}
	if err := quantizeGGUFResultError(core.Ok("ok")); err != nil {
		t.Fatalf("quantizeGGUFResultError(ok) = %v", err)
	}
	if err := quantizeGGUFResultError(core.Result{Value: "bad", OK: false}); err == nil || !core.Contains(err.Error(), "core result failed") {
		t.Fatalf("quantizeGGUFResultError(non-error) = %v", err)
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
