// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

func TestQuantizeModelPackToGGUF_Q8RoundTrip_Good(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
		{Name: "model.norm.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	output := core.PathJoin(t.TempDir(), "out-q8")

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ8_0,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack() error = %v", err)
	}
	if result.RequestedFormat != QuantizeQ8_0 || result.Format != QuantizeQ8_0 {
		t.Fatalf("formats = requested:%q used:%q", result.RequestedFormat, result.Format)
	}
	if result.TensorCount != 2 || result.QuantizedTensors != 2 {
		t.Fatalf("tensor counts = %+v", result)
	}
	if result.WeightPath != core.PathJoin(output, "model.gguf") {
		t.Fatalf("WeightPath = %q", result.WeightPath)
	}

	info, err := ReadInfo(output)
	if err != nil {
		t.Fatalf("ReadInfo(output) error = %v", err)
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

	if stat := core.Stat(core.PathJoin(output, "tokenizer.json")); !stat.OK {
		t.Fatalf("tokenizer.json was not preserved: %v", stat.Value)
	}
	if stat := core.Stat(core.PathJoin(output, "model.gguf")); !stat.OK {
		t.Fatalf("model.gguf was not produced: %v", stat.Value)
	}
}

func TestQuantizeModelPackToGGUF_Q4KMNative_Good(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "gemma3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{256, 2}, Data: ascendingFloat32s(512)},
	})
	output := core.PathJoin(t.TempDir(), "out-q4k")

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ4_K_M,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack() error = %v", err)
	}
	if result.RequestedFormat != QuantizeQ4_K_M || result.Format != QuantizeQ4_K {
		t.Fatalf("formats = requested:%q used:%q", result.RequestedFormat, result.Format)
	}
	if len(result.Notes) != 0 {
		t.Fatalf("notes = %v, want none for native q4_k", result.Notes)
	}
	info, err := ReadInfo(output)
	if err != nil {
		t.Fatalf("ReadInfo(output) error = %v", err)
	}
	if info.QuantType != "q4_k_m" || info.QuantBits != 4 || info.QuantGroup != 256 {
		t.Fatalf("quant info = %+v", info)
	}
}

func TestQuantizeModelPackToGGUF_RejectsNonSafetensors_Bad(t *testing.T) {
	source := t.TempDir()
	writeModelPackFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(source, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestGGUF(t, core.PathJoin(source, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: TensorTypeQ8_0, Dims: []uint64{32, 2}}},
	)

	_, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     QuantizeQ8_0,
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

	_, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     QuantizeQ8_0,
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
		input     QuantizeFormat
		requested QuantizeFormat
		used      QuantizeFormat
		notes     int
	}{
		{input: "", requested: QuantizeQ8_0, used: QuantizeQ8_0},
		{input: "Q4-K-M", requested: QuantizeQ4_K_M, used: QuantizeQ4_K},
		{input: " q4_0 ", requested: QuantizeQ4_0, used: QuantizeQ4_0},
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
	if _, _, _, err := resolveGGUFQuantizeFormat("iq4_nl"); err == nil {
		t.Fatal("expected unsupported quant format error")
	}
}

func TestSafetensorDecodeFloatData_Good(t *testing.T) {
	f32 := make([]byte, 8)
	binary.LittleEndian.PutUint32(f32[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(f32[4:8], math.Float32bits(-2.25))
	got, err := safetensors.DecodeFloatData("F32", f32, 2)
	if err != nil {
		t.Fatalf("decode F32: %v", err)
	}
	if got[0] != 1.5 || got[1] != -2.25 {
		t.Fatalf("F32 values = %+v", got)
	}

	f16 := make([]byte, 4)
	binary.LittleEndian.PutUint16(f16[0:2], float32ToFloat16(1.5))
	binary.LittleEndian.PutUint16(f16[2:4], float32ToFloat16(-2))
	got, err = safetensors.DecodeFloatData("F16", f16, 2)
	if err != nil {
		t.Fatalf("decode F16: %v", err)
	}
	if got[0] != 1.5 || got[1] != -2 {
		t.Fatalf("F16 values = %+v", got)
	}

	bf16 := make([]byte, 4)
	binary.LittleEndian.PutUint16(bf16[0:2], uint16(math.Float32bits(3.5)>>16))
	binary.LittleEndian.PutUint16(bf16[2:4], uint16(math.Float32bits(-4)>>16))
	got, err = safetensors.DecodeFloatData("BF16", bf16, 2)
	if err != nil {
		t.Fatalf("decode BF16: %v", err)
	}
	if got[0] != 3.5 || got[1] != -4 {
		t.Fatalf("BF16 values = %+v", got)
	}

	f64 := make([]byte, 16)
	binary.LittleEndian.PutUint64(f64[0:8], math.Float64bits(6.25))
	binary.LittleEndian.PutUint64(f64[8:16], math.Float64bits(-7.5))
	got, err = safetensors.DecodeFloatData("F64", f64, 2)
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
		if _, err := safetensors.DecodeFloatData(tc.dtype, tc.raw, 1); err == nil {
			t.Fatalf("safetensors.DecodeFloatData(%s) expected error", tc.dtype)
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

// TestReadDenseSafetensors_OversizeWindow_Ugly covers a STRUCTURALLY
// VALID header (ReadIndex parses it without error) whose declared
// data_offsets describe a tensor payload that runs past the actual file
// bytes. The whole-file read path bounded this via `end > len(data)`;
// the streaming path must reject it (one fstat) before sizing the reused
// scratch — otherwise a corrupt-huge ByteLen would make a giant
// allocation. This is the malformed case the bad-header-len / bad-JSON
// fixtures cannot reach, because both of those fail inside ReadIndex.
func TestReadDenseSafetensors_OversizeWindow_Ugly(t *testing.T) {
	// Both cases use a structurally valid header (ReadIndex parses it) so
	// the per-tensor bound check is the only line that can reject them.
	// Without it, make([]byte, maxByteLen) attempts the declared size and
	// panics, where the old whole-file path returned a clean error.
	cases := []struct {
		name        string
		dataOffsets string
	}{
		// Magnitude arm: ~1 TiB window, far past EOF but in int64 range —
		// caught by `end > fileSize`.
		{"magnitude", "[0,1099511627776]"},
		// Overflow arm: DataStart(0)+ByteLen(MaxInt64) is in range here but
		// a non-zero DataStart would wrap negative; MaxInt64 itself still
		// exceeds fileSize. The companion bypass test proves `end < DataStart`
		// is the arm that matters when the sum overflows.
		{"overflow_maxint", "[0,9223372036854775807]"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := core.PathJoin(dir, "oversize.safetensors")
			header := `{"big.weight":{"dtype":"F32","shape":[2],"data_offsets":` + tc.dataOffsets + `}}`
			headerBytes := []byte(header)
			out := make([]byte, 8+len(headerBytes)+4)
			binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
			copy(out[8:], headerBytes)
			if result := core.WriteFile(path, out, 0o644); !result.OK {
				t.Fatalf("write oversize safetensors: %v", result.Value)
			}
			_, err := readDenseSafetensors(path)
			if err == nil {
				t.Fatal("expected oversize-window error, got nil")
			}
			if !core.Contains(err.Error(), "offsets exceed payload") {
				t.Fatalf("oversize-window error = %v, want 'offsets exceed payload'", err)
			}
		})
	}
}

func TestDecodeDenseSafetensor_InvalidEntries_Bad(t *testing.T) {
	payload := make([]byte, 16)
	cases := []safetensors.HeaderEntry{
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
	q8, err := quantizeGGUFTensor(denseSafetensor{Name: "q8.weight", Shape: []uint64{32}, Data: values}, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("quantize q8: %v", err)
	}
	if q8.Type != TensorTypeQ8_0 || len(q8.Data) != 34 {
		t.Fatalf("q8 tensor = %+v len=%d", q8, len(q8.Data))
	}
	q4, err := quantizeGGUFTensor(denseSafetensor{Name: "q4.weight", Shape: []uint64{32}, Data: values}, QuantizeQ4_0)
	if err != nil {
		t.Fatalf("quantize q4: %v", err)
	}
	if q4.Type != TensorTypeQ4_0 || len(q4.Data) != 18 {
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
	if _, err := quantizeGGUFTensor(denseSafetensor{Name: "bad", Shape: []uint64{32}, Data: ascendingFloat32s(32)}, "q3_0"); err == nil {
		t.Fatal("expected unsupported resolved format error")
	}
	if _, err := quantizeGGUFTensor(denseSafetensor{Name: "bad", Shape: []uint64{32}, Data: ascendingFloat32s(31)}, QuantizeQ8_0); err == nil {
		t.Fatal("expected data block size error")
	}
	if _, err := quantizeGGUFTensor(denseSafetensor{Name: "bad", Shape: []uint64{31}, Data: ascendingFloat32s(32)}, QuantizeQ8_0); err == nil {
		t.Fatal("expected shape block size error")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := quantizeGGUFTensors(cancelled, []denseSafetensor{{Name: "x", Shape: []uint64{32}, Data: ascendingFloat32s(32)}}, QuantizeQ8_0); err != context.Canceled {
		t.Fatalf("quantizeGGUFTensors(cancelled) = %v, want context.Canceled", err)
	}
}

func TestQuantizeModelPackToGGUF_ValidationErrors_Bad(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := QuantizeModelPack(cancelled, QuantizeOptions{}); err != context.Canceled {
		t.Fatalf("QuantizeModelPack(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{}); err == nil {
		t.Fatal("expected source path validation error")
	}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{}); err == nil {
		t.Fatal("expected output path validation error")
	}
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{SourcePack: sourcePackFromDir(source), OutputPath: core.PathJoin(t.TempDir(), "model.gguf")}); err == nil {
		t.Fatal("expected output directory validation error")
	}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{SourcePack: sourcePackFromDir(source), OutputPath: source}); err == nil {
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

func TestQuantize_QuantizeModelPack_FormatRoundTrip_Good(t *testing.T) {
	// Round-trip the GGUF quant formats whose encoder block layout matches
	// the canonical ggml type-size table (lib/gguflib/gguflib.c) through the
	// public QuantizeModelPack -> ReadInfo path, exercising the previously-
	// uncovered quantizeQ5_0 and quantizeQ5_K (+ quantizeKBlock / packKScales)
	// encoders and asserting the produced tensor's ggml type, bit width and
	// block size survive the write -> parse trip.
	//
	// Q2_K / Q3_K / Q6_K / Q8_K have their own end-to-end round-trip in
	// TestQuantizeModelPack_AllKQuants_Good (quantize_kquant_test.go),
	// alongside payload-level decode assertions — their encoders were
	// rewritten to the canonical ggml block layout (q2_k 84, q3_k 110,
	// q6_k 210, q8_k 292), fixing the prior under/over-emit write bug.
	cases := []struct {
		format    QuantizeFormat
		blockLen  int    // elements per block: 32 for *_0, 256 for K-quants
		typeName  string // ggml type name reported by ReadInfo
		bits      int
		blockSize int
	}{
		{QuantizeQ5_0, 32, "q5_0", 5, 32},
		{QuantizeQ5_K, 256, "q5_k", 5, 256},
	}
	for _, tc := range cases {
		t.Run(string(tc.format), func(t *testing.T) {
			// One weight whose rows are a whole number of blocks, plus a
			// 1-D norm the quantiser passes through.
			rows := 2
			source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
				{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{tc.blockLen, rows}, Data: ascendingFloat32s(tc.blockLen * rows)},
				{Name: "model.norm.weight", Shape: []int{tc.blockLen}, Data: ascendingFloat32s(tc.blockLen)},
			})
			output := core.PathJoin(t.TempDir(), "out-"+string(tc.format))

			result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
				SourcePack: sourcePackFromDir(source),
				OutputPath: output,
				Format:     tc.format,
			})
			if err != nil {
				t.Fatalf("QuantizeModelPack(%s) error = %v", tc.format, err)
			}
			if result.RequestedFormat != tc.format || result.Format != tc.format {
				t.Fatalf("formats = requested:%q used:%q, want %q", result.RequestedFormat, result.Format, tc.format)
			}
			if result.TensorCount != 2 || result.QuantizedTensors != 2 {
				t.Fatalf("tensor counts = %+v, want 2/2", result)
			}

			info, err := ReadInfo(output)
			if err != nil {
				t.Fatalf("ReadInfo(output) error = %v", err)
			}
			if !info.Valid() {
				t.Fatalf("GGUF validation issues = %+v", info.ValidationIssues)
			}
			if info.TensorCount != 2 {
				t.Fatalf("info.TensorCount = %d, want 2", info.TensorCount)
			}
			// The 2-D weight is the quantised tensor; assert its decoded
			// ggml type round-tripped bit-exact.
			weight := findTensorByName(info.Tensors, "model.layers.0.self_attn.q_proj.weight")
			if weight == nil {
				t.Fatalf("quantised weight tensor missing from %+v", info.Tensors)
			}
			if weight.TypeName != tc.typeName || weight.Bits != tc.bits || weight.BlockSize != tc.blockSize {
				t.Fatalf("weight tensor = type:%q bits:%d block:%d, want %q/%d/%d",
					weight.TypeName, weight.Bits, weight.BlockSize, tc.typeName, tc.bits, tc.blockSize)
			}
		})
	}
}

func findTensorByName(tensors []TensorInfo, name string) *TensorInfo {
	for i := range tensors {
		if tensors[i].Name == name {
			return &tensors[i]
		}
	}
	return nil
}

func TestQuantize_ValidationSummary_Good(t *testing.T) {
	// Mixed issues: one carries a tensor name (rendered code:tensor), one
	// does not (rendered code only); joined with ", " in order.
	issues := []ValidationIssue{
		{Severity: GGUFValidationError, Code: "shape_mismatch", Tensor: "blk.0.attn_q.weight"},
		{Severity: GGUFValidationWarning, Code: "missing_alignment"},
	}
	if got := ValidationSummary(issues); got != "shape_mismatch:blk.0.attn_q.weight, missing_alignment" {
		t.Fatalf("ValidationSummary() = %q", got)
	}
}

func TestQuantize_ValidationSummary_Bad(t *testing.T) {
	// A single tensor-scoped error must render as code:tensor with no
	// trailing separator.
	issues := []ValidationIssue{
		{Severity: GGUFValidationError, Code: "dtype_unsupported", Tensor: "blk.3.ffn_up.weight"},
	}
	if got := ValidationSummary(issues); got != "dtype_unsupported:blk.3.ffn_up.weight" {
		t.Fatalf("ValidationSummary(single tensor) = %q", got)
	}
}

func TestQuantize_ValidationSummary_Ugly(t *testing.T) {
	// Boundary: no issues at all yields the sentinel, never an empty string.
	if got := ValidationSummary(nil); got != "unknown validation failure" {
		t.Fatalf("ValidationSummary(nil) = %q, want sentinel", got)
	}
	if got := ValidationSummary([]ValidationIssue{}); got != "unknown validation failure" {
		t.Fatalf("ValidationSummary(empty) = %q, want sentinel", got)
	}
	// An issue with an empty Code and empty Tensor still renders a slot.
	if got := ValidationSummary([]ValidationIssue{{Severity: GGUFValidationWarning}}); got != "" {
		t.Fatalf("ValidationSummary(blank code) = %q, want one empty slot", got)
	}
}

// writeGGUFToTempFile runs fn against a fresh temp *core.OSFile, closes it, and
// returns the bytes written — the readback arbiter for the low-level GGUF
// header writers, which take a *core.OSFile rather than an io.Writer.
func writeGGUFToTempFile(t *testing.T, fn func(*core.OSFile) error) []byte {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "out.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create temp file: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	if err := fn(file); err != nil {
		file.Close()
		t.Fatalf("write fn error = %v", err)
	}
	file.Close()
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("read temp file: %v", read.Value)
	}
	return read.Value.([]byte)
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

func sourcePackFromDir(dir string) mp.ModelPack {
	return mp.ModelPack{
		Root:        dir,
		Path:        dir,
		Format:      mp.ModelPackFormatSafetensors,
		WeightFiles: []string{core.PathJoin(dir, "model.safetensors")},
	}
}

func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

const modelPackTokenizerJSON = `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`
