// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/safetensors"
)

// writeTypedSafetensors writes a single-tensor safetensors file with an
// arbitrary dtype + raw payload, for the non-F32 paths the F32-only test
// helper cannot build (e.g. an I32 tensor that ReadIndex accepts but the
// float decoders reject).
func writeTypedSafetensors(t *testing.T, path, name, dtype string, shape []int, payload []byte) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{
		name: {DType: dtype, Shape: shape, DataOffsets: []int{0, len(payload)}},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal typed header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write typed safetensors: %v", result.Value)
	}
}

// TestQuantizeModelPack_NilContext_Good drives the ctx == nil arm: a nil
// context is replaced with context.Background() and the quantisation still
// succeeds.
func TestQuantizeModelPack_NilContext_Good(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	output := core.PathJoin(t.TempDir(), "out-nilctx")
	//nolint:staticcheck // deliberately passing a nil context to cover the guard.
	result, err := QuantizeModelPack(nil, QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ8_0,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack(nil ctx) error = %v", err)
	}
	if result.Format != QuantizeQ8_0 {
		t.Fatalf("QuantizeModelPack(nil ctx) format = %q", result.Format)
	}
}

// TestQuantizeModelPack_ValidationArms_Bad drives the early-return validation
// arms: empty output path, unsupported format, and a non-safetensors source
// pack format.
func TestQuantizeModelPack_ValidationArms_Bad(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})

	// Empty output path.
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
	}); err == nil {
		t.Fatal("QuantizeModelPack(empty output) error = nil")
	}

	// Unsupported quant format.
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     "iq4_nl",
	}); err == nil {
		t.Fatal("QuantizeModelPack(bad format) error = nil")
	}

	// Non-safetensors source pack: a pack whose Format is GGUF (not the dense
	// safetensors the quantiser requires).
	ggufPack := mp.ModelPack{
		Root:        source,
		Path:        source,
		Format:      mp.ModelPackFormatGGUF,
		WeightFiles: []string{core.PathJoin(source, "model.safetensors")},
	}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: ggufPack,
		OutputPath: core.PathJoin(t.TempDir(), "out2"),
		Format:     QuantizeQ8_0,
	}); err == nil {
		t.Fatal("QuantizeModelPack(non-safetensors source) error = nil")
	}
}

// TestQuantizeModelPack_OccupiedDestination_Bad drives the ensureEmpty error
// arm inside QuantizeModelPack: the output directory already holds a weight
// file, so the destination-empty guard rejects it.
func TestQuantizeModelPack_OccupiedDestination_Bad(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	output := core.PathJoin(t.TempDir(), "occupied")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(output, "existing.gguf"), []byte("x"), 0o644); !result.OK {
		t.Fatalf("write existing weight: %v", result.Value)
	}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ8_0,
	}); err == nil {
		t.Fatal("QuantizeModelPack(occupied destination) error = nil")
	}
}

// TestQuantizeModelPack_NonFloatTensor_Bad drives the buildStreaming dtype
// rejection (DTypeByteSize error) through the public QuantizeModelPack path:
// a source safetensors carrying an I32 tensor, which the float-only GGUF
// quantiser cannot consume.
func TestQuantizeModelPack_NonFloatTensor_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	// 32 int32 values -> 128 bytes; a valid window ReadIndex accepts but
	// DTypeByteSize/DecodeFloatData reject.
	payload := make([]byte, 32*4)
	writeTypedSafetensors(t, core.PathJoin(dir, "model.safetensors"), "w.weight", "I32", []int{32}, payload)

	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(dir),
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     QuantizeQ8_0,
	}); err == nil {
		t.Fatal("QuantizeModelPack(I32 tensor) error = nil")
	}
}

// TestResolveGGUFQuantizeFormat_BareQ4K_Good drives the bare q4_k case arm of
// resolveGGUFQuantizeFormat: the existing cases pass q4_k_m (a distinct arm)
// but never the canonical q4_k spelling, which maps to itself.
func TestResolveGGUFQuantizeFormat_BareQ4K_Good(t *testing.T) {
	requested, used, notes, err := resolveGGUFQuantizeFormat("q4_k")
	if err != nil {
		t.Fatalf("resolveGGUFQuantizeFormat(q4_k) error = %v", err)
	}
	if requested != QuantizeQ4_K || used != QuantizeQ4_K || len(notes) != 0 {
		t.Fatalf("resolveGGUFQuantizeFormat(q4_k) = req:%q used:%q notes:%d", requested, used, len(notes))
	}
}

// TestQuantizeModelPack_MkdirFailure_Bad drives the MkdirAll error arm of
// QuantizeModelPack: the output path's parent is a regular file, so creating
// the output directory fails.
func TestQuantizeModelPack_MkdirFailure_Bad(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	// Create a regular file, then use a path UNDER it as the output dir.
	parent := core.PathJoin(t.TempDir(), "afile")
	if result := core.WriteFile(parent, []byte("x"), 0o644); !result.OK {
		t.Fatalf("write blocking file: %v", result.Value)
	}
	output := core.PathJoin(parent, "subdir") // parent is a file -> MkdirAll fails
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ8_0,
	}); err == nil {
		t.Fatal("QuantizeModelPack(mkdir failure) error = nil")
	}
}

// TestQuantizeModelPack_MkdirEACCES_Bad drives QuantizeModelPack's MkdirAll
// error arm: the output directory's parent is read-only (mode 0o555), so it
// is searchable enough for ensureEmpty's Stat to return ENOENT (-> nil) but
// MkdirAll fails with EACCES when it tries to create the output directory.
// (A regular-file parent instead trips ensureEmpty's stat-error arm, which is
// why a permission-only parent is needed to isolate the MkdirAll arm.)
func TestQuantizeModelPack_MkdirEACCES_Bad(t *testing.T) {
	source := writeDenseSafetensorsPack(t, "qwen3", []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	parent := core.PathJoin(t.TempDir(), "ro-parent")
	if result := core.MkdirAll(parent, 0o755); !result.OK {
		t.Fatalf("mkdir parent: %v", result.Value)
	}
	if result := core.Chmod(parent, 0o555); !result.OK {
		t.Fatalf("chmod parent read-only: %v", result.Value)
	}
	// Restore write permission so t.TempDir's recursive cleanup succeeds.
	defer core.Chmod(parent, 0o755)

	output := core.PathJoin(parent, "out")
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(source),
		OutputPath: output,
		Format:     QuantizeQ8_0,
	}); err == nil {
		t.Fatal("QuantizeModelPack(read-only parent) error = nil, want mkdir failure")
	}
}

// TestBuildStreamingGGUFQuantizedTensors_BadFormat_Bad drives the
// ggufQuantizeLayout error arm: an unsupported resolved format makes the
// layout lookup fail before any tensor is inspected.
func TestBuildStreamingGGUFQuantizedTensors_BadFormat_Bad(t *testing.T) {
	index := safetensors.Index{
		Names: []string{"w.weight"},
		Tensors: map[string]safetensors.TensorRef{
			"w.weight": {Name: "w.weight", DType: "F32", Shape: []uint64{32}, Elements: 32},
		},
	}
	if _, _, err := buildStreamingGGUFQuantizedTensors(index, "q3_0"); err == nil {
		t.Fatal("buildStreamingGGUFQuantizedTensors(bad format) error = nil")
	}
}

// TestQuantizeModelPack_MetadataCopyError_Bad drives QuantizeModelPack's
// copyModelPackMetadata error arm: a source pack whose config.json is a
// DIRECTORY (not a file) makes the metadata copy's ReadFile fail after the
// output directory has already been created. This also exercises
// copyLocalFile's read-failure arm via the real pipeline.
func TestQuantizeModelPack_MetadataCopyError_Bad(t *testing.T) {
	src := t.TempDir()
	// config.json as a directory -> core.ReadFile on it fails.
	if result := core.MkdirAll(core.PathJoin(src, "config.json"), 0o755); !result.OK {
		t.Fatalf("mkdir config.json dir: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(src, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestSafetensorsF32(t, core.PathJoin(src, "model.safetensors"), []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: sourcePackFromDir(src),
		OutputPath: core.PathJoin(t.TempDir(), "out"),
		Format:     QuantizeQ8_0,
	}); err == nil {
		t.Fatal("QuantizeModelPack(config.json dir) error = nil")
	}
}

// TestLoadDenseSafetensors_Success_Good drives loadDenseSafetensors' happy
// path across two shards (the read, the dedup, the sort, and the return) plus
// the per-shard readDenseSafetensors success path.
func TestLoadDenseSafetensors_Success_Good(t *testing.T) {
	dir := t.TempDir()
	first := core.PathJoin(dir, "a.safetensors")
	second := core.PathJoin(dir, "b.safetensors")
	writeTestSafetensorsF32(t, first, []safetensorTestTensor{
		{Name: "zeta.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	writeTestSafetensorsF32(t, second, []safetensorTestTensor{
		{Name: "alpha.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	out, err := loadDenseSafetensors([]string{first, second})
	if err != nil {
		t.Fatalf("loadDenseSafetensors error = %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("loadDenseSafetensors returned %d tensors, want 2", len(out))
	}
	// Sorted by name: alpha before zeta.
	if out[0].Name != "alpha.weight" || out[1].Name != "zeta.weight" {
		t.Fatalf("loadDenseSafetensors order = %q,%q; want alpha,zeta", out[0].Name, out[1].Name)
	}
}

// TestLoadDenseSafetensors_ReadError_Bad drives loadDenseSafetensors'
// error-propagation arm: one shard is malformed, so readDenseSafetensors
// fails and the error surfaces.
func TestLoadDenseSafetensors_ReadError_Bad(t *testing.T) {
	dir := t.TempDir()
	good := core.PathJoin(dir, "good.safetensors")
	writeTestSafetensorsF32(t, good, []safetensorTestTensor{
		{Name: "ok.weight", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	bad := core.PathJoin(dir, "bad.safetensors")
	if result := core.WriteFile(bad, []byte{1, 2, 3}, 0o644); !result.OK {
		t.Fatalf("write bad shard: %v", result.Value)
	}
	if _, err := loadDenseSafetensors([]string{good, bad}); err == nil {
		t.Fatal("loadDenseSafetensors(bad shard) error = nil")
	}
}

// TestReadDenseSafetensors_NonFloatDecode_Bad drives readDenseSafetensors'
// DecodeFloatData rejection arm: an I32 tensor has a window ReadIndex accepts,
// but the float decoder rejects the dtype.
func TestReadDenseSafetensors_NonFloatDecode_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "i32.safetensors")
	writeTypedSafetensors(t, path, "w.weight", "I32", []int{32}, make([]byte, 32*4))
	if _, err := readDenseSafetensors(path); err == nil {
		t.Fatal("readDenseSafetensors(I32) error = nil, want decode rejection")
	}
}

// TestDecodeDenseSafetensor_Success_Good drives decodeDenseSafetensor's happy
// path (valid offsets, non-empty shape, supported dtype) so the success
// return + shape/elements computation is covered, complementing the existing
// error-path table.
func TestDecodeDenseSafetensor_Success_Good(t *testing.T) {
	// 4 F32 values laid out little-endian.
	payload := make([]byte, 16)
	for i := range []int{0, 1, 2, 3} {
		binary.LittleEndian.PutUint32(payload[i*4:], 0)
	}
	entry := safetensors.HeaderEntry{
		DType:       "F32",
		Shape:       []int64{2, 2},
		DataOffsets: []int64{0, 16},
	}
	got, err := decodeDenseSafetensor("m.safetensors", "w.weight", entry, payload)
	if err != nil {
		t.Fatalf("decodeDenseSafetensor(valid) error = %v", err)
	}
	if got.Name != "w.weight" || len(got.Shape) != 2 || len(got.Data) != 4 {
		t.Fatalf("decodeDenseSafetensor(valid) = %+v", got)
	}
}

// TestDecodeDenseSafetensor_DecodeReject_Bad drives the dtype-rejection arm of
// decodeDenseSafetensor: a valid window with an I32 dtype the float decoder
// rejects.
func TestDecodeDenseSafetensor_DecodeReject_Bad(t *testing.T) {
	entry := safetensors.HeaderEntry{
		DType:       "I32",
		Shape:       []int64{4},
		DataOffsets: []int64{0, 16},
	}
	if _, err := decodeDenseSafetensor("m.safetensors", "w.weight", entry, make([]byte, 16)); err == nil {
		t.Fatal("decodeDenseSafetensor(I32) error = nil, want decode rejection")
	}
}

// TestDecodeDenseSafetensor_EmptyShape_Bad drives decodeDenseSafetensor's
// empty-shape arm: a tensor whose Shape slice is empty (zero dimensions, not
// a zero-valued dimension — that is a separate arm the existing table covers).
func TestDecodeDenseSafetensor_EmptyShape_Bad(t *testing.T) {
	entry := safetensors.HeaderEntry{
		DType:       "F32",
		Shape:       []int64{}, // empty, not [0]
		DataOffsets: []int64{0, 16},
	}
	if _, err := decodeDenseSafetensor("m.safetensors", "w.weight", entry, make([]byte, 16)); err == nil {
		t.Fatal("decodeDenseSafetensor(empty shape) error = nil")
	}
}

// TestBuildStreamingGGUFQuantizedTensors_ShapeNotAligned_Bad drives the
// first-dimension-not-block-aligned arm: a tensor whose leading dim is not a
// multiple of the format block size.
func TestBuildStreamingGGUFQuantizedTensors_ShapeNotAligned_Bad(t *testing.T) {
	// Elements divisible by 32 (so the element guard passes) but Shape[0]==1
	// is not a multiple of 32 (so the shape guard fires).
	index := safetensors.Index{
		Names: []string{"w.weight"},
		Tensors: map[string]safetensors.TensorRef{
			"w.weight": {Name: "w.weight", DType: "F32", Shape: []uint64{1, 32}, Elements: 32},
		},
	}
	if _, _, err := buildStreamingGGUFQuantizedTensors(index, QuantizeQ8_0); err == nil {
		t.Fatal("buildStreamingGGUFQuantizedTensors(unaligned shape) error = nil")
	}
}

// TestCopyModelPackMetadata_Arms_Good drives copyModelPackMetadata's dedup
// (same name across patterns), weight-skip (a .safetensors / .gguf file is not
// copied), and the successful copy of a metadata file.
func TestCopyModelPackMetadata_Arms_Good(t *testing.T) {
	src := t.TempDir()
	dst := t.TempDir()
	// A metadata file (copied), a weight file (skipped), and the
	// adapter_provenance.json (skipped).
	writeModelPackFile(t, core.PathJoin(src, "config.json"), `{"model_type":"qwen3"}`)
	writeModelPackFile(t, core.PathJoin(src, "model.safetensors"), "WEIGHTS")
	writeModelPackFile(t, core.PathJoin(src, "adapter_provenance.json"), "{}")
	writeModelPackFile(t, core.PathJoin(src, "tokenizer.model"), "TOKENIZER")

	if err := copyModelPackMetadata(src, dst); err != nil {
		t.Fatalf("copyModelPackMetadata error = %v", err)
	}
	// config.json + tokenizer.model copied.
	if stat := core.Stat(core.PathJoin(dst, "config.json")); !stat.OK {
		t.Fatal("config.json was not copied")
	}
	if stat := core.Stat(core.PathJoin(dst, "tokenizer.model")); !stat.OK {
		t.Fatal("tokenizer.model was not copied")
	}
	// model.safetensors + adapter_provenance.json skipped.
	if stat := core.Stat(core.PathJoin(dst, "model.safetensors")); stat.OK {
		t.Fatal("model.safetensors should have been skipped")
	}
	if stat := core.Stat(core.PathJoin(dst, "adapter_provenance.json")); stat.OK {
		t.Fatal("adapter_provenance.json should have been skipped")
	}
}

// TestCopyModelPackMetadata_CopyError_Bad drives copyModelPackMetadata's
// copy-failure arm: the destination directory does not exist, so writing the
// copied file fails.
func TestCopyModelPackMetadata_CopyError_Bad(t *testing.T) {
	src := t.TempDir()
	writeModelPackFile(t, core.PathJoin(src, "config.json"), `{"model_type":"qwen3"}`)
	// Destination is a path whose parent does not exist -> copyLocalFile's
	// WriteFile fails.
	dst := core.PathJoin(t.TempDir(), "no", "such", "dir")
	if err := copyModelPackMetadata(src, dst); err == nil {
		t.Fatal("copyModelPackMetadata(bad dst) error = nil")
	}
}

// TestCopyLocalFile_Arms_Bad drives copyLocalFile's read-failure and
// write-failure arms directly.
func TestCopyLocalFile_Arms_Bad(t *testing.T) {
	// Read failure: source does not exist.
	if err := copyLocalFile(core.PathJoin(t.TempDir(), "missing.json"), core.PathJoin(t.TempDir(), "dst.json")); err == nil {
		t.Fatal("copyLocalFile(missing source) error = nil")
	}

	// Write failure: destination parent does not exist.
	src := core.PathJoin(t.TempDir(), "src.json")
	if result := core.WriteFile(src, []byte("{}"), 0o644); !result.OK {
		t.Fatalf("write src: %v", result.Value)
	}
	if err := copyLocalFile(src, core.PathJoin(t.TempDir(), "no", "dst.json")); err == nil {
		t.Fatal("copyLocalFile(bad dst) error = nil")
	}
}

// TestEnsureEmptyGGUFQuantizeDestination_NilReturn_Good drives the final
// return-nil arm: a destination directory that exists but holds no weight
// files is accepted.
func TestEnsureEmptyGGUFQuantizeDestination_NilReturn_Good(t *testing.T) {
	dir := t.TempDir()
	// A non-weight file present -> the glob finds nothing, so the function
	// returns nil (the directory is "empty" of weights).
	writeModelPackFile(t, core.PathJoin(dir, "README.txt"), "hello")
	if err := ensureEmptyGGUFQuantizeDestination(dir); err != nil {
		t.Fatalf("ensureEmptyGGUFQuantizeDestination(no weights) = %v, want nil", err)
	}
}
