// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// TestWriteQuantizedGGUF_CreateFailure_Bad drives writeQuantizedGGUF's
// create-failure arm: an output path that is an existing directory makes
// core.Create fail before any header is written.
func TestWriteQuantizedGGUF_CreateFailure_Bad(t *testing.T) {
	dir := t.TempDir() // exists as a directory -> Create fails
	tensors := []ggufQuantizedTensor{{
		Name:  "model.norm.weight",
		Type:  TensorTypeQ8_0,
		Shape: []uint64{32},
		Data:  quantizeQ8_0(ascendingFloat32s(32)),
	}}
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	if err := writeQuantizedGGUF(dir, metadata, tensors); err == nil {
		t.Fatal("writeQuantizedGGUF(dir path) error = nil, want create failure")
	}
}

// TestWriteQuantizedGGUF_MultiTensorPadding_Good writes several tensors with
// distinct ragged sizes through the non-streaming writer so the inter-tensor
// padding + sequential data-write loop runs across more than one tensor (the
// single-tensor happy-path test only writes one). Read back via ReadInfo.
func TestWriteQuantizedGGUF_MultiTensorPadding_Good(t *testing.T) {
	output := core.PathJoin(t.TempDir(), "multi.gguf")
	tensors := []ggufQuantizedTensor{
		{Name: "model.layers.0.attn.weight", Type: TensorTypeQ8_0, Shape: []uint64{32, 3}, Data: quantizeQ8_0(ascendingFloat32s(96))},
		{Name: "model.layers.1.attn.weight", Type: TensorTypeQ8_0, Shape: []uint64{32, 2}, Data: quantizeQ8_0(ascendingFloat32s(64))},
		{Name: "model.norm.weight", Type: TensorTypeQ8_0, Shape: []uint64{32}, Data: quantizeQ8_0(ascendingFloat32s(32))},
	}
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	if err := writeQuantizedGGUF(output, metadata, tensors); err != nil {
		t.Fatalf("writeQuantizedGGUF(multi) error = %v", err)
	}
	info, err := ReadInfo(output)
	if err != nil {
		t.Fatalf("ReadInfo(multi) error = %v", err)
	}
	if !info.Valid() || info.TensorCount != 3 {
		t.Fatalf("multi-tensor info = %+v", info)
	}
}

// TestWriteQuantizedGGUFStream_ControlArms_Bad drives the validation/control
// arms of writeQuantizedGGUFStream that the happy path skips: an unsupported
// resolved format, a create failure (directory output), and a cancelled
// context detected inside the per-tensor loop.
func TestWriteQuantizedGGUFStream_ControlArms_Bad(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "model.layers.0.k_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	tensors, refs, err := buildStreamingGGUFQuantizedTensors(index, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("build streaming tensors: %v", err)
	}

	// Unsupported resolved format -> ggufQuantizeLayout returns an error
	// before any file is created.
	if err := writeQuantizedGGUFStream(context.Background(), core.PathJoin(t.TempDir(), "x.gguf"), nil, tensors, refs, "q3_0", 32); err == nil {
		t.Fatal("writeQuantizedGGUFStream(bad format) error = nil")
	}

	// Create failure: output path is a directory.
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	if err := writeQuantizedGGUFStream(context.Background(), t.TempDir(), metadata, tensors, refs, QuantizeQ8_0, 32); err == nil {
		t.Fatal("writeQuantizedGGUFStream(dir output) error = nil")
	}

	// Cancelled context: the per-tensor loop's ctx.Err() check fires.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := writeQuantizedGGUFStream(cancelled, core.PathJoin(t.TempDir(), "cancel.gguf"), metadata, tensors, refs, QuantizeQ8_0, 32); err != context.Canceled {
		t.Fatalf("writeQuantizedGGUFStream(cancelled) = %v, want context.Canceled", err)
	}
}

// TestWriteQuantizedGGUFStream_ChunkNormalisation_Good drives the
// chunkElements normalisation arms: a non-positive chunk falls back to the
// package default, and a chunk smaller than the block size is rounded up to
// one block. Both still produce a valid GGUF.
func TestWriteQuantizedGGUFStream_ChunkNormalisation_Good(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "model.layers.0.k_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)

	for _, chunk := range []int{0, -5, 1} {
		tensors, refs, err := buildStreamingGGUFQuantizedTensors(index, QuantizeQ8_0)
		if err != nil {
			t.Fatalf("build streaming tensors: %v", err)
		}
		output := core.PathJoin(t.TempDir(), "chunk.gguf")
		if err := writeQuantizedGGUFStream(context.Background(), output, metadata, tensors, refs, QuantizeQ8_0, chunk); err != nil {
			t.Fatalf("writeQuantizedGGUFStream(chunk=%d) error = %v", chunk, err)
		}
		info, err := ReadInfo(output)
		if err != nil {
			t.Fatalf("ReadInfo(chunk=%d) error = %v", chunk, err)
		}
		if !info.Valid() || info.TensorCount != 1 {
			t.Fatalf("chunk=%d info = %+v", chunk, info)
		}
	}
}

// TestWriteQuantizedGGUFTensorStream_Arms_Bad drives the standalone tensor-
// stream writer's control arms: an unsupported format (default switch arm)
// and an OpenReader failure (the source ref points at a missing file).
func TestWriteQuantizedGGUFTensorStream_Arms_Bad(t *testing.T) {
	created := core.Create(core.PathJoin(t.TempDir(), "sink.bin"))
	if !created.OK {
		t.Fatalf("create sink: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	ref := safetensors.TensorRef{
		Name:     "w",
		Path:     core.PathJoin(t.TempDir(), "missing.safetensors"),
		DType:    "F32",
		Shape:    []uint64{32},
		Elements: 32,
	}

	// Unsupported format -> default arm returns before opening the reader.
	if _, err := writeQuantizedGGUFTensorStream(context.Background(), file, ref, "q3_0", 32); err == nil {
		t.Fatal("writeQuantizedGGUFTensorStream(bad format) error = nil")
	}

	// OpenReader fails: the source file does not exist.
	if _, err := writeQuantizedGGUFTensorStream(context.Background(), file, ref, QuantizeQ8_0, 32); err == nil {
		t.Fatal("writeQuantizedGGUFTensorStream(missing source) error = nil")
	}
}

// TestWriteQuantizedGGUFTensorStream_Q4_0_Good drives the QuantizeQ4_0 switch
// arm of writeQuantizedGGUFTensorStream (the resolved-quantiser selection) and
// the chunk loop's write + accumulate, then checks the byte count.
func TestWriteQuantizedGGUFTensorStream_Q4_0_Good(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "w", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	ref := index.Tensors["w"]

	output := core.PathJoin(t.TempDir(), "out.bin")
	created := core.Create(output)
	if !created.OK {
		t.Fatalf("create sink: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	written, err := writeQuantizedGGUFTensorStream(context.Background(), file, ref, QuantizeQ4_0, 32)
	file.Close()
	if err != nil {
		t.Fatalf("writeQuantizedGGUFTensorStream(q4_0) error = %v", err)
	}
	// 64 elements / 32 per block = 2 blocks * 18 bytes = 36.
	if written != 36 {
		t.Fatalf("writeQuantizedGGUFTensorStream(q4_0) wrote %d, want 36", written)
	}
}

// TestWriteQuantizedGGUFTensorStream_WriteFailure_Bad drives the file.Write
// error arm inside the chunk loop: a read-only output file makes the chunk
// write fail with EBADF after the source chunk is read and quantised.
func TestWriteQuantizedGGUFTensorStream_WriteFailure_Bad(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "w", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	ref := index.Tensors["w"]

	file := readOnlyOSFile(t) // writes fail with EBADF
	defer file.Close()
	if _, err := writeQuantizedGGUFTensorStream(context.Background(), file, ref, QuantizeQ8_0, 32); err == nil {
		t.Fatal("writeQuantizedGGUFTensorStream(read-only sink) error = nil")
	}
}

// TestWriteQuantizedGGUFTensorStream_ChunkReadError_Bad drives the
// ReadFloat32Chunk error arm: a ref whose Elements count is inflated far
// beyond the actual source payload so OpenReader succeeds but the chunk read
// runs off the end of the file and returns EOF.
func TestWriteQuantizedGGUFTensorStream_ChunkReadError_Bad(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "w", Shape: []int{32}, Data: ascendingFloat32s(32)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	ref := index.Tensors["w"]
	ref.Elements = 32 * 1000 // far more than the file actually holds

	created := core.Create(core.PathJoin(t.TempDir(), "sink.bin"))
	if !created.OK {
		t.Fatalf("create sink: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()
	if _, err := writeQuantizedGGUFTensorStream(context.Background(), file, ref, QuantizeQ8_0, 32); err == nil {
		t.Fatal("writeQuantizedGGUFTensorStream(inflated elements) error = nil, want chunk-read failure")
	}
}

// TestWriteQuantizedGGUFStream_PerTensorError_Bad drives the per-tensor error
// return inside writeQuantizedGGUFStream's loop: a two-tensor index whose
// SECOND ref points at a missing source file, so tensor 0 streams cleanly but
// tensor 1's OpenReader fails mid-loop.
func TestWriteQuantizedGGUFStream_PerTensorError_Bad(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "a", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
		{Name: "b", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	tensors, refs, err := buildStreamingGGUFQuantizedTensors(index, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("build streaming tensors: %v", err)
	}
	if len(refs) != 2 {
		t.Fatalf("expected 2 refs, got %d", len(refs))
	}
	// Point the second tensor's source at a path that does not exist so its
	// OpenReader fails after the first tensor has written.
	for i := range refs {
		if refs[i].Name == "b" {
			refs[i].Path = core.PathJoin(t.TempDir(), "missing.safetensors")
		}
	}

	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	output := core.PathJoin(t.TempDir(), "partial.gguf")
	if err := writeQuantizedGGUFStream(context.Background(), output, metadata, tensors, refs, QuantizeQ8_0, 32); err == nil {
		t.Fatal("writeQuantizedGGUFStream(second tensor missing) error = nil")
	}
}

// TestWriteQuantizedGGUFTensorStream_CancelledChunkLoop_Bad drives the
// ctx.Err() check inside the chunk loop: a real source ref is opened, but the
// context is already cancelled so the first chunk iteration aborts.
func TestWriteQuantizedGGUFTensorStream_CancelledChunkLoop_Bad(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "w", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	ref := index.Tensors["w"]

	created := core.Create(core.PathJoin(t.TempDir(), "sink.bin"))
	if !created.OK {
		t.Fatalf("create sink: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := writeQuantizedGGUFTensorStream(cancelled, file, ref, QuantizeQ8_0, 32); err != context.Canceled {
		t.Fatalf("writeQuantizedGGUFTensorStream(cancelled) = %v, want context.Canceled", err)
	}
}

// readOnlyOSFile opens a freshly-seeded file read-only and returns the
// *core.OSFile. Writing to it fails with EBADF, which is exactly the
// mid-write I/O failure the low-level GGUF header writers guard against — a
// clean injection point that needs no mock filesystem.
func readOnlyOSFile(t *testing.T) *core.OSFile {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "ro.bin")
	if w := core.WriteFile(path, []byte("seed"), 0o644); !w.OK {
		t.Fatalf("seed read-only file: %v", w.Value)
	}
	open := core.Open(path) // O_RDONLY
	if !open.OK {
		t.Fatalf("open read-only: %v", open.Value)
	}
	return open.Value.(*core.OSFile)
}

// TestWriteGGUF_WriteFailures_Bad drives the file.Write error arms of the
// low-level GGUF header writers by handing each a read-only file descriptor.
// These are the mid-write I/O-failure guards (the `if _, err := file.Write
// (...); err != nil` arms) that the happy-path readback tests cannot reach.
func TestWriteGGUF_WriteFailures_Bad(t *testing.T) {
	t.Run("writeGGUFStringValue_small", func(t *testing.T) {
		file := readOnlyOSFile(t)
		defer file.Close()
		if err := writeGGUFStringValue(file, "general.architecture"); err == nil {
			t.Fatal("writeGGUFStringValue(read-only) error = nil")
		}
	})

	t.Run("writeGGUFStringValue_large", func(t *testing.T) {
		// > 248 bytes forces the separate length-prefix write, whose error arm
		// differs from the small stacked path.
		file := readOnlyOSFile(t)
		defer file.Close()
		big := string(make([]byte, 4096))
		if err := writeGGUFStringValue(file, big); err == nil {
			t.Fatal("writeGGUFStringValue(read-only, large) error = nil")
		}
	})

	t.Run("writeGGUFMetadataValue_uint32", func(t *testing.T) {
		file := readOnlyOSFile(t)
		defer file.Close()
		if err := writeGGUFMetadataValue(file, ValueTypeUint32, uint32(7)); err == nil {
			t.Fatal("writeGGUFMetadataValue(read-only uint32) error = nil")
		}
	})

	t.Run("writeGGUFMetadataEntry", func(t *testing.T) {
		// The entry writer's first action is the key string write, which fails.
		file := readOnlyOSFile(t)
		defer file.Close()
		entry := ggufMetadataEntry{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"}
		if err := writeGGUFMetadataEntry(file, entry); err == nil {
			t.Fatal("writeGGUFMetadataEntry(read-only) error = nil")
		}
	})

	t.Run("writeGGUFTensorInfo", func(t *testing.T) {
		file := readOnlyOSFile(t)
		defer file.Close()
		tensor := ggufQuantizedTensor{Name: "blk.0.w", Type: TensorTypeQ8_0, Shape: []uint64{32, 2}}
		if err := writeGGUFTensorInfo(file, tensor); err == nil {
			t.Fatal("writeGGUFTensorInfo(read-only) error = nil")
		}
	})

	t.Run("writePadding", func(t *testing.T) {
		file := readOnlyOSFile(t)
		defer file.Close()
		if err := writePadding(file, 16); err == nil {
			t.Fatal("writePadding(read-only) error = nil")
		}
	})

	t.Run("writeQuantizedGGUFHeader", func(t *testing.T) {
		// The header writer's first action is the 24-byte header write.
		file := readOnlyOSFile(t)
		defer file.Close()
		tensors := []ggufQuantizedTensor{{Name: "blk.0.w", Type: TensorTypeQ8_0, Shape: []uint64{32}, Data: quantizeQ8_0(ascendingFloat32s(32))}}
		metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
		if err := writeQuantizedGGUFHeader(file, metadata, tensors); err == nil {
			t.Fatal("writeQuantizedGGUFHeader(read-only) error = nil")
		}
	})
}

// TestWriteGGUFTensorInfo_HighRank_Good drives writeGGUFTensorInfo's heap
// fallback for the dimension buffer: a tensor with enough dimensions that the
// packed ndim+dims+type+offset block exceeds the 64-byte stack scratch takes
// the make([]byte, bufLen) path. Read back through the temp-file arbiter.
func TestWriteGGUFTensorInfo_HighRank_Good(t *testing.T) {
	// 8 dims -> 4 + 8*8 + 4 + 8 = 80 bytes > 64-byte stack buffer.
	tensor := ggufQuantizedTensor{
		Name:  "blk.0.w",
		Type:  TensorTypeQ8_0,
		Shape: []uint64{2, 2, 2, 2, 2, 2, 2, 2},
	}
	got := writeGGUFToTempFile(t, func(f *core.OSFile) error {
		return writeGGUFTensorInfo(f, tensor)
	})
	// name length prefix (8) + "blk.0.w" (7) + 80-byte dim block.
	wantLen := 8 + len("blk.0.w") + 80
	if len(got) != wantLen {
		t.Fatalf("writeGGUFTensorInfo(8-dim) wrote %d bytes, want %d", len(got), wantLen)
	}
}

// TestWriteQuantizedGGUFStream_TensorSizeMismatch_Bad drives the
// streamed-bytes-vs-expected guard: a tensor whose declared Size does not
// match the bytes the quantiser actually emits for its source ref must be
// rejected. Here the ref carries 64 F32 elements (-> 68 q8_0 bytes) but the
// tensor metadata advertises a deliberately wrong Size.
func TestWriteQuantizedGGUFStream_TensorSizeMismatch_Bad(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "w", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	tensors, refs, err := buildStreamingGGUFQuantizedTensors(index, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("build streaming tensors: %v", err)
	}
	// Corrupt the advertised size so the post-write equality check fails.
	tensors[0].Size = 9999

	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	output := core.PathJoin(t.TempDir(), "mismatch.gguf")
	if err := writeQuantizedGGUFStream(context.Background(), output, metadata, tensors, refs, QuantizeQ8_0, 32); err == nil {
		t.Fatal("writeQuantizedGGUFStream(size mismatch) error = nil")
	}
}
