// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"testing"
	"unsafe"

	"dappco.re/go/mlx/internal/metaltest"
)

// TestLoadDirMmap_RealGemma4_12B4bit_HeaderTruth mmaps the REAL
// mlx-community/gemma-4-12B-it-4bit checkpoint — a 2-shard, 1341-tensor,
// ~6.7GB 4-bit-quantized production checkpoint — and pins field values read
// directly from the shard's on-disk header (independently confirmed by
// inspecting model-00001-of-00002.safetensors's raw JSON header and the
// index.json's weight_map with a standalone Python script, not derived from
// this package's own output). The synthetic fixtures elsewhere in this
// package only ever use 1-3-tensor blobs with tiny (<100 byte) data_offsets;
// this is the header-parse path's only exposure to: a real quantization
// dtype pairing (U32 packed codes + BF16 scale/bias companions), a 262,144-
// row embedding table, and — a shape a synthetic blob cannot produce — a
// data_offset (5,351,622,730) that exceeds math.MaxInt32, proving the
// parser's []int offsets survive a real multi-GB shard without truncation.
// mmap-only (LoadDirMmap): the ~6.7GB checkpoint is never read onto the
// heap, only mapped and spot-checked.
func TestLoadDirMmap_RealGemma4_12B4bit_HeaderTruth(t *testing.T) {
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-12B-it-4bit")

	dm, err := LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap(real gemma-4 12B 4bit): %v", err)
	}
	defer dm.Close()

	if len(dm.Shards) != 2 {
		t.Fatalf("shards = %d, want 2 (model-00001-of-00002 + model-00002-of-00002)", len(dm.Shards))
	}
	// Pinned from the real model.safetensors.index.json weight_map — independently
	// counted (python: len(json.load(open(index))['weight_map'])) == 1341, not
	// derived from this package's own output.
	const wantTensors = 1341
	if len(dm.Tensors) != wantTensors {
		t.Fatalf("merged tensor count = %d, want %d (real index.json weight_map size)", len(dm.Tensors), wantTensors)
	}

	// language_model.model.embed_tokens.weight: the packed 4-bit codes, dtype U32,
	// shape [262144, 480] — pinned straight from the shard's raw on-disk header.
	weight, ok := dm.Tensors["language_model.model.embed_tokens.weight"]
	if !ok {
		t.Fatal("language_model.model.embed_tokens.weight missing from real checkpoint")
	}
	if weight.Dtype != "U32" {
		t.Errorf("embed_tokens.weight dtype = %q, want U32", weight.Dtype)
	}
	if len(weight.Shape) != 2 || weight.Shape[0] != 262144 || weight.Shape[1] != 480 {
		t.Errorf("embed_tokens.weight shape = %v, want [262144 480]", weight.Shape)
	}
	// 262144 * 480 * 4 bytes (U32) = 503,316,480 — the real on-disk data_offsets span
	// (2923360304..3426676784) from the shard header, not recomputed from Shape here.
	const wantWeightBytes = 503316480
	if len(weight.Data) != wantWeightBytes {
		t.Errorf("embed_tokens.weight byte span = %d, want %d", len(weight.Data), wantWeightBytes)
	}

	// language_model.model.embed_tokens.biases: the BF16 quantization companion,
	// shape [262144, 60], data_offsets span 31,457,280 bytes on disk.
	biases, ok := dm.Tensors["language_model.model.embed_tokens.biases"]
	if !ok {
		t.Fatal("language_model.model.embed_tokens.biases missing from real checkpoint")
	}
	if biases.Dtype != "BF16" {
		t.Errorf("embed_tokens.biases dtype = %q, want BF16", biases.Dtype)
	}
	if len(biases.Shape) != 2 || biases.Shape[0] != 262144 || biases.Shape[1] != 60 {
		t.Errorf("embed_tokens.biases shape = %v, want [262144 60]", biases.Shape)
	}
	const wantBiasesBytes = 31457280 // real on-disk span: 862289936 - 830832656
	if len(biases.Data) != wantBiasesBytes {
		t.Errorf("embed_tokens.biases byte span = %d, want %d", len(biases.Data), wantBiasesBytes)
	}

	// Both tensors must be VIEWS into shard mmaps (zero-copy), not heap copies — the
	// property the whole DirMapping/LoadDirMmap design exists for, now proven at real
	// multi-GB scale rather than only on a synthetic few-byte fixture.
	for name, tensor := range map[string]Tensor{"weight": weight, "biases": biases} {
		if len(tensor.Data) == 0 {
			continue
		}
		ptr := uintptr(unsafe.Pointer(&tensor.Data[0]))
		inShard := false
		for _, sh := range dm.Shards {
			if len(sh.Data) == 0 {
				continue
			}
			base := uintptr(unsafe.Pointer(&sh.Data[0]))
			if ptr >= base && ptr < base+uintptr(len(sh.Data)) {
				inShard = true
				break
			}
		}
		if !inShard {
			t.Errorf("%s: Data is not a view into any shard mmap — zero-copy broken on a real checkpoint", name)
		}
	}
	t.Logf("real gemma-4 12B-4bit: %d tensors across %d shards, embed_tokens U32[262144,480] + BF16[262144,60] byte spans pinned from the real on-disk header", len(dm.Tensors), len(dm.Shards))
}
