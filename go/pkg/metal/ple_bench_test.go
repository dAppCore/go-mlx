// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Per-Layer Embedding (PLE) bench coverage map (W7-E, Wave 7).
//
// Gemma 4 E2B / E4B carry massive Per-Layer Embedding tables that
// inflate total parameter counts (5.1B / 8B) without participating in
// every forward pass — only the per-layer slice is fetched per layer.
//
// IDEAS.md §2: "The PLE tables are only used for quick lookups
// per layer. They should remain in fast local storage (or mapped CPU
// RAM) and only the specific embedding slice for the current layer
// should be fetched via mlx_take during the forward pass."
//
// Coverage:
//   - Take (mlx_take) on PLE-sized lookup tables: per-layer fetch cost
//     at varying table sizes (proxying E2B vs E4B PLE block sizes).
//   - Embedding.Forward — the standard token embedding (separate
//     concern from PLE but uses similar gather mechanics; benched here
//     as the comparator).
//   - Sweep on table_size × hidden combinations to surface the
//     bandwidth-bound vs latency-bound regime split.

import "testing"

// --- PLE-table Take (per-layer slice fetch) ---

// E2B-scale PLE block: typical numLayers × hiddenSizePerLayerInput.
// Gemma 4 E2B has hidden_size_per_layer_input typically 256, and
// numLayers ≈ 26 — so the per-layer PLE block is ~256 × 256.
// We bench the gather of a single layer's slice from a packed
// table of shape [numLayers, perLayerInputSize].
func BenchmarkPLE_TakeLayerSlice_NumLayers32_PerLayer256(b *testing.B) {
	table := RandomUniform(-1, 1, []int32{32, 256}, DTypeFloat32)
	indices := FromValues([]int32{15}, 1) // layer 15
	defer Free(table, indices)
	Materialize(table, indices)
	b.SetBytes(int64(256 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Take(table, indices, 0)
		Materialize(y)
		Free(y)
	}
}

// E4B-scale PLE block: hidden_size_per_layer_input 512, numLayers 38.
func BenchmarkPLE_TakeLayerSlice_NumLayers38_PerLayer512(b *testing.B) {
	table := RandomUniform(-1, 1, []int32{38, 512}, DTypeFloat32)
	indices := FromValues([]int32{20}, 1)
	defer Free(table, indices)
	Materialize(table, indices)
	b.SetBytes(int64(512 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Take(table, indices, 0)
		Materialize(y)
		Free(y)
	}
}

// PLE block as a full embedding pattern: vocab_size × per_layer_input.
// Per IDEAS.md, this is the table that "shouldn't live in VRAM" — but
// when it does, Take cost scales with the lookup, not the table size.
func BenchmarkPLE_TakeFromLargeTable_Vocab262k_PerLayer256(b *testing.B) {
	table := RandomUniform(-1, 1, []int32{262208, 256}, DTypeFloat32)
	// Lookup 1 token's slice — single fetch path.
	indices := FromValues([]int32{42}, 1)
	defer Free(table, indices)
	Materialize(table, indices)
	b.SetBytes(int64(256 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Take(table, indices, 0)
		Materialize(y)
		Free(y)
	}
}

// 32-token batch fetch from large PLE table — typical prefill.
func BenchmarkPLE_TakeBatch32_Vocab262k_PerLayer256(b *testing.B) {
	table := RandomUniform(-1, 1, []int32{262208, 256}, DTypeFloat32)
	// 32 distinct tokens.
	idsData := make([]int32, 32)
	for i := range idsData {
		idsData[i] = int32((i * 100) % 262208)
	}
	indices := FromValues(idsData, 32)
	defer Free(table, indices)
	Materialize(table, indices)
	b.SetBytes(int64(32 * 256 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Take(table, indices, 0)
		Materialize(y)
		Free(y)
	}
}

// --- Standard Embedding.Forward (token embedding lookup) ---

// Gemma 4 input embedding: vocab_size × hidden_size.
// Hidden 1024 (E2B) and 3072 (E4B), vocab 262208.
func BenchmarkEmbedding_Forward_E2B_Decode(b *testing.B) {
	w := RandomUniform(-0.05, 0.05, []int32{262208, 1024}, DTypeFloat32)
	defer Free(w)
	Materialize(w)
	emb := &Embedding{Weight: w}
	indices := FromValues([]int32{42}, 1)
	defer Free(indices)
	Materialize(indices)
	b.SetBytes(int64(1024 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := emb.Forward(indices)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkEmbedding_Forward_E2B_Prefill32(b *testing.B) {
	w := RandomUniform(-0.05, 0.05, []int32{262208, 1024}, DTypeFloat32)
	defer Free(w)
	Materialize(w)
	emb := &Embedding{Weight: w}
	idsData := make([]int32, 32)
	for i := range idsData {
		idsData[i] = int32((i * 100) % 262208)
	}
	indices := FromValues(idsData, 32)
	defer Free(indices)
	Materialize(indices)
	b.SetBytes(int64(32 * 1024 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := emb.Forward(indices)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkEmbedding_Forward_E4B_Decode(b *testing.B) {
	w := RandomUniform(-0.05, 0.05, []int32{262208, 3072}, DTypeFloat32)
	defer Free(w)
	Materialize(w)
	emb := &Embedding{Weight: w}
	indices := FromValues([]int32{42}, 1)
	defer Free(indices)
	Materialize(indices)
	b.SetBytes(int64(3072 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := emb.Forward(indices)
		Materialize(y)
		Free(y)
	}
}

// --- Per-layer input tensor compose (PLE forward integration) ---

// computePerLayerInputs in gemma4.go normalises PLE outputs to the
// expected per-layer shape. This bench measures the slice-and-split
// pattern in isolation: large PLE output → per-layer slice.
//
// Synthetic: a [numLayers × perLayerInput] precomputed PLE tensor and
// a per-layer Take on axis 0.
func BenchmarkPLE_PerLayerSplit_NumLayers26_PerLayer256(b *testing.B) {
	// Precomputed PLE: [numLayers, perLayerInput].
	ple := RandomUniform(-1, 1, []int32{26, 256}, DTypeFloat32)
	defer Free(ple)
	Materialize(ple)
	b.ReportAllocs()
	// Bench the full per-layer split: iterate 26 layers, do 26 Takes.
	for b.Loop() {
		slices := make([]*Array, 26)
		for i := range 26 {
			idx := FromValues([]int32{int32(i)}, 1)
			slices[i] = Take(ple, idx, 0)
			Free(idx)
		}
		Materialize(slices...)
		Free(slices...)
	}
}

func BenchmarkPLE_PerLayerInputViewsSplitAll_Graph(b *testing.B) {
	combined := RandomUniform(-1, 1, []int32{1, 1, 26, 256}, DTypeFloat32)
	defer Free(combined)
	Materialize(combined)
	squeezeAxis2 := []int{2}
	b.ReportAllocs()
	for b.Loop() {
		slices := make([]*Array, 26)
		for i := range slices {
			sliced := SliceAxis(combined, 2, int32(i), int32(i+1))
			slices[i] = Squeeze(sliced, squeezeAxis2...)
			Free(sliced)
		}
		Free(slices...)
	}
}

func BenchmarkPLE_PerLayerInputViewsStreamed_Graph(b *testing.B) {
	combined := RandomUniform(-1, 1, []int32{1, 1, 26, 256}, DTypeFloat32)
	defer Free(combined)
	Materialize(combined)
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			HiddenSizePerLayerInput: 256,
			NumHiddenLayers:         26,
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		for i := int32(0); i < model.Cfg.NumHiddenLayers; i++ {
			slice := model.perLayerInputForLayer(combined, 1, 1, i)
			Free(slice)
		}
	}
}

func BenchmarkPLE_SplitPerLayerInputTensor_Graph(b *testing.B) {
	combined := RandomUniform(-1, 1, []int32{1, 1, 26, 256}, DTypeFloat32)
	defer Free(combined)
	Materialize(combined)
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			HiddenSizePerLayerInput: 256,
			NumHiddenLayers:         26,
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		slices := model.splitPerLayerInputTensor(combined.Clone())
		Free(slices...)
	}
}

// --- Take on alternate axis (rare but exercises strided-take) ---

// Per IDEAS.md, "the specific embedding slice for the current layer
// should be fetched via mlx_take during the forward pass" — typically
// axis 0 (layer dim), but some routing pass slice on axis 1 (per-token
// per-layer feature). Bench both for completeness.
func BenchmarkPLE_Take_Axis1_Slice(b *testing.B) {
	table := RandomUniform(-1, 1, []int32{1024, 26}, DTypeFloat32)
	// Pick layer 15 along axis 1.
	indices := FromValues([]int32{15}, 1)
	defer Free(table, indices)
	Materialize(table, indices)
	b.ReportAllocs()
	for b.Loop() {
		y := Take(table, indices, 1)
		Materialize(y)
		Free(y)
	}
}

// --- Reshape after Take (slice → per-layer tensor shape) ---

// Gemma 4 computePerLayerInputs reshapes the PLE output. Bench the
// Take+Reshape combo to expose any reshape-strided-copy cost.
func BenchmarkPLE_TakePlusReshape(b *testing.B) {
	table := RandomUniform(-1, 1, []int32{26, 256}, DTypeFloat32)
	indices := FromValues([]int32{15}, 1)
	defer Free(table, indices)
	Materialize(table, indices)
	b.ReportAllocs()
	for b.Loop() {
		gathered := Take(table, indices, 0)
		reshaped := Reshape(gathered, 1, 1, 256)
		Materialize(reshaped)
		Free(gathered, reshaped)
	}
}
