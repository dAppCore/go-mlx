// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Usage-in-situ for qwen3_moe.go — the sparse Qwen3-MoE model the loader builds
// from a config carrying num_experts / num_experts_per_tok / moe_intermediate.
// These examples drive the pure-Go surface (layer classification, the decode
// family token, model-info population, layer count) on hand-built structs — no
// model load, the AX-11-clean fixture style.

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// MoETextDecodeFamily returns the canonical family token the runtime uses in
// "decode kernels unavailable" diagnostics — a constant for this model.
func ExampleQwen3MoEModel_MoETextDecodeFamily() {
	model := &Qwen3MoEModel{}

	core.Println(model.MoETextDecodeFamily())
	// Output: qwen3_moe
}

// MoETextRuntimeAvailable reports false for an empty model: with no layers,
// none of the per-layer selected-expert decode kernels are linked.
func ExampleQwen3MoEModel_MoETextRuntimeAvailable() {
	model := &Qwen3MoEModel{}

	core.Println(model.MoETextRuntimeAvailable())
	// Output: false
}

// NumLayers reports the decoder layer count — the length of the Layers slice
// the loader sizes to num_hidden_layers.
func ExampleQwen3MoEModel_NumLayers() {
	model := &Qwen3MoEModel{
		Layers: []*Qwen3MoEDecoderLayer{nil, nil, nil},
	}

	core.Println(model.NumLayers())
	// Output: 3
}

// ModelType returns the architecture identifier resolved during load.
func ExampleQwen3MoEModel_ModelType() {
	model := &Qwen3MoEModel{modelType: "qwen3_moe"}

	core.Println(model.ModelType())
	// Output: qwen3_moe
}

// FillModelInfo copies the sparse config metadata — architecture, vocab, layer
// and head counts, hidden size, context length — into a ModelInfo for the API
// surface.
func ExampleQwen3MoEModel_FillModelInfo() {
	cfg := &metal.DenseConfig{}
	cfg.VocabSize = 1000
	cfg.NumHiddenLayers = 2
	cfg.NumAttentionHeads = 8
	cfg.HiddenSize = 1024
	cfg.MaxPositionEmbeddings = 32768
	model := &Qwen3MoEModel{Cfg: cfg, modelType: "qwen3_moe"}

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	core.Println(info.Architecture, info.VocabSize, info.NumLayers, info.NumHeads, info.HiddenSize, info.ContextLength)
	// Output: qwen3_moe 1000 2 8 1024 32768
}

// qwen3MoELayerMask marks which decoder layers are sparse (MoE) vs dense. With
// decoder_sparse_step == 0 every layer after the first is sparse — the common
// "all but layer 0" Qwen3-MoE pattern.
func Example_qwen3MoELayerMask() {
	cfg := &metal.DenseConfig{}
	cfg.NumHiddenLayers = 4
	cfg.DecoderSparseStep = 0

	core.Println(qwen3MoELayerMask(cfg))
	// Output: [false true true true]
}
