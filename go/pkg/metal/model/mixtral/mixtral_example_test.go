// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import (
	core "dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
)

// ExampleLoadMixtral loads a Mixtral checkpoint from a model directory holding
// config.json, tokenizer.json and the safetensors weights. The returned model
// satisfies metal.InternalModel and drives the generate loop. A missing path
// surfaces as an error rather than a panic.
func ExampleLoadMixtral() {
	model, err := LoadMixtral("/path/to/mixtral")
	_, _ = model, err
}

// Example_mixtralMoELayerMask shows the dense/MoE interleave the loader builds:
// decoder_sparse_step=N marks every Nth layer (index i where i%N == N-1) as MoE
// and the rest dense. SparseStep<=0 makes every layer MoE.
func Example_mixtralMoELayerMask() {
	stepped := mixtralMoELayerMask(&MixtralConfig{NumHiddenLayers: 4, SparseStep: 2})
	allMoE := mixtralMoELayerMask(&MixtralConfig{NumHiddenLayers: 3, SparseStep: 0})
	core.Println(stepped, allMoE)
	// Output: [false true false true] [true true true]
}

// ExampleMixtralModel_NewCache allocates one KV cache per decoder layer — the
// caches slice Forward/ForwardMasked expect. The layer entries may be nil; only
// the layer count drives the cache length.
func ExampleMixtralModel_NewCache() {
	model := &MixtralModel{
		Layers: []*MixtralDecoderLayer{
			nil,
			nil,
		},
	}

	caches := model.NewCache()

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.KVCache *metal.KVCache
}

// ExampleMixtralModel_NumLayers reports the decoder depth — the length of the
// Layers slice, which also fixes the NewCache length.
func ExampleMixtralModel_NumLayers() {
	model := &MixtralModel{Layers: make([]*MixtralDecoderLayer, 3)}
	core.Println(model.NumLayers())
	// Output: 3
}

// ExampleMixtralModel_ModelType returns the architecture id the loader stamps on
// the model ("mixtral").
func ExampleMixtralModel_ModelType() {
	model := &MixtralModel{modelType: "mixtral"}
	core.Println(model.ModelType())
	// Output: mixtral
}

// ExampleMixtralModel_MoETextDecodeFamily returns the canonical family token used
// in native-runtime "unavailable" diagnostics.
func ExampleMixtralModel_MoETextDecodeFamily() {
	core.Println((&MixtralModel{}).MoETextDecodeFamily())
	// Output: mixtral
}

// ExampleMixtralModel_FillModelInfo copies vocab/hidden/context sizing (and
// quantization, when present) out of the parsed config into a ModelInfo for the
// generate-side reporter.
func ExampleMixtralModel_FillModelInfo() {
	model := &MixtralModel{Cfg: &MixtralConfig{
		VocabSize:             32000,
		HiddenSize:            4096,
		MaxPositionEmbeddings: 32768,
	}}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength)
	// Output: 32000 4096 32768
}

// ExampleMixtralModel_ApplyLoRA authors a LoRA adapter over the requested target
// projections. With no layers built, the adapter carries the normalised config
// (Scale*Rank → Alpha) but zero adapted layers.
func ExampleMixtralModel_ApplyLoRA() {
	model := &MixtralModel{}
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Scale:        4,
		TargetLayers: []string{"q_proj"},
	})

	core.Println(adapter.Config.TargetKeys, adapter.Config.Rank, adapter.Config.Alpha, adapter.Config.Scale, len(adapter.Layers))
	// Output: [q_proj] 2 8 4 0
}
