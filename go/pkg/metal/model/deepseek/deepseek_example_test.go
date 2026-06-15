// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deepseek

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// ExampleStagedModel_ModelType returns the canonical architecture token the
// loader registry dispatched on. The DeepSeek staged loader normalises every
// recognised variant (deepseek_v2/v3, the *ForCausalLM heads) to "deepseek".
func ExampleStagedModel_ModelType() {
	model := &StagedModel{}

	core.Println(model.ModelType())
	// Output: deepseek
}

// ExampleStagedModel_NumLayers reports the decoder-layer count read from
// num_hidden_layers; the staged loader carries it as metadata even though the
// native sparse-expert decode kernels have not landed.
func ExampleStagedModel_NumLayers() {
	model := &StagedModel{config: stagedConfig{NumHiddenLayers: 2}}

	core.Println(model.NumLayers())
	// Output: 2
}

// ExampleStagedModel_NewCache documents the staged loader's no-KV contract:
// without native decode kernels there is nothing to cache, so NewCache returns
// nil — Forward/ForwardMasked are inert until the MoE kernels land.
func ExampleStagedModel_NewCache() {
	model := &StagedModel{}

	core.Println(model.NewCache() == nil)
	// Output: true
}

// ExampleStagedModel_FillModelInfo copies the config's vocab, hidden width,
// context window, and quantisation parameters into the shared metal.ModelInfo
// the runtime reports to callers.
func ExampleStagedModel_FillModelInfo() {
	model := &StagedModel{config: stagedConfig{
		VocabSize:             32000,
		HiddenSize:            1024,
		MaxPositionEmbeddings: 8192,
		Quantization:          metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	info := metal.ModelInfo{}
	model.FillModelInfo(&info)

	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength, info.QuantBits, info.QuantGroup)
	// Output: 32000 1024 8192 4 64
}

// ExampleStagedModel_DecodeUnavailableError returns the diagnostic callers see
// when they ask the staged loader for native decode — the sparse-expert kernels
// are not landed, so the message names the operation and the missing capability.
func ExampleStagedModel_DecodeUnavailableError() {
	model := &StagedModel{}

	core.Println(model.DecodeUnavailableError("generate").Error())
	// Output: generate: deepseek staged loader has no native sparse-expert decode kernels yet
}

// Example_mlaPlan derives the multi-head latent-attention plan from the config.
// When qk_head_dim is omitted the loader infers it from the no-rope/rope split,
// so a config carrying only the two split dims still produces a complete plan.
func Example_mlaPlan() {
	cfg := stagedConfig{
		KVLoRARank:    512,
		QKNoPEHeadDim: 128,
		QKRoPEHeadDim: 64,
		VHeadDim:      128,
	}
	plan, err := cfg.deepSeekMLAPlan()
	if err != nil {
		core.Println("plan error:", err)
		return
	}

	core.Println(plan.QKHeadDim, plan.KVLoRARank, plan.VHeadDim)
	// Output: 192 512 128
}

// Example_expertCount reports the routed-expert count, reading the first
// populated of num_experts, num_local_experts, n_routed_experts — the three
// fields DeepSeek variants spell the same quantity with.
func Example_expertCount() {
	core.Println(stagedConfig{NRoutedExperts: 64}.expertCount())
	core.Println(stagedConfig{NumLocalExperts: 8}.expertCount())
	// Output:
	// 64
	// 8
}

// Example_architectureNameMapping maps a Hugging Face architecture list to the
// loader registry id: any DeepSeek* architecture resolves to "deepseek", an
// unrelated architecture returns the empty string so the caller falls back to
// the config's model_type field.
func Example_architectureNameMapping() {
	core.Println(firstDeepSeekArchitectureName([]string{"DeepseekV3ForCausalLM"}))
	core.Println(firstDeepSeekArchitectureName([]string{"LlamaForCausalLM"}) == "")
	// Output:
	// deepseek
	// true
}
