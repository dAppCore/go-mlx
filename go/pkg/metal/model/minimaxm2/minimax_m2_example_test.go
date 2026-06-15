// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package minimaxm2

// Package-level examples. The MiniMax-M2 staged loader exposes no exported
// identifiers (the model type, loader and helpers are all package-private), so
// these use the Go whole-package Example_<suffix> form rather than the
// Example<Ident>/ExampleType_Method form, which the testing tool validates
// against exported names.

import (
	core "dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
)

// Example_validateNativeLoad shows the cheap, header-only validation entry: it
// parses config.json plus the safetensors headers and the optional
// jang_config.json, then returns a one-line JANGTQ plan summary without loading
// any weights — safe to run against a multi-hundred-GB pack. The model
// directory must hold config.json and at least one *.safetensors shard.
func Example_validateNativeLoad() {
	summary, err := validateMiniMaxM2NativeLoad("/path/to/minimax-m2", []byte(`{"model_type":"minimax_m2"}`))
	_, _ = summary, err
}

// Example_forward demonstrates the staged-model forward signature. MiniMax-M2
// has no native decode kernels yet, so the staged loader's Forward returns nil;
// DecodeUnavailableError is the surfaced diagnostic. Compile-only: it shows the
// call shape against a nil model.
func Example_forward() {
	var (
		model  *miniMaxM2StagedModel
		tokens *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.Forward(tokens, caches)
	_ = logits
}

// Example_forwardMasked is the explicit-mask variant of Forward; like Forward it
// returns nil on the staged loader until native MiniMax decode kernels are
// linked.
func Example_forwardMasked() {
	var (
		model  *miniMaxM2StagedModel
		tokens *metal.Array
		mask   *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.ForwardMasked(tokens, mask, caches)
	_ = logits
}

// Example_applyLoRA shows the LoRA-attach signature. The staged loader holds no
// live projections, so ApplyLoRA returns nil until decode kernels exist; the
// call shape matches the metal.InternalModel contract.
func Example_applyLoRA() {
	var model *miniMaxM2StagedModel
	if model == nil {
		return
	}
	adapter := model.ApplyLoRA(metal.LoRAConfig{Rank: 2, Alpha: 4})
	_ = adapter
}

// Example_numLayers reports the decoder-layer count taken straight from the
// parsed config.
func Example_numLayers() {
	model := &miniMaxM2StagedModel{
		plan: miniMaxM2NativeLoadPlan{Config: miniMaxM2LoadConfig{NumHiddenLayers: 62}},
	}
	core.Println(model.NumLayers())
	// Output: 62
}

// Example_modelType returns the canonical architecture token used by the loader
// registry and diagnostics.
func Example_modelType() {
	model := &miniMaxM2StagedModel{}
	core.Println(model.ModelType())
	// Output: minimax_m2
}

// Example_newCache shows that the staged loader allocates no KV caches yet —
// decode kernels are not linked, so Forward/decode are unavailable.
func Example_newCache() {
	model := &miniMaxM2StagedModel{}
	core.Println(len(model.NewCache()))
	// Output: 0
}

// Example_decodeUnavailableError is the diagnostic the staged loader returns
// from the decode path: it names the attempted operation and explains that
// native MiniMax decode kernels are not yet linked.
func Example_decodeUnavailableError() {
	model := &miniMaxM2StagedModel{}
	core.Println(model.DecodeUnavailableError("generate").Error())
	// Output: generate: minimax_m2 staged loader has no native decode kernels yet
}

// Example_routerProject shows the dense router projection: each token's hidden
// vector is scored against every expert row (plus the optional per-expert
// correction bias). Here expert 0 reads x0 and expert 1 reads 2*x1, with biases
// +0.5 and -1.
func Example_routerProject() {
	router := miniMaxM2NativeRouterWeights{
		NumExperts: 2,
		HiddenSize: 2,
		Weight:     []float32{1, 0, 0, 2},
		Bias:       []float32{0.5, -1},
	}
	scores, _ := router.Project([][]float32{{3, 4}})
	core.Println(scores[0][0], scores[0][1])
	// Output: 3.5 7
}
