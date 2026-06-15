// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Usage-in-situ for qwen36_moe_staged.go — the staged Qwen 3.6 MoE loader. Like
// its dense sibling it validates a hybrid+sparse config and builds the cache
// plan, but carries no native hybrid linear-attention + sparse-expert decode
// kernels yet. These examples drive the reporting surface on a hand-built
// model; no model load.

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// ModelType is a constant for the staged MoE loader.
func Example_stagedMoEModelType() {
	model := &qwen36MoEStagedModel{}

	core.Println(model.ModelType())
	// Output: qwen3_6_moe
}

// NumLayers reports the configured decoder layer count from the dense config.
func Example_stagedMoENumLayers() {
	cfg := &metal.DenseConfig{}
	cfg.NumHiddenLayers = 48
	model := &qwen36MoEStagedModel{config: cfg}

	core.Println(model.NumLayers())
	// Output: 48
}

// ForwardMasked returns nil: the staged MoE loader has no native decode
// kernels, so the orchestrator must gate on availability before generating.
func Example_stagedMoEForwardMasked() {
	model := &qwen36MoEStagedModel{}

	out := model.ForwardMasked(nil, nil, nil)
	core.Println(out == nil)
	// Output: true
}

// DecodeUnavailableError names the operation and the missing hybrid +
// sparse-expert kernels.
func Example_stagedMoEDecodeUnavailableError() {
	model := &qwen36MoEStagedModel{}

	err := model.DecodeUnavailableError("generate")
	core.Println(err.Error())
	// Output: generate: qwen3_6_moe staged loader has no native hybrid linear-attention and sparse-expert decode kernels yet
}
