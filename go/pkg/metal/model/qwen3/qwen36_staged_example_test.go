// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Usage-in-situ for qwen36_staged.go — the staged Qwen 3.6 dense loader, which
// validates a hybrid config and builds its attention cache plan but has no
// native decode kernels yet ("loadable, not runnable"). These examples drive
// the pure-Go reporting surface on a hand-built staged model; no model load.

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// ModelType is a constant for the staged loader — it always reports the
// canonical qwen3_6 family token regardless of the qwen3_5/qwen3_6 alias the
// config used.
func Example_stagedDenseModelType() {
	model := &qwen36StagedModel{}

	core.Println(model.ModelType())
	// Output: qwen3_6
}

// NumLayers reports the configured decoder layer count.
func Example_stagedDenseNumLayers() {
	model := &qwen36StagedModel{config: qwen36StagedConfig{NumHiddenLayers: 64}}

	core.Println(model.NumLayers())
	// Output: 64
}

// Forward returns nil: the staged loader carries no native hybrid
// linear-attention decode kernels, so the model is loadable for inspection but
// not runnable. The orchestrator gates on this before a decode loop.
func Example_stagedDenseForward() {
	model := &qwen36StagedModel{}

	out := model.Forward(nil, nil)
	core.Println(out == nil)
	// Output: true
}

// DecodeUnavailableError formats the diagnostic the runtime surfaces when a
// caller tries to decode with the staged loader — it names the operation and
// the missing kernels.
func Example_stagedDenseDecodeUnavailableError() {
	model := &qwen36StagedModel{}

	err := model.DecodeUnavailableError("generate")
	core.Println(err.Error())
	// Output: generate: qwen3_6 staged loader has no native hybrid linear-attention decode kernels yet
}

// FillModelInfo copies the staged config metadata into a ModelInfo. When
// max_position_embeddings is absent it falls back to the sliding window for the
// context length.
func Example_stagedDenseFillModelInfo() {
	model := &qwen36StagedModel{config: qwen36StagedConfig{
		VocabSize:     128,
		HiddenSize:    16,
		SlidingWindow: 1024,
	}}

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength)
	// Output: 128 16 1024
}
