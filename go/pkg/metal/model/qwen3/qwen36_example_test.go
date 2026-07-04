// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Usage-in-situ for qwen36.go — the Qwen 3.5/3.6 family detection the dense and
// MoE loaders consult before attempting a native load. These examples are pure
// classification over config values; no model is involved.

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// isQwen36HybridConfig reports whether a dense config is a Qwen 3.6 hybrid the
// native Go loader does not implement — here detected by the model_type, the
// path the loaders' guard takes first.
func Example_isQwen36HybridConfig() {
	cfg := &metal.DenseConfig{}
	cfg.ModelType = "qwen3_6"

	core.Println(isQwen36HybridConfig(cfg))
	// Output: true
}

// A plain dense Qwen 3 config is not hybrid — full_attention layers, no
// fractional partial-rotary factor — so the native dense loader proceeds.
func Example_isQwen36HybridConfig_dense() {
	cfg := &metal.DenseConfig{LayerTypes: []string{"full_attention"}}
	cfg.ModelType = "qwen3"

	core.Println(isQwen36HybridConfig(cfg))
	// Output: false
}

// qwen36NativeGuardMessage renders the diagnostic the loaders return for an
// unsupported hybrid config. The MoE variant additionally names sparse expert
// routing.
func Example_qwen36NativeGuardMessage() {
	core.Println(qwen36NativeGuardMessage("qwen3_6_moe"))
	// Output: qwen3_6_moe hybrid linear attention and sparse expert routing are not implemented in the native Go loader yet
}
