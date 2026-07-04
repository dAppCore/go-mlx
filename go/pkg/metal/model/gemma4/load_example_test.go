// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
)

// ExampleGemma4Model_modelType_multimodal shows how the multimodal model-type
// finaliser stamps a model's surface id. A freshly built model with no config
// resolves to the plain multimodal "gemma4" id.
func Example_finalizeGemma4MultiModalModelType() {
	model := &Gemma4Model{}
	finalizeGemma4MultiModalModelType(model)
	core.Println(model.ModelType())
	// Output: gemma4
}

// Example_gemma4AttnBandFor shows the compiled-decode attention band sizing: the
// band starts at 256 and doubles until it covers the needed key span, then
// clamps to the cache capacity.
func Example_gemma4AttnBandFor() {
	core.Println(gemma4AttnBandFor(10, 4096))   // below floor → 256
	core.Println(gemma4AttnBandFor(257, 4096))  // one doubling → 512
	core.Println(gemma4AttnBandFor(5000, 4096)) // clamped to capacity
	// Output:
	// 256
	// 512
	// 4096
}

// Example_gemma4DecoderLayer_nativeTraceName shows the per-layer native trace
// name a decode step tags its graph with: a zero-padded layer index and the
// phase label.
func Example_gemma4DecoderLayer_nativeTraceName() {
	layer := &Gemma4DecoderLayer{LayerIdx: 7}
	core.Println(layer.nativeTraceName("attn"))
	// Output: gemma4.layer.07.attn
}
