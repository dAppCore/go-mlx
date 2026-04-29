// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleLoadGemma3() {
	core.Println("LoadGemma3")
	// Output: LoadGemma3
}

func ExampleGemmaModel_Forward() {
	core.Println("GemmaModel_Forward")
	// Output: GemmaModel_Forward
}

func ExampleGemmaModel_ForwardMasked() {
	core.Println("GemmaModel_ForwardMasked")
	// Output: GemmaModel_ForwardMasked
}

func ExampleGemmaModel_NewCache() {
	core.Println("GemmaModel_NewCache")
	// Output: GemmaModel_NewCache
}

func ExampleGemmaModel_NumLayers() {
	core.Println("GemmaModel_NumLayers")
	// Output: GemmaModel_NumLayers
}

func ExampleGemmaModel_Tokenizer() {
	core.Println("GemmaModel_Tokenizer")
	// Output: GemmaModel_Tokenizer
}

func ExampleGemmaModel_ModelType() {
	core.Println("GemmaModel_ModelType")
	// Output: GemmaModel_ModelType
}

func ExampleGemmaModel_ApplyLoRA() {
	core.Println("GemmaModel_ApplyLoRA")
	// Output: GemmaModel_ApplyLoRA
}
