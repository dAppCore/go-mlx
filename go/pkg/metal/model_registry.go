// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Model loader registry — architecture id → loader. Replaces the central switch
// in loadModel so loaders are looked up, not hard-coded. RegisterModelLoader is
// exported so future go/model/{family}/{version} packages register themselves
// from init() and import metal one-way (no cycle) — the foundation for moving
// model code out of pkg/metal (go-mlx #45).

// ModelLoader builds an InternalModel from a checkpoint path. configData is the
// raw config.json bytes — staged loaders read shape/quant from it; the rest
// ignore it.
type ModelLoader func(modelPath string, configData []byte) (InternalModel, error)

var modelLoaders = map[string]ModelLoader{}

// RegisterModelLoader registers fn for an architecture id; a later call for the
// same id overrides. Call from init() so model packages register themselves and
// the loader dispatch needs no central switch. No-op on empty arch or nil fn.
func RegisterModelLoader(arch string, fn ModelLoader) {
	if arch == "" || fn == nil {
		return
	}
	modelLoaders[arch] = fn
}

// lookupModelLoader returns the loader for arch, or nil if none is registered.
func lookupModelLoader(arch string) ModelLoader {
	return modelLoaders[arch]
}

func init() {
	// Qwen / Llama style dense family self-registers from package
	// metal/model/qwen3 init() (cmd blank-import)

	// qwen3_6 + qwen3_6_moe self-register from package
	// metal/model/qwen3 init() (cmd blank-import)
	// mixtral self-registers from package metal/model/mixtral init() (cmd blank-import)
	// deepseek self-registers from package metal/model/deepseek init() (cmd blank-import)
	// gpt_oss self-registers from package metal/model/gptoss init() (cmd blank-import)
	// kimi self-registers from package metal/model/kimi init() (cmd blank-import)
	// qwen3_moe self-registers from package metal/model/qwen3 init() (cmd blank-import)
	// bert + bert_rerank self-register from package metal/model/bert init() (cmd blank-import)

	// gemma2 + gemma3 + gemma3_text self-register from package gemma3 init();
	// gemma4_text + gemma4 + gemma4_unified self-register from package
	// gemma4 init() (cmd blank-import). gemma4_unified_text is nested
	// text_config metadata and normalises to gemma4_text, not a load target.
	// minimax_m2 self-registers from package metal/model/minimaxm2 init() (cmd blank-import)
}
