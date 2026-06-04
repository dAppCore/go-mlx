// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Model loader registry — architecture id → loader. Replaces the central switch
// in loadModel so loaders are looked up, not hard-coded. RegisterModelLoader is
// exported so future go/model/{family}/{version} packages register themselves
// from init() and import metal one-way (no cycle) — the foundation for moving
// model code out of pkg/metal (go-mlx #45).

import (
	"dappco.re/go"
)

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

// wrapModelLoad adapts a path-only loader into a ModelLoader, wrapping its error
// with the original loadModel op string so behaviour is preserved.
func wrapModelLoad(op string, fn func(string) (InternalModel, error)) ModelLoader {
	return func(modelPath string, _ []byte) (InternalModel, error) {
		model, err := fn(modelPath)
		if err != nil {
			return nil, core.E("model.loadModel", op, err)
		}
		return model, nil
	}
}

// stagedModelLoad adapts a (path, configData)-staged loader, wrapping its error.
func stagedModelLoad(op string, fn func(string, []byte) (InternalModel, error)) ModelLoader {
	return func(modelPath string, configData []byte) (InternalModel, error) {
		model, err := fn(modelPath, configData)
		if err != nil {
			return nil, core.E("model.loadModel", op, err)
		}
		return model, nil
	}
}

func init() {
	// Qwen / Llama style dense family share the Qwen3 loader (direct return).
	qwen3 := func(modelPath string, _ []byte) (InternalModel, error) { return LoadQwen3(modelPath) }
	for _, arch := range []string{"qwen3", "qwen3_next", "qwen2", "llama", "mistral", "hermes", "granite", "phi", "glm"} {
		RegisterModelLoader(arch, qwen3)
	}

	RegisterModelLoader("qwen3_6", stagedModelLoad("validate qwen3_6 native load",
		func(p string, d []byte) (InternalModel, error) { return loadQwen36StagedModel(p, d) }))
	RegisterModelLoader("qwen3_6_moe", stagedModelLoad("validate qwen3_6_moe native load",
		func(p string, d []byte) (InternalModel, error) { return loadQwen36MoEStagedModel(p, d) }))
	RegisterModelLoader("qwen3_moe", wrapModelLoad("load qwen3_moe native model",
		func(p string) (InternalModel, error) { return LoadQwen3MoE(p) }))
	// mixtral self-registers from package metal/model/mixtral init() (cmd blank-import)
	RegisterModelLoader("deepseek", stagedModelLoad("validate deepseek native load",
		func(p string, d []byte) (InternalModel, error) { return loadMoEStagedModel(p, d, "deepseek") }))
	// gpt_oss self-registers from package metal/model/gptoss init() (cmd blank-import)
	// kimi self-registers from package metal/model/kimi init() (cmd blank-import)
	RegisterModelLoader("bert", stagedModelLoad("validate bert native load",
		func(p string, d []byte) (InternalModel, error) { return loadBERTStagedModel(p, d, "bert") }))
	RegisterModelLoader("bert_rerank", stagedModelLoad("validate bert_rerank native load",
		func(p string, d []byte) (InternalModel, error) { return loadBERTStagedModel(p, d, "bert_rerank") }))

	// gemma2 + gemma3 + gemma3_text self-register from package gemma3 init();
	// gemma4_text + gemma4 self-register from package gemma4 init() (cmd blank-import)
	RegisterModelLoader("minimax_m2", stagedModelLoad("validate minimax_m2 native load",
		func(p string, d []byte) (InternalModel, error) { return loadMiniMaxM2StagedModel(p, d) }))
}
