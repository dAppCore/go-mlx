//go:build darwin && arm64

package metal

import "dappco.re/go/core"

// LoadConfig holds configuration applied during model loading.
type LoadConfig struct {
	ContextLen  int    // Context window size (0 = model default, unbounded KV cache)
	AdapterPath string // Path to LoRA adapter directory (empty = no adapter)
}

// LoadAndInit initialises Metal and loads a model from the given path.
//
//	m, err := metal.LoadAndInit("/Volumes/Data/lem/gemma-3-1b-it-base")
//	m, err := metal.LoadAndInit(path, metal.LoadConfig{ContextLen: 4096})
func LoadAndInit(path string, cfg ...LoadConfig) (*Model, error) {
	Init()
	im, err := loadModel(path)
	if err != nil {
		return nil, core.E("metal.LoadAndInit", "load model", err)
	}
	model := &Model{
		model:     im,
		tokenizer: im.Tokenizer(),
		modelType: im.ModelType(),
	}
	if len(cfg) > 0 {
		if cfg[0].ContextLen > 0 {
			model.contextLen = cfg[0].ContextLen
		}
		if cfg[0].AdapterPath != "" {
			if err := applyLoadedLoRA(im, cfg[0].AdapterPath); err != nil {
				return nil, core.E("metal.LoadAndInit", "load adapter", err)
			}
		}
	}
	return model, nil
}
