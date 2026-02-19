//go:build darwin && arm64

package metal

import "fmt"

// LoadConfig holds configuration applied during model loading.
type LoadConfig struct {
	ContextLen int // Context window size (0 = model default, unbounded KV cache)
}

// LoadAndInit initialises Metal and loads a model from the given path.
// Returns a *Model ready for generation.
func LoadAndInit(path string, cfg ...LoadConfig) (*Model, error) {
	Init()
	im, err := loadModel(path)
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	m := &Model{
		model:     im,
		tokenizer: im.Tokenizer(),
		modelType: im.ModelType(),
	}
	if len(cfg) > 0 && cfg[0].ContextLen > 0 {
		m.contextLen = cfg[0].ContextLen
	}
	return m, nil
}
