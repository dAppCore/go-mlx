//go:build darwin && arm64

package metal

import "fmt"

// LoadAndInit initialises Metal and loads a model from the given path.
// Returns a *Model ready for generation.
func LoadAndInit(path string) (*Model, error) {
	Init()
	im, err := loadModel(path)
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	return &Model{
		model:     im,
		tokenizer: im.Tokenizer(),
		modelType: im.ModelType(),
	}, nil
}
