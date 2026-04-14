//go:build darwin && arm64

package metal

import "dappco.re/go/core"

// LoadConfig holds configuration applied during model loading.
type LoadConfig struct {
	ContextLen  int    // Context window size (0 = model default, unbounded KV cache)
	AdapterPath string // Path to LoRA adapter directory (empty = no adapter)
	Device      DeviceType
}

// LoadAndInit initialises Metal and loads a model from the given path.
//
//	m, err := metal.LoadAndInit("/Volumes/Data/lem/gemma-3-1b-it-base")
//	m, err := metal.LoadAndInit(path, metal.LoadConfig{ContextLen: 4096})
func LoadAndInit(path string, cfg ...LoadConfig) (*Model, error) {
	if !MetalAvailable() {
		return nil, core.E("metal.LoadAndInit", "Metal unavailable", nil)
	}
	loadCfg := LoadConfig{Device: DeviceGPU}
	if len(cfg) > 0 {
		loadCfg = cfg[0]
		if loadCfg.Device == "" {
			loadCfg.Device = DeviceGPU
		}
	}

	var (
		im         InternalModel
		loadErr    error
		adapterErr error
	)
	if err := withDefaultDevice(loadCfg.Device, func() {
		im, loadErr = loadModel(path)
		if loadErr == nil && loadCfg.AdapterPath != "" {
			adapterErr = applyLoadedLoRA(im, loadCfg.AdapterPath)
		}
	}); err != nil {
		return nil, core.E("metal.LoadAndInit", "select device", err)
	}
	if loadErr != nil {
		return nil, core.E("metal.LoadAndInit", "load model", loadErr)
	}
	if adapterErr != nil {
		return nil, core.E("metal.LoadAndInit", "load adapter", adapterErr)
	}

	model := &Model{
		model:     im,
		tokenizer: im.Tokenizer(),
		modelType: im.ModelType(),
		device:    loadCfg.Device,
	}
	if loadCfg.ContextLen > 0 {
		model.contextLen = loadCfg.ContextLen
	}
	return model, nil
}
