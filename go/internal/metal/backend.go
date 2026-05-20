// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

const (
	DefaultLocalContextLen      = 131072
	DefaultLocalParallelSlots   = 1
	DefaultPromptCacheMinTokens = 2048
)

var runtimeMetalAvailable = MetalAvailable

func resolveLoadDevice(device DeviceType) (DeviceType, bool) {
	if device == "" {
		device = DeviceGPU
	}
	return device, false
}

func ensureLoadDeviceAvailable(device DeviceType) error {
	if device == "" {
		device = DeviceGPU
	}
	if !runtimeMetalAvailable() {
		return core.NewError("mlx: no usable Metal device available; refusing native MLX load because CPU fallback can abort this MLX build")
	}
	return nil
}

// LoadConfig holds configuration applied during model loading.
type LoadConfig struct {
	ContextLen           int    // Context window size (0 = local default)
	Gemma4SlidingWindow  int    // Gemma 4 local-attention window cap (0 = model default)
	ParallelSlots        int    // Concurrent inference slots (0 = local default)
	DisablePromptCache   bool   // Disable exact token-prefix prompt cache
	PromptCacheMinTokens int    // Minimum stable prefix tokens before cache reuse
	AdapterPath          string // Path to LoRA adapter directory (empty = no adapter)
	Device               DeviceType
	CachePolicy          string
	KVCacheMode          string
	BatchSize            int
	PrefillChunkSize     int
	ExpectedQuantization int
	MemoryLimitBytes     uint64
	CacheLimitBytes      uint64
	WiredLimitBytes      uint64
}

var (
	setMemoryLimit = SetMemoryLimit
	setCacheLimit  = SetCacheLimit
	setWiredLimit  = SetWiredLimit
)

func applyAllocatorLimits(cfg LoadConfig) {
	if cfg.MemoryLimitBytes > 0 {
		setMemoryLimit(cfg.MemoryLimitBytes)
	}
	if cfg.CacheLimitBytes > 0 {
		setCacheLimit(cfg.CacheLimitBytes)
	}
	if cfg.WiredLimitBytes > 0 {
		setWiredLimit(cfg.WiredLimitBytes)
	}
}

// LoadAndInit initialises Metal and loads a model from the given path.
//
//	m, err := metal.LoadAndInit("/Volumes/Data/lem/gemma-3-1b-it-base")
//	m, err := metal.LoadAndInit(path, metal.LoadConfig{ContextLen: 4096})
func LoadAndInit(path string, cfg ...LoadConfig) (*Model, error) {
	loadCfg := normalizeMetalLoadConfig(LoadConfig{})
	if len(cfg) > 0 {
		loadCfg = normalizeMetalLoadConfig(cfg[0])
	}
	resolvedDevice, fellBack := resolveLoadDevice(loadCfg.Device)
	loadCfg.Device = resolvedDevice
	if fellBack {
		core.Warn("mlx: Metal unavailable, falling back to CPU")
	}
	if err := ensureLoadDeviceAvailable(loadCfg.Device); err != nil {
		return nil, core.E("metal.LoadAndInit", "select device", err)
	}
	applyAllocatorLimits(loadCfg)

	var (
		im         InternalModel
		adapter    *LoRAAdapter
		loadErr    error
		adapterErr error
	)
	if err := withDefaultDevice(loadCfg.Device, func() {
		im, loadErr = loadModel(path)
		if loadErr == nil && loadCfg.AdapterPath != "" {
			adapter, adapterErr = loadLoRAAdapter(im, loadCfg.AdapterPath)
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
	if adapter != nil {
		model.adapter = adapter
		model.adapterInfo = adapterInfoFromLoRA(loadCfg.AdapterPath, adapter)
	}
	applyGemma4SlidingWindow(im, loadCfg.Gemma4SlidingWindow)
	if loadCfg.ContextLen > 0 {
		model.contextLen = loadCfg.ContextLen
	}
	if loadCfg.ParallelSlots > 0 {
		model.parallelSlots = make(chan struct{}, loadCfg.ParallelSlots)
	}
	model.promptCacheEnabled = !loadCfg.DisablePromptCache
	model.promptCacheMinTokens = loadCfg.PromptCacheMinTokens
	model.cachePolicy = loadCfg.CachePolicy
	model.cacheMode = loadCfg.KVCacheMode
	model.batchSizeLimit = loadCfg.BatchSize
	model.prefillChunkSize = loadCfg.PrefillChunkSize
	if loadCfg.ExpectedQuantization > 0 {
		info := model.Info()
		if info.QuantBits > 0 && info.QuantBits != loadCfg.ExpectedQuantization {
			core.Warn("mlx: model quantization differs from memory-plan preference", "model_bits", info.QuantBits, "preferred_bits", loadCfg.ExpectedQuantization)
		}
	}
	return model, nil
}

func applyGemma4SlidingWindow(im InternalModel, window int) {
	if window <= 0 {
		return
	}
	model, ok := im.(*Gemma4Model)
	if !ok || model == nil || model.Cfg == nil {
		return
	}
	if model.Cfg.SlidingWindow <= 0 || model.Cfg.SlidingWindow > int32(window) {
		model.Cfg.SlidingWindow = int32(window)
	}
}

func normalizeMetalLoadConfig(cfg LoadConfig) LoadConfig {
	if cfg.Device == "" {
		cfg.Device = DeviceGPU
	}
	if cfg.ContextLen == 0 {
		cfg.ContextLen = DefaultLocalContextLen
	}
	if cfg.ParallelSlots == 0 {
		cfg.ParallelSlots = DefaultLocalParallelSlots
	}
	if !cfg.DisablePromptCache && cfg.PromptCacheMinTokens == 0 {
		cfg.PromptCacheMinTokens = DefaultPromptCacheMinTokens
	}
	return cfg
}
