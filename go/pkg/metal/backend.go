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
	ContextLen            int    // Context window size (0 = local default)
	ParallelSlots         int    // Concurrent inference slots (0 = local default)
	DisablePromptCache    bool   // Disable exact token-prefix prompt cache
	PromptCacheMinTokens  int    // Minimum stable prefix tokens before cache reuse
	AdapterPath           string // Path to LoRA adapter directory (empty = no adapter)
	Device                DeviceType
	CachePolicy           string
	KVCacheMode           string
	KVCacheStorageDType   string
	PagedKVPageSize       int
	PagedKVPrealloc       bool
	FixedSlidingCacheSize int
	BatchSize             int
	PrefillChunkSize      int
	ExpectedQuantization  int
	MemoryLimitBytes      uint64
	CacheLimitBytes       uint64
	WiredLimitBytes       uint64
}

var (
	setMemoryLimit                   = SetMemoryLimit
	setCacheLimit                    = SetCacheLimit
	setWiredLimit                    = SetWiredLimit
	errMetalTurboQuantKVCachePlanned = core.NewError("mlx: TurboQuant KV cache mode is planned; native TurboQuant cache kernels are not implemented")
)

// minDefaultMLXCacheLimitBytes floors the auto-derived MLX allocator-cache
// limit so a tiny model still keeps a usable buffer-reuse pool.
const minDefaultMLXCacheLimitBytes = 1 << 30 // 1 GiB

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
	if err := validateMetalKVCacheMode(loadCfg.KVCacheMode); err != nil {
		return nil, core.E("metal.LoadAndInit", "cache mode", err)
	}
	if _, ok := parseKVCacheStorageDType(loadCfg.KVCacheStorageDType); !ok && loadCfg.KVCacheStorageDType != "" {
		return nil, core.E("metal.LoadAndInit", "cache storage dtype", core.NewError("unsupported KV cache storage dtype: "+loadCfg.KVCacheStorageDType))
	}
	if loadCfg.PagedKVPageSize < 0 {
		return nil, core.E("metal.LoadAndInit", "paged KV page size", core.NewError("must be >= 0"))
	}
	if loadCfg.FixedSlidingCacheSize < 0 {
		return nil, core.E("metal.LoadAndInit", "fixed Gemma 4 cache size", core.NewError("must be >= 0"))
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
	// Apply the loaded model's declared engine fast-path. This is the single
	// authoritative point every run path (serve, benchmark, tuning) funnels
	// through, so a model runs the kernels it declares without each caller
	// re-deriving them. Inspection paths (InspectLocalPack) don't reach here.
	// The restore is dropped — gates live for the model's process lifetime.
	EngineFeaturesFor(im).Apply()
	// Bound MLX's freed-buffer cache when the caller set no explicit limit.
	// MLX defaults its allocator cache to ~half the device's RAM (≈91 GB on a
	// 192 GB M3 Ultra); under size-diverse prompts — every distinct prompt
	// length allocates transient buffers that are freed to the pool but never
	// reused ("prompts get sent once and never again") — the pool only grows,
	// reaching tens of GB. Short prompts don't reclaim it; only ClearCache does.
	// Cap it to a small multiple of the model's resident weight footprint (read
	// here, post-load, before any generation has perturbed the counter): ample
	// for buffer reuse, never a runaway. An explicit CacheLimitBytes overrides.
	if loadCfg.CacheLimitBytes == 0 {
		if resident := GetActiveMemory(); resident > 0 {
			setCacheLimit(max(2*resident, minDefaultMLXCacheLimitBytes))
		}
	}
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
	model.kvCacheStorageDType = loadCfg.KVCacheStorageDType
	model.pagedKVPageSize = loadCfg.PagedKVPageSize
	model.pagedKVPrealloc = loadCfg.PagedKVPrealloc
	model.fixedSlidingCacheSize = loadCfg.FixedSlidingCacheSize
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

func normalizeMetalLoadConfig(cfg LoadConfig) LoadConfig {
	if cfg.Device == "" {
		cfg.Device = DeviceGPU
	}
	if cfg.ParallelSlots == 0 {
		cfg.ParallelSlots = DefaultLocalParallelSlots
	}
	if !cfg.DisablePromptCache && cfg.PromptCacheMinTokens == 0 {
		cfg.PromptCacheMinTokens = DefaultPromptCacheMinTokens
	}
	cfg.KVCacheStorageDType = normalizeMetalKVCacheStorageDType(cfg.KVCacheStorageDType)
	return cfg
}

func normalizeMetalKVCacheStorageDType(value string) string {
	switch core.Lower(core.Trim(value)) {
	case "", "native", "default":
		return ""
	case "fp16", "float16", "f16":
		return "fp16"
	case "bf16", "bfloat16":
		return "bf16"
	default:
		return core.Trim(value)
	}
}

func validateMetalKVCacheMode(mode string) error {
	switch KVCacheMode(core.Trim(mode)) {
	case KVCacheModeDefault, KVCacheModeFP16, KVCacheModeQ8, KVCacheModeKQ8VQ4, KVCacheModePaged, KVCacheModeFixed, KVCacheModeTurboQuant:
		return nil
	default:
		return core.NewError("mlx: unsupported KV cache mode: " + mode)
	}
}
