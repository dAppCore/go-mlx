// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	// Note: AX-6 - time.Duration is part of the public Metrics API.

	core "dappco.re/go"
	"dappco.re/go/inference"
	coreio "dappco.re/go/io"
	"dappco.re/go/inference/memory"
)

// load_options.go: LoadConfig and its WithX LoadOption functional options —
// context length, slots, prompt cache, quantisation, device, memory plan, KV
// cache policy/mode/dtype, paged/fixed caches, batch/prefill, split inference.

// LoadConfig holds root-package model loading parameters.
type LoadConfig struct {
	ContextLength         int
	ParallelSlots         int
	PromptCache           bool
	PromptCacheMinTokens  int
	Quantization          int
	Device                string
	AdapterPath           string
	Medium                coreio.Medium
	AutoMemoryPlan        bool
	MemoryPlan            *memory.Plan
	CachePolicy           memory.KVCachePolicy
	CacheMode             memory.KVCacheMode
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
	SplitInference        *inference.SplitInferencePlan
	contextLengthExplicit bool
}

// DefaultLoadConfig returns sensible defaults for root-package loading.
func DefaultLoadConfig() LoadConfig {
	return LoadConfig{
		ParallelSlots:        DefaultLocalParallelSlots,
		PromptCache:          true,
		PromptCacheMinTokens: DefaultPromptCacheMinTokens,
		Device:               "gpu",
		AutoMemoryPlan:       true,
	}
}

// LoadOption configures root-package model loading.
type LoadOption func(*LoadConfig)

// WithContextLength bounds the KV cache to the given context window.
func WithContextLength(n int) LoadOption {
	return func(c *LoadConfig) {
		c.ContextLength = n
		c.contextLengthExplicit = n > 0
	}
}

// WithParallelSlots bounds concurrent native inference calls for this model.
// 0 leaves the backend default unchanged.
func WithParallelSlots(n int) LoadOption {
	return func(c *LoadConfig) { c.ParallelSlots = n }
}

// withPromptCacheEnabledOption / withPromptCacheDisabledOption are the two
// package-init singleton closures returned by WithPromptCache. The builder
// only takes a bool so the value space is exhausted by two pre-built
// closures, dropping the per-call alloc to zero and matching the Wave 5
// switch-cached static closure pattern (finite-domain builders return a
// pointer to a pre-existing closure instead of constructing a new one).
var (
	withPromptCacheEnabledOption  LoadOption = func(c *LoadConfig) { c.PromptCache = true }
	withPromptCacheDisabledOption LoadOption = func(c *LoadConfig) { c.PromptCache = false }
)

// WithPromptCache enables or disables exact token-prefix KV caching.
func WithPromptCache(enabled bool) LoadOption {
	if enabled {
		return withPromptCacheEnabledOption
	}
	return withPromptCacheDisabledOption
}

// WithPromptCacheMinTokens sets the minimum prefix length considered cacheable.
func WithPromptCacheMinTokens(n int) LoadOption {
	return func(c *LoadConfig) { c.PromptCacheMinTokens = n }
}

// WithQuantization validates the loaded quantisation width.
func WithQuantization(bits int) LoadOption {
	return func(c *LoadConfig) { c.Quantization = bits }
}

// WithExpectedQuantization tells the native loader which quantisation width the
// planner expects before post-load validation can inspect model metadata.
func WithExpectedQuantization(bits int) LoadOption {
	return func(c *LoadConfig) { c.ExpectedQuantization = bits }
}

// withDeviceGPUOption / withDeviceCPUOption short-cut the two canonical
// device values WithDevice receives in 99% of caller paths. The string
// space is theoretically open (callers can pass any string and have
// normalizeLoadConfig reject it), but the package-level singleton
// closures eliminate the per-call alloc for the two values that actually
// reach this builder — matching the Wave 5 switch-cached static closure
// pattern. The default branch preserves the original semantics for the
// fallback path.
var (
	withDeviceGPUOption LoadOption = func(c *LoadConfig) { c.Device = "gpu" }
	withDeviceCPUOption LoadOption = func(c *LoadConfig) { c.Device = "cpu" }
)

// WithDevice selects the execution device: "gpu" or "cpu".
func WithDevice(device string) LoadOption {
	switch device {
	case "gpu":
		return withDeviceGPUOption
	case "cpu":
		return withDeviceCPUOption
	}
	return func(c *LoadConfig) { c.Device = device }
}

// WithAdapterPath injects a LoRA adapter directory at model load time.
func WithAdapterPath(path string) LoadOption {
	return func(c *LoadConfig) { c.AdapterPath = path }
}

// WithMedium stages model files from the supplied io.Medium before loading.
// The model path passed to LoadModel is interpreted within that medium.
func WithMedium(medium coreio.Medium) LoadOption {
	return func(c *LoadConfig) { c.Medium = medium }
}

// withAutoMemoryPlanEnabledOption / withAutoMemoryPlanDisabledOption are the
// pre-built closures returned by WithAutoMemoryPlan — same switch-cached
// finite-domain pattern as withPromptCacheEnabledOption.
var (
	withAutoMemoryPlanEnabledOption  LoadOption = func(c *LoadConfig) { c.AutoMemoryPlan = true }
	withAutoMemoryPlanDisabledOption LoadOption = func(c *LoadConfig) { c.AutoMemoryPlan = false }
)

// WithAutoMemoryPlan enables or disables measured-device runtime planning.
func WithAutoMemoryPlan(enabled bool) LoadOption {
	if enabled {
		return withAutoMemoryPlanEnabledOption
	}
	return withAutoMemoryPlanDisabledOption
}

// WithMemoryPlan applies an explicit memory plan instead of probing the device.
func WithMemoryPlan(plan memory.Plan) LoadOption {
	return func(c *LoadConfig) {
		cloned := plan
		c.MemoryPlan = &cloned
		c.AutoMemoryPlan = false
	}
}

// withCachePolicy*Option singletons exhaust the memory.KVCachePolicy
// constant set ("", "rotating", "full"). Returning the pre-built closure
// for each known value drops the WithCachePolicy alloc to zero on the
// option-stack hot path — same pattern as withPromptCache*Option.
var (
	withCachePolicyDefaultOption  LoadOption = func(c *LoadConfig) { c.CachePolicy = memory.KVCacheDefault }
	withCachePolicyRotatingOption LoadOption = func(c *LoadConfig) { c.CachePolicy = memory.KVCacheRotating }
	withCachePolicyFullOption     LoadOption = func(c *LoadConfig) { c.CachePolicy = memory.KVCacheFull }
)

// WithCachePolicy selects the KV cache policy used by the native backend.
func WithCachePolicy(policy memory.KVCachePolicy) LoadOption {
	switch policy {
	case memory.KVCacheDefault:
		return withCachePolicyDefaultOption
	case memory.KVCacheRotating:
		return withCachePolicyRotatingOption
	case memory.KVCacheFull:
		return withCachePolicyFullOption
	}
	return func(c *LoadConfig) { c.CachePolicy = policy }
}

// withCacheMode*Option singletons exhaust the memory.KVCacheMode constant
// set ("", "fp16", "q8", "k-q8-v-q4", "paged", "turboquant"). Each known mode returns the
// pre-built closure so WithKVCacheMode allocates nothing on the canonical
// caller paths — same finite-domain pattern as withCachePolicy*Option.
var (
	withCacheModeDefaultOption    LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeDefault }
	withCacheModeFP16Option       LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeFP16 }
	withCacheModeQ8Option         LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeQ8 }
	withCacheModeKQ8VQ4Option     LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeKQ8VQ4 }
	withCacheModePagedOption      LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModePaged }
	withCacheModeTurboQuantOption LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeTurboQuant }
)

// WithKVCacheMode selects the native KV cache storage mode.
func WithKVCacheMode(mode memory.KVCacheMode) LoadOption {
	switch mode {
	case memory.KVCacheModeDefault:
		return withCacheModeDefaultOption
	case memory.KVCacheModeFP16:
		return withCacheModeFP16Option
	case memory.KVCacheModeQ8:
		return withCacheModeQ8Option
	case memory.KVCacheModeKQ8VQ4:
		return withCacheModeKQ8VQ4Option
	case memory.KVCacheModePaged:
		return withCacheModePagedOption
	case memory.KVCacheModeTurboQuant:
		return withCacheModeTurboQuantOption
	}
	return func(c *LoadConfig) { c.CacheMode = mode }
}

// WithKVCacheStorageDType selects the native retained KV storage dtype for
// cache implementations that support typed storage. "" leaves backend-native
// storage.
func WithKVCacheStorageDType(dtype string) LoadOption {
	switch dtype {
	case "", "native", "default":
		return func(c *LoadConfig) { c.KVCacheStorageDType = "" }
	case "fp16", "bf16":
		return func(c *LoadConfig) { c.KVCacheStorageDType = dtype }
	}
	return func(c *LoadConfig) { c.KVCacheStorageDType = dtype }
}

// WithPagedKVPageSize selects the page size for native paged KV caches.
// 0 leaves the backend default.
func WithPagedKVPageSize(n int) LoadOption {
	return func(c *LoadConfig) { c.PagedKVPageSize = n }
}

// WithPagedKVPrealloc selects full-page preallocation for native paged KV
// caches. This is a memory-residency diagnostic option, not a default speed
// path; use only when the lower active+cache footprint is worth the decode cost.
func WithPagedKVPrealloc(enabled bool) LoadOption {
	return func(c *LoadConfig) { c.PagedKVPrealloc = enabled }
}

// WithFixedSlidingCacheSize selects an explicit fixed Gemma 4 KV cache size.
// 0 leaves the backend to derive the size from context or request shape.
func WithFixedSlidingCacheSize(n int) LoadOption {
	return func(c *LoadConfig) { c.FixedSlidingCacheSize = n }
}

// WithBatchSize sets the planner batch shape for native batched generation.
func WithBatchSize(n int) LoadOption {
	return func(c *LoadConfig) { c.BatchSize = n }
}

// WithPrefillChunkSize bounds long prompt prefill passes into token chunks.
func WithPrefillChunkSize(n int) LoadOption {
	return func(c *LoadConfig) { c.PrefillChunkSize = n }
}

// WithAllocatorLimits applies Metal allocator limits in bytes.
func WithAllocatorLimits(memory, cache, wired uint64) LoadOption {
	return func(c *LoadConfig) {
		c.MemoryLimitBytes = memory
		c.CacheLimitBytes = cache
		c.WiredLimitBytes = wired
	}
}

// WithSplitInference attaches a validated split-inference plan to the load
// request. Remote execution is still planned; local plans are accepted so UIs
// can persist the same shape before backend execution lands.
func WithSplitInference(plan inference.SplitInferencePlan) LoadOption {
	return func(c *LoadConfig) {
		c.SplitInference = cloneSplitInferencePlan(plan)
	}
}

func applyLoadOptions(opts []LoadOption) LoadConfig {
	cfg := DefaultLoadConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// normalizeLoadConfig validation errors hoisted to package vars — the
// failure paths are rare in callers but each core.NewError() allocates
// a fresh error value; reusing a single instance per message keeps the
// rare path alloc-free and preserves errors.Is comparability.
var (
	errMlxContextLengthNegative    = core.NewError("mlx: context length must be >= 0")
	errMlxParallelSlotsNegative    = core.NewError("mlx: parallel slots must be >= 0")
	errMlxPromptCacheMinTokensNeg  = core.NewError("mlx: prompt cache minimum tokens must be >= 0")
	errMlxQuantizationNegative     = core.NewError("mlx: quantization bits must be >= 0")
	errMlxBatchSizeNegative        = core.NewError("mlx: batch size must be >= 0")
	errMlxPrefillChunkSizeNegative = core.NewError("mlx: prefill chunk size must be >= 0")
	errMlxExpectedQuantizationNeg  = core.NewError("mlx: expected quantization bits must be >= 0")
	errMlxSplitInferenceRemotePlan = core.NewError("mlx: split inference execution is planned; remote FFN/expert execution is not wired yet")
)

func normalizeLoadConfig(cfg LoadConfig) (LoadConfig, error) {
	if cfg.ContextLength < 0 {
		return LoadConfig{}, errMlxContextLengthNegative
	}
	if cfg.ParallelSlots < 0 {
		return LoadConfig{}, errMlxParallelSlotsNegative
	}
	if cfg.PromptCacheMinTokens < 0 {
		return LoadConfig{}, errMlxPromptCacheMinTokensNeg
	}
	if cfg.PromptCache && cfg.PromptCacheMinTokens == 0 {
		cfg.PromptCacheMinTokens = DefaultPromptCacheMinTokens
	}
	if cfg.Quantization < 0 {
		return LoadConfig{}, errMlxQuantizationNegative
	}
	if cfg.BatchSize < 0 {
		return LoadConfig{}, errMlxBatchSizeNegative
	}
	if cfg.PrefillChunkSize < 0 {
		return LoadConfig{}, errMlxPrefillChunkSizeNegative
	}
	if cfg.ExpectedQuantization < 0 {
		return LoadConfig{}, errMlxExpectedQuantizationNeg
	}
	if cfg.PagedKVPageSize < 0 {
		return LoadConfig{}, core.NewError("mlx: paged KV page size must be >= 0")
	}
	if cfg.FixedSlidingCacheSize < 0 {
		return LoadConfig{}, core.NewError("mlx: fixed Gemma 4 cache size must be >= 0")
	}
	if cfg.SplitInference != nil {
		if err := inference.ValidateSplitInferencePlan(*cfg.SplitInference); err != nil {
			return LoadConfig{}, err
		}
		mode := cfg.SplitInference.Mode
		if mode == "" {
			mode = inference.SplitInferenceModeLocal
		}
		if mode != inference.SplitInferenceModeLocal {
			return LoadConfig{}, errMlxSplitInferenceRemotePlan
		}
	}
	if !memory.IsKnownKVCacheMode(cfg.CacheMode) {
		return LoadConfig{}, core.NewError("mlx: unsupported KV cache mode: " + string(cfg.CacheMode))
	}
	cfg.KVCacheStorageDType = normalizeKVCacheStorageDType(cfg.KVCacheStorageDType)
	if cfg.KVCacheStorageDType == "unsupported" {
		return LoadConfig{}, core.NewError("mlx: unsupported KV cache storage dtype")
	}

	// Fast-path the canonical "", "gpu", "cpu" values that the default
	// LoadConfig and almost every caller provide. core.Lower/Trim each
	// walk the string and Trim allocates a fresh substring for any
	// whitespace input, which dominates a 90%-clean hot path. Skip both
	// scans when the input is already canonical and only fall through
	// to the normalising slow path when the device string actually
	// needs work.
	switch cfg.Device {
	case "gpu", "cpu":
		return cfg, nil
	case "":
		cfg.Device = "gpu"
		return cfg, nil
	}
	device := core.Lower(core.Trim(cfg.Device))
	if device == "" {
		device = "gpu"
	}
	switch device {
	case "gpu", "cpu":
		cfg.Device = device
		return cfg, nil
	default:
		return LoadConfig{}, core.NewError("mlx: unsupported device: " + device)
	}
}

func normalizeKVCacheStorageDType(dtype string) string {
	switch core.Lower(core.Trim(dtype)) {
	case "", "native", "default":
		return ""
	case "fp16", "float16", "f16":
		return "fp16"
	case "bf16", "bfloat16":
		return "bf16"
	default:
		return "unsupported"
	}
}

func cloneSplitInferencePlan(plan inference.SplitInferencePlan) *inference.SplitInferencePlan {
	// plan is already a value-copy taken on parameter receive — mutating
	// its slice/map fields in place builds the cloned shape without the
	// extra `cloned := plan` struct-copy the prior form paid. Returning
	// &plan escapes the parameter to heap, replacing the two-copy
	// (parameter + cloned local) pattern with one heap-allocated value.
	//
	// core.SliceClone still short-circuits to nil for nil-input slices,
	// keeping the typical "Components present, Notes empty" plan shape
	// alloc-light for the slice/map sub-fields.
	plan.LocalSlice.Components = core.SliceClone(plan.LocalSlice.Components)
	plan.LocalSlice.Notes = core.SliceClone(plan.LocalSlice.Notes)
	plan.LocalSlice.Labels = cloneInferenceLabels(plan.LocalSlice.Labels)
	plan.Endpoints = cloneInferenceSplitEndpoints(plan.Endpoints)
	plan.Labels = cloneInferenceLabels(plan.Labels)
	return &plan
}
