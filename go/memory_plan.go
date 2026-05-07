// SPDX-Licence-Identifier: EUPL-1.2

package mlx

const MemoryGiB uint64 = 1 << 30

// MemoryClass names the local Apple memory tier driving runtime policy.
type MemoryClass string

const (
	MemoryClassUnknown    MemoryClass = "unknown"
	MemoryClassApple16GB  MemoryClass = "apple-silicon-16gb"
	MemoryClassApple24GB  MemoryClass = "apple-silicon-24gb"
	MemoryClassApple32GB  MemoryClass = "apple-silicon-32gb"
	MemoryClassApple64GB  MemoryClass = "apple-silicon-64gb"
	MemoryClassApple96GB  MemoryClass = "apple-silicon-96gb"
	MemoryClassApple128GB MemoryClass = "apple-silicon-128gb-plus"
)

// KVCachePolicy names the cache shape selected by the planner.
type KVCachePolicy string

const (
	KVCacheDefault  KVCachePolicy = ""
	KVCacheRotating KVCachePolicy = "rotating"
	KVCacheFull     KVCachePolicy = "full"
)

// KVCacheMode names the physical KV storage strategy used by the native cache.
type KVCacheMode string

const (
	KVCacheModeDefault KVCacheMode = ""
	KVCacheModeFP16    KVCacheMode = "fp16"
	KVCacheModeQ8      KVCacheMode = "q8"
	KVCacheModeKQ8VQ4  KVCacheMode = "k-q8-v-q4"
	KVCacheModePaged   KVCacheMode = "paged"
)

// MemoryPlanInput supplies measured hardware and optional model metadata.
type MemoryPlanInput struct {
	Device    DeviceInfo
	Pack      *ModelPack
	ModelInfo *ModelInfo
}

// MemoryPlan is the local runtime policy derived from measured device memory.
type MemoryPlan struct {
	MachineClass               MemoryClass   `json:"machine_class"`
	Architecture               string        `json:"architecture,omitempty"`
	DeviceMemoryBytes          uint64        `json:"device_memory_bytes,omitempty"`
	RecommendedWorkingSetBytes uint64        `json:"recommended_working_set_bytes,omitempty"`
	ContextLength              int           `json:"context_length"`
	CachePolicy                KVCachePolicy `json:"cache_policy"`
	CacheMode                  KVCacheMode   `json:"cache_mode,omitempty"`
	BatchSize                  int           `json:"batch_size"`
	PrefillChunkSize           int           `json:"prefill_chunk_size"`
	ParallelSlots              int           `json:"parallel_slots"`
	PromptCache                bool          `json:"prompt_cache"`
	PromptCacheMinTokens       int           `json:"prompt_cache_min_tokens"`
	PreferredQuantization      int           `json:"preferred_quantization,omitempty"`
	ModelQuantization          int           `json:"model_quantization,omitempty"`
	ModelQuantizationType      string        `json:"model_quantization_type,omitempty"`
	ModelQuantizationFamily    string        `json:"model_quantization_family,omitempty"`
	MemoryLimitBytes           uint64        `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes            uint64        `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes            uint64        `json:"wired_limit_bytes,omitempty"`
	EstimatedKVCacheBytes      uint64        `json:"estimated_kv_cache_bytes,omitempty"`
	EstimatedKVCacheModeBytes  uint64        `json:"estimated_kv_cache_mode_bytes,omitempty"`
	KVCacheSavingsRatio        float64       `json:"kv_cache_savings_ratio,omitempty"`
	Notes                      []string      `json:"notes,omitempty"`
}

// PlanMemory chooses opinionated local inference settings from measured memory.
func PlanMemory(input MemoryPlanInput) MemoryPlan {
	deviceMemory := input.Device.MemorySize
	workingSet := input.Device.MaxRecommendedWorkingSetSize
	if workingSet == 0 {
		workingSet = deviceMemory
	}
	class := memoryClassForBytes(deviceMemory)
	plan := baseMemoryPlan(class)
	plan.MachineClass = class
	plan.Architecture = input.Device.Architecture
	plan.DeviceMemoryBytes = deviceMemory
	plan.RecommendedWorkingSetBytes = workingSet
	plan.MemoryLimitBytes = percentBytes(workingSet, 85)
	plan.CacheLimitBytes = percentBytes(workingSet, 8)
	plan.WiredLimitBytes = percentBytes(workingSet, 75)

	modelContext, modelQuant, modelQuantType, modelQuantFamily, modelArchitecture := modelMemoryHints(input)
	if modelContext > 0 && modelContext < plan.ContextLength {
		plan.ContextLength = modelContext
		plan.Notes = append(plan.Notes, "context capped by model metadata")
	}
	plan.ModelQuantization = modelQuant
	plan.ModelQuantizationType = modelQuantType
	plan.ModelQuantizationFamily = modelQuantFamily
	if modelQuant > 0 && modelQuant < plan.PreferredQuantization {
		plan.Notes = append(plan.Notes, "model quantization is below machine-class preference")
	}
	applyModelArchitectureMemoryHints(&plan, modelArchitecture)
	plan.EstimatedKVCacheBytes = estimateKVCacheBytes(plan, input, KVCacheModeFP16)
	plan.EstimatedKVCacheModeBytes = estimateKVCacheBytes(plan, input, plan.CacheMode)
	if plan.EstimatedKVCacheBytes > 0 && plan.EstimatedKVCacheModeBytes > 0 && plan.EstimatedKVCacheModeBytes < plan.EstimatedKVCacheBytes {
		plan.KVCacheSavingsRatio = 1 - float64(plan.EstimatedKVCacheModeBytes)/float64(plan.EstimatedKVCacheBytes)
	}
	return plan
}

func memoryClassForBytes(bytes uint64) MemoryClass {
	if bytes == 0 {
		return MemoryClassUnknown
	}
	switch gib := (bytes + MemoryGiB - 1) / MemoryGiB; {
	case gib <= 18:
		return MemoryClassApple16GB
	case gib <= 26:
		return MemoryClassApple24GB
	case gib <= 40:
		return MemoryClassApple32GB
	case gib <= 80:
		return MemoryClassApple64GB
	case gib <= 112:
		return MemoryClassApple96GB
	default:
		return MemoryClassApple128GB
	}
}

func baseMemoryPlan(class MemoryClass) MemoryPlan {
	switch class {
	case MemoryClassApple16GB:
		return MemoryPlan{
			ContextLength:         8192,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModeKQ8VQ4,
			BatchSize:             1,
			PrefillChunkSize:      512,
			ParallelSlots:         1,
			PromptCache:           false,
			PromptCacheMinTokens:  0,
			PreferredQuantization: 4,
		}
	case MemoryClassApple24GB:
		return MemoryPlan{
			ContextLength:         16384,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModeQ8,
			BatchSize:             1,
			PrefillChunkSize:      768,
			ParallelSlots:         1,
			PromptCache:           true,
			PromptCacheMinTokens:  4096,
			PreferredQuantization: 4,
		}
	case MemoryClassApple32GB:
		return MemoryPlan{
			ContextLength:         32768,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModeQ8,
			BatchSize:             1,
			PrefillChunkSize:      1024,
			ParallelSlots:         1,
			PromptCache:           true,
			PromptCacheMinTokens:  4096,
			PreferredQuantization: 4,
		}
	case MemoryClassApple64GB:
		return MemoryPlan{
			ContextLength:         65536,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModePaged,
			BatchSize:             2,
			PrefillChunkSize:      2048,
			ParallelSlots:         1,
			PromptCache:           true,
			PromptCacheMinTokens:  DefaultPromptCacheMinTokens,
			PreferredQuantization: 4,
		}
	case MemoryClassApple96GB:
		return MemoryPlan{
			ContextLength:         DefaultLocalContextLength,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModePaged,
			BatchSize:             4,
			PrefillChunkSize:      4096,
			ParallelSlots:         2,
			PromptCache:           true,
			PromptCacheMinTokens:  DefaultPromptCacheMinTokens,
			PreferredQuantization: 8,
		}
	case MemoryClassApple128GB:
		return MemoryPlan{
			ContextLength:         DefaultLocalContextLength,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModePaged,
			BatchSize:             6,
			PrefillChunkSize:      4096,
			ParallelSlots:         2,
			PromptCache:           true,
			PromptCacheMinTokens:  DefaultPromptCacheMinTokens,
			PreferredQuantization: 8,
		}
	default:
		return MemoryPlan{
			ContextLength:         DefaultLocalContextLength,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModeQ8,
			BatchSize:             1,
			PrefillChunkSize:      1024,
			ParallelSlots:         DefaultLocalParallelSlots,
			PromptCache:           true,
			PromptCacheMinTokens:  DefaultPromptCacheMinTokens,
			PreferredQuantization: 4,
		}
	}
}

func estimateKVCacheBytes(plan MemoryPlan, input MemoryPlanInput, mode KVCacheMode) uint64 {
	if plan.ContextLength <= 0 {
		return 0
	}
	layers, hidden := kvEstimateShape(input, plan.MachineClass)
	if layers <= 0 || hidden <= 0 {
		return 0
	}
	elements := uint64(plan.ContextLength) * uint64(layers) * uint64(hidden) * 2
	switch mode {
	case KVCacheModeKQ8VQ4:
		// K uses one byte, V uses four logical bits. The current native cache
		// stores q4 values in int8 lanes until packed kernels are available.
		return elements * 3 / 4
	case KVCacheModeQ8:
		return elements
	default:
		return elements * 2
	}
}

func kvEstimateShape(input MemoryPlanInput, class MemoryClass) (layers, hidden int) {
	if input.ModelInfo != nil {
		layers = input.ModelInfo.NumLayers
		hidden = input.ModelInfo.HiddenSize
	}
	if input.Pack != nil {
		if layers == 0 {
			layers = input.Pack.NumLayers
		}
		if hidden == 0 {
			hidden = input.Pack.HiddenSize
		}
	}
	if layers > 0 && hidden > 0 {
		return layers, hidden
	}
	switch class {
	case MemoryClassApple16GB, MemoryClassApple24GB:
		return 28, 2048
	case MemoryClassApple32GB:
		return 32, 3072
	case MemoryClassApple64GB:
		return 40, 4096
	default:
		return 48, 5120
	}
}

func modelMemoryHints(input MemoryPlanInput) (contextLength, quantization int, quantType, quantFamily, architecture string) {
	if input.Pack != nil {
		contextLength = input.Pack.ContextLength
		quantization = input.Pack.QuantBits
		quantType = input.Pack.QuantType
		quantFamily = input.Pack.QuantFamily
		architecture = input.Pack.Architecture
	}
	if input.ModelInfo != nil {
		if input.ModelInfo.Architecture != "" {
			architecture = input.ModelInfo.Architecture
		}
		if input.ModelInfo.ContextLength > 0 {
			contextLength = input.ModelInfo.ContextLength
		}
		if input.ModelInfo.QuantBits > 0 {
			quantization = input.ModelInfo.QuantBits
		}
	}
	return contextLength, quantization, quantType, quantFamily, architecture
}

func applyModelArchitectureMemoryHints(plan *MemoryPlan, architecture string) {
	switch normalizeKnownArchitecture(architecture) {
	case "qwen3_moe":
		plan.Notes = append(plan.Notes, "Qwen3-MoE sparse expert routing increases memory pressure; prefer compact KV cache modes on constrained Apple memory")
		if plan.MachineClass == MemoryClassApple24GB || plan.MachineClass == MemoryClassApple32GB {
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "Qwen3-MoE uses asymmetric K@q8,V@q4 cache below 64GB")
		}
	case "qwen3_next":
		plan.Notes = append(plan.Notes, "Qwen3-Next uses nested text_config metadata; keep context and cache policy tied to text model limits")
	}
}

func percentBytes(value uint64, percent uint64) uint64 {
	if value == 0 {
		return 0
	}
	return value * percent / 100
}

var memoryPlannerDeviceInfo = GetDeviceInfo

func applyMemoryPlanToLoadConfig(modelPath string, cfg LoadConfig) LoadConfig {
	var plan MemoryPlan
	if cfg.MemoryPlan != nil {
		plan = *cfg.MemoryPlan
	} else if cfg.AutoMemoryPlan {
		var pack *ModelPack
		if inspected, err := InspectModelPack(modelPath, WithPackRequireChatTemplate(false)); err == nil {
			pack = &inspected
		}
		plan = PlanMemory(MemoryPlanInput{
			Device: memoryPlannerDeviceInfo(),
			Pack:   pack,
		})
	} else {
		return cfg
	}

	cfg.MemoryPlan = &plan
	if plan.ContextLength > 0 && (cfg.ContextLength == 0 || cfg.ContextLength == DefaultLocalContextLength) {
		cfg.ContextLength = plan.ContextLength
	}
	if plan.ParallelSlots > 0 && (cfg.ParallelSlots == 0 || cfg.ParallelSlots == DefaultLocalParallelSlots) {
		cfg.ParallelSlots = plan.ParallelSlots
	}
	if !plan.PromptCache {
		cfg.PromptCache = false
	} else if plan.PromptCacheMinTokens > 0 && (cfg.PromptCacheMinTokens == 0 || cfg.PromptCacheMinTokens == DefaultPromptCacheMinTokens) {
		cfg.PromptCacheMinTokens = plan.PromptCacheMinTokens
	}
	if cfg.CachePolicy == "" {
		cfg.CachePolicy = plan.CachePolicy
	}
	if cfg.CacheMode == "" {
		cfg.CacheMode = plan.CacheMode
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = plan.BatchSize
	}
	if cfg.PrefillChunkSize == 0 {
		cfg.PrefillChunkSize = plan.PrefillChunkSize
	}
	if cfg.ExpectedQuantization == 0 {
		cfg.ExpectedQuantization = plan.PreferredQuantization
	}
	if cfg.MemoryLimitBytes == 0 {
		cfg.MemoryLimitBytes = plan.MemoryLimitBytes
	}
	if cfg.CacheLimitBytes == 0 {
		cfg.CacheLimitBytes = plan.CacheLimitBytes
	}
	if cfg.WiredLimitBytes == 0 {
		cfg.WiredLimitBytes = plan.WiredLimitBytes
	}
	return cfg
}
