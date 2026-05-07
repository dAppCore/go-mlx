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
	BatchSize                  int           `json:"batch_size"`
	PrefillChunkSize           int           `json:"prefill_chunk_size"`
	ParallelSlots              int           `json:"parallel_slots"`
	PromptCache                bool          `json:"prompt_cache"`
	PromptCacheMinTokens       int           `json:"prompt_cache_min_tokens"`
	PreferredQuantization      int           `json:"preferred_quantization,omitempty"`
	ModelQuantization          int           `json:"model_quantization,omitempty"`
	MemoryLimitBytes           uint64        `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes            uint64        `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes            uint64        `json:"wired_limit_bytes,omitempty"`
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

	modelContext, modelQuant := modelMemoryHints(input)
	if modelContext > 0 && modelContext < plan.ContextLength {
		plan.ContextLength = modelContext
		plan.Notes = append(plan.Notes, "context capped by model metadata")
	}
	plan.ModelQuantization = modelQuant
	if modelQuant > 0 && modelQuant < plan.PreferredQuantization {
		plan.Notes = append(plan.Notes, "model quantization is below machine-class preference")
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
			BatchSize:             1,
			PrefillChunkSize:      1024,
			ParallelSlots:         DefaultLocalParallelSlots,
			PromptCache:           true,
			PromptCacheMinTokens:  DefaultPromptCacheMinTokens,
			PreferredQuantization: 4,
		}
	}
}

func modelMemoryHints(input MemoryPlanInput) (contextLength, quantization int) {
	if input.Pack != nil {
		contextLength = input.Pack.ContextLength
		quantization = input.Pack.QuantBits
	}
	if input.ModelInfo != nil {
		if input.ModelInfo.ContextLength > 0 {
			contextLength = input.ModelInfo.ContextLength
		}
		if input.ModelInfo.QuantBits > 0 {
			quantization = input.ModelInfo.QuantBits
		}
	}
	return contextLength, quantization
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
