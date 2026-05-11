// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference/quant/jang"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

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
	Pack      *mp.ModelPack
	ModelInfo *ModelInfo
}

// MemoryPlan is the local runtime policy derived from measured device memory.
type MemoryPlan struct {
	MachineClass                  MemoryClass                    `json:"machine_class"`
	Architecture                  string                         `json:"architecture,omitempty"`
	DeviceMemoryBytes             uint64                         `json:"device_memory_bytes,omitempty"`
	RecommendedWorkingSetBytes    uint64                         `json:"recommended_working_set_bytes,omitempty"`
	ContextLength                 int                            `json:"context_length"`
	CachePolicy                   KVCachePolicy                  `json:"cache_policy"`
	CacheMode                     KVCacheMode                    `json:"cache_mode,omitempty"`
	BatchSize                     int                            `json:"batch_size"`
	PrefillChunkSize              int                            `json:"prefill_chunk_size"`
	ParallelSlots                 int                            `json:"parallel_slots"`
	PromptCache                   bool                           `json:"prompt_cache"`
	PromptCacheMinTokens          int                            `json:"prompt_cache_min_tokens"`
	PreferredQuantization         int                            `json:"preferred_quantization,omitempty"`
	ModelQuantization             int                            `json:"model_quantization,omitempty"`
	ModelQuantizationType         string                         `json:"model_quantization_type,omitempty"`
	ModelQuantizationFamily       string                         `json:"model_quantization_family,omitempty"`
	ModelPackedQuantization       *jang.PackedProfile `json:"model_packed_quantization,omitempty"`
	ModelWeightBytes              uint64                         `json:"model_weight_bytes,omitempty"`
	ModelForwardSkeletonValidated bool                           `json:"model_forward_skeleton_validated,omitempty"`
	ModelForwardSkeletonBytes     uint64                         `json:"model_forward_skeleton_bytes,omitempty"`
	ExpertResidency               ExpertResidencyPlan            `json:"expert_residency,omitempty"`
	MemoryLimitBytes              uint64                         `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes               uint64                         `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes               uint64                         `json:"wired_limit_bytes,omitempty"`
	EstimatedKVCacheBytes         uint64                         `json:"estimated_kv_cache_bytes,omitempty"`
	EstimatedKVCacheModeBytes     uint64                         `json:"estimated_kv_cache_mode_bytes,omitempty"`
	KVCacheSavingsRatio           float64                        `json:"kv_cache_savings_ratio,omitempty"`
	Notes                         []string                       `json:"notes,omitempty"`
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

	modelContext, modelQuant, modelQuantType, modelQuantFamily, modelArchitecture, modelWeightBytes := modelMemoryHints(input)
	if modelContext > 0 && modelContext < plan.ContextLength {
		plan.ContextLength = modelContext
		plan.Notes = append(plan.Notes, "context capped by model metadata")
	}
	plan.ModelQuantization = modelQuant
	plan.ModelQuantizationType = modelQuantType
	plan.ModelQuantizationFamily = modelQuantFamily
	if input.Pack != nil {
		plan.ModelPackedQuantization = jang.ClonePackedProfile(input.Pack.PackedQuantization)
		if skel, _ := input.Pack.MiniMaxM2LayerSkeleton.(*MiniMaxM2LayerForwardSkeleton); skel != nil {
			plan.ModelForwardSkeletonValidated = true
			plan.ModelForwardSkeletonBytes = skel.EstimatedBytes()
			plan.Notes = append(plan.Notes, "MiniMax M2 first-layer tensor skeleton validated from safetensors metadata")
		}
	}
	plan.ModelWeightBytes = modelWeightBytes
	if modelQuant > 0 && modelQuant < plan.PreferredQuantization {
		plan.Notes = append(plan.Notes, "model quantization is below machine-class preference")
	}
	applyModelArchitectureMemoryHints(&plan, modelArchitecture)
	applyModelQuantizationMemoryHints(&plan)
	applyExpertResidencyMemoryHints(&plan, input.Pack, modelArchitecture)
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
	if !memoryPlanUsesGenerationKVCache(input) {
		return 0
	}
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

func modelMemoryHints(input MemoryPlanInput) (contextLength, quantization int, quantType, quantFamily, architecture string, weightBytes uint64) {
	if input.Pack != nil {
		contextLength = input.Pack.ContextLength
		quantization = input.Pack.QuantBits
		quantType = input.Pack.QuantType
		quantFamily = input.Pack.QuantFamily
		architecture = input.Pack.Architecture
		weightBytes = input.Pack.WeightBytes
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
	return contextLength, quantization, quantType, quantFamily, architecture, weightBytes
}

func applyModelArchitectureMemoryHints(plan *MemoryPlan, architecture string) {
	normalized := normalizeKnownArchitecture(architecture)
	if profile, ok := profile.LookupArchitectureProfile(architecture); ok {
		normalized = profile.ID
	}
	switch normalized {
	case "qwen3_moe":
		plan.Notes = append(plan.Notes, "Qwen3-MoE sparse expert routing increases memory pressure; prefer compact KV cache modes on constrained Apple memory")
		if plan.MachineClass == MemoryClassApple24GB || plan.MachineClass == MemoryClassApple32GB {
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "Qwen3-MoE uses asymmetric K@q8,V@q4 cache below 64GB")
		}
	case "qwen3_next":
		plan.Notes = append(plan.Notes, "Qwen3-Next uses nested text_config metadata; keep context and cache policy tied to text model limits")
	case "minimax_m2":
		plan.Notes = append(plan.Notes, "MiniMax M2 MoE has a large routed-expert footprint; keep prefill narrow and prefer paged cache on Apple unified memory")
		plan.ParallelSlots = 1
		plan.BatchSize = 1
		if plan.PrefillChunkSize > 2048 {
			plan.PrefillChunkSize = 2048
		}
		if plan.ContextLength > 32768 {
			plan.ContextLength = 32768
			plan.Notes = append(plan.Notes, "MiniMax M2 context capped for 96GB-class local inference")
		}
		if plan.MachineClass == MemoryClassApple16GB || plan.MachineClass == MemoryClassApple24GB || plan.MachineClass == MemoryClassApple32GB {
			plan.ContextLength = minPositive(plan.ContextLength, 8192)
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "MiniMax M2 requires asymmetric compact KV cache below 64GB")
		}
	case "bert":
		applyEncoderMemoryHints(plan, "BERT embedding encoder")
	case "bert_rerank":
		applyEncoderMemoryHints(plan, "BERT cross-encoder rerank")
	}
}

func applyEncoderMemoryHints(plan *MemoryPlan, label string) {
	plan.CachePolicy = KVCacheDefault
	plan.CacheMode = KVCacheModeDefault
	plan.PromptCache = false
	plan.PromptCacheMinTokens = 0
	if plan.PrefillChunkSize == 0 || plan.PrefillChunkSize > 512 {
		plan.PrefillChunkSize = 512
	}
	switch plan.MachineClass {
	case MemoryClassApple16GB, MemoryClassApple24GB:
		if plan.BatchSize < 8 {
			plan.BatchSize = 8
		}
	case MemoryClassApple32GB:
		if plan.BatchSize < 16 {
			plan.BatchSize = 16
		}
	case MemoryClassApple64GB, MemoryClassApple96GB:
		if plan.BatchSize < 32 {
			plan.BatchSize = 32
		}
	case MemoryClassApple128GB:
		if plan.BatchSize < 48 {
			plan.BatchSize = 48
		}
	default:
		if plan.BatchSize < 4 {
			plan.BatchSize = 4
		}
	}
	plan.Notes = append(plan.Notes, label+" uses pooled sequence outputs and does not allocate generation KV cache")
}

func memoryPlanUsesGenerationKVCache(input MemoryPlanInput) bool {
	architecture := ""
	if input.ModelInfo != nil {
		architecture = input.ModelInfo.Architecture
	}
	if input.Pack != nil && input.Pack.Architecture != "" {
		architecture = input.Pack.Architecture
	}
	return modelPackUsesGenerationKVCache(input.Pack, architecture)
}

func applyModelQuantizationMemoryHints(plan *MemoryPlan) {
	if plan.ModelQuantizationFamily != "jang" && plan.ModelQuantizationType != "jangtq" {
		return
	}
	plan.Notes = append(plan.Notes, "JANGTQ/JANG mixed precision protects attention while compressing routed experts; fit estimates should use measured weight bytes over uniform-bit heuristics")
}

func applyExpertResidencyMemoryHints(plan *MemoryPlan, pack *mp.ModelPack, architecture string) {
	if plan == nil {
		return
	}
	if pack != nil {
		if mm, _ := pack.MiniMaxM2.(*MiniMaxM2TensorPlan); mm != nil {
			plan.ExpertResidency = PlanMiniMaxM2ExpertResidency(*mm, *plan, nil)
			plan.Notes = append(plan.Notes, "MiniMax M2 lazy expert residency enabled by memory planner")
			return
		}
		if pack.Architecture != "" {
			architecture = pack.Architecture
		}
	}
	profile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok || !profile.MoE {
		return
	}
	plan.ExpertResidency = ExpertResidencyPlan{
		Enabled:                 true,
		Mode:                    ExpertResidencyModeLazy,
		Architecture:            profile.ID,
		MaxResidentExperts:      genericMoEResidentExpertLimit(plan.MachineClass),
		PageInBatchSize:         1,
		EvictionPolicy:          ExpertEvictionLRU,
		FirstUseLatencyExpected: true,
		Notes:                   []string{"MoE model uses lazy expert residency until backend-specific expert byte estimates are available"},
	}
	plan.Notes = append(plan.Notes, "lazy expert residency enabled for MoE architecture")
}

func genericMoEResidentExpertLimit(class MemoryClass) int {
	switch class {
	case MemoryClassApple16GB, MemoryClassApple24GB:
		return 2
	case MemoryClassApple32GB:
		return 4
	case MemoryClassApple64GB:
		return 8
	case MemoryClassApple96GB:
		return 16
	case MemoryClassApple128GB:
		return 24
	default:
		return 2
	}
}

func minPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}

func percentBytes(value uint64, percent uint64) uint64 {
	if value == 0 {
		return 0
	}
	return value * percent / 100
}

var memoryPlannerDeviceInfo = safeRuntimeDeviceInfo

func applyMemoryPlanToLoadConfig(modelPath string, cfg LoadConfig) LoadConfig {
	var plan MemoryPlan
	if cfg.MemoryPlan != nil {
		plan = *cfg.MemoryPlan
	} else if cfg.AutoMemoryPlan {
		var pack *mp.ModelPack
		if inspected, err := InspectModelPack(modelPath, mp.WithPackRequireChatTemplate(false)); err == nil {
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
