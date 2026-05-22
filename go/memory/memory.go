// SPDX-Licence-Identifier: EUPL-1.2

// Package memory is the go-mlx local-inference memory planner. It maps
// measured Apple-silicon hardware + optional model metadata to a
// runtime policy (context length, KV cache shape, batch size, prompt
// cache, MoE expert residency) that fits the device class without
// over-allocating.
//
//	plan := memory.NewPlan(memory.Input{Device: dev, Pack: pack, ModelInfo: info})
//	if plan.ContextLength > 0 { … }
package memory

import (
	"time"

	"dappco.re/go/inference/quant/jang"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

// GiB is the number of bytes in a gibibyte.
const GiB uint64 = 1 << 30

// Class names the local Apple memory tier driving runtime policy.
type Class string

const (
	ClassUnknown    Class = "unknown"
	ClassApple16GB  Class = "apple-silicon-16gb"
	ClassApple24GB  Class = "apple-silicon-24gb"
	ClassApple32GB  Class = "apple-silicon-32gb"
	ClassApple64GB  Class = "apple-silicon-64gb"
	ClassApple96GB  Class = "apple-silicon-96gb"
	ClassApple128GB Class = "apple-silicon-128gb-plus"
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

// ExpertResidencyMode names how routed MoE experts are kept resident.
type ExpertResidencyMode string

const (
	ExpertResidencyModeOff    ExpertResidencyMode = ""
	ExpertResidencyModePinned ExpertResidencyMode = "pinned"
	ExpertResidencyModeLazy   ExpertResidencyMode = "lazy"
)

// ExpertEvictionPolicy names the cold-expert eviction strategy.
type ExpertEvictionPolicy string

const (
	ExpertEvictionLRU ExpertEvictionPolicy = "lru"
)

// DeviceInfo carries the measured device memory the planner consults.
// Mirrors the mlx-root metal.DeviceInfo struct so the memory package
// stays driver-internal-free.
type DeviceInfo struct {
	Architecture                 string
	MaxBufferLength              uint64
	MaxRecommendedWorkingSetSize uint64
	MemorySize                   uint64
}

// ModelInfo carries the optional model metadata the planner consults.
// Mirrors the mlx-root ModelInfo identity used at the package boundary.
type ModelInfo struct {
	Architecture  string
	VocabSize     int
	NumLayers     int
	HiddenSize    int
	QuantBits     int
	QuantGroup    int
	ContextLength int
}

// Input supplies measured hardware and optional model metadata.
type Input struct {
	Device    DeviceInfo
	Pack      *mp.ModelPack
	ModelInfo *ModelInfo
}

// ExpertResidencyStats records measured hot-load, page-in, and eviction
// behaviour. Backends can feed this directly into workload bench reports.
type ExpertResidencyStats struct {
	ResidentExperts     int           `json:"resident_experts,omitempty"`
	PeakResidentExperts int           `json:"peak_resident_experts,omitempty"`
	HotLoads            int           `json:"hot_loads,omitempty"`
	ColdLoads           int           `json:"cold_loads,omitempty"`
	PageIns             int           `json:"page_ins,omitempty"`
	PageOuts            int           `json:"page_outs,omitempty"`
	Hits                int           `json:"hits,omitempty"`
	LoadedBytes         uint64        `json:"loaded_bytes,omitempty"`
	EvictedBytes        uint64        `json:"evicted_bytes,omitempty"`
	FirstUseLatency     time.Duration `json:"first_use_latency,omitempty"`
	TotalLoadDuration   time.Duration `json:"total_load_duration,omitempty"`
}

// ExpertResidencyPlan is a backend-neutral MoE residency policy. It is
// small enough for memory planners and benchmark reports while still
// explicit about hot experts, resident limits, and expected first-use
// pressure.
type ExpertResidencyPlan struct {
	Enabled                 bool                 `json:"enabled"`
	Mode                    ExpertResidencyMode  `json:"mode,omitempty"`
	Architecture            string               `json:"architecture,omitempty"`
	TotalExperts            int                  `json:"total_experts,omitempty"`
	ExpertsPerToken         int                  `json:"experts_per_token,omitempty"`
	HotExpertIDs            []int                `json:"hot_expert_ids,omitempty"`
	StartupExpertIDs        []int                `json:"startup_expert_ids,omitempty"`
	HotExperts              int                  `json:"hot_experts,omitempty"`
	MaxResidentExperts      int                  `json:"max_resident_experts,omitempty"`
	PageInBatchSize         int                  `json:"page_in_batch_size,omitempty"`
	EvictionPolicy          ExpertEvictionPolicy `json:"eviction_policy,omitempty"`
	EstimatedExpertBytes    uint64               `json:"estimated_expert_bytes,omitempty"`
	EstimatedResidentBytes  uint64               `json:"estimated_resident_bytes,omitempty"`
	MaxResidentBytes        uint64               `json:"max_resident_bytes,omitempty"`
	FirstUseLatencyExpected bool                 `json:"first_use_latency_expected,omitempty"`
	Notes                   []string             `json:"notes,omitempty"`
}

// Plan is the local runtime policy derived from measured device memory.
type Plan struct {
	MachineClass                  Class               `json:"machine_class"`
	Architecture                  string              `json:"architecture,omitempty"`
	DeviceMemoryBytes             uint64              `json:"device_memory_bytes,omitempty"`
	RecommendedWorkingSetBytes    uint64              `json:"recommended_working_set_bytes,omitempty"`
	ContextLength                 int                 `json:"context_length"`
	CachePolicy                   KVCachePolicy       `json:"cache_policy"`
	CacheMode                     KVCacheMode         `json:"cache_mode,omitempty"`
	BatchSize                     int                 `json:"batch_size"`
	PrefillChunkSize              int                 `json:"prefill_chunk_size"`
	ParallelSlots                 int                 `json:"parallel_slots"`
	PromptCache                   bool                `json:"prompt_cache"`
	PromptCacheMinTokens          int                 `json:"prompt_cache_min_tokens"`
	PreferredQuantization         int                 `json:"preferred_quantization,omitempty"`
	ModelQuantization             int                 `json:"model_quantization,omitempty"`
	ModelQuantizationType         string              `json:"model_quantization_type,omitempty"`
	ModelQuantizationFamily       string              `json:"model_quantization_family,omitempty"`
	ModelPackedQuantization       *jang.PackedProfile `json:"model_packed_quantization,omitempty"`
	ModelWeightBytes              uint64              `json:"model_weight_bytes,omitempty"`
	ModelForwardSkeletonValidated bool                `json:"model_forward_skeleton_validated,omitempty"`
	ModelForwardSkeletonBytes     uint64              `json:"model_forward_skeleton_bytes,omitempty"`
	ExpertResidency               ExpertResidencyPlan `json:"expert_residency,omitempty"`
	MemoryLimitBytes              uint64              `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes               uint64              `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes               uint64              `json:"wired_limit_bytes,omitempty"`
	EstimatedKVCacheBytes         uint64              `json:"estimated_kv_cache_bytes,omitempty"`
	EstimatedKVCacheModeBytes     uint64              `json:"estimated_kv_cache_mode_bytes,omitempty"`
	KVCacheSavingsRatio           float64             `json:"kv_cache_savings_ratio,omitempty"`
	Notes                         []string            `json:"notes,omitempty"`
}

// Defaults that mirror the mlx-root local-inference baselines. Kept
// here so the memory package is self-contained.
const (
	defaultLocalContextLength   = 131072
	defaultLocalParallelSlots   = 1
	defaultPromptCacheMinTokens = 2048
)

// NewPlan chooses opinionated local inference settings from measured memory.
//
//	plan := memory.NewPlan(memory.Input{Device: dev, Pack: pack})
func NewPlan(input Input) Plan {
	deviceMemory := input.Device.MemorySize
	workingSet := input.Device.MaxRecommendedWorkingSetSize
	if workingSet == 0 {
		workingSet = deviceMemory
	}
	class := classForBytes(deviceMemory)
	plan := baseClassPlan(class)
	plan.MachineClass = class
	plan.Architecture = input.Device.Architecture
	plan.DeviceMemoryBytes = deviceMemory
	plan.RecommendedWorkingSetBytes = workingSet
	plan.MemoryLimitBytes = percentBytes(workingSet, 85)
	plan.CacheLimitBytes = percentBytes(workingSet, 8)
	plan.WiredLimitBytes = percentBytes(workingSet, 75)

	modelContext, modelQuant, modelQuantType, modelQuantFamily, modelArchitecture, modelWeightBytes := modelHints(input)
	if modelContext > 0 && modelContext < plan.ContextLength {
		plan.ContextLength = modelContext
		plan.Notes = append(plan.Notes, "context capped by model metadata")
	}
	plan.ModelQuantization = modelQuant
	plan.ModelQuantizationType = modelQuantType
	plan.ModelQuantizationFamily = modelQuantFamily
	if input.Pack != nil {
		plan.ModelPackedQuantization = jang.ClonePackedProfile(input.Pack.PackedQuantization)
	}
	plan.ModelWeightBytes = modelWeightBytes
	if modelQuant > 0 && modelQuant < plan.PreferredQuantization {
		plan.Notes = append(plan.Notes, "model quantization is below machine-class preference")
	}
	// Resolve the canonical architecture once and look up the
	// profile registry exactly once for the whole NewPlan call.
	// usesGenerationKVCache, applyArchitectureHints, and
	// applyGenericMoEResidency all need the same profile lookup; the
	// profile package clones the entry on every Lookup so caching
	// here saves two clones (plus their child-slice allocations) per
	// plan.
	resolvedArch := modelArchitecture
	if input.Pack != nil && input.Pack.Architecture != "" {
		resolvedArch = input.Pack.Architecture
	}
	resolvedProfile, profileFound := profile.LookupArchitectureProfile(resolvedArch)
	var profilePtr *profile.ModelArchitectureProfile
	if profileFound {
		profilePtr = &resolvedProfile
	}
	applyArchitectureHints(&plan, modelArchitecture, profilePtr)
	applyQuantizationHints(&plan)
	applyGenericMoEResidency(&plan, input.Pack, profilePtr)
	// Both KV-cache estimates use the same gating + shape — compute
	// once, scale the element count for each mode. usesGenerationKV
	// + kvEstimateShape used to run twice per plan.
	if usesGenerationKVCacheWithProfile(input, profilePtr) && plan.ContextLength > 0 {
		if layers, hidden := kvEstimateShape(input, plan.MachineClass); layers > 0 && hidden > 0 {
			elements := uint64(plan.ContextLength) * uint64(layers) * uint64(hidden) * 2
			plan.EstimatedKVCacheBytes = elements * 2 // FP16 = 2 bytes/element
			plan.EstimatedKVCacheModeBytes = scaleKVElements(elements, plan.CacheMode)
		}
	}
	if plan.EstimatedKVCacheBytes > 0 && plan.EstimatedKVCacheModeBytes > 0 && plan.EstimatedKVCacheModeBytes < plan.EstimatedKVCacheBytes {
		plan.KVCacheSavingsRatio = 1 - float64(plan.EstimatedKVCacheModeBytes)/float64(plan.EstimatedKVCacheBytes)
	}
	return plan
}

// ClassForBytes returns the Class corresponding to the supplied memory
// size in bytes. Exported so callers that already know the device
// memory can pre-compute the class without a full plan.
//
//	class := memory.ClassForBytes(96 * memory.GiB)
func ClassForBytes(bytes uint64) Class { return classForBytes(bytes) }

func classForBytes(bytes uint64) Class {
	if bytes == 0 {
		return ClassUnknown
	}
	switch gib := (bytes + GiB - 1) / GiB; {
	case gib <= 18:
		return ClassApple16GB
	case gib <= 26:
		return ClassApple24GB
	case gib <= 40:
		return ClassApple32GB
	case gib <= 80:
		return ClassApple64GB
	case gib <= 112:
		return ClassApple96GB
	default:
		return ClassApple128GB
	}
}

func baseClassPlan(class Class) Plan {
	switch class {
	case ClassApple16GB:
		return Plan{
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
	case ClassApple24GB:
		return Plan{
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
	case ClassApple32GB:
		return Plan{
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
	case ClassApple64GB:
		return Plan{
			ContextLength:         65536,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModePaged,
			BatchSize:             2,
			PrefillChunkSize:      4096,
			ParallelSlots:         1,
			PromptCache:           true,
			PromptCacheMinTokens:  defaultPromptCacheMinTokens,
			PreferredQuantization: 4,
		}
	case ClassApple96GB:
		return Plan{
			ContextLength:         defaultLocalContextLength,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModePaged,
			BatchSize:             4,
			PrefillChunkSize:      4096,
			ParallelSlots:         2,
			PromptCache:           true,
			PromptCacheMinTokens:  defaultPromptCacheMinTokens,
			PreferredQuantization: 8,
		}
	case ClassApple128GB:
		return Plan{
			ContextLength:         defaultLocalContextLength,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModePaged,
			BatchSize:             6,
			PrefillChunkSize:      4096,
			ParallelSlots:         2,
			PromptCache:           true,
			PromptCacheMinTokens:  defaultPromptCacheMinTokens,
			PreferredQuantization: 8,
		}
	default:
		return Plan{
			ContextLength:         defaultLocalContextLength,
			CachePolicy:           KVCacheRotating,
			CacheMode:             KVCacheModeQ8,
			BatchSize:             1,
			PrefillChunkSize:      1024,
			ParallelSlots:         defaultLocalParallelSlots,
			PromptCache:           true,
			PromptCacheMinTokens:  defaultPromptCacheMinTokens,
			PreferredQuantization: 4,
		}
	}
}

func estimateKVCacheBytes(plan Plan, input Input, mode KVCacheMode) uint64 {
	return estimateKVCacheBytesWithProfile(plan, input, mode, nil)
}

func estimateKVCacheBytesWithProfile(plan Plan, input Input, mode KVCacheMode, profileHint *profile.ModelArchitectureProfile) uint64 {
	if !usesGenerationKVCacheWithProfile(input, profileHint) {
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
	return scaleKVElements(elements, mode)
}

// scaleKVElements maps the raw element count to bytes for the given
// KV cache mode. Hoisted from estimateKVCacheBytes so NewPlan can
// run the gating + shape compute once and call this twice instead.
func scaleKVElements(elements uint64, mode KVCacheMode) uint64 {
	switch mode {
	case KVCacheModeKQ8VQ4:
		return elements * 3 / 4
	case KVCacheModeQ8:
		return elements
	default:
		return elements * 2
	}
}

func kvEstimateShape(input Input, class Class) (layers, hidden int) {
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
	case ClassApple16GB, ClassApple24GB:
		return 28, 2048
	case ClassApple32GB:
		return 32, 3072
	case ClassApple64GB:
		return 40, 4096
	default:
		return 48, 5120
	}
}

func modelHints(input Input) (contextLength, quantization int, quantType, quantFamily, architecture string, weightBytes uint64) {
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

func applyArchitectureHints(plan *Plan, architecture string, profileHint *profile.ModelArchitectureProfile) {
	// Profile registry is authoritative when it matches — skip the
	// normalize allocation entirely in that case. Only fall through
	// to normalize for architectures the registry does not know.
	var normalized string
	switch {
	case profileHint != nil:
		normalized = profileHint.ID
	default:
		if p, ok := profile.LookupArchitectureProfile(architecture); ok {
			normalized = p.ID
		} else {
			normalized = normalizeKnownArchitecture(architecture)
		}
	}
	switch normalized {
	case "qwen2":
		plan.Notes = append(plan.Notes, "Qwen2.x uses the native Qwen decoder; long contexts benefit from paged or compact KV cache modes on Apple unified memory")
	case "qwen3_moe":
		plan.Notes = append(plan.Notes, "Qwen3-MoE sparse expert routing increases memory pressure; prefer compact KV cache modes on constrained Apple memory")
		if plan.MachineClass == ClassApple24GB || plan.MachineClass == ClassApple32GB {
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "Qwen3-MoE uses asymmetric K@q8,V@q4 cache below 64GB")
		}
	case "qwen3_6":
		plan.Notes = append(plan.Notes, "Qwen3.6 uses hybrid linear attention; native Go kernels are pending, so prefer the mlx_lm fallback backend")
		plan.ParallelSlots = 1
		if plan.PrefillChunkSize > 2048 {
			plan.PrefillChunkSize = 2048
		}
	case "qwen3_6_moe":
		plan.Notes = append(plan.Notes, "Qwen3.6-MoE uses hybrid linear attention plus routed experts; native Go kernels are pending, so prefer the mlx_lm fallback backend")
		plan.ParallelSlots = 1
		if plan.PrefillChunkSize > 2048 {
			plan.PrefillChunkSize = 2048
		}
		if plan.MachineClass == ClassApple16GB || plan.MachineClass == ClassApple24GB || plan.MachineClass == ClassApple32GB {
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "Qwen3.6-MoE uses asymmetric K@q8,V@q4 cache below 64GB")
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
		if plan.MachineClass == ClassApple16GB || plan.MachineClass == ClassApple24GB || plan.MachineClass == ClassApple32GB {
			plan.ContextLength = minPositive(plan.ContextLength, 8192)
			plan.CacheMode = KVCacheModeKQ8VQ4
			plan.Notes = append(plan.Notes, "MiniMax M2 requires asymmetric compact KV cache below 64GB")
		}
	case "bert":
		applyEncoderHints(plan, encoderHintBert)
	case "bert_rerank":
		applyEncoderHints(plan, encoderHintBertRerank)
	}
}

func applyEncoderHints(plan *Plan, label string) {
	plan.CachePolicy = KVCacheDefault
	plan.CacheMode = KVCacheModeDefault
	plan.PromptCache = false
	plan.PromptCacheMinTokens = 0
	if plan.PrefillChunkSize == 0 || plan.PrefillChunkSize > 512 {
		plan.PrefillChunkSize = 512
	}
	switch plan.MachineClass {
	case ClassApple16GB, ClassApple24GB:
		if plan.BatchSize < 8 {
			plan.BatchSize = 8
		}
	case ClassApple32GB:
		if plan.BatchSize < 16 {
			plan.BatchSize = 16
		}
	case ClassApple64GB, ClassApple96GB:
		if plan.BatchSize < 32 {
			plan.BatchSize = 32
		}
	case ClassApple128GB:
		if plan.BatchSize < 48 {
			plan.BatchSize = 48
		}
	default:
		if plan.BatchSize < 4 {
			plan.BatchSize = 4
		}
	}
	plan.Notes = append(plan.Notes, label)
}

// Pre-computed encoder hint strings — applyEncoderHints used to build
// these by concatenating a per-call label with a constant suffix at
// runtime. With only two call sites it is cheaper to pre-compute the
// full strings as package-level constants and pass the matching one in.
const (
	encoderHintBert       = "BERT embedding encoder uses pooled sequence outputs and does not allocate generation KV cache"
	encoderHintBertRerank = "BERT cross-encoder rerank uses pooled sequence outputs and does not allocate generation KV cache"
)

func usesGenerationKVCache(input Input) bool {
	return usesGenerationKVCacheWithProfile(input, nil)
}

func usesGenerationKVCacheWithProfile(input Input, profileHint *profile.ModelArchitectureProfile) bool {
	// Cheapest checks first — Pack-resident flags short-circuit
	// without touching the architecture string or the profile
	// registry. Most callers that pass Embedding/Rerank packs return
	// here.
	if input.Pack != nil {
		if input.Pack.Embedding != nil || input.Pack.Rerank != nil {
			return false
		}
		if input.Pack.ArchitectureProfile != nil && (input.Pack.ArchitectureProfile.Embeddings || input.Pack.ArchitectureProfile.Rerank) {
			return false
		}
	}
	// Caller may have already done the registry lookup — use the
	// cached profile instead of touching the registry again.
	if profileHint != nil {
		if profileHint.Embeddings || profileHint.Rerank {
			return false
		}
		return true
	}
	// Fall through to the legacy single-call path.
	architecture := ""
	if input.Pack != nil && input.Pack.Architecture != "" {
		architecture = input.Pack.Architecture
	} else if input.ModelInfo != nil {
		architecture = input.ModelInfo.Architecture
	}
	if p, ok := profile.LookupArchitectureProfile(architecture); ok && (p.Embeddings || p.Rerank) {
		return false
	}
	return true
}

func applyQuantizationHints(plan *Plan) {
	if plan.ModelQuantizationFamily != "jang" && plan.ModelQuantizationType != "jangtq" {
		return
	}
	plan.Notes = append(plan.Notes, "JANGTQ/JANG mixed precision protects attention while compressing routed experts; fit estimates should use measured weight bytes over uniform-bit heuristics")
}

func applyGenericMoEResidency(plan *Plan, pack *mp.ModelPack, profileHint *profile.ModelArchitectureProfile) {
	if plan == nil {
		return
	}
	if profileHint == nil || !profileHint.MoE {
		return
	}
	p := *profileHint
	plan.ExpertResidency = ExpertResidencyPlan{
		Enabled:                 true,
		Mode:                    ExpertResidencyModeLazy,
		Architecture:            p.ID,
		MaxResidentExperts:      genericMoEResidentExpertLimit(plan.MachineClass),
		PageInBatchSize:         1,
		EvictionPolicy:          ExpertEvictionLRU,
		FirstUseLatencyExpected: true,
		Notes:                   []string{"MoE model uses lazy expert residency until backend-specific expert byte estimates are available"},
	}
	plan.Notes = append(plan.Notes, "lazy expert residency enabled for MoE architecture")
}

func genericMoEResidentExpertLimit(class Class) int {
	switch class {
	case ClassApple16GB, ClassApple24GB:
		return 2
	case ClassApple32GB:
		return 4
	case ClassApple64GB:
		return 8
	case ClassApple96GB:
		return 16
	case ClassApple128GB:
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

// normalizeKnownArchitecture canonicalises an architecture identifier
// so the planner can match the variations seen in HF configs. Kept
// private inside memory so the package is self-contained.
func normalizeKnownArchitecture(value string) string {
	value = lowerASCII(trimSpace(value))
	value = replaceASCII(value, '-', '_')
	value = replaceASCII(value, '.', '_')
	switch value {
	case "qwen2_5", "qwen25":
		return "qwen2"
	case "qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_6_text", "qwen35", "qwen36":
		return "qwen3_6"
	case "qwen3_5_moe", "qwen3_6_moe", "qwen35_moe", "qwen36_moe":
		return "qwen3_6_moe"
	case "minimaxm2", "minimax_m2":
		return "minimax_m2"
	case "mixtral":
		return "mixtral"
	case "mistral":
		return "mistral"
	case "phi", "phi3", "phi4":
		return "phi"
	case "deepseek", "deepseek_v3", "deepseek_r1":
		return "deepseek"
	case "gptoss", "gpt_oss", "gpt_oss_model":
		return "gpt_oss"
	case "bert":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	default:
		return value
	}
}

func lowerASCII(s string) string {
	// Fast path — most architecture identifiers are already lowercase
	// after the first canonicalisation pass. Scan once; if there is
	// nothing to convert, return the input unchanged to skip both the
	// byte-slice allocation and the return-side string copy.
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			b := []byte(s)
			b[i] = c + ('a' - 'A')
			for j := i + 1; j < len(b); j++ {
				if b[j] >= 'A' && b[j] <= 'Z' {
					b[j] += 'a' - 'A'
				}
			}
			return string(b)
		}
	}
	return s
}

func trimSpace(s string) string {
	end := len(s)
	if end == 0 {
		return s
	}
	// Fast path — most canonicalised architecture strings have no
	// leading or trailing whitespace. One bounds check per end and we
	// return the input slice header unchanged.
	if !isSpaceASCII(s[0]) && !isSpaceASCII(s[end-1]) {
		return s
	}
	start := 0
	for start < end && isSpaceASCII(s[start]) {
		start++
	}
	for end > start && isSpaceASCII(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isSpaceASCII(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'
}

func replaceASCII(s string, old, new byte) string {
	// Fast path — most identifiers never contain the sentinel byte we
	// rewrite (dots, dashes). Scan once; if there is nothing to
	// replace, return the input unchanged to skip both the byte-slice
	// allocation and the return-side string copy.
	for i := 0; i < len(s); i++ {
		if s[i] == old {
			b := []byte(s)
			b[i] = new
			for j := i + 1; j < len(b); j++ {
				if b[j] == old {
					b[j] = new
				}
			}
			return string(b)
		}
	}
	return s
}
