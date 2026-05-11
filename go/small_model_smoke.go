// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

const (
	DefaultSmallModelSmokeMaxWeightBytes     = 26 * MemoryGiB
	DefaultSmallModelSmokeQuantization       = 4
	DefaultSmallModelSmokeMaxContextLength   = 8192
	DefaultSmallModelSmokeMaxBatchSize       = 1
	DefaultSmallModelSmokeMaxPrefillChunk    = 1024
	DefaultSmallModelSmokeMaxTokens          = 8
	DefaultSmallModelSmokePromptCacheMinSize = 256
)

// SmallModelSmokeConfig configures a laptop-safe native MLX smoke pass.
type SmallModelSmokeConfig struct {
	ModelPath              string              `json:"model_path,omitempty"`
	MaxWeightBytes         uint64              `json:"max_weight_bytes,omitempty"`
	RequiredQuantization   int                 `json:"required_quantization,omitempty"`
	MaxContextLength       int                 `json:"max_context_length,omitempty"`
	MaxBatchSize           int                 `json:"max_batch_size,omitempty"`
	MaxPrefillChunkSize    int                 `json:"max_prefill_chunk_size,omitempty"`
	Device                 DeviceInfo          `json:"device,omitempty"`
	IncludeWorkloadBench   bool                `json:"include_workload_bench"`
	IncludeChatTemplate    bool                `json:"include_chat_template"`
	Workload               WorkloadBenchConfig `json:"workload,omitempty"`
	AdditionalLoadOptions  []LoadOption        `json:"-"`
	RequireNativeLoadable  bool                `json:"require_native_loadable"`
	RequireValidModelPack  bool                `json:"require_valid_model_pack"`
	RequireKnownWeightSize bool                `json:"require_known_weight_size"`
}

// SmallModelSmokeBudget records the conservative load/no-load decision.
type SmallModelSmokeBudget struct {
	SafeToLoad           bool   `json:"safe_to_load"`
	Reason               string `json:"reason,omitempty"`
	MaxWeightBytes       uint64 `json:"max_weight_bytes"`
	RequiredQuantization int    `json:"required_quantization,omitempty"`
	WeightBytes          uint64 `json:"weight_bytes,omitempty"`
	Quantization         int    `json:"quantization,omitempty"`
	NativeLoadable       bool   `json:"native_loadable"`
	ValidModelPack       bool   `json:"valid_model_pack"`
}

// SmallModelSmokeLoadPlan is the MLX load shape produced by the smoke planner.
type SmallModelSmokeLoadPlan struct {
	ContextLength        int           `json:"context_length"`
	ParallelSlots        int           `json:"parallel_slots"`
	PromptCache          bool          `json:"prompt_cache"`
	PromptCacheMinTokens int           `json:"prompt_cache_min_tokens,omitempty"`
	Quantization         int           `json:"quantization,omitempty"`
	CachePolicy          KVCachePolicy `json:"cache_policy,omitempty"`
	CacheMode            KVCacheMode   `json:"cache_mode,omitempty"`
	BatchSize            int           `json:"batch_size"`
	PrefillChunkSize     int           `json:"prefill_chunk_size"`
	MemoryLimitBytes     uint64        `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes      uint64        `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes      uint64        `json:"wired_limit_bytes,omitempty"`
}

// SmallModelSmokePlan is a metadata-only decision about whether a model should
// be touched by a native Apple smoke run.
type SmallModelSmokePlan struct {
	ModelPath  string                  `json:"model_path"`
	Pack       mp.ModelPack               `json:"pack"`
	Budget     SmallModelSmokeBudget   `json:"budget"`
	MemoryPlan MemoryPlan              `json:"memory_plan"`
	Load       SmallModelSmokeLoadPlan `json:"load"`
	Notes      []string                `json:"notes,omitempty"`
}

// SmallModelSmokeReport captures a guarded native smoke run.
type SmallModelSmokeReport struct {
	Plan       SmallModelSmokePlan  `json:"plan"`
	Skipped    bool                 `json:"skipped"`
	SkipReason string               `json:"skip_reason,omitempty"`
	Bench      *WorkloadBenchReport `json:"bench,omitempty"`
	Error      string               `json:"error,omitempty"`
}

// DefaultSmallModelSmokeConfig returns the Apple-local smoke defaults: q4 only,
// at most 26GiB of weights, and an 8K smoke context even on larger machines.
func DefaultSmallModelSmokeConfig() SmallModelSmokeConfig {
	fast := DefaultFastEvalConfig()
	fast.MaxTokens = DefaultSmallModelSmokeMaxTokens
	fast.Prompt = "Write one short sentence about native Apple inference."
	fast.CachePrompt = fast.Prompt
	fast.IncludeMemvidKVBlockWarm = true
	fast.MemvidKVBlockSize = DefaultCacheBlockSize
	return SmallModelSmokeConfig{
		MaxWeightBytes:         DefaultSmallModelSmokeMaxWeightBytes,
		RequiredQuantization:   DefaultSmallModelSmokeQuantization,
		MaxContextLength:       DefaultSmallModelSmokeMaxContextLength,
		MaxBatchSize:           DefaultSmallModelSmokeMaxBatchSize,
		MaxPrefillChunkSize:    DefaultSmallModelSmokeMaxPrefillChunk,
		IncludeWorkloadBench:   true,
		RequireNativeLoadable:  true,
		RequireValidModelPack:  true,
		RequireKnownWeightSize: true,
		Workload: WorkloadBenchConfig{
			FastEval:            fast,
			IncludeKVCacheBench: true,
		},
	}
}

// EvaluateSmallModelSmokeBudget evaluates the load budget for an inspected pack.
func EvaluateSmallModelSmokeBudget(pack mp.ModelPack, cfg SmallModelSmokeConfig) SmallModelSmokeBudget {
	cfg = normalizeSmallModelSmokeConfig(cfg)
	budget := SmallModelSmokeBudget{
		SafeToLoad:           true,
		MaxWeightBytes:       cfg.MaxWeightBytes,
		RequiredQuantization: cfg.RequiredQuantization,
		WeightBytes:          pack.WeightBytes,
		Quantization:         pack.QuantBits,
		NativeLoadable:       pack.NativeLoadable,
		ValidModelPack:       pack.Valid(),
	}
	switch {
	case cfg.RequireValidModelPack && !pack.Valid():
		budget.SafeToLoad = false
		budget.Reason = "model pack has validation issues"
	case cfg.RequireNativeLoadable && !pack.NativeLoadable:
		budget.SafeToLoad = false
		budget.Reason = "model pack is not native-loadable by go-mlx"
	case cfg.RequireKnownWeightSize && pack.WeightBytes == 0:
		budget.SafeToLoad = false
		budget.Reason = "model weight size is unknown"
	case cfg.RequiredQuantization > 0 && pack.QuantBits == 0:
		budget.SafeToLoad = false
		budget.Reason = core.Sprintf("model quantization is unknown; q%d is required for this smoke run", cfg.RequiredQuantization)
	case cfg.RequiredQuantization > 0 && pack.QuantBits != cfg.RequiredQuantization:
		budget.SafeToLoad = false
		budget.Reason = core.Sprintf("model is q%d; q%d is required for this smoke run", pack.QuantBits, cfg.RequiredQuantization)
	case cfg.MaxWeightBytes > 0 && pack.WeightBytes > cfg.MaxWeightBytes:
		budget.SafeToLoad = false
		budget.Reason = core.Sprintf("model weights use %d bytes; smoke budget is %d bytes", pack.WeightBytes, cfg.MaxWeightBytes)
	}
	return budget
}

// PlanSmallModelSmoke inspects a model and builds a safe load shape without
// loading weights.
func PlanSmallModelSmoke(modelPath string, cfg SmallModelSmokeConfig) (SmallModelSmokePlan, error) {
	cfg = normalizeSmallModelSmokeConfig(cfg)
	if modelPath == "" {
		modelPath = cfg.ModelPath
	}
	if modelPath == "" {
		return SmallModelSmokePlan{}, core.NewError("mlx: small model smoke requires a model path")
	}
	pack, err := InspectModelPack(modelPath, smallModelSmokePackOptions(cfg)...)
	if err != nil {
		return SmallModelSmokePlan{}, err
	}
	if !cfg.IncludeChatTemplate {
		pack.ChatTemplate = ""
	}
	memoryPlan := PlanMemory(MemoryPlanInput{Device: cfg.Device, Pack: &pack})
	plan := SmallModelSmokePlan{
		ModelPath:  modelPath,
		Pack:       pack,
		Budget:     EvaluateSmallModelSmokeBudget(pack, cfg),
		MemoryPlan: memoryPlan,
		Load:       smallModelSmokeLoadPlan(memoryPlan, cfg),
	}
	if cfg.MaxContextLength > 0 && memoryPlan.ContextLength > cfg.MaxContextLength {
		plan.Notes = append(plan.Notes, core.Sprintf("smoke context capped from %d to %d tokens", memoryPlan.ContextLength, cfg.MaxContextLength))
	}
	if !plan.Budget.SafeToLoad && plan.Budget.Reason != "" {
		plan.Notes = append(plan.Notes, plan.Budget.Reason)
	}
	return plan, nil
}

// RunSmallModelSmoke performs a guarded load and workload bench for a small
// local model. Oversize or non-q4 models are reported as skipped, not loaded.
func RunSmallModelSmoke(ctx context.Context, cfg SmallModelSmokeConfig) (*SmallModelSmokeReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeSmallModelSmokeConfig(cfg)
	plan, err := PlanSmallModelSmoke(cfg.ModelPath, cfg)
	if err != nil {
		return nil, err
	}
	report := &SmallModelSmokeReport{Plan: plan}
	if !plan.Budget.SafeToLoad {
		report.Skipped = true
		report.SkipReason = plan.Budget.Reason
		return report, nil
	}
	model, err := LoadModel(plan.ModelPath, smallModelSmokeLoadOptions(plan, cfg)...)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	defer model.Close()
	if !cfg.IncludeWorkloadBench {
		return report, nil
	}
	bench, err := RunModelWorkloadBench(ctx, model, cfg.Workload)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	report.Bench = bench
	return report, nil
}

func normalizeSmallModelSmokeConfig(cfg SmallModelSmokeConfig) SmallModelSmokeConfig {
	def := DefaultSmallModelSmokeConfig()
	if cfg.MaxWeightBytes == 0 {
		cfg.MaxWeightBytes = def.MaxWeightBytes
	}
	if cfg.RequiredQuantization == 0 {
		cfg.RequiredQuantization = def.RequiredQuantization
	}
	if cfg.MaxContextLength == 0 {
		cfg.MaxContextLength = def.MaxContextLength
	}
	if cfg.MaxBatchSize == 0 {
		cfg.MaxBatchSize = def.MaxBatchSize
	}
	if cfg.MaxPrefillChunkSize == 0 {
		cfg.MaxPrefillChunkSize = def.MaxPrefillChunkSize
	}
	if cfg.Workload.FastEval.Prompt == "" && cfg.Workload.FastEval.MaxTokens == 0 {
		cfg.Workload = def.Workload
	}
	if !cfg.IncludeWorkloadBench {
		cfg.IncludeWorkloadBench = def.IncludeWorkloadBench
	}
	if !cfg.RequireNativeLoadable {
		cfg.RequireNativeLoadable = def.RequireNativeLoadable
	}
	if !cfg.RequireValidModelPack {
		cfg.RequireValidModelPack = def.RequireValidModelPack
	}
	if !cfg.RequireKnownWeightSize {
		cfg.RequireKnownWeightSize = def.RequireKnownWeightSize
	}
	return cfg
}

func smallModelSmokePackOptions(cfg SmallModelSmokeConfig) []mp.ModelPackOption {
	opts := []mp.ModelPackOption{mp.WithPackRequireChatTemplate(false)}
	if cfg.RequiredQuantization > 0 {
		opts = append(opts, mp.WithPackQuantization(cfg.RequiredQuantization))
	}
	return opts
}

func smallModelSmokeLoadPlan(plan MemoryPlan, cfg SmallModelSmokeConfig) SmallModelSmokeLoadPlan {
	contextLength := plan.ContextLength
	if cfg.MaxContextLength > 0 && (contextLength == 0 || contextLength > cfg.MaxContextLength) {
		contextLength = cfg.MaxContextLength
	}
	batchSize := maxPositive(plan.BatchSize, 1)
	if cfg.MaxBatchSize > 0 && batchSize > cfg.MaxBatchSize {
		batchSize = cfg.MaxBatchSize
	}
	prefillChunkSize := maxPositive(plan.PrefillChunkSize, 512)
	if cfg.MaxPrefillChunkSize > 0 && prefillChunkSize > cfg.MaxPrefillChunkSize {
		prefillChunkSize = cfg.MaxPrefillChunkSize
	}
	promptCacheMinTokens := plan.PromptCacheMinTokens
	if promptCacheMinTokens == 0 && plan.PromptCache {
		promptCacheMinTokens = DefaultSmallModelSmokePromptCacheMinSize
	}
	return SmallModelSmokeLoadPlan{
		ContextLength:        contextLength,
		ParallelSlots:        maxPositive(plan.ParallelSlots, 1),
		PromptCache:          plan.PromptCache,
		PromptCacheMinTokens: promptCacheMinTokens,
		Quantization:         cfg.RequiredQuantization,
		CachePolicy:          plan.CachePolicy,
		CacheMode:            plan.CacheMode,
		BatchSize:            batchSize,
		PrefillChunkSize:     prefillChunkSize,
		MemoryLimitBytes:     plan.MemoryLimitBytes,
		CacheLimitBytes:      plan.CacheLimitBytes,
		WiredLimitBytes:      plan.WiredLimitBytes,
	}
}

func smallModelSmokeLoadOptions(plan SmallModelSmokePlan, cfg SmallModelSmokeConfig) []LoadOption {
	load := plan.Load
	opts := []LoadOption{
		WithMemoryPlan(plan.MemoryPlan),
		WithContextLength(load.ContextLength),
		WithParallelSlots(load.ParallelSlots),
		WithPromptCache(load.PromptCache),
		WithPromptCacheMinTokens(load.PromptCacheMinTokens),
		WithQuantization(load.Quantization),
		WithExpectedQuantization(load.Quantization),
		WithCachePolicy(load.CachePolicy),
		WithKVCacheMode(load.CacheMode),
		WithBatchSize(load.BatchSize),
		WithPrefillChunkSize(load.PrefillChunkSize),
		WithAllocatorLimits(load.MemoryLimitBytes, load.CacheLimitBytes, load.WiredLimitBytes),
	}
	opts = append(opts, cfg.AdditionalLoadOptions...)
	return opts
}
