// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/profile"
)

const (
	// ProductionLaneName is the local agentic runtime lane exercised by the
	// driver-profile benchmark artefacts.
	ProductionLaneName = "gemma4-e2b-it-q6"
	// ProductionLaneModelID is the Hugging Face repository for the target lane.
	ProductionLaneModelID = "mlx-community/gemma-4-e2b-it-6bit"
	// ProductionLaneArchitecture is the canonical architecture reported by
	// model-pack inspection for the target lane.
	ProductionLaneArchitecture = "gemma4_text"
	// ProductionLaneChatTemplate is the chat renderer used for the target lane.
	ProductionLaneChatTemplate = "gemma4"
	// ProductionLaneQuantBits is the product default Gemma 4 E2B weight tier.
	ProductionLaneQuantBits = 6
	// ProductionLaneActiveParameterEstimate is the approximate active parameter
	// count per E2B forward pass used for the memory-bandwidth throughput
	// model. The official E2B assistant lane still has to validate this against
	// measured Apple Silicon bandwidth before promotion.
	ProductionLaneActiveParameterEstimate = 2300000000
	// ProductionLaneProductDefaultQuantBits is the app-facing Gemma 4 E2B
	// default when memory planning says it fits without falling back.
	ProductionLaneProductDefaultQuantBits = 6
	// ProductionLaneQualityQuantBits is the app-facing quality-first choice for
	// machines with enough memory headroom.
	ProductionLaneQualityQuantBits = 8
	// ProductionLaneConstrainedQuantBits is the explicit lower-memory fallback
	// for phones, older machines, or very long retained contexts.
	ProductionLaneConstrainedQuantBits = 4
	// ProductionLaneArchivedBaselineName identifies the old q4 smoke/control
	// lane kept for regression comparison, not the product default.
	ProductionLaneArchivedBaselineName = "gemma4-e2b-it-q4"
	// ProductionLaneArchivedBaselineModelID is the archived q4 control pack.
	ProductionLaneArchivedBaselineModelID = "mlx-community/gemma-4-e2b-it-4bit"
	// ProductionLaneArchivedBaselineQuantBits is the q4 control width.
	ProductionLaneArchivedBaselineQuantBits = 4
	// ProductionLaneContextLength is the short explicit profile/smoke context.
	// Default driver-profile runs leave context unset and resolve from the
	// loaded model instead.
	ProductionLaneContextLength = 4096
	// ProductionLaneLongContextLength is the opencode-sized diagnostic context.
	ProductionLaneLongContextLength = 32768
	// ProductionLaneLongContextPrefillChunkSize is the proven large-context
	// Gemma 4 prefill chunk size for digestible model ingestion.
	ProductionLaneLongContextPrefillChunkSize = 512
	// ProductionLaneLongContextPromptChunkBytes is the proven large-context
	// prompt chunk size for avoiding repeated giant-string tokenisation.
	ProductionLaneLongContextPromptChunkBytes = 4096
	// ProductionLanePagedKVPageSize is the accepted paged K/V block size for
	// retained-state runs. It is a storage-layout default, not a context cutoff.
	ProductionLanePagedKVPageSize = 2048
	// ProductionLaneRetainedKVCacheDType is the accepted K/V storage dtype for
	// retained-state Gemma 4 runs.
	ProductionLaneRetainedKVCacheDType = "fp16"
	// ProductionLaneHyperLongContextLength is the Gemma 4 E2B/E4B 128Ki stress
	// ceiling used by 100k retained-state and warm build-up profiles.
	ProductionLaneHyperLongContextLength = 131072
	// ProductionLaneLongFormMaxTokens is the retained-state chapter turn
	// allowance used by long-form workflows that need an explicit per-turn cap.
	ProductionLaneLongFormMaxTokens = 8192
	// ProductionLaneMaxTokens leaves Gemma 4 driver-profile generation uncapped
	// at the production-lane layer. The driver resolves 0 from loaded model
	// capacity, so short probes must opt in with an explicit max-token override.
	ProductionLaneMaxTokens = 0
	// ProductionLaneRuns is the target driver-profile run count.
	ProductionLaneRuns = 3

	// Runtime gate names used by the accepted Gemma 4 fast lane.
	Gemma4FastRuntimeGateExpertIDMatVec         = "GO_MLX_ENABLE_EXPERT_ID_MATVEC"
	Gemma4FastRuntimeGateExpertIDFused          = "GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION"
	Gemma4FastRuntimeGateSortedExpertPrefill    = "GO_MLX_ENABLE_SORTED_EXPERT_PREFILL"
	Gemma4FastRuntimeGateNativeMLPMatVec        = "GO_MLX_ENABLE_NATIVE_MLP_MATVEC"
	Gemma4FastRuntimeGateNativeLinearMatVec     = "GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC"
	Gemma4FastRuntimeGateNativeRouterMatVec     = "GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC"
	Gemma4FastRuntimeGateNativeRouterTopK       = "GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK"
	Gemma4FastRuntimeGateNativeQ6Bitstream      = "GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC"
	Gemma4FastRuntimeGateNativeAttentionOMatVec = "GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC"
	Gemma4FastRuntimeGateFixedGemma4Cache       = "GO_MLX_ENABLE_FIXED_GEMMA4_CACHE"
	Gemma4FastRuntimeGateFixedGemma4Sliding     = "GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND"
	Gemma4FastRuntimeGateFixedGemma4SharedMask  = "GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK"
	Gemma4FastRuntimeGateNativeFixedSliding     = "GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION"
	Gemma4FastRuntimeGateDirectGreedyToken      = "GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN"
	Gemma4FastRuntimeGateGenerationStream       = "GO_MLX_ENABLE_GENERATION_STREAM"
	Gemma4FastRuntimeGateAsyncDecodePrefetch    = "GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH"
	Gemma4FastRuntimeGatePagedDecodeFastConcat  = "GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT"
	Gemma4FastRuntimeGateNativePagedAttention   = "GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION"
)

// defaultGemma4FastRuntimeGates is derived from the model-owned engine feature
// declaration (metal.DefaultEngineFeatures) so there is a single source of truth
// for the accepted fast-path set — this package no longer maintains its own
// copy. The Gemma4FastRuntimeGate* name constants above remain (the broader gate
// registry + diagnostics reference them); only the accepted-set membership moved
// to metal, where models declare it.
var defaultGemma4FastRuntimeGates = metal.DefaultEngineFeatures().GateNames()

// ProductionLane describes the current package-owned local runtime target.
type ProductionLane struct {
	Name             string `json:"name"`
	ModelID          string `json:"model_id"`
	Architecture     string `json:"architecture"`
	ChatTemplate     string `json:"chat_template"`
	QuantBits        int    `json:"quant_bits"`
	ContextLength    int    `json:"context_length"`
	MaxTokens        int    `json:"max_tokens"`
	Runs             int    `json:"runs"`
	Prompt           string `json:"prompt"`
	IncludeOutput    bool   `json:"include_output"`
	TraceTokenPhases bool   `json:"trace_token_phases"`
}

// ProductionQuantizationTier describes one app-facing model tier. The q4 tier
// remains available as a constrained fallback and benchmark control, while the
// default user path should prefer q6 once that pack is validated.
type ProductionQuantizationTier struct {
	Name                              string `json:"name"`
	ModelID                           string `json:"model_id"`
	Bits                              int    `json:"bits"`
	Purpose                           string `json:"purpose"`
	ActiveWeightReadBytesPerToken     uint64 `json:"active_weight_read_bytes_per_token,omitempty"`
	MinimumWorkingSetBytes            uint64 `json:"minimum_working_set_bytes,omitempty"`
	LongContextMinimumWorkingSetBytes uint64 `json:"long_context_minimum_working_set_bytes,omitempty"`
	ProductDefault                    bool   `json:"product_default,omitempty"`
	QualityFirst                      bool   `json:"quality_first,omitempty"`
	ConstrainedOnly                   bool   `json:"constrained_only,omitempty"`
	ArchivedControl                   bool   `json:"archived_control,omitempty"`
	StepDownToBits                    int    `json:"step_down_to_bits,omitempty"`
}

// ProductionQuantizationPackSupport records MLX-community Gemma 4 E2B pack
// types that the loader/reporting surface recognises. This is broader than the
// app tier ladder: q6/q8/q4 are product choices, while q5, bf16, and MXFP packs
// are explicit bench/R&D targets until they earn product roles.
type ProductionQuantizationPackSupport struct {
	Name           string `json:"name"`
	ModelID        string `json:"model_id"`
	Bits           int    `json:"bits"`
	QuantMode      string `json:"quant_mode"`
	QuantGroup     int    `json:"quant_group,omitempty"`
	ProductRole    string `json:"product_role,omitempty"`
	Supported      bool   `json:"supported"`
	RequiresBench  bool   `json:"requires_bench,omitempty"`
	RequiresNative bool   `json:"requires_native,omitempty"`
}

// ProductionGemma4BenchmarkTarget records one Gemma 4 family member that the
// Goal 3/4 harness understands. Working targets must load/generate in go-mlx;
// ProductionThroughputCandidate is narrower and means production-compare may
// apply the 100 tok/s floor to that pack.
type ProductionGemma4BenchmarkTarget struct {
	Name                          string `json:"name"`
	ModelID                       string `json:"model_id"`
	Architecture                  string `json:"architecture"`
	Role                          string `json:"role"`
	ContextLength                 int    `json:"context_length,omitempty"`
	QuantBits                     int    `json:"quant_bits,omitempty"`
	WorkingTarget                 bool   `json:"working_target"`
	ProductionThroughputCandidate bool   `json:"production_throughput_candidate"`
	MinimumDecodeTokensPerSec     int    `json:"minimum_decode_tokens_per_sec,omitempty"`
	Notes                         string `json:"notes,omitempty"`
}

// ProductionArchitectureStatusReport is the machine-readable native-runtime
// completion matrix for the production lane.
type ProductionArchitectureStatusReport struct {
	TotalArchitectures        int                         `json:"total_architectures"`
	NativeArchitectures       int                         `json:"native_architectures"`
	MetadataOnlyArchitectures int                         `json:"metadata_only_architectures"`
	NativeIDs                 []string                    `json:"native_ids,omitempty"`
	MetadataOnlyIDs           []string                    `json:"metadata_only_ids,omitempty"`
	RemainingGaps             []ProductionArchitectureGap `json:"remaining_gaps,omitempty"`
}

// ProductionArchitectureGap records one recognised architecture that still
// lacks a production-native load/generate or encoder/scorer implementation.
type ProductionArchitectureGap struct {
	ID            string   `json:"id"`
	Family        string   `json:"family,omitempty"`
	Generation    bool     `json:"generation,omitempty"`
	Chat          bool     `json:"chat,omitempty"`
	Embeddings    bool     `json:"embeddings,omitempty"`
	Rerank        bool     `json:"rerank,omitempty"`
	MoE           bool     `json:"moe,omitempty"`
	ParserID      string   `json:"parser_id,omitempty"`
	ToolParserID  string   `json:"tool_parser_id,omitempty"`
	MissingNative string   `json:"missing_native"`
	NextWork      []string `json:"next_work,omitempty"`
	Notes         []string `json:"notes,omitempty"`
}

// ProductionQuantizationPolicy is the machine-readable ladder the app can use
// when choosing an official Gemma 4 E2B pack.
type ProductionQuantizationPolicy struct {
	TargetModelID            string                              `json:"target_model_id"`
	AssistantModelID         string                              `json:"assistant_model_id,omitempty"`
	DefaultBits              int                                 `json:"default_bits"`
	QualityBits              int                                 `json:"quality_bits"`
	ConstrainedBits          int                                 `json:"constrained_bits"`
	ArchivedBaseline         string                              `json:"archived_baseline,omitempty"`
	ActiveParameterEstimate  uint64                              `json:"active_parameter_estimate,omitempty"`
	DecodeThroughputEstimate string                              `json:"decode_throughput_estimate,omitempty"`
	RequiredBenchmarkMetrics []string                            `json:"required_benchmark_metrics,omitempty"`
	Tiers                    []ProductionQuantizationTier        `json:"tiers"`
	SupportedPacks           []ProductionQuantizationPackSupport `json:"supported_packs"`
}

// ProductionQuantizationSelectionInput carries the app's current hardware and
// workload preference into the Gemma 4 E2B quantisation chooser.
type ProductionQuantizationSelectionInput struct {
	Device              memory.DeviceInfo `json:"device"`
	ContextLength       int               `json:"context_length,omitempty"`
	QualityFirst        bool              `json:"quality_first,omitempty"`
	ConstrainedFallback bool              `json:"constrained_fallback,omitempty"`
}

// ProductionQuantizationChoice is the app-facing selected tier plus the
// memory-fit decision that led to it.
type ProductionQuantizationChoice struct {
	Tier                       ProductionQuantizationTier `json:"tier"`
	Fits                       bool                       `json:"fits"`
	RequestedBits              int                        `json:"requested_bits,omitempty"`
	WorkingSetBytes            uint64                     `json:"working_set_bytes,omitempty"`
	RequiredWorkingSet         uint64                     `json:"required_working_set_bytes,omitempty"`
	LongContextSelection       bool                       `json:"long_context_selection,omitempty"`
	StepDownFromBits           int                        `json:"step_down_from_bits,omitempty"`
	StepDownWorkingSetBytes    uint64                     `json:"step_down_working_set_bytes,omitempty"`
	StepDownRequiredWorkingSet uint64                     `json:"step_down_required_working_set_bytes,omitempty"`
	Reason                     string                     `json:"reason"`
}

var (
	// Production policy defaults are package-init singletons. Public default
	// accessors return defensive slice copies so callers cannot mutate global
	// policy state.
	defaultProductionQuantizationRequiredBenchmarkMetrics = []string{
		"load_duration",
		"peak_memory_bytes",
		"retained_restore_duration",
		"raw_decode_tokens_per_sec",
		"active_weight_read_bytes_per_token",
		"memory_bandwidth_bytes_per_sec",
		"long_output_quality_flags",
		"step_down_working_set_bytes",
		"context_length",
	}
	defaultProductionQuantizationTiers = []ProductionQuantizationTier{
		{
			Name:                              "quality",
			ModelID:                           "mlx-community/gemma-4-e2b-it-8bit",
			Bits:                              ProductionLaneQualityQuantBits,
			Purpose:                           "prefer when hardware and retained-context memory headroom allow it",
			ActiveWeightReadBytesPerToken:     productionQuantizationActiveWeightReadBytes(ProductionLaneQualityQuantBits),
			MinimumWorkingSetBytes:            32 * memory.GiB,
			LongContextMinimumWorkingSetBytes: 64 * memory.GiB,
			QualityFirst:                      true,
			ProductDefault:                    false,
			StepDownToBits:                    ProductionLaneProductDefaultQuantBits,
		},
		{
			Name:                              "default",
			ModelID:                           "mlx-community/gemma-4-e2b-it-6bit",
			Bits:                              ProductionLaneProductDefaultQuantBits,
			Purpose:                           "normal app default; lowest tier expected to avoid consistent 4-bit quality loss",
			ActiveWeightReadBytesPerToken:     productionQuantizationActiveWeightReadBytes(ProductionLaneProductDefaultQuantBits),
			MinimumWorkingSetBytes:            16 * memory.GiB,
			LongContextMinimumWorkingSetBytes: 24 * memory.GiB,
			ProductDefault:                    true,
			StepDownToBits:                    ProductionLaneConstrainedQuantBits,
		},
		{
			Name:                              "constrained",
			ModelID:                           ProductionLaneArchivedBaselineModelID,
			Bits:                              ProductionLaneConstrainedQuantBits,
			Purpose:                           "explicit low-memory fallback for phones, older machines, or very long retained contexts",
			ActiveWeightReadBytesPerToken:     productionQuantizationActiveWeightReadBytes(ProductionLaneConstrainedQuantBits),
			MinimumWorkingSetBytes:            8 * memory.GiB,
			LongContextMinimumWorkingSetBytes: 12 * memory.GiB,
			ConstrainedOnly:                   true,
			ArchivedControl:                   true,
		},
	}
	defaultProductionQuantizationPackSupport = []ProductionQuantizationPackSupport{
		{
			Name:          "mxfp4",
			ModelID:       "mlx-community/gemma-4-e2b-it-mxfp4",
			Bits:          4,
			QuantMode:     "mxfp4",
			QuantGroup:    32,
			ProductRole:   "research",
			Supported:     true,
			RequiresBench: true,
		},
		{
			Name:          "mxfp8",
			ModelID:       "mlx-community/gemma-4-e2b-it-mxfp8",
			Bits:          8,
			QuantMode:     "mxfp8",
			QuantGroup:    32,
			ProductRole:   "research",
			Supported:     true,
			RequiresBench: true,
		},
		{
			Name:        "4bit",
			ModelID:     ProductionLaneArchivedBaselineModelID,
			Bits:        ProductionLaneConstrainedQuantBits,
			QuantMode:   "affine",
			QuantGroup:  64,
			ProductRole: "constrained",
			Supported:   true,
		},
		{
			Name:          "5bit",
			ModelID:       "mlx-community/gemma-4-e2b-it-5bit",
			Bits:          5,
			QuantMode:     "affine",
			QuantGroup:    64,
			ProductRole:   "bench",
			Supported:     true,
			RequiresBench: true,
		},
		{
			Name:        "6bit",
			ModelID:     ProductionLaneModelID,
			Bits:        ProductionLaneProductDefaultQuantBits,
			QuantMode:   "affine",
			QuantGroup:  64,
			ProductRole: "default",
			Supported:   true,
		},
		{
			Name:        "8bit",
			ModelID:     "mlx-community/gemma-4-e2b-it-8bit",
			Bits:        ProductionLaneQualityQuantBits,
			QuantMode:   "affine",
			QuantGroup:  64,
			ProductRole: "quality",
			Supported:   true,
		},
		{
			Name:           "bf16",
			ModelID:        "mlx-community/gemma-4-e2b-it-bf16",
			Bits:           16,
			QuantMode:      "bf16",
			ProductRole:    "quality-control",
			Supported:      true,
			RequiresBench:  true,
			RequiresNative: true,
		},
	}
	defaultProductionGemma4BenchmarkTargets = []ProductionGemma4BenchmarkTarget{
		{
			Name:                          "e2b",
			ModelID:                       OfficialGemma4E2BTargetLock().ModelID,
			Architecture:                  "gemma4_text",
			Role:                          "coder-production",
			ContextLength:                 ProductionLaneHyperLongContextLength,
			QuantBits:                     ProductionLaneProductDefaultQuantBits,
			WorkingTarget:                 true,
			ProductionThroughputCandidate: true,
			MinimumDecodeTokensPerSec:     ProductionMTPPromotionMinDecodeTokensPerSec,
			Notes:                         "primary coder pack; production compare applies the 100 tok/s floor with MTP/quant evidence",
		},
		{
			Name:                          "e4b",
			ModelID:                       "google/gemma-4-E4B-it",
			Architecture:                  "gemma4_text",
			Role:                          "coder-production",
			ContextLength:                 ProductionLaneHyperLongContextLength,
			QuantBits:                     ProductionLaneProductDefaultQuantBits,
			WorkingTarget:                 true,
			ProductionThroughputCandidate: true,
			MinimumDecodeTokensPerSec:     ProductionMTPPromotionMinDecodeTokensPerSec,
			Notes:                         "larger coder pack; production compare applies the 100 tok/s floor only after pack-specific go-mlx artefacts exist",
		},
		{
			Name:          "12b-unified",
			ModelID:       DefaultOfficialGemma412BUnifiedSourceLock().ModelID,
			Architecture:  "gemma4_unified",
			Role:          "unified-validation",
			ContextLength: DefaultOfficialGemma412BUnifiedSourceLock().TextConfig.MaxPositionEmbeddings,
			QuantBits:     6,
			WorkingTarget: true,
			Notes:         "validation-only Unified pack; useful to keep working, but not scheduled for the 100 tok/s production floor",
		},
		{
			Name:          "31b",
			ModelID:       "google/gemma-4-31B-it",
			Architecture:  "gemma4_text",
			Role:          "large-validation",
			ContextLength: ProductionLaneHyperLongContextLength * 2,
			WorkingTarget: true,
			Notes:         "large dense validation target; not a production-throughput candidate",
		},
		{
			Name:          "26b-moe",
			ModelID:       "google/gemma-4-26B-it",
			Architecture:  "gemma4_text",
			Role:          "moe-validation",
			ContextLength: ProductionLaneHyperLongContextLength * 2,
			WorkingTarget: true,
			Notes:         "MoE validation target; must keep router/expert path working, but not a 100 tok/s production candidate",
		},
	}
	defaultProductionQuantizationPolicy = ProductionQuantizationPolicy{
		TargetModelID:            OfficialGemma4E2BTargetLock().ModelID,
		AssistantModelID:         OfficialGemma4E2BAssistantLock().ModelID,
		DefaultBits:              ProductionLaneProductDefaultQuantBits,
		QualityBits:              ProductionLaneQualityQuantBits,
		ConstrainedBits:          ProductionLaneConstrainedQuantBits,
		ArchivedBaseline:         ProductionLaneArchivedBaselineModelID,
		ActiveParameterEstimate:  ProductionLaneActiveParameterEstimate,
		DecodeThroughputEstimate: "tok/s ~= measured memory bandwidth bytes/sec / active weight read bytes/token",
		RequiredBenchmarkMetrics: defaultProductionQuantizationRequiredBenchmarkMetrics,
		Tiers:                    defaultProductionQuantizationTiers,
		SupportedPacks:           defaultProductionQuantizationPackSupport,
	}
)

// DefaultProductionLane returns the Gemma 4 E2B q6 target used for production
// local agentic profiling. Qwen lanes remain contract-covered alternatives,
// but they do not replace the baseline without changing this descriptor.
func DefaultProductionLane() ProductionLane {
	return ProductionLane{
		Name:             ProductionLaneName,
		ModelID:          ProductionLaneModelID,
		Architecture:     ProductionLaneArchitecture,
		ChatTemplate:     ProductionLaneChatTemplate,
		QuantBits:        ProductionLaneQuantBits,
		ContextLength:    ProductionLaneContextLength,
		MaxTokens:        ProductionLaneMaxTokens,
		Runs:             ProductionLaneRuns,
		Prompt:           DefaultNewSessionText,
		IncludeOutput:    false,
		TraceTokenPhases: true,
	}
}

// DefaultProductionQuantizationPolicy returns the app-facing Gemma 4 E2B
// quantisation ladder. It intentionally lives beside, not inside,
// DefaultProductionLane so historical q4 benchmark artefacts remain stable.
func DefaultProductionQuantizationPolicy() ProductionQuantizationPolicy {
	policy := defaultProductionQuantizationPolicy
	policy.RequiredBenchmarkMetrics = append([]string(nil), policy.RequiredBenchmarkMetrics...)
	policy.Tiers = append([]ProductionQuantizationTier(nil), policy.Tiers...)
	policy.SupportedPacks = append([]ProductionQuantizationPackSupport(nil), policy.SupportedPacks...)
	return policy
}

// DefaultProductionQuantizationPackSupport returns every MLX-community Gemma 4
// E2B pack type that the production lane recognises for product selection,
// benchmark selection, or R&D validation.
func DefaultProductionQuantizationPackSupport() []ProductionQuantizationPackSupport {
	return append([]ProductionQuantizationPackSupport(nil), defaultProductionQuantizationPackSupport...)
}

// DefaultProductionGemma4BenchmarkTargets returns the five Gemma 4 family
// targets that Goal 3/4 tracks. Only E2B/E4B are production-throughput
// candidates; the larger Unified/dense/MoE packs are working validation targets
// unless a future policy explicitly promotes them.
func DefaultProductionGemma4BenchmarkTargets() []ProductionGemma4BenchmarkTarget {
	return append([]ProductionGemma4BenchmarkTarget(nil), defaultProductionGemma4BenchmarkTargets...)
}

// ProductionQuantizationPackByName resolves a supported pack by its short name
// ("6bit", "mxfp8") or model ID. It is intended for benchmark harnesses that
// need a concrete pack target without changing the app-facing q6/q8/q4 ladder.
func ProductionQuantizationPackByName(name string) (ProductionQuantizationPackSupport, bool) {
	needle := core.Lower(core.Trim(name))
	if needle == "" {
		return ProductionQuantizationPackSupport{}, false
	}
	for _, pack := range defaultProductionQuantizationPackSupport {
		if core.Lower(pack.Name) == needle || core.Lower(pack.ModelID) == needle {
			return pack, true
		}
	}
	return ProductionQuantizationPackSupport{}, false
}

// DefaultProductionArchitectureStatus reports native-runtime completion for
// every recognised architecture. It is intentionally derived from
// profile.BuiltinArchitectureProfiles so GOAL.md progress, CLI reports, and
// model-pack planning all share one source of truth.
func DefaultProductionArchitectureStatus() ProductionArchitectureStatusReport {
	profiles := profile.BuiltinArchitectureProfiles()
	report := ProductionArchitectureStatusReport{
		TotalArchitectures: len(profiles),
		NativeIDs:          make([]string, 0, len(profiles)),
		MetadataOnlyIDs:    make([]string, 0, len(profiles)),
		RemainingGaps:      make([]ProductionArchitectureGap, 0, len(profiles)),
	}
	for _, prof := range profiles {
		if prof.NativeRuntime {
			report.NativeArchitectures++
			report.NativeIDs = append(report.NativeIDs, prof.ID)
			continue
		}
		report.MetadataOnlyArchitectures++
		report.MetadataOnlyIDs = append(report.MetadataOnlyIDs, prof.ID)
		report.RemainingGaps = append(report.RemainingGaps, productionArchitectureGap(prof))
	}
	return report
}

func productionArchitectureGap(prof profile.ModelArchitectureProfile) ProductionArchitectureGap {
	return ProductionArchitectureGap{
		ID:            prof.ID,
		Family:        prof.Family,
		Generation:    prof.Generation,
		Chat:          prof.Chat,
		Embeddings:    prof.Embeddings,
		Rerank:        prof.Rerank,
		MoE:           prof.MoE,
		ParserID:      prof.ParserID,
		ToolParserID:  prof.ToolParserID,
		MissingNative: productionArchitectureMissingNative(prof),
		NextWork:      productionArchitectureNextWork(prof),
		Notes:         append([]string(nil), prof.Notes...),
	}
}

func productionArchitectureMissingNative(prof profile.ModelArchitectureProfile) string {
	if prof.Embeddings {
		return "embedding encoder"
	}
	if prof.Rerank {
		return "rerank scorer"
	}
	if prof.MoE {
		if prof.ID == "qwen3_6_moe" {
			return "hybrid linear attention plus sparse expert router"
		}
		if prof.ID == "deepseek" {
			return "MoE router plus MLA attention variants"
		}
		if prof.ID == "gpt_oss" {
			return "MoE router plus channel parser validation"
		}
		return "sparse expert router"
	}
	if prof.ID == "qwen3_6" {
		return "hybrid linear attention"
	}
	return "native loader"
}

func productionArchitectureNextWork(prof profile.ModelArchitectureProfile) []string {
	switch prof.ID {
	case "qwen3_6":
		return []string{"linear_attention_kernel", "native_load_generate_smoke", "retained_state_smoke"}
	case "qwen3_6_moe":
		return []string{"linear_attention_kernel", "sparse_expert_router", "native_load_generate_smoke"}
	case "qwen3_moe", "mixtral", "kimi":
		return []string{"sparse_expert_router", "selected_expert_matvec", "native_load_generate_smoke"}
	case "deepseek":
		return []string{"sparse_expert_router", "mla_attention_variant", "native_load_generate_smoke"}
	case "gpt_oss":
		return []string{"channel_parser_validation", "sparse_expert_router", "native_load_generate_smoke"}
	case "bert":
		return []string{"encoder_loader", "pooled_embedding_output", "no_generation_kv_smoke"}
	case "bert_rerank":
		return []string{"cross_encoder_loader", "score_head_output", "no_generation_kv_smoke"}
	default:
		return []string{"native_loader", "native_smoke"}
	}
}

func productionQuantizationActiveWeightReadBytes(bits int) uint64 {
	if bits <= 0 {
		return 0
	}
	return (uint64(ProductionLaneActiveParameterEstimate)*uint64(bits) + 7) / 8
}

// SelectProductionQuantizationTier chooses the app-facing Gemma 4 E2B tier.
// The normal path is q6; q8 is opt-in for quality when memory headroom allows,
// and q4 is used only for explicit constrained mode or when q6 does not fit the
// requested retained-context shape.
func SelectProductionQuantizationTier(input ProductionQuantizationSelectionInput) ProductionQuantizationChoice {
	policy := defaultProductionQuantizationPolicy
	defaultTier := productionQuantizationTierByBits(policy, policy.DefaultBits)
	qualityTier := productionQuantizationTierByBits(policy, policy.QualityBits)
	constrainedTier := productionQuantizationTierByBits(policy, policy.ConstrainedBits)

	workingSet := productionQuantizationWorkingSet(input.Device)
	longContext := input.ContextLength >= ProductionLaneLongContextLength
	requestedBits := policy.DefaultBits
	if input.QualityFirst {
		requestedBits = policy.QualityBits
	}

	if input.ConstrainedFallback {
		return productionQuantizationChoice(constrainedTier, workingSet, longContext, policy.ConstrainedBits, "constrained fallback requested")
	}
	if input.QualityFirst {
		if workingSet == 0 {
			return productionQuantizationStepDownChoice(defaultTier, qualityTier, workingSet, longContext, requestedBits, "quality q8 requires measured memory headroom; using q6 default")
		}
		choice := productionQuantizationChoice(qualityTier, workingSet, longContext, requestedBits, "quality tier selected with sufficient headroom")
		if choice.Fits {
			return choice
		}
		defaultChoice := productionQuantizationStepDownChoice(defaultTier, qualityTier, workingSet, longContext, requestedBits, "quality q8 does not fit requested memory/context; using q6 default")
		if defaultChoice.Fits {
			return defaultChoice
		}
	}
	choice := productionQuantizationChoice(defaultTier, workingSet, longContext, requestedBits, "default q6 tier selected")
	if choice.Fits {
		return choice
	}
	fallback := productionQuantizationStepDownChoice(constrainedTier, defaultTier, workingSet, longContext, requestedBits, "q6 does not fit requested memory/context; using q4 fallback")
	if fallback.Fits {
		return fallback
	}
	fallback.Reason = "q4 is the smallest supported tier but still exceeds the measured working set"
	return fallback
}

func productionQuantizationTierByBits(policy ProductionQuantizationPolicy, bits int) ProductionQuantizationTier {
	for _, tier := range policy.Tiers {
		if tier.Bits == bits {
			return tier
		}
	}
	return ProductionQuantizationTier{}
}

func productionQuantizationChoice(tier ProductionQuantizationTier, workingSet uint64, longContext bool, requestedBits int, reason string) ProductionQuantizationChoice {
	required := productionQuantizationRequiredWorkingSet(tier, longContext)
	fits := workingSet == 0 || required == 0 || workingSet >= required
	return ProductionQuantizationChoice{
		Tier:                 tier,
		Fits:                 fits,
		RequestedBits:        requestedBits,
		WorkingSetBytes:      workingSet,
		RequiredWorkingSet:   required,
		LongContextSelection: longContext,
		Reason:               reason,
	}
}

func productionQuantizationStepDownChoice(tier, failedTier ProductionQuantizationTier, workingSet uint64, longContext bool, requestedBits int, reason string) ProductionQuantizationChoice {
	choice := productionQuantizationChoice(tier, workingSet, longContext, requestedBits, reason)
	choice.StepDownFromBits = failedTier.Bits
	choice.StepDownWorkingSetBytes = workingSet
	choice.StepDownRequiredWorkingSet = productionQuantizationRequiredWorkingSet(failedTier, longContext)
	return choice
}

func productionQuantizationRequiredWorkingSet(tier ProductionQuantizationTier, longContext bool) uint64 {
	required := tier.MinimumWorkingSetBytes
	if longContext && tier.LongContextMinimumWorkingSetBytes > required {
		required = tier.LongContextMinimumWorkingSetBytes
	}
	return required
}

func productionQuantizationWorkingSet(device memory.DeviceInfo) uint64 {
	if device.MaxRecommendedWorkingSetSize > 0 {
		return device.MaxRecommendedWorkingSetSize
	}
	return device.MemorySize
}

// DefaultGemma4FastRuntimeGates returns runtime gates promoted into the q6
// production default. Runtime gates remain opt-in until they beat the no-gate
// q6 E2B path on full-output go-mlx self-benchmarks; direct greedy is promoted
// because the q6 self-bench produced the same greedy token hash while reducing
// 49k-context decode wall time. The fast lane still owns context, paged-cache,
// and long-prefill defaults.
//
// The result is a defensive copy of the package-init singleton so callers
// cannot accidentally mutate the production default gate list.
func DefaultGemma4FastRuntimeGates() []string {
	return append([]string(nil), defaultGemma4FastRuntimeGates...)
}

// DefaultGemma4FastRuntimeGateCount returns the number of promoted runtime
// gates without allocating a defensive slice copy.
func DefaultGemma4FastRuntimeGateCount() int {
	return len(defaultGemma4FastRuntimeGates)
}

// DefaultGemma4FastRuntimeGate returns a promoted runtime gate by index without
// allocating a defensive slice copy.
func DefaultGemma4FastRuntimeGate(index int) (string, bool) {
	if index < 0 || index >= len(defaultGemma4FastRuntimeGates) {
		return "", false
	}
	return defaultGemma4FastRuntimeGates[index], true
}
