// SPDX-Licence-Identifier: EUPL-1.2

package mlx

const (
	// ProductionLaneName is the local agentic runtime lane exercised by the
	// driver-profile benchmark artefacts.
	ProductionLaneName = "gemma4-e2b-it-q4"
	// ProductionLaneModelID is the Hugging Face repository for the target lane.
	ProductionLaneModelID = "mlx-community/gemma-4-e2b-it-4bit"
	// ProductionLaneArchitecture is the canonical architecture reported by
	// model-pack inspection for the target lane.
	ProductionLaneArchitecture = "gemma4_text"
	// ProductionLaneChatTemplate is the chat renderer used for the target lane.
	ProductionLaneChatTemplate = "gemma4"
	// ProductionLaneQuantBits is the archived q4 smoke/control baseline. It is
	// not the product default once official Google E2B 6-bit packs are validated.
	ProductionLaneQuantBits = 4
	// ProductionLaneProductDefaultQuantBits is the app-facing Gemma 4 E2B
	// default when memory planning says it fits without falling back.
	ProductionLaneProductDefaultQuantBits = 6
	// ProductionLaneQualityQuantBits is the app-facing quality-first choice for
	// machines with enough memory headroom.
	ProductionLaneQualityQuantBits = 8
	// ProductionLaneConstrainedQuantBits is the explicit lower-memory fallback
	// for phones, older machines, or very long retained contexts.
	ProductionLaneConstrainedQuantBits = 4
	// ProductionLaneContextLength is the driver-profile context used by GOAL.md.
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
	// ProductionLaneLongFormMaxTokens is the default per-turn long-form
	// generation allowance.
	ProductionLaneLongFormMaxTokens = 8192
	// ProductionLaneMaxTokens is the target driver-profile token budget.
	ProductionLaneMaxTokens = 128
	// ProductionLaneRuns is the target driver-profile run count.
	ProductionLaneRuns = 3

	// Runtime gate names used by the accepted Gemma 4 fast lane.
	Gemma4FastRuntimeGateExpertIDMatVec        = "GO_MLX_ENABLE_EXPERT_ID_MATVEC"
	Gemma4FastRuntimeGateExpertIDFused         = "GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION"
	Gemma4FastRuntimeGateSortedExpertPrefill   = "GO_MLX_ENABLE_SORTED_EXPERT_PREFILL"
	Gemma4FastRuntimeGateNativeMLPMatVec       = "GO_MLX_ENABLE_NATIVE_MLP_MATVEC"
	Gemma4FastRuntimeGateNativeLinearMatVec    = "GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC"
	Gemma4FastRuntimeGateNativeRouterMatVec    = "GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC"
	Gemma4FastRuntimeGateNativeRouterTopK      = "GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK"
	Gemma4FastRuntimeGateFixedGemma4Cache      = "GO_MLX_ENABLE_FIXED_GEMMA4_CACHE"
	Gemma4FastRuntimeGateFixedGemma4Sliding    = "GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND"
	Gemma4FastRuntimeGateFixedGemma4SharedMask = "GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK"
	Gemma4FastRuntimeGateNativeFixedSliding    = "GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION"
	Gemma4FastRuntimeGateDirectGreedyToken     = "GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN"
	Gemma4FastRuntimeGateGenerationStream      = "GO_MLX_ENABLE_GENERATION_STREAM"
	Gemma4FastRuntimeGateAsyncDecodePrefetch   = "GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH"
	Gemma4FastRuntimeGatePagedDecodeFastConcat = "GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT"
	Gemma4FastRuntimeGateNativePagedAttention  = "GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION"
)

var defaultGemma4FastRuntimeGates = []string{
	Gemma4FastRuntimeGateExpertIDMatVec,
	Gemma4FastRuntimeGateExpertIDFused,
	Gemma4FastRuntimeGateSortedExpertPrefill,
	Gemma4FastRuntimeGateNativeMLPMatVec,
	Gemma4FastRuntimeGateNativeLinearMatVec,
	Gemma4FastRuntimeGateNativeRouterMatVec,
	Gemma4FastRuntimeGateNativeRouterTopK,
	Gemma4FastRuntimeGateDirectGreedyToken,
	Gemma4FastRuntimeGateGenerationStream,
	Gemma4FastRuntimeGateAsyncDecodePrefetch,
	Gemma4FastRuntimeGatePagedDecodeFastConcat,
}

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
	Name            string `json:"name"`
	Bits            int    `json:"bits"`
	Purpose         string `json:"purpose"`
	ProductDefault  bool   `json:"product_default,omitempty"`
	QualityFirst    bool   `json:"quality_first,omitempty"`
	ConstrainedOnly bool   `json:"constrained_only,omitempty"`
	ArchivedControl bool   `json:"archived_control,omitempty"`
}

// ProductionQuantizationPolicy is the machine-readable ladder the app can use
// when choosing an official Gemma 4 E2B pack.
type ProductionQuantizationPolicy struct {
	TargetModelID    string                       `json:"target_model_id"`
	DefaultBits      int                          `json:"default_bits"`
	QualityBits      int                          `json:"quality_bits"`
	ConstrainedBits  int                          `json:"constrained_bits"`
	ArchivedBaseline string                       `json:"archived_baseline,omitempty"`
	Tiers            []ProductionQuantizationTier `json:"tiers"`
}

// DefaultProductionLane returns the Gemma 4 E2B q4 target used for production
// local agentic profiling as an archived baseline. Qwen lanes remain
// contract-covered alternatives, but they do not replace the baseline without
// changing this descriptor.
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
	return ProductionQuantizationPolicy{
		TargetModelID:    "google/gemma-4-E2B-it",
		DefaultBits:      ProductionLaneProductDefaultQuantBits,
		QualityBits:      ProductionLaneQualityQuantBits,
		ConstrainedBits:  ProductionLaneConstrainedQuantBits,
		ArchivedBaseline: ProductionLaneModelID,
		Tiers: []ProductionQuantizationTier{
			{
				Name:           "quality",
				Bits:           ProductionLaneQualityQuantBits,
				Purpose:        "prefer when hardware and retained-context memory headroom allow it",
				QualityFirst:   true,
				ProductDefault: false,
			},
			{
				Name:           "default",
				Bits:           ProductionLaneProductDefaultQuantBits,
				Purpose:        "normal app default; lowest tier expected to avoid consistent 4-bit quality loss",
				ProductDefault: true,
			},
			{
				Name:            "constrained",
				Bits:            ProductionLaneConstrainedQuantBits,
				Purpose:         "explicit low-memory fallback for phones, older machines, or very long retained contexts",
				ConstrainedOnly: true,
				ArchivedControl: true,
			},
		},
	}
}

// DefaultGemma4FastRuntimeGates returns the accepted Gemma 4 runtime gates used
// by the current packed expert-ID fast lane. Rejected diagnostic gates such as
// full native layer/model wrappers are intentionally excluded.
//
// The result shares the package-init singleton — callers in this codebase only
// range over it (cmd/mlx/main.go) and never mutate or store-then-mutate. The
// slice is immutable after package init; treat it as read-only.
func DefaultGemma4FastRuntimeGates() []string {
	return defaultGemma4FastRuntimeGates
}
