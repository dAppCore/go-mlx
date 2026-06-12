// SPDX-Licence-Identifier: EUPL-1.2

// Package mlx provides Apple Metal GPU inference via mlx-c bindings.
//
// This package implements the [inference.Backend] interface from
// dappco.re/go/inference for Apple Silicon (M1-M4) GPUs.
// Import it blank to register the "metal" backend automatically:
//
//	import _ "dappco.re/go/mlx"
//
// Build mlx-c before use:
//
//	go generate ./...
//
// # Generate text
//
//	model, err := inference.LoadModel("/path/to/model/")
//	if err != nil { log.Fatal(err) }
//	defer model.Close()
//
//	ctx := context.Background()
//	for token := range model.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(128)) {
//	    fmt.Print(token.Text)
//	}
//	if err := model.Err(); err != nil { log.Fatal(err) }
//
// # Multi-turn chat
//
// Chat applies the model's native template (Gemma3, Qwen3, Llama3):
//
//	for token := range model.Chat(ctx, []inference.Message{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "Translate 'hello' to French."},
//	}, inference.WithMaxTokens(64)) {
//	    fmt.Print(token.Text)
//	}
//
// # Batch classification
//
// Classify runs a single forward pass per prompt (prefill only, no decoding):
//
//	results, err := model.Classify(ctx, []string{
//	    "Bonjour, comment allez-vous?",
//	    "The quarterly report shows growth.",
//	}, inference.WithTemperature(0))
//	for index, result := range results {
//	    fmt.Printf("prompt %d → %q\n", index, result.Token.Text)
//	}
//
// # Batch generation
//
//	results, err := model.BatchGenerate(ctx, []string{
//	    "The capital of France is",
//	    "Water boils at",
//	}, inference.WithMaxTokens(32))
//	for index, result := range results {
//	    for _, token := range result.Tokens {
//	        fmt.Print(token.Text)
//	    }
//	    fmt.Println()
//	}
//
// # Performance metrics
//
// After any inference call, retrieve timing and memory statistics:
//
//	for token := range model.Generate(ctx, prompt, inference.WithMaxTokens(128)) {
//	    fmt.Print(token.Text)
//	}
//	metrics := model.Metrics()
//	fmt.Printf("decode: %.0f tok/s, peak GPU: %d MB\n",
//	    metrics.DecodeTokensPerSec, metrics.PeakMemoryBytes/1024/1024)
//
// # Model info
//
//	modelInfo := model.Info()
//	fmt.Printf("%s %d-layer, %d-bit quantised\n",
//	    modelInfo.Architecture, modelInfo.NumLayers, modelInfo.QuantBits)
//
// # Model discovery
//
//	discoveredModels, err := inference.Discover("/path/to/models/")
//	for _, discoveredModel := range discoveredModels {
//	    fmt.Printf("%s (%s, %d-bit)\n", discoveredModel.Path, discoveredModel.ModelType, discoveredModel.QuantBits)
//	}
//
// # Metal memory controls
//
// These control the Metal allocator directly, not individual models:
//
//	mlx.SetCacheLimit(4 << 30)  // 4 GB cache limit
//	mlx.SetMemoryLimit(32 << 30) // 32 GB hard limit
//
//	// Between chat turns, reclaim prompt cache memory:
//	mlx.ClearCache()
//	model1.Close()
//	mlx.GC() // run Go finalizers for CGO-owned memory without importing runtime
//
//	fmt.Printf("active: %d MB, peak: %d MB\n",
//	    mlx.GetActiveMemory()/1024/1024, mlx.GetPeakMemory()/1024/1024)
package mlx

import (
	// Note: AX-6 - time.Duration is part of the public Metrics API.
	"time"

	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build

// GC runs Go garbage collection for MLX CGO lifecycle cleanup.
//
// Use this after closing large models when prompt/model memory must be
// reclaimed promptly, without importing runtime at call sites.
func GC() { metal.RuntimeGC() }

// SeedRandom resets MLX's default random sequence for subsequent sampling.
func SeedRandom(seed uint64) error { return metal.SeedRandom(seed) }

const (
	// DefaultLocalContextLength is the opt-in local cap used by production
	// lanes and explicit workstation profiles. Default loads leave context
	// length at 0 so the native model metadata can supply the full window.
	DefaultLocalContextLength = 131072
	// DefaultLocalParallelSlots keeps one foreground native request active.
	DefaultLocalParallelSlots = 1
	// DefaultPromptCacheMinTokens avoids cache overhead for short prompts.
	DefaultPromptCacheMinTokens = 2048
)

// Token is a generated token from the RFC-style root API.
type Token struct {
	ID    int32
	Value string
	Text  string
}

// Metrics reports performance counters from the last inference call.
type Metrics struct {
	PromptTokens               int                          `json:"prompt_tokens"`
	GeneratedTokens            int                          `json:"generated_tokens"`
	FirstTokenDuration         time.Duration                `json:"first_token_duration,omitempty"`
	PrefillDuration            time.Duration                `json:"prefill_duration"`
	DecodeDuration             time.Duration                `json:"decode_duration"`
	TotalDuration              time.Duration                `json:"total_duration"`
	PrefillTokensPerSec        float64                      `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec         float64                      `json:"decode_tokens_per_sec"`
	PeakMemoryBytes            uint64                       `json:"peak_memory_bytes"`
	ActiveMemoryBytes          uint64                       `json:"active_memory_bytes"`
	CacheMemoryBytes           uint64                       `json:"cache_memory_bytes"`
	ProcessVirtualMemoryBytes  uint64                       `json:"process_virtual_memory_bytes"`
	ProcessResidentMemoryBytes uint64                       `json:"process_resident_memory_bytes"`
	ProcessPeakResidentBytes   uint64                       `json:"process_peak_resident_bytes"`
	PromptCacheHits            int                          `json:"prompt_cache_hits,omitempty"`
	PromptCacheMisses          int                          `json:"prompt_cache_misses,omitempty"`
	PromptCacheHitTokens       int                          `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens      int                          `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreDuration time.Duration                `json:"prompt_cache_restore_duration,omitempty"`
	CacheProfile               *CacheProfile                `json:"cache_profile,omitempty"`
	TurboQuantKVPayload        *TurboQuantKVPayloadEstimate `json:"turboquant_kv_payload,omitempty"`
	TokenPhases                []TokenPhaseTrace            `json:"token_phases,omitempty"`
	MTP                        *MTPMetrics                  `json:"mtp,omitempty"`
	Adapter                    lora.AdapterInfo             `json:"adapter"`
	// DecodeLane/DecodeLaneReason name the decode loop that served the
	// generation (pipelined vs serial + the first failed eligibility
	// condition); CompiledLayerHits counts whole-layer compiled steps.
	DecodeLane        string `json:"decode_lane,omitempty"`
	DecodeLaneReason  string `json:"decode_lane_reason,omitempty"`
	CompiledLayerHits uint64 `json:"compiled_layer_hits,omitempty"`
}

// TurboQuantKVPayloadEstimate summarises the compressed TurboQuant K/V payload
// currently retained by a generation cache. PayloadBytes is section data before
// alignment padding; PaddedPayloadBytes is the actual retained binary span.
type TurboQuantKVPayloadEstimate struct {
	Pages                     int     `json:"pages"`
	PageVectors               uint64  `json:"page_vectors,omitempty"`
	PageElements              uint64  `json:"page_elements,omitempty"`
	KeyCentroidBytes          uint64  `json:"key_centroid_bytes,omitempty"`
	KeyQJLSignBytes           uint64  `json:"key_qjl_sign_bytes,omitempty"`
	KeyNormBytes              uint64  `json:"key_norm_bytes,omitempty"`
	KeyResidualNormBytes      uint64  `json:"key_residual_norm_bytes,omitempty"`
	ValueCentroidBytes        uint64  `json:"value_centroid_bytes,omitempty"`
	ValueNormBytes            uint64  `json:"value_norm_bytes,omitempty"`
	OutlierMaskBytes          uint64  `json:"outlier_mask_bytes,omitempty"`
	PayloadBytes              uint64  `json:"payload_bytes,omitempty"`
	PaddedPayloadBytes        uint64  `json:"padded_payload_bytes,omitempty"`
	AlignmentPaddingBytes     uint64  `json:"alignment_padding_bytes,omitempty"`
	FP16BaselineBytes         uint64  `json:"fp16_baseline_bytes,omitempty"`
	PayloadToFP16Ratio        float64 `json:"payload_to_fp16_ratio,omitempty"`
	PaddedPayloadToFP16Ratio  float64 `json:"padded_payload_to_fp16_ratio,omitempty"`
	PayloadSavingsRatio       float64 `json:"payload_savings_ratio,omitempty"`
	PaddedPayloadSavingsRatio float64 `json:"padded_payload_savings_ratio,omitempty"`
}

// MTPMetrics records attached multi-token-prediction drafter counters.
type MTPMetrics struct {
	DraftTokenSchedule     []int         `json:"draft_token_schedule,omitempty"`
	ProposedTokens         int           `json:"proposed_tokens,omitempty"`
	AcceptedTokens         int           `json:"accepted_tokens,omitempty"`
	RejectedTokens         int           `json:"rejected_tokens,omitempty"`
	TargetVerifyCalls      int           `json:"target_verify_calls,omitempty"`
	TargetCalls            int           `json:"target_calls,omitempty"`
	DraftCalls             int           `json:"draft_calls,omitempty"`
	AcceptanceRate         float64       `json:"acceptance_rate,omitempty"`
	VisibleTokensPerSec    float64       `json:"visible_tokens_per_sec,omitempty"`
	TargetTokensPerSec     float64       `json:"target_tokens_per_sec,omitempty"`
	WarmDecodeTokensPerSec float64       `json:"warm_decode_tokens_per_sec,omitempty"`
	WallDuration           time.Duration `json:"wall_duration,omitempty"`
	RestoreDuration        time.Duration `json:"restore_duration,omitempty"`
	TargetVerifyDuration   time.Duration `json:"target_verify_duration,omitempty"`
	TargetDuration         time.Duration `json:"target_duration,omitempty"`
	DraftDuration          time.Duration `json:"draft_duration,omitempty"`
	PeakMemoryBytes        uint64        `json:"peak_memory_bytes,omitempty"`
}

// CacheProfile reports the model/cache topology observed after a generation
// turn. Gemma 4 uses this to prove local sliding caches stay bounded while
// global owner layers carry the retained long-context state.
type CacheProfile struct {
	Architecture       string `json:"architecture,omitempty"`
	TotalCaches        int    `json:"total_caches"`
	LocalCaches        int    `json:"local_caches"`
	GlobalCaches       int    `json:"global_caches"`
	SharedLayers       int    `json:"shared_layers"`
	CachelessLayers    int    `json:"cacheless_layers"`
	LocalWindowTokens  int    `json:"local_window_tokens"`
	MaxLocalTokens     int    `json:"max_local_tokens"`
	MaxLocalCapacity   int    `json:"max_local_capacity"`
	MaxGlobalTokens    int    `json:"max_global_tokens"`
	MaxGlobalCapacity  int    `json:"max_global_capacity"`
	MaxCacheTokens     int    `json:"max_cache_tokens"`
	MaxCacheCapacity   int    `json:"max_cache_capacity"`
	MaxProcessedTokens int    `json:"max_processed_tokens"`
	FullCaches         int    `json:"full_caches"`
	RotatingCaches     int    `json:"rotating_caches"`
	FixedCaches        int    `json:"fixed_caches"`
	PagedCaches        int    `json:"paged_caches"`
	QuantizedCaches    int    `json:"quantized_caches"`
	UnknownCaches      int    `json:"unknown_caches"`
	UnboundedCaches    int    `json:"unbounded_caches"`
	LocalWindowLeaked  bool   `json:"local_window_leaked"`
}

// TokenPhaseTrace reports the coarse decode-loop cost for one generated token.
type TokenPhaseTrace struct {
	Step                   int                `json:"step"`
	TokenID                int32              `json:"token_id"`
	TokenText              string             `json:"token_text,omitempty"`
	FinalToken             bool               `json:"final_token,omitempty"`
	TotalDuration          time.Duration      `json:"total_duration,omitempty"`
	LogitsDuration         time.Duration      `json:"logits_duration,omitempty"`
	SampleDuration         time.Duration      `json:"sample_duration,omitempty"`
	SampleEvalDuration     time.Duration      `json:"sample_eval_duration,omitempty"`
	TokenReadDuration      time.Duration      `json:"token_read_duration,omitempty"`
	DecodeTextDuration     time.Duration      `json:"decode_text_duration,omitempty"`
	ProbeTokenDuration     time.Duration      `json:"probe_token_duration,omitempty"`
	YieldDuration          time.Duration      `json:"yield_duration,omitempty"`
	NextInputDuration      time.Duration      `json:"next_input_duration,omitempty"`
	ForwardDuration        time.Duration      `json:"forward_duration,omitempty"`
	PrefetchDuration       time.Duration      `json:"prefetch_duration,omitempty"`
	PrefetchLogitsDuration time.Duration      `json:"prefetch_logits_duration,omitempty"`
	PrefetchCacheDuration  time.Duration      `json:"prefetch_cache_duration,omitempty"`
	MaterializeDuration    time.Duration      `json:"materialize_duration,omitempty"`
	DetachDuration         time.Duration      `json:"detach_duration,omitempty"`
	CacheProbeDuration     time.Duration      `json:"cache_probe_duration,omitempty"`
	OtherDuration          time.Duration      `json:"other_duration,omitempty"`
	NativeEvents           []NativePhaseTrace `json:"native_events,omitempty"`
}

// NativePhaseTrace reports an optional native materialisation event captured
// during a decode forward pass.
type NativePhaseTrace struct {
	Name     string        `json:"name"`
	Duration time.Duration `json:"duration"`
	Error    string        `json:"error,omitempty"`
	Pages    int           `json:"pages,omitempty"`
	Tokens   int           `json:"tokens,omitempty"`
}

// ClassifyResult holds the sampled token for a single prompt and optional logits.
type ClassifyResult struct {
	Token  Token
	Logits []float32
}

// BatchResult holds the streamed tokens for a single prompt in a batch call.
type BatchResult struct {
	Tokens []Token
	Err    error
}

// AttentionSnapshot contains post-RoPE key tensors extracted from KV caches.
type AttentionSnapshot struct {
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	Keys          [][][]float32
	Queries       [][][]float32
	Architecture  string
}

// HasQueries reports whether query tensors are present in the snapshot.
func (s *AttentionSnapshot) HasQueries() bool {
	// len(nil) == 0 — the explicit s.Queries != nil check is redundant,
	// and dropping it lets the inliner fold the single bounds load into
	// a fused nil-check + length compare instead of a three-step chain.
	return s != nil && len(s.Queries) > 0
}

// ModelInfo describes a loaded model. The definition lives in spine so
// subpackages can consume it without importing root.
type ModelInfo = spine.ModelInfo
