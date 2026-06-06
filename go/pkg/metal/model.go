// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/profile"
)

// InternalModel is the common interface for all transformer model architectures.
type InternalModel interface {
	// Forward runs the model forward pass on token IDs with KV caches.
	Forward(tokens *Array, caches []Cache) *Array

	// ForwardMasked runs the forward pass with an explicit attention mask.
	// mask shape: [B, 1, L, L] — additive mask (0 = attend, -inf = ignore).
	// Used for batched inference with padded sequences.
	ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array

	// NewCache creates per-layer KV caches for generation.
	NewCache() []Cache

	// NumLayers returns the number of transformer layers.
	NumLayers() int

	// Tokenizer returns the model's tokenizer.
	Tokenizer() *Tokenizer

	// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
	ModelType() string

	// ApplyLoRA wraps target projection layers with LoRA adapters for training.
	// Returns the adapter which holds references to all LoRA layers.
	ApplyLoRA(cfg LoRAConfig) *LoRAAdapter
}

// LastTokenLogitsModel is an optional fast prefill path for architectures that
// can project only the final sequence position instead of allocating
// [batch, sequence, vocab] logits for long context warmup.
type LastTokenLogitsModel interface {
	ForwardLastTokenLogits(tokens *Array, mask *Array, caches []Cache) *Array
}

// GreedyTokenModel is an optional decode path for deterministic generation.
// It returns the next token directly, avoiding a retained logits tensor when
// sampling is exactly Greedy and no repeat penalty or probe sink is active.
type GreedyTokenModel interface {
	ForwardGreedyToken(tokens *Array, mask *Array, caches []Cache) *Array
}

// SuppressedGreedyTokenModel can produce a Greedy token while masking out
// template or modality token IDs that must not be sampled.
type SuppressedGreedyTokenModel interface {
	ForwardGreedyTokenWithSuppression(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32) *Array
}

// QueryHeadCounter optionally reports a model's number of attention query heads.
// Attention/KV extraction uses it to size per-head output; a model that cannot
// report a head count is treated as 0. Dispatching on this capability instead of
// a concrete type-switch lets model types live outside package metal (go-mlx #45).
type QueryHeadCounter interface {
	NumQueryHeads() int
}

// LoRALinearResolver optionally resolves a LoRA-targetable linear projection by
// layer index and projection path (e.g. "self_attn.q_proj"), returning nil for an
// unknown layer or path. Dispatching on this capability instead of a concrete
// type-switch lets model types live outside package metal (go-mlx #45).
type LoRALinearResolver interface {
	ResolveLoRALinear(layerIdx int, projPath string) *Linear
}

// DenseSplitParts exposes the dense decoder components needed by split
// inference without tying pkg/metal to a concrete model package.
type DenseSplitParts interface {
	SplitEmbedding() *Embedding
	SplitDecoderLayers() []*DenseDecoderLayer
	SplitNorm() *RMSNormModule
	SplitOutput() *Linear
	SplitConfig() *DenseConfig
}

// CacheTopologyRecorder optionally records architecture-specific KV-cache
// topology (e.g. Gemma 4's local/global sliding-window layout) into a
// CacheProfile, on top of the generic per-cache pass. Dispatching on this
// capability instead of a concrete type-switch lets model types live outside
// package metal (go-mlx #45).
type CacheTopologyRecorder interface {
	RecordCacheTopology(profile *CacheProfile, caches []Cache)
}

// AttentionCacheLayouter optionally maps each transformer layer to its KV-cache
// index for architectures with a non-identity cache layout (e.g. Gemma 4's shared
// local/global windows). Models without a custom layout get the identity mapping.
// Dispatching on this capability instead of a concrete type-switch lets model
// types live outside package metal (go-mlx #45).
type AttentionCacheLayouter interface {
	AttentionCacheLayout(numLayers, numCaches int) []int
}

// ModelCloser optionally releases a model's Metal weight arrays on Close.
// Architectures with native weights implement it; staged loaders that hold no
// arrays of their own do not (Close is a no-op for them, as before). Dispatching
// on this capability instead of a concrete type-switch lets model types live
// outside package metal (go-mlx #45).
type ModelCloser interface {
	CloseModel()
}

// FixedSlidingPrefillLimiter optionally reports the largest safe prefill chunk for
// a fixed-size sliding-window cache (Gemma 4), or 0 when not applicable.
// Dispatching on this capability instead of a concrete type assertion lets model
// types live outside package metal (go-mlx #45).
type FixedSlidingPrefillLimiter interface {
	FixedSlidingPrefillChunkLimit(caches []Cache) int
}

// FixedSlidingCacheModel optionally declares that a model uses the fixed-size
// sliding-window KV cache (Gemma 4) rather than the generic paged cache when a
// bounded context is configured. Dispatching on this capability instead of a
// concrete type-switch lets model types live outside package metal (go-mlx #45).
type FixedSlidingCacheModel interface {
	UsesFixedSlidingCache() bool
}

// ThoughtChannelSuppressorModel optionally declares that a model's chat prompt
// needs the empty thought-channel suppressor when reasoning is off — the large
// Gemma 4 variants (26B/31B, num_attention_heads>=16) ghost a thought channel
// otherwise, and the suppressor makes them answer directly. Dispatching on this
// capability instead of a concrete type-switch lets model types live outside
// package metal (go-mlx #45).
type ThoughtChannelSuppressorModel interface {
	NeedsThoughtChannelSuppressor() bool
}

// ModelInfoReporter optionally fills architecture-specific metadata (vocab size,
// hidden size, context length, quantization, head count, …) into a ModelInfo.
// Dispatching on this capability instead of a concrete type-switch lets model
// types live outside package metal (go-mlx #45).
type ModelInfoReporter interface {
	FillModelInfo(info *ModelInfo)
}

// MoETextRuntimeReporter optionally reports whether a sparse-MoE model's native
// selected-expert decode kernels are linked and ready for text generation, and
// the canonical architecture family used in diagnostics when they are not.
// Dispatching on this capability instead of a concrete type-switch lets the
// qwen-family MoE model types (qwen3_moe, mixtral, kimi, gpt_oss) live outside
// package metal (go-mlx #45).
type MoETextRuntimeReporter interface {
	// MoETextRuntimeAvailable reports whether every layer's dense and sparse
	// parts are populated such that native text decode can run.
	MoETextRuntimeAvailable() bool
	// MoETextDecodeFamily returns the canonical family token for unavailable
	// diagnostics (e.g. "qwen3_moe"), independent of the detected model type.
	MoETextDecodeFamily() string
}

// DecodeUnavailableReporter lets staged model packages report why Generate
// cannot run yet without forcing pkg/metal to know their concrete types.
type DecodeUnavailableReporter interface {
	DecodeUnavailableError(operation string) error
}

// HybridAttentionLayerPlan describes one layer in an architecture whose K/V
// caches do not map one-to-one with decoder layers. Cacheless layers set
// RequiresKV=false; cache-owning layers set CacheIndex to the cache slot used by
// that layer.
type HybridAttentionLayerPlan struct {
	Layer      int
	Kind       string
	Window     int
	RequiresKV bool
	CacheIndex int
}

// HybridAttentionCachePlan reports the cache layout for hybrid attention
// models such as Qwen3.6, where linear-attention layers are cacheless and
// full-attention layers own K/V cache slots.
type HybridAttentionCachePlan struct {
	Layers            []HybridAttentionLayerPlan
	CacheIndexByLayer []int
	CachelessLayers   int
	GlobalLayers      int
}

// HybridAttentionCachePlanner lets model packages expose non-identity cache
// topology without pkg/metal depending on their concrete staged types.
type HybridAttentionCachePlanner interface {
	HybridAttentionCachePlan() (HybridAttentionCachePlan, bool)
}

// QuantizationConfig holds quantization parameters from config.json.
type QuantizationConfig struct {
	GroupSize int    `json:"group_size"`
	Bits      int    `json:"bits"`
	Mode      string `json:"mode"`
}

func NormalizeQuantizationMode(mode string) string {
	mode = core.Lower(core.Trim(mode))
	if mode == "" {
		return "affine"
	}
	return mode
}

func IsAffineQuantizationMode(mode string) bool {
	return NormalizeQuantizationMode(mode) == "affine"
}

// mxfp8DenseFallback keeps the older-metallib dense-matmul fallback available as
// an in-code diagnostic — off by default (native MLX kernels on v0.31.1+), and
// NEVER ambient env (an env-readable compute toggle is external control). Set it
// locally only to drive an old metallib that lacks MXFP8 qmm.
var mxfp8DenseFallback = false

func RequiresDenseQuantizedMatmulFallback(mode string) bool {
	// Older local metallib builds exposed MXFP8 dequantize without MXFP8 qmm.
	return NormalizeQuantizationMode(mode) == "mxfp8" && mxfp8DenseFallback
}

func weightCandidates(name string) []string {
	return WeightCandidates(name)
}

// WeightCandidates returns the standard model/language_model aliases for a
// checkpoint tensor name.
func WeightCandidates(name string) []string {
	candidates := []string{name}
	if core.HasPrefix(name, "model.") {
		suffix := core.TrimPrefix(name, "model.")
		return append(candidates,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	}
	return append(candidates,
		"model."+name,
		"language_model."+name,
		"language_model.model."+name,
		"model.language_model."+name,
		"model.language_model.model."+name,
	)
}

// ResolveWeight looks up a weight with optional "language_model." prefix.
func ResolveWeight(weights map[string]*Array, name string) *Array {
	for _, candidate := range weightCandidates(name) {
		if w, ok := weights[candidate]; ok {
			return w
		}
	}
	return nil
}

// HasResolvedWeight reports whether a weight exists under the standard model
// and language_model aliases.
func HasResolvedWeight(weights map[string]*Array, name string) bool {
	for _, candidate := range weightCandidates(name) {
		if _, ok := weights[candidate]; ok {
			return true
		}
	}
	return false
}

func hasResolvedWeight(weights map[string]*Array, name string) bool {
	return HasResolvedWeight(weights, name)
}

func probeModelType(data []byte) (string, error) {
	var probe struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
		TextConfig    struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
		return "", core.E("model.probeModelType", "parse model_type", nil)
	}
	// The resolution order and the family refinements (a Gemma-4 multimodal
	// wrapper → its text tower; a BERT encoder whose architectures name a
	// cross-encoder → bert_rerank) live in the registry — the single home
	// shared with the gguf/hf/config probes, so they can never disagree. The
	// loader does not name-branch on a model family: a new family is supported
	// by adding registry data, not editing this dispatch. (A "Gemma4Assistant*"
	// architecture resolves to "gemma4_assistant", caught below by the
	// attached-drafter guard, instead of mis-loading as a standalone model.)
	return profile.ResolveArchitecture(probe.ModelType, probe.TextConfig.ModelType, probe.Architectures), nil
}

// NormalizeProbeModelType canonicalises a raw model_type string to the
// registry arch key. Exported so models on the metal SDK can normalise their
// own config's model_type to match registration.
func NormalizeProbeModelType(value string) string { return normalizeProbeModelType(value) }

func normalizeProbeModelType(value string) string {
	// Single source of truth — the model-type alias table lives in
	// profile.NormalizeArchitecture, shared with the gguf/hf/model config
	// probes. This was a verbatim third copy of that switch; it now delegates
	// so the arm set (including the gemma4 unified family) can never drift
	// between the metal dispatch and the config-probe path again.
	return profile.NormalizeArchitecture(value)
}

func compactArchitectureName(value string) string {
	return CompactArchitectureName(value)
}

// CompactArchitectureName normalises architecture names for family detection.
// Model packages use it when config.json relies on `architectures` aliases.
func CompactArchitectureName(value string) string {
	compact := core.Lower(value)
	compact = core.Replace(compact, "_", "")
	compact = core.Replace(compact, "-", "")
	return core.Replace(compact, ".", "")
}

// loadModel auto-detects the model architecture from config.json and loads it.
// Supports "gemma3", "gemma3_text", "gemma2", "gemma4", "gemma4_text",
// "qwen3", "qwen3_next", "qwen2", "llama", and recognized staged
// architectures such as "qwen3_6" and "minimax_m2". Gemma 4 assistant checkpoints are
// attached MTP drafters; load them through LoadGemma4AssistantPair or the
// public LoadSpeculativePair path rather than as standalone InternalModel
// values.
func loadModel(modelPath string) (InternalModel, error) {
	root := ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("model.loadModel", "load config", err)
	}
	data := []byte(str)

	modelType, err := probeModelType(data)
	if err != nil {
		return nil, core.E("model.loadModel", "parse model_type", err)
	}

	// Attached-only architectures (e.g. MTP assistant drafters) are declared
	// not-standalone in the registry; they load beside a target, never alone.
	if profile.AttachedOnlyArchitecture(modelType) {
		return nil, core.E("model.loadModel", modelType+" is an attached drafter, not a standalone model; load it beside its target via LoadSpeculativePair", nil)
	}
	// Dispatch via the loader registry (model_registry.go) — no central switch.
	if loader := lookupModelLoader(modelType); loader != nil {
		return loader(modelPath, data)
	}
	return nil, core.E("model.loadModel", "unsupported architecture: "+modelType, nil)
}
