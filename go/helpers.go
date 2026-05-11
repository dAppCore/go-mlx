// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/memory"
)

// firstNonEmpty returns the first non-empty string after trimming whitespace.
// Shared across dataset_stream / kv_snapshot_index / memvid_chapter_smoke /
// model_pack and the legacy hf_fit alias surface.
//
//	value := firstNonEmpty(primary, fallback)
func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

// firstPositive returns the first positive value from a list.
//
//	n := firstPositive(headDim*heads, hidden)
func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

// modelInfoToMemory converts an mlx-root ModelInfo into the structural
// mirror used by go-mlx/memory/, go-mlx/agent/, and other subpackages
// that cannot import mlx-root. Shared by session_agent_darwin.go,
// fast_eval_runner.go, etc.
//
//	out := modelInfoToMemory(info)
func modelInfoToMemory(info ModelInfo) memory.ModelInfo {
	return memory.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
}

// modelInfoToBundle converts mlx.ModelInfo to bundle.ModelInfo.
// Used by session_darwin.go + fast_eval_runner.go callers.
//
//	out := modelInfoToBundle(info)
func modelInfoToBundle(info ModelInfo) bundle.ModelInfo {
	return bundle.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       info.Adapter,
	}
}

// sampleFromGenerateConfig converts mlx.GenerateConfig sampler fields
// into bundle.Sampler. Used by fast_eval_runner.go.
//
//	s := sampleFromGenerateConfig(cfg)
func sampleFromGenerateConfig(cfg GenerateConfig) bundle.Sampler {
	return bundle.Sampler{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

// renderTokensText concatenates Token.Text || Token.Value across a token
// slice. Used by memvid_chapter_smoke when no Text was reported.
//
//	text := renderTokensText(tokens)
func renderTokensText(tokens []Token) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(firstNonEmpty(token.Text, token.Value))
	}
	return builder.String()
}

// indexString locates substr inside s, returning its index or -1.
// Shared between hf_fit and openai.go.
//
//	pos := indexString(haystack, needle)
func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
