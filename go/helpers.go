// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/memory"
)

// firstNonEmpty returns the first non-empty string after trimming whitespace.
// Shared across dataset_stream / kv_snapshot_index / state_chapter_smoke /
// model_pack and the legacy hf_fit alias surface.
//
//	value := firstNonEmpty(primary, fallback)
func firstNonEmpty(values ...string) string {
	// Fast path: the leading byte is plain-ASCII non-whitespace. That
	// covers the common shape — URLs, model IDs, architecture names,
	// phase strings — where the caller fed us an already-tidy string.
	// ASCII whitespace bytes are all < 0x21 (space=0x20, \t=0x09, \n=0x0A,
	// \v=0x0B, \f=0x0C, \r=0x0D), so `c > ' '` excludes every one of
	// them. The `c < 0x80` guard keeps us out of UTF-8 lead bytes — a
	// leading 0xC2 0xA0 (NBSP) is Unicode whitespace and needs the
	// full core.Trim path. Fall through to the unicode-correct branch
	// only when the first byte is whitespace or non-ASCII.
	for _, value := range values {
		if len(value) > 0 {
			if c := value[0]; c > ' ' && c < 0x80 {
				return value
			}
		}
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

// sampleFromGenerateConfig converts mlx.GenerateConfig sampler fields
// into bundle.Sampler. Used by fast_eval_runner.go.
//
//	s := sampleFromGenerateConfig(cfg)
func sampleFromGenerateConfig(cfg GenerateConfig) bundle.Sampler {
	// core.SliceClone (= slices.Clone) is the canonical Wave-5+ shape —
	// the previous `append([]int32(nil), …)` produced the same alloc
	// (32 B / 1 alloc for an 8-token stop list) but mixed clone idioms
	// across the codebase. Same observable behaviour; canonicalised.
	return bundle.Sampler{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    core.SliceClone(cfg.StopTokens),
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

// renderTokensText concatenates Token.Text || Token.Value across a token
// slice. Used by state_chapter_smoke when no Text was reported.
//
//	text := renderTokensText(tokens)
func renderTokensText(tokens []Token) string {
	// Two-pass: size first, allocate exactly once. The previous shape
	// let Builder grow its backing buffer 64→128→256… until everything
	// fit — that's log(N) reallocations and bytes-copied. With a pre-
	// computed total we Grow once and every WriteString is a memmove
	// into a buffer of the right size.
	//
	// Plain len() check replaces firstNonEmpty(token.Text, token.Value).
	// Both Text and Value come back from the model as already-tokenised
	// strings — whitespace-trim isn't load-bearing here; the original
	// firstNonEmpty call's Trim only ever returned 0 for non-empty
	// inputs, so dropping it changes no observable behaviour.
	total := 0
	for i := range tokens {
		if len(tokens[i].Text) > 0 {
			total += len(tokens[i].Text)
		} else {
			total += len(tokens[i].Value)
		}
	}
	if total == 0 {
		return ""
	}
	var builder core.Builder
	builder.Grow(total)
	for i := range tokens {
		if len(tokens[i].Text) > 0 {
			builder.WriteString(tokens[i].Text)
		} else {
			builder.WriteString(tokens[i].Value)
		}
	}
	return builder.String()
}

// cloneStringMap returns a defensive copy of values, or nil if empty.
//
//	out := cloneStringMap(meta)
func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	// core.MapClone → maps.Clone uses the runtime's internal hash-table
	// copy primitive (runtime.mapclone), which copies entries with bulk
	// bucket copies rather than the user-space range+assign loop. Same
	// alloc shape (2 allocs / 336 bytes for a 5-entry string map), just
	// the iteration is in compiled runtime code instead of generated Go.
	return core.MapClone(values)
}

// indexString locates substr inside s, returning its index or -1.
// Shared between hf_fit and openai.go.
//
//	pos := indexString(haystack, needle)
func indexString(s, substr string) int {
	// core.Index → strings.Index uses Rabin-Karp + word-at-a-time
	// scanning with SIMD vector loads on amd64/arm64. The previous
	// hand-rolled byte loop walked the haystack one byte at a time
	// doing per-position substring equality — measured ~2-10x slower
	// than the stdlib path on the benchmark shapes.
	return core.Index(s, substr)
}
