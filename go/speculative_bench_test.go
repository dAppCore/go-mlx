// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the CPU-only side of speculative.go — tokeniser-probe
// equality, the Gemma 4 assistant result → decode.Result converter, and
// the default-probe list builder. Per AX-11 — the converter runs once
// per speculative generation; int32SlicesEqual runs once per tokeniser
// probe per pair-validation; gemma4AssistantModelInfo runs on every pair
// validation that goes through the assistant attach path.
//
// Functions that touch the loaded Model/draft or call into metal
// (GenerateSpeculative, LoadSpeculativePair, validateSpeculative* —
// they all reach a *Model.Tokenizer() / .Info() that requires a real
// model, or attach via gemma4.AttachGemma4Assistant) are intentionally
// OUT of scope.
//
// Run:    go test -bench='BenchmarkSpeculative' -benchmem -run='^$' ./go

package mlx

import (
	"testing"
	"time"

	"dappco.re/go/inference/decode"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// Sinks defeat compiler DCE. Distinct from other bench files in this package.
var (
	specBenchSinkResult decode.Result
	specBenchSinkProbes []string
	specBenchSinkBool   bool
)

// specBenchAssistantResult mirrors the shape returned by the native
// Gemma 4 assistant generator. Token count and accept/reject counters
// reflect the typical short-answer assistant trace.
func specBenchAssistantResult(tokenCount int) gemma4.Gemma4AssistantGenerateResult {
	tokens := make([]metal.Token, tokenCount)
	for i := range tokens {
		tokens[i] = metal.Token{ID: int32(i + 1), Text: "tok"}
	}
	return gemma4.Gemma4AssistantGenerateResult{
		Tokens:          tokens,
		Text:            "The quick brown fox jumps over the lazy dog.",
		PromptTokens:    2048,
		TargetTokens:    tokenCount,
		DraftTokens:     tokenCount + 4,
		AcceptedTokens:  tokenCount - 2,
		RejectedTokens:  2,
		TargetCalls:     1,
		DraftCalls:      1,
		Duration:        500 * time.Millisecond,
		PrefillDuration: 50 * time.Millisecond,
		TargetDuration:  300 * time.Millisecond,
		DraftDuration:   150 * time.Millisecond,
	}
}

// --- gemma4AssistantGenerateResultToDecode — per-generation converter ---

func BenchmarkSpeculative_Gemma4AssistantGenerateResultToDecode_32Tokens(b *testing.B) {
	result := specBenchAssistantResult(32)
	prompt := "Continue the story:"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkResult = gemma4AssistantGenerateResultToDecode(prompt, result)
	}
}

func BenchmarkSpeculative_Gemma4AssistantGenerateResultToDecode_256Tokens(b *testing.B) {
	result := specBenchAssistantResult(256)
	prompt := "Continue the story:"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkResult = gemma4AssistantGenerateResultToDecode(prompt, result)
	}
}

// Zero draft tokens — exercises the acceptance-rate divide-by-zero guard.
func BenchmarkSpeculative_Gemma4AssistantGenerateResultToDecode_ZeroDraft(b *testing.B) {
	result := specBenchAssistantResult(32)
	result.DraftTokens = 0
	result.AcceptedTokens = 0
	prompt := "Continue the story:"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkResult = gemma4AssistantGenerateResultToDecode(prompt, result)
	}
}

// --- speculativeTokenizerProbes — default + custom probe-list build ---

func BenchmarkSpeculative_SpeculativeTokenizerProbes_DefaultSet(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkProbes = speculativeTokenizerProbes(nil)
	}
}

func BenchmarkSpeculative_SpeculativeTokenizerProbes_CustomSet(b *testing.B) {
	probes := []string{
		"hello world",
		"Translate 'hello' to French.",
		"The quick brown fox jumps over the lazy dog.",
		"Answer in one short sentence.",
		"Summarise the following passage briefly.",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkProbes = speculativeTokenizerProbes(probes)
	}
}

// --- int32SlicesEqual — pair-validation equality check ---

// Equal vectors — happy path that must scan the whole slice.
func BenchmarkSpeculative_Int32SlicesEqual_Equal_5(b *testing.B) {
	a := []int32{1, 2, 3, 4, 5}
	c := []int32{1, 2, 3, 4, 5}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkBool = int32SlicesEqual(a, c)
	}
}

func BenchmarkSpeculative_Int32SlicesEqual_Equal_64(b *testing.B) {
	a := make([]int32, 64)
	c := make([]int32, 64)
	for i := range a {
		a[i] = int32(i)
		c[i] = int32(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkBool = int32SlicesEqual(a, c)
	}
}

// Different lengths — early exit on the len check.
func BenchmarkSpeculative_Int32SlicesEqual_DiffLen(b *testing.B) {
	a := []int32{1, 2, 3, 4, 5}
	c := []int32{1, 2, 3, 4}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkBool = int32SlicesEqual(a, c)
	}
}

// Tail mismatch — worst case: full scan, fails on last element.
func BenchmarkSpeculative_Int32SlicesEqual_TailMismatch_64(b *testing.B) {
	a := make([]int32, 64)
	c := make([]int32, 64)
	for i := range a {
		a[i] = int32(i)
		c[i] = int32(i)
	}
	c[63] = -1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specBenchSinkBool = int32SlicesEqual(a, c)
	}
}
