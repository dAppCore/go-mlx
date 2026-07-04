// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// AX-11 allocation baselines for the MTP assistant decode path. Every benchmark
// drives the tiny synthetic pair (no real checkpoint load), seeds the target
// caches + backbone hidden ONCE outside the timed loop, and reports allocs/op +
// B/op for one decode-step operation. The synthetic config keeps the GPU work
// trivial so the measurement isolates the Go-side allocation cost of the step
// orchestration (slicing, shape scratch, result wrapping), not the kernels.

// benchTinyAssistantPrefill loads a tiny pair, seeds its caches with a short
// prefill, and returns the pair, caches, prefill logits, and backbone hidden.
// The caller owns all four and must free/close them. Used by the decode-step
// benchmarks so the per-op loop measures only the step under test.
func benchTinyAssistantPrefill(b *testing.B, ordered bool) (*Gemma4AssistantPair, []metal.Cache, *metal.Array, *metal.Array) {
	b.Helper()
	pair := loadTinyGemma4AssistantPair(b, ordered)
	caches := pair.Target.NewCache()
	prefill := metal.FromValues([]int32{1, 2, 3}, 3)
	prefillInput := metal.Reshape(prefill, 1, 3)
	prefillLogits, previousHidden := pair.Target.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	if err := metal.Eval(prefillLogits, previousHidden); err != nil {
		b.Fatalf("bench prefill: %v", err)
	}
	metal.Free(prefill, prefillInput)
	metal.DetachCaches(caches)
	return pair, caches, prefillLogits, previousHidden
}

// BenchmarkGemma4AssistantDecode_DraftStep measures one MTP draft step
// (draftStepActivations + lm head + argmax) on the dense drafter.
func BenchmarkGemma4AssistantDecode_DraftStep(b *testing.B) {
	requireMetalRuntime(b)

	pair, caches, prefillLogits, previousHidden := benchTinyAssistantPrefill(b, false)
	defer pair.Close()
	defer metal.FreeCaches(caches)
	defer metal.Free(prefillLogits, previousHidden)

	warm, err := pair.DraftStep(3, previousHidden, caches)
	if err != nil {
		b.Fatalf("warm DraftStep: %v", err)
	}
	if err := metal.Eval(warm.Logits, warm.Token, warm.Hidden); err != nil {
		warm.Close()
		b.Fatalf("warm eval: %v", err)
	}
	warm.Close()

	b.ReportAllocs()
	for b.Loop() {
		result, err := pair.DraftStep(3, previousHidden, caches)
		if err != nil {
			b.Fatalf("DraftStep: %v", err)
		}
		if err := metal.Eval(result.Logits, result.Token, result.Hidden); err != nil {
			result.Close()
			b.Fatalf("eval DraftStep: %v", err)
		}
		result.Close()
	}
}

// BenchmarkGemma4AssistantDecode_DraftBlock measures a chained two-token greedy
// draft block (draftStepGreedy x2 + per-step eval + token readback).
func BenchmarkGemma4AssistantDecode_DraftBlock(b *testing.B) {
	requireMetalRuntime(b)

	pair, caches, prefillLogits, previousHidden := benchTinyAssistantPrefill(b, false)
	defer pair.Close()
	defer metal.FreeCaches(caches)
	defer metal.Free(prefillLogits, previousHidden)

	if warm, err := pair.DraftBlock(3, previousHidden, caches, 2); err != nil {
		b.Fatalf("warm DraftBlock: %v", err)
	} else {
		warm.Close()
	}

	b.ReportAllocs()
	for b.Loop() {
		block, err := pair.DraftBlock(3, previousHidden, caches, 2)
		if err != nil {
			b.Fatalf("DraftBlock: %v", err)
		}
		block.Close()
	}
}

// BenchmarkGemma4AssistantDecode_VerifyDraftBlock_Accept measures the verify
// path on an all-accepted single-token draft (clone caches, target forward,
// batched greedy graph, accept walk, carry logits + hidden).
func BenchmarkGemma4AssistantDecode_VerifyDraftBlock_Accept(b *testing.B) {
	requireMetalRuntime(b)

	pair, caches, prefillLogits, previousHidden := benchTinyAssistantPrefill(b, false)
	defer pair.Close()
	defer metal.FreeCaches(caches)
	defer metal.Free(prefillLogits, previousHidden)

	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		b.Fatalf("greedy target token: %v", err)
	}
	draft := []int32{targetToken}

	if warm, err := pair.VerifyDraftBlock(prefillLogits, draft, caches); err != nil {
		b.Fatalf("warm VerifyDraftBlock: %v", err)
	} else {
		warm.Close()
	}

	b.ReportAllocs()
	for b.Loop() {
		result, err := pair.VerifyDraftBlock(prefillLogits, draft, caches)
		if err != nil {
			b.Fatalf("VerifyDraftBlock(accept): %v", err)
		}
		result.Close()
	}
}

// BenchmarkGemma4AssistantDecode_VerifyDraftBlock_Reject measures the reject
// path on a single bad draft token (the partial/full-reject branch: replacement
// + cache truncate/rebuild fallback).
func BenchmarkGemma4AssistantDecode_VerifyDraftBlock_Reject(b *testing.B) {
	requireMetalRuntime(b)

	pair, caches, prefillLogits, previousHidden := benchTinyAssistantPrefill(b, false)
	defer pair.Close()
	defer metal.FreeCaches(caches)
	defer metal.Free(prefillLogits, previousHidden)

	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		b.Fatalf("greedy target token: %v", err)
	}
	draft := []int32{(targetToken + 1) % 10}

	if warm, err := pair.VerifyDraftBlock(prefillLogits, draft, caches); err != nil {
		b.Fatalf("warm VerifyDraftBlock(reject): %v", err)
	} else {
		warm.Close()
	}

	b.ReportAllocs()
	for b.Loop() {
		result, err := pair.VerifyDraftBlock(prefillLogits, draft, caches)
		if err != nil {
			b.Fatalf("VerifyDraftBlock(reject): %v", err)
		}
		result.Close()
	}
}

// BenchmarkGemma4AssistantDecode_DraftStepActivations measures just the
// assistant backbone activations (token embed + concat + pre-projection +
// drafter layers + norm + post-projection), isolating the per-step layer stack
// from the lm head and sampler.
func BenchmarkGemma4AssistantDecode_DraftStepActivations(b *testing.B) {
	requireMetalRuntime(b)

	pair, caches, prefillLogits, previousHidden := benchTinyAssistantPrefill(b, false)
	defer pair.Close()
	defer metal.FreeCaches(caches)
	defer metal.Free(prefillLogits, previousHidden)

	warmNormed, warmHidden, err := pair.draftStepActivations(3, previousHidden, caches)
	if err != nil {
		b.Fatalf("warm draftStepActivations: %v", err)
	}
	if err := metal.Eval(warmNormed, warmHidden); err != nil {
		metal.Free(warmNormed, warmHidden)
		b.Fatalf("warm eval activations: %v", err)
	}
	metal.Free(warmNormed, warmHidden)

	b.ReportAllocs()
	for b.Loop() {
		normed, hidden, err := pair.draftStepActivations(3, previousHidden, caches)
		if err != nil {
			b.Fatalf("draftStepActivations: %v", err)
		}
		if err := metal.Eval(normed, hidden); err != nil {
			metal.Free(normed, hidden)
			b.Fatalf("eval activations: %v", err)
		}
		metal.Free(normed, hidden)
	}
}
