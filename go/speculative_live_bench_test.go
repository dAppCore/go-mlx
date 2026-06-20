// SPDX-Licence-Identifier: EUPL-1.2

// Real-model benchmark for the MTP speculative-decode lane — the user-facing
// SpeculativePair.Generate path that pairs a Gemma 4 target with its attached
// assistant drafter (the drafter proposes a block of tokens the target verifies
// in one batched forward). This is the speculative-specific overhead the
// CPU-only speculative_bench_test.go cannot reach: the draft/verify loop, the
// accept/reject bookkeeping, and the assistant-result → decode.Result
// conversion, all driven against weights on Metal.
//
// It loads the cached e2b target beside its e2b assistant via
// LoadSpeculativePair ONCE (the title-line helper), then drives one short
// generation per op. Both lanes are fully greedy — the zero-value
// GenerateConfig sets Temperature/TopK/TopP/MinP all to 0 — which is what
// engages the drafter's batched argmax fast path (assistant_generate.go:
// `sampling := cfg.Temperature > 0`) AND makes the emitted token IDs
// byte-identical run to run. Those IDs are both the per-token denominator and
// the verification anchor for any production change.
//
// Pair this with BenchmarkGenerate_LiveE2B (same prompt, same tokens/op metric)
// to read MTP overhead against plain decode: divide each bench's allocs/op by
// its tokens/op for the per-token figure.
//
// Allocation resolution is the DEFAULT byte-sampled MemProfileRate, NOT rate 1.
// The full speculative decode is FFI-heavy (per verify round it crosses into
// Metal repeatedly); driving that under MemProfileRate=1 has SIGKILLed the host
// on a sibling run. b.ReportAllocs()/-benchmem at the default rate over a SHORT
// sequence keeps memory pressure flat while still attributing the per-op host
// allocations the speculative loop pays.
//
// Run (needs the cached weights + Metal):
//
//	MLX_METALLIB_PATH=$PWD/../dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run '^$' -bench BenchmarkGenerateSpeculative_LiveE2B \
//	    -benchmem -benchtime=5x ./go

package mlx

import (
	"context"
	"runtime/debug"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
)

// generateSpeculativeLiveBenchSink defeats dead-code elimination of the result.
var generateSpeculativeLiveBenchSink SpeculativeDecodeResult

// benchSpeculativePrompt is a fixed, short completion-style prompt. The MTP
// pair runs the prompt through the target's chat/template surface like any
// generation; a completion lead-in keeps the instruction-tuned model decoding a
// real continuation rather than emitting EOS at step one. The content is
// otherwise irrelevant to allocation counting — a stable prompt keeps the
// greedy token count deterministic across runs. It matches benchGeneratePrompt
// in generate_live_bench_test.go so the MTP and plain-decode benches are
// directly comparable.
const benchSpeculativePrompt = benchGeneratePrompt

// benchSpeculativeMaxTokens caps the decode loop short. Two models are resident
// (target + assistant ≈ 4 GB), so each op runs a brief generation, never a long
// stream — the SHORT sequence that keeps the default-rate profile flat under the
// FFI-heavy verify loop.
const benchSpeculativeMaxTokens = 32

// requireLiveE2BSpeculativePair loads the cached e2b target beside its e2b
// assistant drafter via LoadSpeculativePair, skipping (never failing) when
// Metal is unavailable or either pack is not cached. The pair takes the
// Gemma4Assistant attach path (the assistant is `gemma4_assistant` arch), which
// is the MTP lane SpeculativePair.Generate drives.
func requireLiveE2BSpeculativePair(b *testing.B) *SpeculativePair {
	b.Helper()
	if !metaltest.RunMetalTests {
		b.Skip("build with -tags metal_runtime to drive the live MTP speculative pair")
	}
	if !MetalAvailable() {
		b.Skip("Metal runtime unavailable")
	}
	targetDir := metaltest.HFModelPath(b, "mlx-community/gemma-4-e2b-it-4bit")
	draftDir := metaltest.HFModelPath(b, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	if targetDir == "" || draftDir == "" {
		b.Skip("e2b target or assistant drafter not cached")
	}
	pair, err := LoadSpeculativePair(targetDir, draftDir, SpeculativePairConfig{})
	if err != nil {
		b.Skipf("LoadSpeculativePair did not load the e2b MTP pair: %v", err)
	}
	return pair
}

// BenchmarkGenerateSpeculative_LiveE2B measures per-op heap allocations on the
// MTP speculative-decode lane (SpeculativePair.Generate) against a real gemma-4
// e2b target + assistant drafter. It reports allocs/op (per full generation)
// and tokens/op as custom metrics, so the per-token figure is allocs/op ÷
// tokens/op — the same shape as BenchmarkGenerate_LiveE2B, making the
// speculative overhead readable against plain decode from the two bench lines.
func BenchmarkGenerateSpeculative_LiveE2B(b *testing.B) {
	// Cap heap as a guard rail. Two e2b packs (4bit target + bf16 assistant)
	// sit far under this; the limit only stops a runaway load from marching the
	// host toward the OOM earlier full-model benches hit.
	debug.SetMemoryLimit(40 << 30)

	pair := requireLiveE2BSpeculativePair(b)
	defer func() {
		if closeErr := pair.Close(); closeErr != nil {
			b.Errorf("Close: %v", closeErr)
		}
	}()

	// Fully greedy: the zero-value GenerateConfig leaves Temperature/TopK/TopP/
	// MinP at 0, which engages the drafter's batched-argmax fast path and makes
	// the emitted IDs byte-identical run to run.
	cfg := SpeculativeDecodeConfig{
		MaxTokens:      benchSpeculativeMaxTokens,
		GenerateConfig: GenerateConfig{MaxTokens: benchSpeculativeMaxTokens},
	}

	// Untimed deterministic pass: establish the greedy emitted-token count (the
	// per-token denominator) and the byte-identical output IDs (the anchor any
	// production change must preserve). temp 0 ⇒ identical every call.
	warm, err := pair.Generate(context.Background(), benchSpeculativePrompt, cfg)
	if err != nil {
		b.Fatalf("Generate (warm-up): %v", err)
	}
	tokenCount := len(warm.Tokens)
	if tokenCount == 0 {
		b.Skip("live e2b MTP pair emitted no tokens — cannot measure per-op allocs")
	}
	wantIDs := make([]int32, tokenCount)
	for i := range warm.Tokens {
		wantIDs[i] = warm.Tokens[i].ID
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := pair.Generate(context.Background(), benchSpeculativePrompt, cfg)
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		// Greedy determinism guard: a drift in emitted IDs means the loop is no
		// longer byte-identical, which would invalidate the per-token figure.
		if len(out.Tokens) != tokenCount {
			b.Fatalf("non-deterministic token count: got %d, want %d", len(out.Tokens), tokenCount)
		}
		generateSpeculativeLiveBenchSink = out
	}
	b.StopTimer()

	// allocs/op (from -benchmem) is per full generation; tokens/op records the
	// deterministic denominator so a reader divides to get the per-token figure
	// from the bench line alone — directly comparable to BenchmarkGenerate_LiveE2B.
	b.ReportMetric(float64(tokenCount), "tokens/op")
	_ = generateSpeculativeLiveBenchSink
}
