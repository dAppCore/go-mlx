// SPDX-Licence-Identifier: EUPL-1.2

// Real-model benchmark for the top-level buffered Generate path — the
// integrated user-facing generate (prompt → decode loop → token stream →
// string). It loads the cached gemma-4 e2b 4bit pack via the shared live
// harness and drives one short greedy generation per op, so allocs/op
// divided by the (deterministic) emitted token count gives the per-token
// heap allocations a real request actually pays.
//
// This exercises the shared generateTokensWithConfig core: the per-call
// core.NewBuilder()/Grow and parser.NewProcessor, then per token the
// filteredRootTokenSeq closure (filter.Process + Token struct + yield) and
// builder.WriteString. GenerateStream and GenerateTokens ride the same
// core (GenerateStream just wraps GenerateTokens in a channel), so this one
// bench covers the root generate path; the channel hop is the only thing
// it does not measure.
//
// Greedy (temperature 0) makes the emitted token IDs byte-identical run to
// run, which is both the per-token denominator and the verification anchor
// for any production change.
//
// Run (needs the cached weights + Metal):
//
//	MLX_METALLIB_PATH=$PWD/../dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run '^$' -bench BenchmarkGenerate_LiveE2B \
//	    -benchmem -benchtime=5x ./go

package mlx

import (
	"runtime"
	"runtime/debug"
	"testing"
)

// generateLiveBenchSink defeats dead-code elimination of the Generate result.
var generateLiveBenchSink string

// benchGeneratePrompt is a fixed, short completion-style prompt. The raw
// Generate path feeds the prompt to the engine with no chat-template wrapping,
// so a completion lead-in ("Once upon a time,") keeps the instruction-tuned
// model decoding rather than emitting EOS at step one (an instruction-style
// prompt would). The content is otherwise irrelevant to allocation counting;
// a stable prompt keeps the greedy token count deterministic across runs,
// matching the completion prompts the live generate tests already drive.
const benchGeneratePrompt = "Once upon a time,"

// benchGenerateMaxTokens caps the decode loop at a short output to honour the
// model-load RAM budget — one short generate per op, never a long stream.
const benchGenerateMaxTokens = 64

// BenchmarkGenerate_LiveE2B measures per-token heap allocations on the
// top-level buffered Model.Generate path against a real gemma-4 e2b model.
// It reports allocs/token and tokens/op as custom metrics so the per-token
// figure is read directly off the bench line (allocs/op is per full
// generation, not per token).
func BenchmarkGenerate_LiveE2B(b *testing.B) {
	// Cap heap so a runaway load can never march the host to the 143 GB OOM
	// that earlier full-model benches hit. The e2b 4bit pack sits far under
	// this; the limit is a guard rail, not a working figure.
	debug.SetMemoryLimit(60 << 30)

	model := requireLiveE2BModel(b) // skips if Metal off / weights uncached / load flake

	// Exact-count every allocation from here on. The one-time LoadModel above
	// stays coarsely sampled (its tokenizer-JSON / safetensors allocs are not
	// what this bench attributes), so this profiles ONLY the generate path at
	// full resolution — the default byte-weighted MemProfileRate (~1 sample
	// per 512 KB) under-resolves the small per-token allocs and would report a
	// builder regrow or a small engine site as a deceptive "0 samples". With
	// rate 1 a `-list` zero is a true zero. Restored at function exit so the
	// process-wide rate is not left at 1 for any later bench in the binary.
	prevRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() { runtime.MemProfileRate = prevRate }()

	// Untimed deterministic pass to count the greedy token output. Generate
	// returns a string, so the emitted-token count is established here (temp
	// 0 ⇒ identical every call) and used as the per-token denominator.
	var tokenCount int
	for range model.GenerateTokens(nil, benchGeneratePrompt,
		WithMaxTokens(benchGenerateMaxTokens), WithTemperature(0)) {
		tokenCount++
	}
	if tokenCount == 0 {
		b.Skip("live e2b model emitted no tokens — cannot measure per-token allocs")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := model.Generate(benchGeneratePrompt,
			WithMaxTokens(benchGenerateMaxTokens), WithTemperature(0))
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		generateLiveBenchSink = out
	}
	b.StopTimer()

	// allocs/op (from -benchmem) is per full generation; tokens/op records the
	// deterministic denominator so a reader divides allocs/op by tokens/op to
	// get the per-token figure from the bench line alone.
	b.ReportMetric(float64(tokenCount), "tokens/op")
	_ = generateLiveBenchSink // keep the last result live past the loop
}
