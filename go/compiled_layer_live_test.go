// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// TestCompiledLayerDecode_LiveModel proves the whole-layer compiled decode on
// a real model: byte-exact greedy output against the default decode path (the
// closure traces the same kernels the default path dispatches), with the
// compiled hit counter proving the closure actually served, and decode rates
// logged for both lanes.
//
//	go test -tags model_eval -run TestCompiledLayerDecode_LiveModel -count=1 dappco.re/go/mlx
func TestCompiledLayerDecode_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	// The serve regime: paged cache mode + a bounded context puts every layer
	// on FixedKVCache (hybrid gemma4 swaps paged for fixed storage) — the
	// regime the compiled layer closure serves. A bare LoadModel runs rotating
	// caches, which the closure correctly declines.
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	const prompt = "Write a long, detailed story about a clockmaker who repairs time itself."
	ctx := context.Background()

	gen := func(label string) (string, float64) {
		t.Helper()
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("%s: NewSession: %v", label, err)
		}
		defer sess.Close()
		if err := sess.Prefill(prompt); err != nil {
			t.Fatalf("%s: Prefill: %v", label, err)
		}
		text := core.NewBuilder()
		tokens := 0
		start := time.Now()
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(200), WithTemperature(0)) {
			text.WriteString(tok.Text)
			tokens++
		}
		rate := float64(tokens) / time.Since(start).Seconds()
		if err := sess.Err(); err != nil {
			t.Fatalf("%s: generate: %v", label, err)
		}
		t.Logf("%s: %.1f tok/s (%d tok)", label, rate, tokens)
		return text.String(), rate
	}

	// Uncompiled decode path — the exactness AND perf baseline. gemma4 declares
	// CompiledLayerDecode in its EngineFeatures, so the baseline lane forces the
	// gate off.
	restoreOff := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, false)
	defaultText, defaultRate := gen("uncompiled decode")
	restoreOff()

	// Whole-layer compiled decode.
	restore := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true)
	hitsBefore := gemma4.CompiledLayerDecodeHits()
	compiledText, compiledRate := gen("compiled layer decode")
	hits := gemma4.CompiledLayerDecodeHits() - hitsBefore
	restore()

	if hits == 0 {
		t.Errorf("compiled layer decode never served — every layer declined the closure")
	}
	t.Logf("compiled layer decode served %d layer steps", hits)

	if compiledText != defaultText {
		t.Errorf("compiled layer decode diverged from the default path:\n  default  %q\n  compiled %q", defaultText, compiledText)
	}
	t.Logf("rates: default %.1f · compiled %.1f tok/s", defaultRate, compiledRate)
}

// TestPipelinedDecode_LiveModel proves the one-ahead pipelined decode loop on
// a real model in the serve regime: byte-exact greedy output vs the serial
// compiled loop, EOS-discard leaving the session state identical (a second
// generation from the same session must match between modes), and the decode
// rate gain from overlapping the host graph encode with the GPU compute.
//
//	go test -tags model_eval -run TestPipelinedDecode_LiveModel -count=1 dappco.re/go/mlx
func TestPipelinedDecode_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx := context.Background()

	// Three generations in one session per lane. The first ends on EOS, so
	// the pipelined loop must discard its speculated forward; the appended
	// second question then attends over that cache — a phantom forward
	// shifts every position after it and diverges the answer. The third is
	// a long generation for the rate.
	run := func(label string, pipelined bool) (turns [3]string, rate float64) {
		t.Helper()
		restore := metal.SetRuntimeGate(metal.GatePipelinedDecode, pipelined)
		defer restore()

		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("%s: NewSession: %v", label, err)
		}
		defer sess.Close()
		if err := sess.Prefill("Q: What is 17 multiplied by 4? A:"); err != nil {
			t.Fatalf("%s: Prefill: %v", label, err)
		}
		gen := func(slot int, maxTokens int) int {
			text := core.NewBuilder()
			tokens := 0
			for tok := range sess.GenerateStream(ctx, WithMaxTokens(maxTokens), WithTemperature(0)) {
				text.WriteString(tok.Text)
				tokens++
			}
			if err := sess.Err(); err != nil {
				t.Fatalf("%s: generate %d: %v", label, slot, err)
			}
			turns[slot] = text.String()
			return tokens
		}
		gen(0, 64) // ends on EOS — exercises the speculation discard
		if err := sess.AppendPrompt("\nQ: What is 25 multiplied by 3? A:"); err != nil {
			t.Fatalf("%s: AppendPrompt: %v", label, err)
		}
		gen(1, 64) // attends across the discarded forward's position
		if err := sess.AppendPrompt("\nNow write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("%s: AppendPrompt story: %v", label, err)
		}
		start := time.Now()
		tokens := gen(2, 200)
		rate = float64(tokens) / time.Since(start).Seconds()
		t.Logf("%s: q1 %q · q2 %q · story %.1f tok/s (%d tok)", label, turns[0], turns[1], rate, tokens)
		return turns, rate
	}

	serialTurns, serialRate := run("serial compiled", false)
	pipeTurns, pipeRate := run("pipelined", true)

	for i := range serialTurns {
		if pipeTurns[i] != serialTurns[i] {
			t.Errorf("pipelined turn %d diverged from serial:\n  serial    %q\n  pipelined %q", i, serialTurns[i], pipeTurns[i])
		}
	}
	t.Logf("rates: serial %.1f · pipelined %.1f tok/s", serialRate, pipeRate)
}

// TestCompiledLayerDecode_WideHead_LiveModel probes whether the current
// metallib serves the 512-wide sdpa_vector kernel: with the wide-SDPA
// diagnostic on, the global owner layer (headDim 512) and its shared-KV
// consumers become closure-eligible. Byte-exactness is asserted against the
// wide-off compiled lane; the hit counter shows whether the holdout layers
// joined. If the kernel is genuinely missing the trace panics, poisons, and
// falls back — the test then reports unchanged hits rather than failing
// exactness.
//
//	go test -tags model_eval -run TestCompiledLayerDecode_WideHead_LiveModel -count=1 dappco.re/go/mlx
func TestCompiledLayerDecode_WideHead_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	const prompt = "Write a long, detailed story about a clockmaker who repairs time itself."
	ctx := context.Background()

	gen := func(label string) (string, float64, uint64) {
		t.Helper()
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("%s: NewSession: %v", label, err)
		}
		defer sess.Close()
		if err := sess.Prefill(prompt); err != nil {
			t.Fatalf("%s: Prefill: %v", label, err)
		}
		hitsBefore := gemma4.CompiledLayerDecodeHits()
		text := core.NewBuilder()
		tokens := 0
		start := time.Now()
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(200), WithTemperature(0)) {
			text.WriteString(tok.Text)
			tokens++
		}
		rate := float64(tokens) / time.Since(start).Seconds()
		if err := sess.Err(); err != nil {
			t.Fatalf("%s: generate: %v", label, err)
		}
		hits := gemma4.CompiledLayerDecodeHits() - hitsBefore
		t.Logf("%s: %.1f tok/s (%d tok, %d compiled layer steps)", label, rate, tokens, hits)
		return text.String(), rate, hits
	}

	restoreGate := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true)
	defer restoreGate()

	baseText, baseRate, baseHits := gen("compiled, wide off")

	restoreWide := metal.SetFixedAttentionDiagnostics(true, false, false)
	wideText, wideRate, wideHits := gen("compiled, wide SDPA on")
	restoreWide()

	if wideText != baseText {
		t.Errorf("wide-SDPA lane diverged from the wide-off compiled lane:\n  wide-off %q\n  wide-on  %q", baseText, wideText)
	}
	t.Logf("rates: wide-off %.1f · wide-on %.1f tok/s · layer steps %d -> %d", baseRate, wideRate, baseHits, wideHits)
	if wideHits <= baseHits {
		t.Logf("wide-SDPA did not add compiled layers — the 512-wide kernel is still unavailable on this metallib")
	}
}

// TestCompiledLayerDecode_SlidingWindowCrossing_LiveModel decodes far past the
// sliding-window capacity so the owner layers cross from the pre-cap regime
// (offset-indexed write) into the post-cap regime (rotate-and-write via shift
// indices) mid-generation — the transition a real conversation hits. Output
// must stay byte-exact against the default path across the boundary.
//
//	go test -tags model_eval -run TestCompiledLayerDecode_SlidingWindowCrossing_LiveModel -count=1 dappco.re/go/mlx
func TestCompiledLayerDecode_SlidingWindowCrossing_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	info := m.Info()
	if info.SlidingWindow <= 0 {
		t.Skipf("model declares no sliding window")
	}
	// Enough decode tokens to fill the sliding caches and keep rotating well
	// past capacity.
	maxTokens := info.SlidingWindow + 128

	const prompt = "Write a long, detailed story about a clockmaker who repairs time itself."
	ctx := context.Background()

	gen := func(label string) (string, int) {
		t.Helper()
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("%s: NewSession: %v", label, err)
		}
		defer sess.Close()
		if err := sess.Prefill(prompt); err != nil {
			t.Fatalf("%s: Prefill: %v", label, err)
		}
		text := core.NewBuilder()
		tokens := 0
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(maxTokens), WithTemperature(0)) {
			text.WriteString(tok.Text)
			tokens++
		}
		if err := sess.Err(); err != nil {
			t.Fatalf("%s: generate: %v", label, err)
		}
		t.Logf("%s: %d tok (window %d)", label, tokens, info.SlidingWindow)
		return text.String(), tokens
	}

	restoreOff := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, false)
	defaultText, defaultTokens := gen("uncompiled decode")
	restoreOff()
	if defaultTokens < info.SlidingWindow {
		t.Skipf("greedy generation ended after %d tokens — never crossed the %d-token sliding window", defaultTokens, info.SlidingWindow)
	}

	restore := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true)
	hitsBefore := gemma4.CompiledLayerDecodeHits()
	compiledText, _ := gen("compiled layer decode")
	hits := gemma4.CompiledLayerDecodeHits() - hitsBefore

	restorePipe := metal.SetRuntimeGate(metal.GatePipelinedDecode, true)
	pipelinedText, _ := gen("pipelined decode")
	restorePipe()
	restore()

	if hits == 0 {
		t.Errorf("compiled layer decode never served across the window crossing")
	}
	t.Logf("compiled layer decode served %d layer steps", hits)

	if compiledText != defaultText {
		t.Errorf("compiled layer decode diverged across the sliding-window crossing:\n  default  %q\n  compiled %q", defaultText, compiledText)
	}
	if pipelinedText != defaultText {
		t.Errorf("pipelined decode diverged across the sliding-window crossing:\n  default   %q\n  pipelined %q", defaultText, pipelinedText)
	}
}
