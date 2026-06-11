// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"crypto/sha256"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

// Determinism probes for the bf16 activation stream. Greedy decode must be
// bit-deterministic run to run — the state system's byte-exact sleep/wake
// story depends on it. mlx-lm on the same snapshot is hash-identical across
// runs, so any fork here is ours. 256 tokens keeps the probe inside the
// sliding window (pre-cap only), excluding the post-cap unit from the
// suspect set; the known fork reproduces by ~token 20.
//
// Trace caveat: compiled-layer trace keys do not carry gate state, so each
// gate configuration must run in a FRESH process — invoke one test per
// `go test -run` call, never both in one binary run.

func decodeDeterminismProbe(t *testing.T, pairs int) {
	t.Helper()
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	ctx := context.Background()
	run := func() string {
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("NewSession: %v", err)
		}
		defer sess.Close()
		if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		text := core.NewBuilder()
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(256), WithTemperature(0)) {
			text.WriteString(tok.Text)
		}
		if err := sess.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return text.String()
	}
	reference := run()
	for pair := 1; pair <= pairs; pair++ {
		got := run()
		if got != reference {
			i := 0
			for i < len(reference) && i < len(got) && reference[i] == got[i] {
				i++
			}
			t.Fatalf("non-deterministic at pair %d, first byte diff at %d:\n  a %q\n  b %q",
				pair, i, reference[max(0, i-40):min(len(reference), i+40)], got[max(0, i-40):min(len(got), i+40)])
		}
	}
	t.Logf("deterministic across %d repeat runs", pairs)
}

// TestDecodeDeterminism_LiveModel — everything on (the shipping config).
//
//	go test -tags model_eval -run 'TestDecodeDeterminism_LiveModel$' -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_LiveModel(t *testing.T) {
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_GemmMLP_LiveModel — the custom fused MLP kernels off
// (gemm via MLX quantized_matmul, the ops mlx-lm itself runs). If this is
// deterministic while the default probe forks, the fused MLP kernels are the
// culprit. MUST run in its own process (trace keys do not carry gate state).
//
//	go test -tags model_eval -run TestDecodeDeterminism_GemmMLP_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_GemmMLP_LiveModel(t *testing.T) {
	restore := metal.SetRuntimeGate(metal.GateNativeMLPMatVec, false)
	defer restore()
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_SerialCompiled_LiveModel — one-ahead pipeline off,
// compiled layers on. Splits loop structure from layer math.
//
//	go test -tags model_eval -run TestDecodeDeterminism_SerialCompiled_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_SerialCompiled_LiveModel(t *testing.T) {
	restore := metal.SetRuntimeGate(metal.GatePipelinedDecode, false)
	defer restore()
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_Uncompiled_LiveModel — pipeline AND compiled layers
// off: the plain serial loop over the uncompiled paths.
//
//	go test -tags model_eval -run TestDecodeDeterminism_Uncompiled_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_Uncompiled_LiveModel(t *testing.T) {
	restorePipe := metal.SetRuntimeGate(metal.GatePipelinedDecode, false)
	defer restorePipe()
	restoreCompiled := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, false)
	defer restoreCompiled()
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_GoSampler_LiveModel — the C++ greedy head unit off
// (DirectGreedyToken gate): token selection goes through the Go sampler path
// instead of the compiled q4 last-token + argmax unit.
//
//	go test -tags model_eval -run TestDecodeDeterminism_GoSampler_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_GoSampler_LiveModel(t *testing.T) {
	restore := metal.SetRuntimeGate(metal.GateDirectGreedyToken, false)
	defer restore()
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_SyncEval_LiveModel — pipeline, compiled layers, AND
// async prefetch off: the most synchronous decode the engine has. If this is
// deterministic while every async config forks, the non-determinism is in
// the async eval orchestration (in-flight batches, buffer-pool reuse), not
// in any kernel's math — consistent with every isolated kernel probe
// hashing identical.
//
//	go test -tags model_eval -run TestDecodeDeterminism_SyncEval_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_SyncEval_LiveModel(t *testing.T) {
	restorePipe := metal.SetRuntimeGate(metal.GatePipelinedDecode, false)
	defer restorePipe()
	restoreCompiled := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, false)
	defer restoreCompiled()
	restorePrefetch := metal.SetRuntimeGate(metal.GateAsyncDecodePrefetch, false)
	defer restorePrefetch()
	decodeDeterminismProbe(t, 4)
}

// logitsFingerprint is one decode step's logits identity: the float64 mean
// catches a single-LSB change anywhere in the vector; max id/value catch the
// argmax flip itself.
type logitsFingerprint struct {
	step       int
	meanBits   uint64
	maxLogit   float32
	maxTokenID int32
}

// TestDecodeDeterminism_LogitsFingerprint_LiveModel localises the fork: two
// identical sessions record per-step logits fingerprints; the first step
// whose fingerprint differs is where the varying op lands. A difference at
// step 0 means a single forward is internally non-deterministic; stability
// for k steps then divergence implicates accumulated state (cache writes).
//
//	go test -tags model_eval -run TestDecodeDeterminism_LogitsFingerprint_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_LogitsFingerprint_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	ctx := context.Background()

	run := func() []logitsFingerprint {
		var prints []logitsFingerprint
		sink := probe.SinkFunc(func(event probe.Event) {
			if event.Kind != probe.KindLogits || event.Logits == nil {
				return
			}
			prints = append(prints, logitsFingerprint{
				step:       event.Step,
				meanBits:   math.Float64bits(event.Logits.MeanLogit),
				maxLogit:   event.Logits.MaxLogit,
				maxTokenID: event.Logits.MaxTokenID,
			})
		})
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("NewSession: %v", err)
		}
		defer sess.Close()
		if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		for range sess.GenerateStream(ctx, WithMaxTokens(48), WithTemperature(0), WithProbeSink(sink)) {
		}
		if err := sess.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return prints
	}

	a, b := run(), run()
	if len(a) == 0 || len(b) == 0 {
		t.Fatalf("no logits probes captured (a=%d b=%d)", len(a), len(b))
	}
	steps := min(len(a), len(b))
	for i := 0; i < steps; i++ {
		if a[i] != b[i] {
			t.Logf("first fingerprint divergence at probe %d (step %d):", i, a[i].step)
			t.Logf("  a: meanBits=%016x max=%v id=%d", a[i].meanBits, a[i].maxLogit, a[i].maxTokenID)
			t.Logf("  b: meanBits=%016x max=%v id=%d", b[i].meanBits, b[i].maxLogit, b[i].maxTokenID)
			return
		}
	}
	t.Logf("all %d fingerprints identical — the varying op is downstream of the logits summary", steps)
}

// TestDecodeDeterminism_CacheHash_LiveModel discriminates write-vs-forward:
// generate exactly ONE token in two identical sessions and hash every cache
// tensor. Differing hashes = the step-0 cache WRITES vary run to run;
// identical hashes = the step-1 forward itself varies on identical state.
//
//	go test -tags model_eval -run TestDecodeDeterminism_CacheHash_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_CacheHash_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	ctx := context.Background()

	run := func(decodeTokens int) []string {
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("NewSession: %v", err)
		}
		defer sess.Close()
		if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		if decodeTokens > 0 {
			for range sess.GenerateStream(ctx, WithMaxTokens(decodeTokens), WithTemperature(0)) {
			}
			if err := sess.Err(); err != nil {
				t.Fatalf("generate: %v", err)
			}
		}
		snapshot, err := sess.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
		if err != nil {
			t.Fatalf("CaptureKV: %v", err)
		}
		hashes := make([]string, 0, len(snapshot.Layers))
		for _, layer := range snapshot.Layers {
			sum := sha256.Sum256(layer.KeyBytes)
			sumV := sha256.Sum256(layer.ValueBytes)
			hashes = append(hashes, core.Sprintf("%x:%x", sum[:6], sumV[:6]))
		}
		return hashes
	}

	compare := func(label string, a, b []string) int {
		if len(a) != len(b) {
			t.Fatalf("%s: layer counts differ: %d vs %d", label, len(a), len(b))
		}
		diffs := 0
		for i := range a {
			if a[i] != b[i] {
				diffs++
				if diffs <= 3 {
					t.Logf("%s: cache %d differs: %s vs %s", label, i, a[i], b[i])
				}
			}
		}
		t.Logf("%s: %d of %d caches differ", label, diffs, len(a))
		return diffs
	}

	prefillDiffs := compare("post-prefill", run(0), run(0))
	if prefillDiffs > 0 {
		t.Logf("the PREFILL writes vary — the decode loop is downstream of the problem")
		return
	}
	compare("post-1-token", run(1), run(1))
}
