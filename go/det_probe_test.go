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
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
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

// decodeDeterminismProbe loads the model, then applies gates — the loader
// applies the model's declared EngineFeatures (gates ON) over anything set
// earlier, so a gate flip only sticks POST-load. Round 1 set gates before
// LoadModel and silently measured the all-on path in every config.
func decodeDeterminismProbe(t *testing.T, pairs int, gates map[metal.Gate]bool) {
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
	for gate, enabled := range gates {
		restore := metal.SetRuntimeGate(gate, enabled)
		defer restore()
	}
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
	decodeDeterminismProbe(t, 4, nil)
}

// TestDecodeDeterminism_GemmMLP_LiveModel — the custom fused MLP kernels off
// (gemm via MLX quantized_matmul, the ops mlx-lm itself runs). If this is
// deterministic while the default probe forks, the fused MLP kernels are the
// culprit. MUST run in its own process (trace keys do not carry gate state).
//
//	go test -tags model_eval -run TestDecodeDeterminism_GemmMLP_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_GemmMLP_LiveModel(t *testing.T) {
	decodeDeterminismProbe(t, 4, map[metal.Gate]bool{metal.GateNativeMLPMatVec: false})
}

// TestDecodeDeterminism_SerialCompiled_LiveModel — one-ahead pipeline off,
// compiled layers on. Splits loop structure from layer math.
//
//	go test -tags model_eval -run TestDecodeDeterminism_SerialCompiled_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_SerialCompiled_LiveModel(t *testing.T) {
	decodeDeterminismProbe(t, 4, map[metal.Gate]bool{metal.GatePipelinedDecode: false})
}

// TestDecodeDeterminism_Uncompiled_LiveModel — pipeline AND compiled layers
// off: the plain serial loop over the uncompiled paths.
//
//	go test -tags model_eval -run TestDecodeDeterminism_Uncompiled_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_Uncompiled_LiveModel(t *testing.T) {
	decodeDeterminismProbe(t, 4, map[metal.Gate]bool{
		metal.GatePipelinedDecode:     false,
		metal.GateCompiledLayerDecode: false,
	})
}

// TestDecodeDeterminism_GoSampler_LiveModel — the C++ greedy head unit off
// (DirectGreedyToken gate): token selection goes through the Go sampler path
// instead of the compiled q4 last-token + argmax unit.
//
//	go test -tags model_eval -run TestDecodeDeterminism_GoSampler_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_GoSampler_LiveModel(t *testing.T) {
	decodeDeterminismProbe(t, 4, map[metal.Gate]bool{metal.GateDirectGreedyToken: false})
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
	decodeDeterminismProbe(t, 4, map[metal.Gate]bool{
		metal.GatePipelinedDecode:     false,
		metal.GateCompiledLayerDecode: false,
		metal.GateAsyncDecodePrefetch: false,
	})
}

// TestDecodeDeterminism_PLIPieces_LiveModel hammers the two kernels of the
// per-layer-input tensor path with the REAL model weights — the segment the
// cache-hash pattern indicts (layer-0 K/V clean, every later layer varying =
// the once-per-forward PLI tensor varying). (a) the quantized per-layer
// embedding gather; (b) the PerLayerModelProj matmul at its irregular output
// width. Any hash change across repeats names the op.
//
//	go test -tags model_eval -run TestDecodeDeterminism_PLIPieces_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_PLIPieces_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	metalModel, ok := m.model.(*metal.Model)
	if !ok {
		t.Fatalf("model is %T, want *metal.Model", m.model)
	}
	g, ok := metalModel.UnderlyingModel().(*gemma4.Gemma4Model)
	if !ok {
		t.Fatalf("underlying model is %T, want *gemma4.Gemma4Model", metalModel.UnderlyingModel())
	}

	hashArray := func(arr *metal.Array) [32]byte {
		t.Helper()
		f32 := metal.AsType(arr, metal.DTypeFloat32)
		if err := metal.Eval(f32); err != nil {
			t.Fatalf("Eval: %v", err)
		}
		floats := f32.Floats()
		bytes := make([]byte, 0, len(floats)*4)
		for _, f := range floats {
			u := math.Float32bits(f)
			bytes = append(bytes, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
		}
		metal.Free(f32)
		return sha256.Sum256(bytes)
	}

	probe := func(name string, build func() *metal.Array) {
		t.Helper()
		first := build()
		reference := hashArray(first)
		metal.Free(first)
		for i := 0; i < 200; i++ {
			arr := build()
			got := hashArray(arr)
			metal.Free(arr)
			if got != reference {
				t.Fatalf("%s non-deterministic at repeat %d", name, i)
			}
		}
		t.Logf("%s: 200 repeats hash-identical", name)
	}

	tokens := metal.FromValues([]int32{236776}, 1, 1)
	defer metal.Free(tokens)
	probe("per-layer embed gather", func() *metal.Array {
		return g.EmbedTokensPerLayer.Forward(tokens)
	})
	probe("main embed gather", func() *metal.Array {
		return g.EmbedTokens.Forward(tokens)
	})

	hidden := g.EmbedTokens.Forward(tokens)
	defer metal.Free(hidden)
	probe("per-layer model proj", func() *metal.Array {
		return g.PerLayerModelProj.Forward(hidden)
	})
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

	firstHashes := run(1)
	t.Logf("first-run cache-1 hash: %s", firstHashes[1])
	prefillDiffs := compare("post-prefill", run(0), run(0))
	if prefillDiffs > 0 {
		t.Logf("the PREFILL writes vary — the decode loop is downstream of the problem")
		return
	}
	compare("post-1-token", run(1), run(1))
}

// TestDecodeDeterminism_PhaseHash_LiveModel — round 4: name the op. Runs the
// forking config (uncompiled + synchronous), hashes every layer-phase tensor
// of the FIRST decode forward in two identical sessions, and reports the
// first phase whose value hash differs. Phase order per layer: attention ->
// attention_residual -> [ffn stages] -> ffn -> output (the per-layer-input
// block sits between ffn and output). Caveat: hashing materialises per
// phase, which steers pool behaviour — if the fork vanishes under this
// instrument, that is itself evidence (the stale read needs the batched
// graph's buffer-reuse pattern).
//
//	go test -tags model_eval -run TestDecodeDeterminism_PhaseHash_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_PhaseHash_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	restorePipe := metal.SetRuntimeGate(metal.GatePipelinedDecode, false)
	defer restorePipe()
	restoreCompiled := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, false)
	defer restoreCompiled()
	restorePrefetch := metal.SetRuntimeGate(metal.GateAsyncDecodePrefetch, false)
	defer restorePrefetch()

	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	ctx := context.Background()

	run := func() []metal.NativePhaseValueHash {
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("NewSession: %v", err)
		}
		defer sess.Close()
		// Plumbing check: flag on for prefill AND decode; prefill phases are a
		// prefix with L>1 shapes, decode phases follow. Trim later if noisy.
		metal.SetNativePhaseValueHashCapture(true)
		defer metal.SetNativePhaseValueHashCapture(false)
		if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		for range sess.GenerateStream(ctx, WithMaxTokens(1), WithTemperature(0)) {
		}
		if err := sess.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return metal.TakeNativePhaseValueHashes()
	}

	a, b := run(), run()
	if len(a) == 0 || len(b) == 0 {
		t.Fatalf("no phase hashes captured (a=%d b=%d)", len(a), len(b))
	}
	if len(a) != len(b) {
		t.Logf("phase counts differ: %d vs %d (sequence mismatch)", len(a), len(b))
	}
	steps := min(len(a), len(b))
	diffs := 0
	for i := 0; i < steps; i++ {
		if a[i].Name != b[i].Name {
			t.Fatalf("phase sequence diverged at %d: %q vs %q", i, a[i].Name, b[i].Name)
		}
		if a[i].Hash != b[i].Hash {
			diffs++
			if diffs <= 6 {
				t.Logf("phase %d %s differs: %s vs %s", i, a[i].Name, a[i].Hash, b[i].Hash)
			}
		}
	}
	if diffs == 0 {
		t.Logf("all %d phase hashes identical — the fork vanished under per-phase materialisation (pool-pattern dependent)", steps)
	} else {
		t.Logf("%d of %d phases differ; first varying phase named above", diffs, steps)
	}
}

// TestDecodeDeterminism_FusedGateUpOnly_LiveModel — inside the compiled
// closures, only the fused gate+up GELU-split kernel runs; the down
// projection takes gemm. Forks here = the GELU-split kernel is the culprit.
//
//	go test -tags model_eval -run TestDecodeDeterminism_FusedGateUpOnly_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_FusedGateUpOnly_LiveModel(t *testing.T) {
	metal.SetTracedMLPFusedStages(true, false)
	defer metal.SetTracedMLPFusedStages(true, true)
	decodeDeterminismProbe(t, 4, nil)
}

// TestDecodeDeterminism_FusedDownOnly_LiveModel — inside the compiled
// closures, gate+up take gemm + GeluGateMul; only the fused down matvec
// kernel runs. Forks here = the down matvec kernel is the culprit.
//
//	go test -tags model_eval -run TestDecodeDeterminism_FusedDownOnly_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_FusedDownOnly_LiveModel(t *testing.T) {
	metal.SetTracedMLPFusedStages(false, true)
	defer metal.SetTracedMLPFusedStages(true, true)
	decodeDeterminismProbe(t, 4, nil)
}
