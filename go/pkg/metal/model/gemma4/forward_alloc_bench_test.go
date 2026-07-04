// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"runtime"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// AX-11 alloc benches for the three previously un-benched whole-model forward
// entry points: the model's own fused Forward (forward.go), and the pkg/model
// backend-contract adapter's DecodeForward + incremental Step (backend.go). A
// prior pass already proved the attention sub-path and the per-layer compiled
// closure (compiledDecodeForward) flat; these cover the OUTER wrappers that
// drive the whole layer stack and copy bytes across the contract seam.
//
// Harness split (forced by the code, not a preference):
//   - Forward  -> a REAL cached e2b text model. Forward consumes real token ids,
//     so the per-layer-input (PLE) tower computes; e2b is a PLE model.
//   - DecodeForward + Step -> the synthetic tiny-quant model. Both require a
//     *Gemma4Backend, whose only constructor (NewBackend) hard-rejects PLE
//     models — DecodeForward injects DUMMY token ids (make([]int32,T)), so a PLE
//     model has nothing to derive its per-layer inputs from. The synthetic model
//     is built with hidden_size_per_layer_input:0, the one shape the adapter
//     accepts. So these two CANNOT run on real e2b by design.
//
// Read each FLAT profile (the graph eval dominates -benchmem aggregates):
//
//	go tool pprof -alloc_objects -list 'DecodeForward|Step|forwardHiddenOverride' <memprofile>

// benchModelMemLimit caps the unified-memory allocator for the model-bound
// e2b load (OOM guard: e2b only, one load, modest sequence length).
const benchModelMemLimit = uint64(60) << 30

// BenchmarkGemma4Model_Forward_E2B_Prefill benches the model's own fused
// Forward (full layer stack + final norm + output projection + soft-cap) on a
// REAL e2b text model at a modest prefill length. e2b is a per-layer-input
// model, so this is the one of the three targets that genuinely accepts the real
// weights. The hidden allocs hunted here are HOST-side and do not scale with L
// (the [1,L,vocab] logits tensor does, so L is kept small to stay bounded).
func BenchmarkGemma4Model_Forward_E2B_Prefill(b *testing.B) {
	if !metaltest.RunMetalTests {
		b.Skip("build with -tags metal_runtime to run the real-e2b Forward bench")
	}
	if !metal.MetalAvailable() {
		b.Skip("Metal runtime unavailable")
	}
	metal.SetMemoryLimit(benchModelMemLimit)
	modelPath := metaltest.HFModelPath(b, "mlx-community/gemma-4-e2b-it-6bit")

	// Load ONCE — the model is multi-GB; never inside b.Loop().
	m, err := LoadGemma4(modelPath)
	if err != nil {
		b.Fatalf("LoadGemma4(%s): %v", modelPath, err)
	}
	b.Cleanup(func() { closeGemma4(m) })

	// Modest prefill: L>1 exercises the mask builders + per-layer-input split
	// without materialising a large logits tensor each iteration.
	const L = 16
	ids := make([]int32, L)
	for i := range ids {
		ids[i] = int32(2 + i%6) // in-vocab, deterministic — token ids are pinned
	}

	// Warm once (compiles graph / primes kernels) so the bench measures steady
	// state, not first-call compilation.
	{
		tokens := metal.FromValues(ids, 1, L)
		caches := m.NewCache()
		out := m.Forward(tokens, caches)
		if err := metal.Eval(out); err != nil {
			b.Fatalf("warm-up Forward eval: %v", err)
		}
		metal.Free(tokens, out)
		metal.FreeCaches(caches)
	}

	runtime.MemProfileRate = 1 // after load: profile the per-iteration host allocs, not the load
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		tokens := metal.FromValues(ids, 1, L)
		caches := m.NewCache()
		out := m.Forward(tokens, caches)
		if err := metal.Eval(out); err != nil {
			b.Fatalf("Forward eval: %v", err)
		}
		metal.Free(tokens, out)
		metal.FreeCaches(caches)
	}
	b.StopTimer()
}

// BenchmarkGemma4Backend_DecodeForward_Synthetic benches the pkg/model
// backend-contract adapter's whole-sequence DecodeForward (precomputed
// embeddings injected into the proven stack via the embedHook over a fresh
// cache). It runs on the synthetic tiny-quant model because NewBackend rejects
// the real e2b PLE model (see file doc). The figures of interest are the
// host-side seam copies: the flat byte buffer, the per-token output slices, and
// the embedHook closure.
func BenchmarkGemma4Backend_DecodeForward_Synthetic(b *testing.B) {
	m := cov4clBuildQuantModel(b, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	backend, err := NewBackend(m)
	if err != nil {
		b.Fatalf("NewBackend: %v", err)
	}

	// A short embedding sequence (the whole-seq contract decode). Embeddings are
	// the model's activation-dtype bytes — gather them once, reuse every iter so
	// the bench isolates DecodeForward, not Embed.
	prompt := []int32{2, 3, 4}
	embs := make([][]byte, len(prompt))
	for i, id := range prompt {
		e, eerr := backend.Embed(id)
		if eerr != nil {
			b.Fatalf("Embed: %v", eerr)
		}
		embs[i] = e
	}

	if _, err := backend.DecodeForward(embs); err != nil { // warm
		b.Fatalf("warm-up DecodeForward: %v", err)
	}

	runtime.MemProfileRate = 1
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		out, derr := backend.DecodeForward(embs)
		if derr != nil {
			b.Fatalf("DecodeForward: %v", derr)
		}
		_ = out
	}
	b.StopTimer()
}

// BenchmarkGemma4Stepper_Step_Synthetic benches the backend's incremental
// DecodeStepper.Step — one token at a time over a PERSISTENT cache (the O(1)/
// token serve shape). Synthetic model for the same NewBackend-rejects-PLE reason.
// The cache fills after max_position_embeddings steps; the loop re-opens a fresh
// session (timer-stopped) before it would overflow so the FLAT profile only
// reflects Step itself, not session setup.
func BenchmarkGemma4Stepper_Step_Synthetic(b *testing.B) {
	m := cov4clBuildQuantModel(b, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	backend, err := NewBackend(m)
	if err != nil {
		b.Fatalf("NewBackend: %v", err)
	}
	emb, err := backend.Embed(3)
	if err != nil {
		b.Fatalf("Embed: %v", err)
	}

	// max_position_embeddings for the synthetic config is 64; leave headroom.
	const cacheCap = 60

	openSession := func() (model interface{ Step([]byte) ([]byte, error) }, closer func()) {
		s, oerr := backend.OpenSession()
		if oerr != nil {
			b.Fatalf("OpenSession: %v", oerr)
		}
		return s, func() {
			if c, ok := s.(interface{ Close() error }); ok {
				_ = c.Close()
			}
		}
	}

	sess, closeSess := openSession()
	defer func() { closeSess() }()
	if _, serr := sess.Step(emb); serr != nil { // warm
		b.Fatalf("warm-up Step: %v", serr)
	}
	steps := 1

	runtime.MemProfileRate = 1
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if steps >= cacheCap {
			b.StopTimer()
			closeSess()
			sess, closeSess = openSession()
			steps = 0
			b.StartTimer()
		}
		out, serr := sess.Step(emb)
		if serr != nil {
			b.Fatalf("Step: %v", serr)
		}
		_ = out
		steps++
	}
	b.StopTimer()
}
