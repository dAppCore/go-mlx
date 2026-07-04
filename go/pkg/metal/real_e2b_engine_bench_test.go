// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package metal

import (
	"context"
	"os"
	"runtime/debug"
	"testing"
)

// Real-E2B engine benches (AX-11, model-bound pass). The synthetic decode-loop
// micro-benches in this package measure isolated ops over fixture weights; the
// two paths below had NO real-model bench and are the heaviest un-benched
// per-request / per-token engine allocations in the live cgo serve engine
// (pkg/metal):
//
//   - CaptureKV — the KV-cache GPU->CPU snapshot capture. A per-REQUEST path
//     (store-wake / agent-memory checkpoint): every captured layer copies its
//     K/V head bytes back to host and the assembly clones the per-layer shape +
//     head slices. allocs/op is the figure of merit (B/op is floored by the
//     KeyBytes/ValueBytes + logits output, which restore needs verbatim).
//   - Sample/SampleToken — drawn once per EMITTED token (high multiplier: paid
//     on every token of every generation). Benched through the real token
//     producer SampleTokenIDWithSuppressionGuard (sampler.Sample alone returns
//     modified logits, not a token, for temp/top-k/top-p; the categorical draw
//     + Eval boundary + Int() read are the per-token cost) over the production
//     sampler lane across greedy / temperature / top-k / top-p.
//
// Both load the SAME real gemma-4-e2b-it-4bit checkpoint through the cgo engine
// (LoadAndInit, not the no-cgo pkg/native LoadGemma4Dir). ONE prefill feeds both
// benches: the captured snapshot already carries Logits []float32, so the
// sampler input is built from it with no second forward pass and no private
// field access.
//
// AX-11 model-loads gate + OOM guard (same contract as the pkg/native real-e2b
// benches): OPT-IN via E2B_Q4_DIR (skips in `core go qa` / CI, which never set
// it); model loaded ONCE before ResetTimer and reused across b.N; -benchtime=5x
// ceiling enforced; SetMemoryLimit(60 GiB) GC backstop; small ContextLen to
// bound KV prealloc. Run them:
//
//	export MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
//	export E2B_Q4_DIR=~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<rev>
//	go test -tags metal_runtime -run '^$' -bench '^BenchmarkRealE2BCaptureKV$' -benchmem \
//	        -benchtime=5x -memprofile=/tmp/e2b-capturekv.alloc ./pkg/metal
//	go test -tags metal_runtime -run '^$' -bench '^BenchmarkRealE2BSampleToken$' -benchmem \
//	        -benchtime=5x -memprofile=/tmp/e2b-sample.alloc ./pkg/metal

const (
	// A 256-token prompt — a realistic prefilled context for the capture path.
	realE2BEngPromptLen = 256
	// -benchtime ceiling (OOM guard). The session is reused; CaptureKV does not
	// advance position and the sampler benches run against a frozen logits Array,
	// so the working set stays flat regardless of b.N, but the cap is honoured to
	// mirror the pkg/native contract.
	realE2BEngBenchMax = 5
	// ContextLen kept small to bound KV prealloc for the 256-token prompt.
	realE2BEngContextLen = 4096
	// Fixed sampler seed: same seed -> same draw sequence, so the stochastic
	// (temp/top-k/top-p) configs are byte-identity verifiable across a fix.
	realE2BEngSamplerSeed = 0x5EED1234
)

// realE2BEngineDir returns the E2B-4bit snapshot directory, or "" (skip) if
// unset. Mirrors the pkg/native realE2BDir convention.
func realE2BEngineDir() string { return os.Getenv("E2B_Q4_DIR") }

// realE2BEnginePrompt builds a fixed prompt of valid E2B token ids — spread
// across the vocab, deterministic. Distinct stride from the pkg/native fixtures.
func realE2BEnginePrompt() []int32 {
	p := make([]int32, realE2BEngPromptLen)
	for i := range p {
		p[i] = int32(5 + i*59) // 256 ids up to ~15k, comfortably within the 256k vocab
	}
	return p
}

// loadRealE2BEngineModel loads the e2b-4bit checkpoint through the cgo engine,
// once. The caller must Close it. Skips (not fails) when the model is unavailable
// so the bench is opt-in.
func loadRealE2BEngineModel(tb testing.TB) *Model {
	tb.Helper()
	requireMetalRuntime(tb)
	dir := realE2BEngineDir()
	if dir == "" {
		tb.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model bench)")
	}
	m, err := LoadAndInit(dir, LoadConfig{ContextLen: realE2BEngContextLen})
	if err != nil {
		tb.Fatalf("LoadAndInit(%s): %v", dir, err)
	}
	return m
}

// prefilledRealE2BSession loads the model, opens a session, and prefills the
// fixed prompt. Returns the model + session + the captured logits (real, from
// the prefill) for the sampler benches. The caller must Close both.
func prefilledRealE2BSession(tb testing.TB, m *Model) *ModelSession {
	tb.Helper()
	s, ok := m.NewSession().(*ModelSession)
	if !ok {
		tb.Fatalf("NewSession did not return *ModelSession: %T", m.NewSession())
	}
	if err := s.PrefillTokens(context.Background(), realE2BEnginePrompt()); err != nil {
		_ = s.Close()
		tb.Fatalf("PrefillTokens: %v", err)
	}
	return s
}

// BenchmarkRealE2BCaptureKV measures the heap allocations of one KV snapshot
// capture over a real 256-token prefilled session — the per-request store-wake
// path. allocs/op is the figure of merit. The model + prefill happen once before
// ResetTimer; only the repeated CaptureKV is timed.
//
// Clean A/B for the single-owner-heads fix (real gemma-4-e2b-it-4bit, 256-tok
// prefilled session, -benchtime=5x, NO profiler — memprofilerate=1 inflates
// ns/op and must be off when timing):
//
//	             allocs/op   B/op     ns/op
//	pre-fix          583    53.0M    14.4ms
//	after-fix        502    38.8M    14.0ms
//
// The win is allocs/op (-81) and B/op (-14.2M) — the redundant per-head data
// copy off the per-request path. Per AX-11 (flat memory pressure: fewer
// allocs/bytes => less GC sawtooth per request), that is the payoff, NOT
// latency: ns/op is flat (a 14M memcpy is sub-ms at memory bandwidth). The
// commit cfea6b50 message quoted ns/op 81.1ms->16.8ms; that swing was a
// GODEBUG=memprofilerate=1 artifact (the baseline ran with it on, the re-bench
// without), not the fix. Treat the table above as the corrected record; ns/op
// was never the figure of merit here.
func BenchmarkRealE2BCaptureKV(b *testing.B) {
	if b.N > realE2BEngBenchMax {
		b.Skipf("real-e2b capture bench is capped at -benchtime=%dx (OOM guard); got b.N=%d", realE2BEngBenchMax, b.N)
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30)) // GC backstop; restore prior on exit

	m := loadRealE2BEngineModel(b)
	defer func() { _ = m.Close() }()
	s := prefilledRealE2BSession(b, m)
	defer func() { _ = s.Close() }()
	ctx := context.Background()

	// Warm the path once outside timing (first capture allocates lazily-init'd
	// caches the rest reuse), and prove it produces a snapshot.
	if snap, err := s.CaptureKV(ctx); err != nil || snap == nil {
		b.Fatalf("warm CaptureKV: snap=%v err=%v", snap, err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := s.CaptureKV(ctx)
		if err != nil {
			b.Fatalf("CaptureKV (op %d): %v", i, err)
		}
		if snap == nil || snap.NumLayers == 0 {
			b.Fatalf("CaptureKV (op %d): empty snapshot", i)
		}
	}
}

// BenchmarkRealE2BSampleToken measures the per-emitted-token sampler cost over
// real logits across the production configs. The token producer
// (SampleTokenIDWithSuppressionGuard) is the real path: it draws, Evals, reads
// the id. The logits Array is built once from the prefilled session's captured
// logits and reused frozen across b.N, so only the sampler is measured.
func BenchmarkRealE2BSampleToken(b *testing.B) {
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))

	m := loadRealE2BEngineModel(b)
	defer func() { _ = m.Close() }()
	s := prefilledRealE2BSession(b, m)
	defer func() { _ = s.Close() }()

	logits := capturedLogitsArray(b, s)
	defer Free(logits)

	configs := []struct {
		name             string
		temp, topP, minP float32
		topK             int
	}{
		{"greedy", 0, 0, 0, 0},
		{"temperature", 0.7, 0, 0, 0},
		{"topK", 1.0, 0, 0, 40},
		{"topP", 1.0, 0.95, 0, 0},
		{"topKtopP", 0.7, 0.95, 0, 40}, // the Gemma-4 production lane (topKTopPChain)
	}
	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			sampler := NewSamplerWithSuppressionKeyed(cfg.temp, cfg.topP, cfg.minP, cfg.topK, nil, NewSamplerKeys(realE2BEngSamplerSeed))
			defer CloseSampler(sampler)
			// Warm the compiled-sampler / pooled-scratch path once outside timing.
			if tok, err := SampleTokenWithSuppressionGuard(logits, sampler, nil); err != nil {
				b.Fatalf("warm sample: %v", err)
			} else {
				Free(tok)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tok, err := SampleTokenWithSuppressionGuard(logits, sampler, nil)
				if err != nil {
					b.Fatalf("SampleToken (%s op %d): %v", cfg.name, i, err)
				}
				Free(tok)
			}
		})
	}
}

// capturedLogitsArray builds a 1×vocab float32 logits Array from the session's
// captured logits — real model output, no second forward. The snapshot carries
// the last-token logit row (LogitShape == [1, vocab]).
func capturedLogitsArray(tb testing.TB, s *ModelSession) *Array {
	tb.Helper()
	snap, err := s.CaptureKV(context.Background())
	if err != nil {
		tb.Fatalf("CaptureKV for logits: %v", err)
	}
	if len(snap.Logits) == 0 || len(snap.LogitShape) == 0 {
		tb.Fatalf("captured snapshot has no logits (shape=%v len=%d)", snap.LogitShape, len(snap.Logits))
	}
	shape := make([]int, len(snap.LogitShape))
	for i, d := range snap.LogitShape {
		shape[i] = int(d)
	}
	logits := FromValues(snap.Logits, shape...)
	if err := Eval(logits); err != nil {
		Free(logits)
		tb.Fatalf("Eval logits: %v", err)
	}
	Detach(logits)
	return logits
}

// TestRealE2BCaptureKVDeterministic is the byte-identity precondition for the
// CaptureKV alloc-reduction work: two captures of the SAME prefilled session
// must produce identical K/V bytes, tokens, and logits. Any capture-side alloc
// fix is validated by re-running and confirming these are unchanged.
func TestRealE2BCaptureKVDeterministic(t *testing.T) {
	requireMetalRuntime(t)
	dir := realE2BEngineDir()
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model test)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))

	m, err := LoadAndInit(dir, LoadConfig{ContextLen: realE2BEngContextLen})
	if err != nil {
		t.Fatalf("LoadAndInit: %v", err)
	}
	defer func() { _ = m.Close() }()
	s := prefilledRealE2BSession(t, m)
	defer func() { _ = s.Close() }()
	ctx := context.Background()

	a, err := s.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV a: %v", err)
	}
	c, err := s.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV c: %v", err)
	}
	assertKVSnapshotsEqual(t, a, c)
}

// assertKVSnapshotsEqual fails if two snapshots differ in the fields a restore
// reads back. Used as the CaptureKV byte-identity anchor.
func assertKVSnapshotsEqual(t *testing.T, a, c *KVSnapshot) {
	t.Helper()
	if a.NumLayers != c.NumLayers || a.SeqLen != c.SeqLen || a.NumHeads != c.NumHeads || a.HeadDim != c.HeadDim {
		t.Fatalf("snapshot geometry differs: a={L:%d S:%d H:%d D:%d} c={L:%d S:%d H:%d D:%d}",
			a.NumLayers, a.SeqLen, a.NumHeads, a.HeadDim, c.NumLayers, c.SeqLen, c.NumHeads, c.HeadDim)
	}
	if !int32SliceEqual(a.Tokens, c.Tokens) {
		t.Fatalf("snapshot tokens differ")
	}
	if !float32SliceEqual(a.Logits, c.Logits) {
		t.Fatalf("snapshot logits differ")
	}
	if len(a.Layers) != len(c.Layers) {
		t.Fatalf("layer count differs: %d vs %d", len(a.Layers), len(c.Layers))
	}
	for i := range a.Layers {
		la, lc := a.Layers[i], c.Layers[i]
		if !byteSliceEqual(la.KeyBytes, lc.KeyBytes) || !byteSliceEqual(la.ValueBytes, lc.ValueBytes) {
			t.Fatalf("layer %d K/V bytes differ", i)
		}
		if len(la.Heads) != len(lc.Heads) {
			t.Fatalf("layer %d head count differs: %d vs %d", i, len(la.Heads), len(lc.Heads))
		}
		for h := range la.Heads {
			if !byteSliceEqual(la.Heads[h].KeyBytes, lc.Heads[h].KeyBytes) ||
				!byteSliceEqual(la.Heads[h].ValueBytes, lc.Heads[h].ValueBytes) {
				t.Fatalf("layer %d head %d K/V bytes differ", i, h)
			}
			if !float32SliceEqual(la.Heads[h].Key, lc.Heads[h].Key) ||
				!float32SliceEqual(la.Heads[h].Value, lc.Heads[h].Value) {
				t.Fatalf("layer %d head %d K/V floats differ", i, h)
			}
		}
	}
}

// TestRealE2BSampleTokenDeterministic is the byte-identity precondition for the
// sampler alloc-reduction work: under a FIXED seed, each config must produce the
// SAME token id on repeated draws against the same logits. A sampler-side fix is
// validated by re-running and confirming the per-config id is unchanged.
func TestRealE2BSampleTokenDeterministic(t *testing.T) {
	requireMetalRuntime(t)
	dir := realE2BEngineDir()
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the gemma-4-e2b-it-4bit snapshot dir (opt-in real-model test)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30))

	m, err := LoadAndInit(dir, LoadConfig{ContextLen: realE2BEngContextLen})
	if err != nil {
		t.Fatalf("LoadAndInit: %v", err)
	}
	defer func() { _ = m.Close() }()
	s := prefilledRealE2BSession(t, m)
	defer func() { _ = s.Close() }()
	logits := capturedLogitsArray(t, s)
	defer Free(logits)

	configs := []struct {
		name             string
		temp, topP, minP float32
		topK             int
	}{
		{"greedy", 0, 0, 0, 0},
		{"temperature", 0.7, 0, 0, 0},
		{"topK", 1.0, 0, 0, 40},
		{"topP", 1.0, 0.95, 0, 0},
		{"topKtopP", 0.7, 0.95, 0, 40},
	}
	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			id1 := drawOneToken(t, logits, cfg.temp, cfg.topP, cfg.minP, cfg.topK)
			id2 := drawOneToken(t, logits, cfg.temp, cfg.topP, cfg.minP, cfg.topK)
			if id1 != id2 {
				t.Fatalf("%s: fixed-seed draw not deterministic: %d vs %d", cfg.name, id1, id2)
			}
			if id1 < 0 {
				t.Fatalf("%s: negative token id %d", cfg.name, id1)
			}
			t.Logf("%s fixed-seed token id = %d", cfg.name, id1)
		})
	}
}

// drawOneToken samples one token under a fresh fixed-seed key sequence — same
// seed each call, so a deterministic sampler returns the same id.
func drawOneToken(t *testing.T, logits *Array, temp, topP, minP float32, topK int) int32 {
	t.Helper()
	sampler := NewSamplerWithSuppressionKeyed(temp, topP, minP, topK, nil, NewSamplerKeys(realE2BEngSamplerSeed))
	defer CloseSampler(sampler)
	_, id, _, err := SampleTokenIDWithSuppressionGuard(logits, sampler, nil, false)
	if err != nil {
		t.Fatalf("SampleTokenID: %v", err)
	}
	return id
}

func int32SliceEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func float32SliceEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func byteSliceEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
