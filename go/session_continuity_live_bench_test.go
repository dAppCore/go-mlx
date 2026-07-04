// SPDX-Licence-Identifier: EUPL-1.2

// Real-model benchmarks for the session Wake/Sleep continuity paths — the LEM
// continuity feature — measured against the cached gemma-4 e2b 4bit pack
// through the cgo pkg/metal engine (the backend that owns CaptureKV /
// RangeKVBlocks; the no-cgo pkg/native backend has no snapshot path, so the
// save/restore cycle is unmeasurable there). These were the un-benched
// continuity entry points: the synthetic-dispatch surface in ./session was
// already floored against sessionfake, but the REAL-model save (capture +
// serialise) / restore cycle had no allocation figure.
//
// Each bench loads the model ONCE via the shared requireLiveE2BModel harness
// (Metal gate + flake retry + RAM-budget once-guard), prefills a source
// session ONCE, populates the in-memory store ONCE, then loops only the
// save-back or restore call — so allocs/op is the per-turn continuity cost,
// not the one-time load or the prefill. The restore benches reuse a single
// destination session (RestoreKV replaces state + nils agentMemory, built for
// repeated calls); a fresh NewSession per op would measure engine-session
// creation and grow GPU residency, neither of which is the restore path's
// session-own allocation.
//
// Byte-identity of the restored state is pinned in TestSession_SleepWakeByte
// Identity_LiveE2B (capture → hash → Sleep → Wake → recapture → assert
// hash-equal), the precondition any later alloc fix is validated against. The
// benches measure allocations only; they do not assert hashes in the loop.
//
// Run (needs Metal + the cached weights):
//
//	MLX_METALLIB_PATH=$PWD/../dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run '^$' -bench '^BenchmarkSession_.*_LiveE2B$' \
//	    -benchmem -benchtime=5x ./go

//go:build darwin && arm64

package mlx

import (
	"context"
	"runtime"
	"runtime/debug"
	"testing"

	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/agent"
	"dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
)

// Sinks defeat dead-code elimination of the save/restore results.
var (
	sessionContinuityBenchSinkSleep *agent.SleepReport
	sessionContinuityBenchSinkWake  *agent.WakeReport
	sessionContinuityBenchSinkErr   error
	sessionContinuityBenchSinkText  string
)

// sessionContinuityPrompt is a fixed, fact-bearing prompt. Its only role here
// is to populate a non-trivial KV prefix to capture/restore; the content is
// irrelevant to allocation counting, but a stable prompt keeps the prefilled
// token count (and therefore the serialised payload size) constant across runs.
const sessionContinuityPrompt = "The archivist's name is Wren, the vault key is brass, and the codeword is mistral."

// sessionContinuityStoreURIs are the fixed entry/bundle/index URIs the sleep
// benches write under. Reusing one set across ops keeps the content-addressed
// in-memory store deduplicating the identical KV payload (no residency growth
// across b.N) — the OOM guard the 5x benchtime cap backs up.
const (
	sessionContinuityEntryURI  = "mlx://bench/continuity/entry"
	sessionContinuityBundleURI = "mlx://bench/continuity/bundle"
	sessionContinuityIndexURI  = "mlx://bench/continuity/index"
)

// benchContinuityMemoryGuard caps the heap so a runaway load can never march
// the host toward the 143 GB OOM earlier full-model benches hit. The e2b 4bit
// pack sits far under this; it is a guard rail, not a working figure. Returns a
// restore func so the process-wide limit is not left clamped for later benches.
func benchContinuityMemoryGuard() func() {
	prev := debug.SetMemoryLimit(60 << 30)
	return func() { debug.SetMemoryLimit(prev) }
}

// benchContinuityExactAlloc switches the profiler to exact-count mode (rate 1)
// AFTER the one-time model load, so the small per-turn save/restore allocations
// resolve (the default ~512 KB byte-weighted sampling under-resolves them and
// would report a real site as a deceptive "0 samples"). The one-time load above
// stays coarsely sampled. Returns a restore func.
func benchContinuityExactAlloc() func() {
	prev := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	return func() { runtime.MemProfileRate = prev }
}

// benchContinuitySource prefills a fresh source session with the fixed prompt.
// Each save-side bench gets its own source so the captured KV is identical
// across benches; the caller closes it.
func benchContinuitySource(b *testing.B, m *Model) *ModelSession {
	b.Helper()
	src, err := m.NewSession()
	if err != nil {
		b.Fatalf("NewSession(source): %v", err)
	}
	if err := src.Prefill(sessionContinuityPrompt); err != nil {
		_ = src.Close()
		b.Fatalf("Prefill(source): %v", err)
	}
	return src
}

// BenchmarkSession_SleepAgentMemory_LiveE2B measures the per-turn SLEEP cost on
// a real model: capture the retained KV from the GPU, serialise it to State
// blocks, write the bundle manifest + one-entry wake index. This is the heavy
// half of the continuity feature. The source is prefilled once and reused; the
// capture+serialise work (SaveKVBlocksToState) is identical every op. Note that
// SleepAgentMemory reads s.agentMemory (set by the prior op's sleep), so ops
// 2..N auto-populate Parent*URI from op 1 — small string handling on top of the
// invariant heavy work, and exactly the realistic repeated-turn pattern.
func BenchmarkSession_SleepAgentMemory_LiveE2B(b *testing.B) {
	defer benchContinuityMemoryGuard()()
	m := requireLiveE2BModel(b)
	src := benchContinuitySource(b, m)
	defer src.Close()
	store := state.NewInMemoryStore(nil)
	ctx := context.Background()
	opts := agent.SleepOptions{
		EntryURI:  sessionContinuityEntryURI,
		BundleURI: sessionContinuityBundleURI,
		IndexURI:  sessionContinuityIndexURI,
		Title:     "continuity-sleep",
	}

	defer benchContinuityExactAlloc()()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := src.SleepAgentMemory(ctx, store, opts)
		if err != nil {
			b.Fatalf("SleepAgentMemory (op %d): %v", i, err)
		}
		sessionContinuityBenchSinkSleep = report
	}
	b.StopTimer()
	_ = sessionContinuityBenchSinkSleep
}

// BenchmarkSession_WakeAgentMemory_LiveE2B measures the per-turn WAKE cost on a
// real model: plan the wake from the durable index, then restore the KV prefix
// into the session (kv-blocks restore on the cgo engine, which streams the
// blocks back to the GPU). The store is populated ONCE before the timer; the
// loop reuses a single destination session, so the counted allocations are the
// restore path's own (plan + block-source + the restore call), not session
// construction.
func BenchmarkSession_WakeAgentMemory_LiveE2B(b *testing.B) {
	defer benchContinuityMemoryGuard()()
	m := requireLiveE2BModel(b)
	ctx := context.Background()

	// Populate the store once from a throwaway source.
	src := benchContinuitySource(b, m)
	sleep, err := src.SleepAgentMemory(ctx, state.NewInMemoryStore(nil), agent.SleepOptions{
		EntryURI:  sessionContinuityEntryURI,
		BundleURI: sessionContinuityBundleURI,
		IndexURI:  sessionContinuityIndexURI,
		Title:     "continuity-wake-src",
	})
	_ = src.Close()
	if err != nil {
		b.Fatalf("SleepAgentMemory(seed): %v", err)
	}
	// Re-sleep into the store the wake bench will read from, so the bundle +
	// index the loop resolves are resident for the whole run.
	store := state.NewInMemoryStore(nil)
	src2 := benchContinuitySource(b, m)
	sleep, err = src2.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:  sleep.EntryURI,
		BundleURI: sleep.BundleURI,
		IndexURI:  sleep.IndexURI,
		Title:     "continuity-wake-src",
	})
	_ = src2.Close()
	if err != nil {
		b.Fatalf("SleepAgentMemory(seed-2): %v", err)
	}

	dst, err := m.NewSession()
	if err != nil {
		b.Fatalf("NewSession(dest): %v", err)
	}
	defer dst.Close()
	opts := agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI}

	defer benchContinuityExactAlloc()()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := dst.WakeAgentMemory(ctx, store, opts)
		if err != nil {
			b.Fatalf("WakeAgentMemory (op %d): %v", i, err)
		}
		sessionContinuityBenchSinkWake = report
	}
	b.StopTimer()
	_ = sessionContinuityBenchSinkWake
}

// BenchmarkSession_LoadKVFromState_LiveE2B measures the per-turn cost of the
// single-chunk State restore path (LoadKVFromState): load one KV snapshot
// envelope from the store and restore it into the session. Distinct from the
// blocked WakeAgentMemory path above — this is the whole-snapshot variant
// (SaveKVToState writes one chunk, LoadKVFromState reads it back). The store is
// populated once; the loop reuses a single destination session.
func BenchmarkSession_LoadKVFromState_LiveE2B(b *testing.B) {
	defer benchContinuityMemoryGuard()()
	m := requireLiveE2BModel(b)
	ctx := context.Background()

	src := benchContinuitySource(b, m)
	store := state.NewInMemoryStore(nil)
	ref, err := src.SaveKVToState(ctx, store, kv.StateOptions{URI: sessionContinuityEntryURI})
	_ = src.Close()
	if err != nil {
		b.Fatalf("SaveKVToState(seed): %v", err)
	}

	dst, err := m.NewSession()
	if err != nil {
		b.Fatalf("NewSession(dest): %v", err)
	}
	defer dst.Close()

	defer benchContinuityExactAlloc()()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionContinuityBenchSinkErr = dst.LoadKVFromState(ctx, store, ref)
		if sessionContinuityBenchSinkErr != nil {
			b.Fatalf("LoadKVFromState (op %d): %v", i, sessionContinuityBenchSinkErr)
		}
	}
	b.StopTimer()
}

// BenchmarkSession_RestoreBundleFromState_LiveE2B measures the per-turn cost of
// restoring a session from a state-backed *bundle.Bundle (RestoreBundleFromState):
// compatibility check + SnapshotFromState (load the KV chunk via the bundle's
// State ref + verify the embedded hash) + RestoreKV. The bundle is built once
// around a State ref and a real snapshot hash; the loop reuses a single
// destination session.
func BenchmarkSession_RestoreBundleFromState_LiveE2B(b *testing.B) {
	defer benchContinuityMemoryGuard()()
	m := requireLiveE2BModel(b)
	ctx := context.Background()

	// Capture the source KV once: its hash anchors the bundle, and SaveKVToState
	// writes the chunk the bundle's State ref points at.
	src := benchContinuitySource(b, m)
	snapshot, err := src.CaptureKV()
	if err != nil {
		_ = src.Close()
		b.Fatalf("CaptureKV(seed): %v", err)
	}
	kvHash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		_ = src.Close()
		b.Fatalf("HashSnapshot(seed): %v", err)
	}
	store := state.NewInMemoryStore(nil)
	ref, err := src.SaveKVToState(ctx, store, kv.StateOptions{URI: sessionContinuityEntryURI})
	_ = src.Close()
	if err != nil {
		b.Fatalf("SaveKVToState(seed): %v", err)
	}

	info := m.Info()
	b1 := &bundle.Bundle{
		Version: bundle.Version,
		Kind:    bundle.Kind,
		Model: bundle.Model{
			Architecture: info.Architecture,
			NumLayers:    info.NumLayers,
		},
		KVHash: kvHash,
		Refs:   []bundle.Ref{{Kind: bundle.RefState, State: ref}},
	}

	dst, err := m.NewSession()
	if err != nil {
		b.Fatalf("NewSession(dest): %v", err)
	}
	defer dst.Close()

	defer benchContinuityExactAlloc()()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionContinuityBenchSinkErr = dst.RestoreBundleFromState(ctx, b1, store)
		if sessionContinuityBenchSinkErr != nil {
			b.Fatalf("RestoreBundleFromState (op %d): %v", i, sessionContinuityBenchSinkErr)
		}
	}
	b.StopTimer()
}

// BenchmarkSession_GenerateAndSleep_LiveE2B measures the COMPOSITE
// generate-then-sleep turn (GenerateAndSleepAgentMemory): a short greedy
// generation off the prefilled state, then the full sleep-back. ATTRIBUTION:
// the generate half overlaps BenchmarkGenerate_LiveE2B and the sleep half
// overlaps BenchmarkSession_SleepAgentMemory_LiveE2B — read this bench as the
// end-to-end turn cost, and attribute its allocs against those two component
// benches rather than double-counting. The source is reused; each op generates
// a few tokens from the now-longer state and sleeps it (the realistic loop).
func BenchmarkSession_GenerateAndSleep_LiveE2B(b *testing.B) {
	defer benchContinuityMemoryGuard()()
	m := requireLiveE2BModel(b)
	src := benchContinuitySource(b, m)
	defer src.Close()
	store := state.NewInMemoryStore(nil)
	ctx := context.Background()
	opts := agent.SleepOptions{
		EntryURI:  sessionContinuityEntryURI,
		BundleURI: sessionContinuityBundleURI,
		IndexURI:  sessionContinuityIndexURI,
		Title:     "continuity-gen-sleep",
	}

	defer benchContinuityExactAlloc()()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		text, report, err := src.GenerateAndSleepAgentMemory(ctx, store, opts, WithMaxTokens(8), WithTemperature(0))
		if err != nil {
			b.Fatalf("GenerateAndSleepAgentMemory (op %d): %v", i, err)
		}
		sessionContinuityBenchSinkText = text
		sessionContinuityBenchSinkSleep = report
	}
	b.StopTimer()
	_ = sessionContinuityBenchSinkText
	_ = sessionContinuityBenchSinkSleep
}

// TestSession_SleepWakeByteIdentity_LiveE2B is the byte-identity precondition
// the continuity alloc-reduction work is validated against: a real prefilled
// session is captured, hashed, slept to a store, then woken into a FRESH
// session whose recaptured KV must hash-equal the original. Any save/restore
// alloc fix must keep this green — the restored state must be identical, not
// merely "close". Mirrors TestRealE2BPrefillDeterministic's role for the
// prefill-alloc work.
func TestSession_SleepWakeByteIdentity_LiveE2B(t *testing.T) {
	m := requireLiveE2BModel(t)
	ctx := context.Background()

	src, err := m.NewSession()
	if err != nil {
		t.Fatalf("NewSession(source): %v", err)
	}
	defer src.Close()
	if err := src.Prefill(sessionContinuityPrompt); err != nil {
		t.Fatalf("Prefill(source): %v", err)
	}

	// Anchor: the source KV hash before any save/restore.
	srcSnapshot, err := src.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV(source): %v", err)
	}
	srcHash, err := kv.HashSnapshot(srcSnapshot)
	if err != nil {
		t.Fatalf("HashSnapshot(source): %v", err)
	}

	store := state.NewInMemoryStore(nil)
	sleep, err := src.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:  sessionContinuityEntryURI,
		BundleURI: sessionContinuityBundleURI,
		IndexURI:  sessionContinuityIndexURI,
		Title:     "continuity-identity",
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory: %v", err)
	}

	dst, err := m.NewSession()
	if err != nil {
		t.Fatalf("NewSession(dest): %v", err)
	}
	defer dst.Close()
	if _, err := dst.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	}); err != nil {
		t.Fatalf("WakeAgentMemory: %v", err)
	}

	dstSnapshot, err := dst.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV(dest): %v", err)
	}
	dstHash, err := kv.HashSnapshot(dstSnapshot)
	if err != nil {
		t.Fatalf("HashSnapshot(dest): %v", err)
	}

	if srcHash != dstHash {
		t.Fatalf("sleep/wake perturbed KV: source hash %s != restored hash %s", srcHash, dstHash)
	}
	t.Logf("sleep/wake byte-identity held: hash=%s prefixTokens=%d", srcHash, sleep.TokenCount)
}
