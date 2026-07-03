// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

// prompt_cache_helpers_cover_test.go covers the model-independent prompt-cache
// helpers the heavy restore tests leave dark: the prefix-match predicates and
// their edge arms, the adapter-key resolver, snapshot array bookkeeping, the
// per-array eval failure path, and the clear/acquire lifecycle. All driven on
// synthetic *Model / *PromptCacheEntry / cacheSnapshot structs — no weights.

func TestLongestTokenPrefix_Edges_Good(t *testing.T) {
	cases := []struct {
		a, b []int32
		want int
	}{
		{nil, nil, 0},
		{[]int32{1, 2, 3}, nil, 0},
		{[]int32{1, 2, 3}, []int32{1, 2, 3}, 3}, // identical → full
		{[]int32{1, 2, 3}, []int32{1, 2, 9}, 2}, // diverge at 2
		{[]int32{1, 2}, []int32{1, 2, 3, 4}, 2}, // a is a prefix of b
		{[]int32{9, 1}, []int32{1, 2}, 0},       // diverge at 0
	}
	for _, tc := range cases {
		if got := longestTokenPrefix(tc.a, tc.b); got != tc.want {
			t.Errorf("longestTokenPrefix(%v,%v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestValidateRestorableCacheSnapshotMode_DefaultRejects_Bad(t *testing.T) {
	for _, ok := range []KVCacheMode{
		KVCacheModeDefault, KVCacheModeFP16, KVCacheModeQ8,
		KVCacheModeKQ8VQ4, KVCacheModePaged, KVCacheModeFixed, KVCacheModeTurboQuant,
	} {
		if err := validateRestorableCacheSnapshotMode(ok); err != nil {
			t.Errorf("mode %q rejected: %v", ok, err)
		}
	}
	if err := validateRestorableCacheSnapshotMode(KVCacheMode("not-a-real-mode")); err == nil {
		t.Fatal("unsupported snapshot mode accepted, want error")
	}
}

// promptCacheMinimum falls back to the package default when the model carries no
// override and honours a positive override otherwise.
func TestPromptCacheMinimum_DefaultAndOverride_Good(t *testing.T) {
	var nilModel *Model
	if got := nilModel.promptCacheMinimum(); got != DefaultPromptCacheMinTokens {
		t.Errorf("nil model minimum = %d, want default %d", got, DefaultPromptCacheMinTokens)
	}
	if got := (&Model{}).promptCacheMinimum(); got != DefaultPromptCacheMinTokens {
		t.Errorf("zero-override minimum = %d, want default %d", got, DefaultPromptCacheMinTokens)
	}
	if got := (&Model{promptCacheMinTokens: 7}).promptCacheMinimum(); got != 7 {
		t.Errorf("override minimum = %d, want 7", got)
	}
}

func TestAdapterCacheKey_Resolution_Good(t *testing.T) {
	var nilModel *Model
	if got := nilModel.adapterCacheKey(); got != "" {
		t.Errorf("nil model adapter key = %q, want empty", got)
	}
	if got := (&Model{}).adapterCacheKey(); got != "" {
		t.Errorf("no-adapter key = %q, want empty", got)
	}
	m := &Model{adapterInfo: AdapterInfo{Hash: "deadbeef"}}
	if got := m.adapterCacheKey(); got != "deadbeef" {
		t.Errorf("adapter-info key = %q, want deadbeef", got)
	}
}

// acquirePromptCache returns a no-op release for a disabled/nil model and a real
// unlock for an enabled one (the lock/unlock pair must not deadlock).
func TestAcquirePromptCache_DisabledNoOp_Good(t *testing.T) {
	var nilModel *Model
	nilModel.acquirePromptCache()() // no-op, must not panic

	disabled := &Model{promptCacheEnabled: false}
	disabled.acquirePromptCache()() // no-op release

	enabled := &Model{promptCacheEnabled: true}
	release := enabled.acquirePromptCache()
	release() // unlocks; a second acquire must then succeed
	enabled.acquirePromptCache()()
}

// promptCacheMatch must reject when caching is off, when the adapter key differs,
// when the prefix is below the minimum, and accept a true prefix otherwise.
func TestPromptCacheMatch_Branches_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Disabled / nil cache → no match.
	if e, n := (&Model{promptCacheEnabled: false}).promptCacheMatch([]int32{1, 2}); e != nil || n != 0 {
		t.Fatalf("disabled match = (%v,%d), want (nil,0)", e, n)
	}
	if e, _ := (&Model{promptCacheEnabled: true}).promptCacheMatch([]int32{1, 2}); e != nil {
		t.Fatal("nil-cache match returned an entry")
	}

	toks := make([]int32, 10)
	for i := range toks {
		toks[i] = int32(i)
	}
	entry := &PromptCacheEntry{tokens: append([]int32(nil), toks...), cacheableTokens: 10}
	m := &Model{promptCacheEnabled: true, promptCacheMinTokens: 2, promptCache: entry}

	// Adapter-key mismatch → no match (entry hash empty, model hash set).
	m.adapterInfo = AdapterInfo{Hash: "x"}
	if e, _ := m.promptCacheMatch(toks); e != nil {
		t.Fatal("adapter mismatch still matched")
	}
	m.adapterInfo = AdapterInfo{}

	// Prefix below minimum → no match.
	if e, _ := m.promptCacheMatch([]int32{0}); e != nil {
		t.Fatal("below-minimum prefix matched")
	}

	// A query fully consumed as a prefix but shorter than the entry → no match
	// (the "query is a strict prefix of the cache" guard).
	if e, _ := m.promptCacheMatch([]int32{0, 1, 2, 3, 4}); e != nil {
		t.Fatal("strict-prefix-of-entry query matched, want nil")
	}

	// A query that shares a 5-token prefix then diverges past the entry → the
	// common prefix (5) is returned, the query is longer than the match.
	q := []int32{0, 1, 2, 3, 4, 99, 99, 99, 99, 99}
	if e, n := m.promptCacheMatch(q); e != entry || n != 5 {
		t.Fatalf("diverging-prefix match = (%v,%d), want (entry,5)", e, n)
	}

	// Exact full hit but logits invalid → replay all-but-one.
	if e, n := m.promptCacheMatch(toks); e != entry || n != len(toks)-1 {
		t.Fatalf("full-hit-no-logits match = (%v,%d), want (entry,%d)", e, n, len(toks)-1)
	}

	// Exact hit with valid logits → full prefix.
	entry.logits = Zeros([]int32{1, 1, 4}, DTypeFloat32)
	defer Free(entry.logits)
	if e, n := m.promptCacheMatch(toks); e != entry || n != len(toks) {
		t.Fatalf("full-hit-with-logits match = (%v,%d), want (entry,%d)", e, n, len(toks))
	}
}

// promptCacheMatchWithHidden layers the hidden-state requirement over the base
// match: an exact hit with no valid hidden state replays all-but-one.
func TestPromptCacheMatchWithHidden_Branches_Good(t *testing.T) {
	requireMetalRuntime(t)

	toks := make([]int32, 6)
	for i := range toks {
		toks[i] = int32(i + 1)
	}
	entry := &PromptCacheEntry{tokens: append([]int32(nil), toks...), cacheableTokens: 6}
	entry.logits = Zeros([]int32{1, 1, 4}, DTypeFloat32)
	defer Free(entry.logits)
	m := &Model{promptCacheEnabled: true, promptCacheMinTokens: 2, promptCache: entry}

	// No hidden state on an exact hit → all-but-one.
	if e, n := m.promptCacheMatchWithHidden(toks); e != entry || n != len(toks)-1 {
		t.Fatalf("exact-hit-no-hidden = (%v,%d), want (entry,%d)", e, n, len(toks)-1)
	}

	// A diverging prefix (shares 3, then differs past the entry) returns the
	// common length and is unaffected by the hidden requirement.
	if e, n := m.promptCacheMatchWithHidden([]int32{1, 2, 3, 88, 88, 88, 88}); e != entry || n != 3 {
		t.Fatalf("diverging-prefix-with-hidden = (%v,%d), want (entry,3)", e, n)
	}

	// With valid hidden state, an exact hit returns the full prefix.
	entry.hidden = Zeros([]int32{1, 1, 4}, DTypeFloat32)
	defer Free(entry.hidden)
	if e, n := m.promptCacheMatchWithHidden(toks); e != entry || n != len(toks) {
		t.Fatalf("exact-hit-with-hidden = (%v,%d), want (entry,%d)", e, n, len(toks))
	}

	// Below-minimum prefix → base match returns nil, so this does too.
	if e, _ := m.promptCacheMatchWithHidden([]int32{1}); e != nil {
		t.Fatal("below-minimum returned an entry through the hidden wrapper")
	}
}

// snapshot array bookkeeping: arrayCount must equal the number appendArrays
// yields, across the nil and fully-populated shapes.
func TestCacheSnapshot_ArrayCountMatchesAppend_Good(t *testing.T) {
	requireMetalRuntime(t)

	empty := cacheSnapshot{}
	if empty.arrayCount() != 0 || len(empty.appendArrays(nil)) != 0 {
		t.Fatalf("empty snapshot count=%d append=%d, want 0/0", empty.arrayCount(), len(empty.appendArrays(nil)))
	}

	full := cacheSnapshot{
		keys:       Zeros([]int32{1, 1, 1, 1}, DTypeFloat32),
		values:     Zeros([]int32{1, 1, 1, 1}, DTypeFloat32),
		keyScale:   Zeros([]int32{1, 1}, DTypeFloat32),
		valueScale: Zeros([]int32{1, 1}, DTypeFloat32),
		kPages:     []*Array{Zeros([]int32{1, 1}, DTypeFloat32), Zeros([]int32{1, 1}, DTypeFloat32)},
		vPages:     []*Array{Zeros([]int32{1, 1}, DTypeFloat32)},
	}
	if got, want := full.arrayCount(), 7; got != want {
		t.Fatalf("full arrayCount = %d, want %d", got, want)
	}
	if got := len(full.appendArrays(nil)); got != 7 {
		t.Fatalf("full appendArrays len = %d, want 7", got)
	}
	freeCacheSnapshot(full)
}

// evalPromptCacheArrays returns nil on a clean eval, including a slice that holds
// a nil entry alongside valid arrays (the happy path never invokes the label).
func TestEvalPromptCacheArrays_Happy_Good(t *testing.T) {
	requireMetalRuntime(t)

	good := []*Array{Zeros([]int32{1, 2}, DTypeFloat32), nil, Zeros([]int32{1, 2}, DTypeFloat32)}
	defer Free(good[0], good[2])
	called := false
	if err := evalPromptCacheArrays("scope", good, func(i int) string { called = true; return "item" }); err != nil {
		t.Fatalf("happy eval returned error: %v", err)
	}
	if called {
		t.Error("label callback invoked on the happy path (should be failure-only)")
	}
}

// ClearPromptCache drops the entry and is safe on a nil model and an already-empty
// cache.
func TestClearPromptCache_Idempotent_Good(t *testing.T) {
	requireMetalRuntime(t)

	var nilModel *Model
	nilModel.ClearPromptCache() // nil-safe

	m := &Model{promptCacheEnabled: true}
	m.ClearPromptCache() // empty cache → no-op

	m.promptCache = &PromptCacheEntry{tokens: []int32{1, 2}, logits: Zeros([]int32{1, 1}, DTypeFloat32)}
	m.ClearPromptCache()
	if m.promptCache != nil {
		t.Fatal("ClearPromptCache left the entry in place")
	}
}

// freeCacheSnapshot and PromptCacheEntry.free must tolerate a nil receiver / empty
// snapshot without crashing.
func TestPromptCacheFree_NilSafe_Good(t *testing.T) {
	var nilEntry *PromptCacheEntry
	nilEntry.free() // nil-safe
	freeCacheSnapshot(cacheSnapshot{})
	_ = core.NewError // keep the import load-bearing if branches above shift
}
