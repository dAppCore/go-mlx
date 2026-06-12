// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Speculative trim journal tests — synthetic position-coded K/V rows driven
// straight into the cache types (no model, no weights). Row i carries the
// value float32(i) in every element, so content assertions identify exactly
// which token positions a window holds.

// specTestRow builds a [1,1,1,D] K/V row whose elements all equal pos.
func specTestRow(pos int, d int) *Array {
	values := make([]float32, d)
	for i := range values {
		values[i] = float32(pos)
	}
	return FromValues(values, 1, 1, 1, d)
}

// specTestRows builds a [1,1,n,D] block of rows pos..pos+n-1.
func specTestRows(pos, n, d int) *Array {
	values := make([]float32, n*d)
	for r := 0; r < n; r++ {
		for i := 0; i < d; i++ {
			values[r*d+i] = float32(pos + r)
		}
	}
	return FromValues(values, 1, 1, n, d)
}

// specFeedSingle pushes rows pos..pos+n-1 one token at a time.
func specFeedSingle(t *testing.T, c Cache, pos, n, d int) {
	t.Helper()
	for i := 0; i < n; i++ {
		k := specTestRow(pos+i, d)
		v := specTestRow(pos+i, d)
		outK, outV := c.Update(k, v, 1)
		Free(outK, outV, k, v)
	}
}

// specVisibleRows reads the visible K rows of a cache as per-row leading
// values (row content is constant per row by construction).
func specVisibleRows(t *testing.T, c Cache, d int) []float32 {
	t.Helper()
	state, owned := CacheReadState(c)
	defer Free(owned...)
	if len(state) < 2 || state[0] == nil {
		return nil
	}
	flat := state[0].Floats()
	if len(flat)%d != 0 {
		t.Fatalf("visible K elements = %d, want a multiple of row width %d", len(flat), d)
	}
	rows := make([]float32, 0, len(flat)/d)
	for i := 0; i < len(flat); i += d {
		rows = append(rows, flat[i])
	}
	if visible := c.Len(); visible > 0 && visible < len(rows) {
		rows = rows[:visible]
	}
	return rows
}

func specAssertRows(t *testing.T, got []float32, want ...float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("visible rows = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("visible rows = %v, want %v", got, want)
		}
	}
}

// A rotated sliding window must trim a speculative block exactly: restore the
// pre-verify window, replay only the committed prefix. This is the case stock
// rotating caches refuse (the oldest rows were dropped at append time).
func TestCacheSpeculative_RotatingTrimAcrossRotation_Good(t *testing.T) {
	const d = 2
	c := NewRotatingKVCache(4)
	defer c.Reset()
	specFeedSingle(t, c, 1, 4, d) // window [1 2 3 4], at cap

	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want armed")
	}
	k := specTestRows(5, 3, d) // speculative block [5 6 7]
	v := specTestRows(5, 3, d)
	outK, outV := c.Update(k, v, 3)
	Free(outK, outV, k, v)
	if c.Offset() != 7 {
		t.Fatalf("post-verify Offset() = %d, want 7", c.Offset())
	}

	// Accept 1 of 3: drop 2 → the window must hold [2 3 4 5] exactly, as if
	// token 5 had been decoded autoregressively.
	if !CacheTruncateTo(c, c.Len()-2) {
		t.Fatal("CacheTruncateTo() = false, want journal-backed trim across rotation")
	}
	c.DisarmSpeculativeTrim()
	if c.Offset() != 5 {
		t.Fatalf("post-trim Offset() = %d, want 5", c.Offset())
	}
	specAssertRows(t, specVisibleRows(t, c, d), 2, 3, 4, 5)

	// Reference: a fresh cache fed the same committed tokens one at a time.
	ref := NewRotatingKVCache(4)
	defer ref.Reset()
	specFeedSingle(t, ref, 1, 5, d)
	specAssertRows(t, specVisibleRows(t, ref, d), 2, 3, 4, 5)
	if ref.Offset() != c.Offset() {
		t.Fatalf("trimmed Offset() = %d, want AR reference %d", c.Offset(), ref.Offset())
	}
}

// A fully rejected block must restore the pre-verify window bit-for-bit.
func TestCacheSpeculative_RotatingFullRejectRestores_Good(t *testing.T) {
	const d = 2
	c := NewRotatingKVCache(4)
	defer c.Reset()
	specFeedSingle(t, c, 1, 5, d) // rotated: window [2 3 4 5], offset 5

	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want armed")
	}
	k := specTestRows(6, 2, d)
	v := specTestRows(6, 2, d)
	outK, outV := c.Update(k, v, 2)
	Free(outK, outV, k, v)

	if !CacheTruncateTo(c, c.Len()-2) { // reject both
		t.Fatal("CacheTruncateTo() = false, want full-reject restore")
	}
	c.DisarmSpeculativeTrim()
	if c.Offset() != 5 {
		t.Fatalf("post-restore Offset() = %d, want 5", c.Offset())
	}
	specAssertRows(t, specVisibleRows(t, c, d), 2, 3, 4, 5)
}

// Below the cap the base offset-rewind path serves an armed cache; the
// journal must not get in the way and disarm must be clean.
func TestCacheSpeculative_RotatingBelowCapBasePath_Good(t *testing.T) {
	const d = 2
	c := NewRotatingKVCache(8)
	defer c.Reset()
	specFeedSingle(t, c, 1, 3, d)

	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want armed")
	}
	k := specTestRows(4, 2, d)
	v := specTestRows(4, 2, d)
	outK, outV := c.Update(k, v, 2)
	Free(outK, outV, k, v)

	if !CacheTruncateTo(c, 4) { // accept 1 of 2
		t.Fatal("CacheTruncateTo() = false, want base-path trim below cap")
	}
	c.DisarmSpeculativeTrim()
	if c.Offset() != 4 || c.Len() != 4 {
		t.Fatalf("post-trim Offset()/Len() = %d/%d, want 4/4", c.Offset(), c.Len())
	}
	specAssertRows(t, specVisibleRows(t, c, d), 1, 2, 3, 4)
}

// Without an armed journal a rotated window still refuses to trim — the
// caller's rebuild fallback contract is unchanged.
func TestCacheSpeculative_RotatingUnarmedRotatedRefuses_Bad(t *testing.T) {
	const d = 2
	c := NewRotatingKVCache(4)
	defer c.Reset()
	specFeedSingle(t, c, 1, 6, d) // rotated

	if CacheTruncateTo(c, c.Len()-1) {
		t.Fatal("CacheTruncateTo() = true, want refusal without a journal on a rotated window")
	}
}

// The fixed sliding cache past its cap: a structural overflow update must
// journal and trim back to the committed prefix exactly.
func TestCacheSpeculative_FixedPastCapTrim_Good(t *testing.T) {
	const d = 2
	c := NewFixedKVCache(4)
	defer c.Reset()
	specFeedSingle(t, c, 1, 4, d) // at cap: offset 4 == maxSize

	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want armed")
	}
	k := specTestRows(5, 3, d)
	v := specTestRows(5, 3, d)
	outK, outV := c.Update(k, v, 3)
	Free(outK, outV, k, v)
	if c.Offset() != 7 {
		t.Fatalf("post-verify Offset() = %d, want 7", c.Offset())
	}

	if !CacheTruncateTo(c, c.Len()-2) { // accept 1 of 3
		t.Fatal("CacheTruncateTo() = false, want journal-backed trim past the cap")
	}
	c.DisarmSpeculativeTrim()
	if c.Offset() != 5 {
		t.Fatalf("post-trim Offset() = %d, want 5", c.Offset())
	}
	specAssertRows(t, specVisibleRows(t, c, d), 2, 3, 4, 5)
}

// A linear write that lands exactly on the cap is still a dead-row write; the
// armed counter journal rewinds it (the unarmed path conservatively refuses).
func TestCacheSpeculative_FixedTrimAtExactCap_Good(t *testing.T) {
	const d = 2
	c := NewFixedKVCache(4)
	defer c.Reset()
	specFeedSingle(t, c, 1, 2, d) // offset 2

	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want armed")
	}
	k := specTestRows(3, 2, d) // offset reaches exactly maxSize (4)
	v := specTestRows(3, 2, d)
	outK, outV := c.Update(k, v, 2)
	Free(outK, outV, k, v)
	if c.Offset() != 4 {
		t.Fatalf("post-verify Offset() = %d, want 4", c.Offset())
	}

	if !CacheTruncateTo(c, 3) { // accept 1 of 2
		t.Fatal("CacheTruncateTo() = false, want counter-journal trim at exact cap")
	}
	c.DisarmSpeculativeTrim()
	if c.Offset() != 3 || c.Len() != 3 {
		t.Fatalf("post-trim Offset()/Len() = %d/%d, want 3/3", c.Offset(), c.Len())
	}
	specAssertRows(t, specVisibleRows(t, c, d), 1, 2, 3)
}

// Arming while a pipelined adoption is staged must refuse — growing or
// replacing storage under a staged step is the #73 stale-borrowed-band race.
func TestCacheSpeculative_FixedPendingArmedRefusesArm_Bad(t *testing.T) {
	c := NewFixedKVCache(4)
	defer c.Reset()
	c.ArmPending()
	defer c.DiscardPending()
	if c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = true, want refusal while a pipelined adoption is staged")
	}
}

// A trim that reaches past the journalled update cannot be honoured — the
// caller must fall back rather than trust a partial undo.
func TestCacheSpeculative_TrimBeyondJournalRefuses_Ugly(t *testing.T) {
	const d = 2
	c := NewRotatingKVCache(4)
	defer c.Reset()
	specFeedSingle(t, c, 1, 5, d) // rotated

	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want armed")
	}
	k := specTestRows(6, 2, d)
	v := specTestRows(6, 2, d)
	outK, outV := c.Update(k, v, 2)
	Free(outK, outV, k, v)

	if CacheTruncateTo(c, c.Len()-3) { // drop 3 > journalled 2
		t.Fatal("CacheTruncateTo() = true, want refusal when the trim reaches past the journalled update")
	}
	c.DisarmSpeculativeTrim()
}

// Arming a cache set is all-or-nothing: one unsupported cache disarms the lot
// so the caller takes the clone-verify fallback for the whole round.
func TestCacheSpeculative_CachesArmAllOrNothing_Ugly(t *testing.T) {
	supported := NewKVCache()
	defer supported.Reset()
	unsupported := NewQuantizedKVCache(0, 8, 8)
	defer unsupported.Reset()

	if CachesArmSpeculativeTrim([]Cache{supported, unsupported}) {
		t.Fatal("CachesArmSpeculativeTrim() = true, want refusal when any cache lacks journal support")
	}
	// And a fully supported set arms.
	rotating := NewRotatingKVCache(4)
	defer rotating.Reset()
	fixed := NewFixedKVCache(4)
	defer fixed.Reset()
	caches := []Cache{supported, rotating, fixed}
	if !CachesArmSpeculativeTrim(caches) {
		t.Fatal("CachesArmSpeculativeTrim() = false, want armed for KVCache+Rotating+Fixed")
	}
	CachesDisarmSpeculativeTrim(caches)
}

// The unbounded cache needs no journal — its truncate is already exact.
func TestCacheSpeculative_KVCacheTrimExact_Good(t *testing.T) {
	const d = 2
	c := NewKVCache()
	defer c.Reset()
	specFeedSingle(t, c, 1, 3, d)
	if !c.ArmSpeculativeTrim() {
		t.Fatal("ArmSpeculativeTrim() = false, want trivially armed")
	}
	k := specTestRows(4, 3, d)
	v := specTestRows(4, 3, d)
	outK, outV := c.Update(k, v, 3)
	Free(outK, outV, k, v)
	if !CacheTruncateTo(c, 4) {
		t.Fatal("CacheTruncateTo() = false, want exact offset rewind")
	}
	c.DisarmSpeculativeTrim()
	specAssertRows(t, specVisibleRows(t, c, d), 1, 2, 3, 4)
}
