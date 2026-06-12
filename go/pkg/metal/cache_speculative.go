// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// cache_speculative.go gives the K/V caches an EXACT speculative rollback: a
// one-update-deep journal armed around a batched MTP verify forward, so the
// rejected suffix of a draft block can be trimmed even after the sliding
// window has rotated (the case stock rotating caches refuse — they drop the
// oldest tokens at append time, so a tail trim cannot restore them).
//
// The shape follows the MTPLX Gemma4RollbackRotatingKVCache idea (Apache-2.0
// reference): save the pre-update storage handles plus a copy of the update
// inputs, and trim = restore the pre-update state then replay only the
// committed prefix through the normal update path. Rebuilt here for go-mlx's
// own cache types and ownership rules rather than ported line-by-line.
//
//	if metal.CachesArmSpeculativeTrim(caches) {           // before the verify forward
//	    logits, hidden := target.ForwardAllTokenLogitsAndHidden(block, caches)
//	    ...decide accepted prefix k...
//	    metal.CachesTruncateTo(caches, cache.Len()-(len(block)-k)) // exact, even rotated
//	    metal.CachesDisarmSpeculativeTrim(caches)          // commit: drop the journal
//	}
package metal

// SpeculativeTrimCapable marks caches that can arm a one-update journal so a
// speculative verify block can be rolled back exactly. Arming may refuse —
// e.g. a FixedKVCache with a pipelined adoption staged (the #73 race class) —
// in which case the caller must fall back to the clone-verify regime.
type SpeculativeTrimCapable interface {
	// ArmSpeculativeTrim journals the NEXT Update so TruncateTo can undo its
	// rejected suffix exactly. Returns false when the cache cannot guarantee
	// the rollback right now.
	ArmSpeculativeTrim() bool
	// DisarmSpeculativeTrim commits the speculative state: the journal is
	// dropped and any retained pre-update storage is released.
	DisarmSpeculativeTrim()
}

// CachesArmSpeculativeTrim arms every cache, all-or-nothing: if any cache
// cannot arm (unsupported type, or a refused arm), every armed cache is
// disarmed again and false is returned so the caller takes the clone-verify
// fallback for the whole round.
//
//	live := metal.CachesArmSpeculativeTrim(caches)
func CachesArmSpeculativeTrim(caches []Cache) bool {
	for i, c := range caches {
		if !cacheArmSpeculativeTrim(c) {
			CachesDisarmSpeculativeTrim(caches[:i])
			return false
		}
	}
	return true
}

// CachesDisarmSpeculativeTrim commits every cache's speculative state (drops
// journals, frees retained pre-update storage). Safe on unarmed caches.
func CachesDisarmSpeculativeTrim(caches []Cache) {
	for _, c := range caches {
		if t, ok := c.(SpeculativeTrimCapable); ok {
			t.DisarmSpeculativeTrim()
		}
	}
}

func cacheArmSpeculativeTrim(c Cache) bool {
	if c == nil {
		return false
	}
	if t, ok := c.(SpeculativeTrimCapable); ok {
		return t.ArmSpeculativeTrim()
	}
	return false
}

// specUpdateJournal is the one-deep undo record for an armed cache. content
// records whether pre-update storage handles were retained (the structural
// update paths — rotation drop+append, fixed-cache overflow); counter-only
// journals cover linear writes whose overwritten rows were dead storage.
type specUpdateJournal struct {
	content    bool
	prevKeys   *Array
	prevValues *Array
	prevOffset int
	prevIdx    int // RotatingKVCache idx / FixedKVCache length
	prevBand   int // FixedKVCache bandCap at journal time (0 = unused)
	updKeys    *Array
	updValues  *Array
	updLen     int
}

// free releases everything the journal retained.
func (j *specUpdateJournal) free() {
	if j == nil {
		return
	}
	Free(j.prevKeys, j.prevValues, j.updKeys, j.updValues)
	j.prevKeys, j.prevValues, j.updKeys, j.updValues = nil, nil, nil, nil
}

// detachPrev releases ownership of the retained pre-update storage without
// freeing it (used once a restore has handed the handles back to the cache,
// or when the journal never superseded the live storage).
func (j *specUpdateJournal) detachPrev() {
	j.prevKeys, j.prevValues = nil, nil
}

// journalCopyPair clones the update inputs so a later replay does not depend
// on caller-owned handles. The copies are lazy graph nodes; they ride the
// verify forward's Eval (or are evaluated at restore time).
func journalCopyPair(k, v *Array) (*Array, *Array) {
	var ck, cv *Array
	if k != nil && k.Valid() {
		ck = Copy(k)
	}
	if v != nil && v.Valid() {
		cv = Copy(v)
	}
	return ck, cv
}

// --- RotatingKVCache -------------------------------------------------------

// ArmSpeculativeTrim journals the next Update for exact rollback. Always
// succeeds for a rotating cache. See SpeculativeTrimCapable.
func (c *RotatingKVCache) ArmSpeculativeTrim() bool {
	if c == nil {
		return false
	}
	c.specArmed = true
	return true
}

// DisarmSpeculativeTrim commits the speculative state and drops the journal.
func (c *RotatingKVCache) DisarmSpeculativeTrim() {
	if c == nil {
		return
	}
	c.specArmed = false
	c.specDropJournal()
}

// specDropJournal frees the journal without touching live storage: a journal
// whose retained handles are still the cache's storage (an update path that
// never swapped, e.g. the rank-degenerate pass-through) releases ownership
// instead of freeing.
func (c *RotatingKVCache) specDropJournal() {
	if c.specJournal == nil {
		return
	}
	if c.specJournal.prevKeys == c.keys {
		c.specJournal.detachPrev()
	}
	c.specJournal.free()
	c.specJournal = nil
}

// specJournalUpdate records the pre-update state for an armed rotating cache.
// Every rotating update is journalled as a content journal: the storage
// handles are retained (suppressing in-place donation for that one update —
// the structural drop+append paths produce fresh buffers anyway) so trim is a
// pure handle swap plus a committed-prefix replay.
func (c *RotatingKVCache) specJournalUpdate(k, v *Array, seqLen int) {
	if !c.specArmed || c.specReplaying {
		return
	}
	c.specDropJournal()
	updK, updV := journalCopyPair(k, v)
	c.specJournal = &specUpdateJournal{
		content:    true,
		prevKeys:   c.keys,
		prevValues: c.values,
		prevOffset: c.offset,
		prevIdx:    c.idx,
		updKeys:    updK,
		updValues:  updV,
		updLen:     seqLen,
	}
}

// specReleaseStorage frees superseded storage handles unless the live journal
// retained them (in which case the journal now owns them).
func (c *RotatingKVCache) specReleaseStorage(oldK, oldV *Array) {
	if c.specJournal != nil && c.specJournal.content && c.specJournal.prevKeys == oldK {
		return // journal owns the handles
	}
	Free(oldK, oldV)
}

// specTrimTo undoes the journalled update down to n visible tokens: restore
// the pre-update storage and counters, then replay only the committed prefix
// of the update through the normal update path. Returns false when no journal
// covers the request.
func (c *RotatingKVCache) specTrimTo(n int) bool {
	j := c.specJournal
	if j == nil || !j.content {
		return false
	}
	dropped := c.Len() - n
	if dropped <= 0 || dropped > j.updLen {
		return false
	}
	committed := j.updLen - dropped
	// Restore the pre-update state. The current storage was produced by the
	// journalled update; the journal holds the pre-update handles.
	if c.keys != j.prevKeys {
		Free(c.keys, c.values)
	}
	c.keys = j.prevKeys
	c.values = j.prevValues
	c.offset = j.prevOffset
	c.idx = j.prevIdx
	j.detachPrev()
	// Replay the committed prefix with the journal quiesced so the replay does
	// not re-journal (and a nested trim cannot fire).
	if committed > 0 && j.updKeys != nil && j.updValues != nil {
		kShape := j.updKeys.Shape()
		vShape := j.updValues.Shape()
		if len(kShape) >= 4 && len(vShape) >= 4 {
			replayK := Slice4(j.updKeys, 0, 0, 0, 0, kShape[0], kShape[1], int32(committed), kShape[3])
			replayV := Slice4(j.updValues, 0, 0, 0, 0, vShape[0], vShape[1], int32(committed), vShape[3])
			c.specReplaying = true
			outK, outV := c.Update(replayK, replayV, committed)
			c.specReplaying = false
			Free(outK, outV, replayK, replayV)
		} else {
			// Rank-degenerate update (pass-through path) — counters only.
			c.offset += committed
			c.idx = min(c.idx+committed, c.maxSize)
		}
	}
	j.free()
	c.specJournal = nil
	return true
}

// --- FixedKVCache ----------------------------------------------------------

// ArmSpeculativeTrim journals the next Update for exact rollback. Refuses
// while a pipelined adoption is staged (pendingArmed): growing/replacing
// storage under a staged step is the #73 stale-borrowed-band race class, and
// a journalled restore would compound it.
func (c *FixedKVCache) ArmSpeculativeTrim() bool {
	if c == nil || c.pendingArmed {
		return false
	}
	c.specArmed = true
	return true
}

// DisarmSpeculativeTrim commits the speculative state and drops the journal.
func (c *FixedKVCache) DisarmSpeculativeTrim() {
	if c == nil {
		return
	}
	c.specArmed = false
	c.specDropJournal()
}

// specDropJournal frees the journal without touching live storage (see the
// RotatingKVCache counterpart).
func (c *FixedKVCache) specDropJournal() {
	if c.specJournal == nil {
		return
	}
	if c.specJournal.prevKeys == c.keys {
		c.specJournal.detachPrev()
	}
	c.specJournal.free()
	c.specJournal = nil
}

// specJournalCounters records a counter-only journal: the update wrote into
// dead rows past the committed length (the linear pre-cap regime), so a trim
// is a counter rewind — no content restore needed.
func (c *FixedKVCache) specJournalCounters(seqLen int) {
	if !c.specArmed {
		return
	}
	c.specDropJournal()
	c.specJournal = &specUpdateJournal{
		prevOffset: c.offset,
		prevIdx:    c.length,
		prevBand:   c.bandCap,
		updLen:     seqLen,
	}
}

// specJournalContent records a full content journal before a structural
// overflow update (sliding rotate / tail rebuild): the pre-update storage
// handles are retained and the update inputs copied so a trim can restore
// and replay exactly.
func (c *FixedKVCache) specJournalContent(k, v *Array, seqLen int) {
	if !c.specArmed {
		return
	}
	c.specDropJournal()
	updK, updV := journalCopyPair(k, v)
	c.specJournal = &specUpdateJournal{
		content:    true,
		prevKeys:   c.keys,
		prevValues: c.values,
		prevOffset: c.offset,
		prevIdx:    c.length,
		prevBand:   c.bandCap,
		updKeys:    updK,
		updValues:  updV,
		updLen:     seqLen,
	}
}

// specReleaseStorage frees superseded storage handles unless the live journal
// retained them.
func (c *FixedKVCache) specReleaseStorage(oldK, oldV *Array) {
	if c.specJournal != nil && c.specJournal.content && c.specJournal.prevKeys == oldK {
		return // journal owns the handles
	}
	Free(oldK, oldV)
}

// specTrimTo undoes the journalled update down to n visible tokens. Counter
// journals rewind the offset/length (the committed rows are already in
// storage); content journals restore the pre-update storage then replay the
// committed prefix through the normal Update path.
func (c *FixedKVCache) specTrimTo(n int) bool {
	j := c.specJournal
	if j == nil {
		return false
	}
	dropped := c.Len() - n
	if dropped <= 0 || dropped > j.updLen {
		return false
	}
	committed := j.updLen - dropped
	if !j.content {
		// Linear write into dead rows: rewind the counters, keep the committed
		// rows visible. bandCap may have grown during the update — growth
		// carried the committed content, so it stays.
		c.offset = j.prevOffset + committed
		c.length = min(c.offset, c.maxSize)
		j.free()
		c.specJournal = nil
		return true
	}
	if c.keys != j.prevKeys {
		Free(c.keys, c.values)
	}
	c.keys = j.prevKeys
	c.values = j.prevValues
	c.offset = j.prevOffset
	c.length = j.prevIdx
	c.bandCap = j.prevBand
	c.shapeCached = false
	j.detachPrev()
	if committed > 0 && j.updKeys != nil && j.updValues != nil {
		kShape := j.updKeys.Shape()
		vShape := j.updValues.Shape()
		if len(kShape) >= 4 && len(vShape) >= 4 {
			replayK := Slice4(j.updKeys, 0, 0, 0, 0, kShape[0], kShape[1], int32(committed), kShape[3])
			replayV := Slice4(j.updValues, 0, 0, 0, 0, vShape[0], vShape[1], int32(committed), vShape[3])
			armed := c.specArmed
			c.specArmed = false // the replay must not re-journal
			outK, outV := c.Update(replayK, replayV, committed)
			c.specArmed = armed
			Free(outK, outV, replayK, replayV)
		} else {
			c.offset += committed
			c.length = min(c.offset, c.maxSize)
		}
	}
	j.free()
	c.specJournal = nil
	return true
}

// --- KVCache ---------------------------------------------------------------

// ArmSpeculativeTrim reports the unbounded cache as trim-capable without a
// journal: its TruncateTo is already exact for any n (the buffer only grows;
// rows past the rewound offset are dead storage the next Update overwrites).
func (c *KVCache) ArmSpeculativeTrim() bool { return c != nil }

// DisarmSpeculativeTrim is a no-op — KVCache keeps no journal.
func (c *KVCache) DisarmSpeculativeTrim() {}
