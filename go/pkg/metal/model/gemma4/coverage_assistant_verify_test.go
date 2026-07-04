// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the input-guard ladders of the two draft-block verifiers. The
// guards run before any target forward, so they are reachable with a bare
// pair (non-nil Target, never forwarded) and crafted inputs — no model load.
// The verifiers' live/cloned forward bodies and their engine-fault rollback
// branches are exercised (happy path) and respectively real-model-only
// (fault paths) elsewhere; this file pins only the deterministic guards.

// cov4vBarePair returns a pair whose Target is non-nil but never forwarded, so
// the guard checks past `pair.Target == nil` are reachable without a checkpoint.
func cov4vBarePair() *Gemma4AssistantPair {
	return &Gemma4AssistantPair{Target: &Gemma4Model{}}
}

// TestVerifyLive_Guards drives verifyDraftBlockLive's input-guard ladder:
// nil pair, nil target, invalid logits, empty draft block, empty caches.
func TestVerifyLive_Guards(t *testing.T) {
	requireMetalRuntime(t)

	logits := seqArray(0.1, 1, 10)
	defer metal.Free(logits)
	caches := []metal.Cache{nil} // one (nil) entry → len(caches) != 0

	// nil pair.
	var nilPair *Gemma4AssistantPair
	if _, err := nilPair.verifyDraftBlockLive(logits, []int32{1}, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("nil pair should be refused")
	}

	// non-nil pair, nil Target.
	if _, err := (&Gemma4AssistantPair{}).verifyDraftBlockLive(logits, []int32{1}, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("nil target should be refused")
	}

	pair := cov4vBarePair()

	// invalid target logits.
	if _, err := pair.verifyDraftBlockLive(nil, []int32{1}, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("nil target logits should be refused")
	}

	// empty draft block.
	if _, err := pair.verifyDraftBlockLive(logits, nil, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("empty draft block should be refused")
	}

	// empty caches.
	if _, err := pair.verifyDraftBlockLive(logits, []int32{1}, nil, gemma4VerifyDecider{}); err == nil {
		t.Fatal("empty caches should be refused")
	}
}

// TestVerifySampledClone_Guards drives verifyDraftBlockSampledClone's matching
// input-guard ladder.
func TestVerifySampledClone_Guards(t *testing.T) {
	requireMetalRuntime(t)

	logits := seqArray(0.1, 1, 10)
	defer metal.Free(logits)
	caches := []metal.Cache{nil}

	var nilPair *Gemma4AssistantPair
	if _, err := nilPair.verifyDraftBlockSampledClone(logits, []int32{1}, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("nil pair should be refused")
	}
	if _, err := (&Gemma4AssistantPair{}).verifyDraftBlockSampledClone(logits, []int32{1}, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("nil target should be refused")
	}

	pair := cov4vBarePair()
	if _, err := pair.verifyDraftBlockSampledClone(nil, []int32{1}, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("nil target logits should be refused")
	}
	if _, err := pair.verifyDraftBlockSampledClone(logits, nil, caches, gemma4VerifyDecider{}); err == nil {
		t.Fatal("empty draft block should be refused")
	}
	if _, err := pair.verifyDraftBlockSampledClone(logits, []int32{1}, nil, gemma4VerifyDecider{}); err == nil {
		t.Fatal("empty caches should be refused")
	}
}
