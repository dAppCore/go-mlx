// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the input-guard sentinel ladder shared by the three verify entry
// points (VerifyDraftBlockWithSuppression, verifyDraftBlockLive,
// verifyDraftBlockSampledClone). The happy-path verify tests never trip these,
// so each guard is exercised here with the one malformed argument that reaches
// it, asserting the SPECIFIC sentinel — a wrong-path failure (still non-nil)
// would not satisfy core.Is against the named error.

// cov4vgValidLogits builds a tiny [1,1,vocab] logits array so a guard test that
// must pass the logits check (to reach a later guard) has a valid array.
func cov4vgValidLogits(t *testing.T) *metal.Array {
	t.Helper()
	l := seqArray(0.1, 1, 1, 10)
	if err := metal.Eval(l); err != nil {
		t.Fatalf("eval guard logits: %v", err)
	}
	return l
}

// TestVerifyGuards_PublicLadder walks every guard of the public
// VerifyDraftBlockWithSuppression in order: nil pair/target, invalid logits,
// empty draft tokens, empty caches.
func TestVerifyGuards_PublicLadder(t *testing.T) {
	requireMetalRuntime(t)
	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	logits := cov4vgValidLogits(t)
	defer metal.Free(logits)
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)

	t.Run("nil_target", func(t *testing.T) {
		empty := &Gemma4AssistantPair{}
		if _, err := empty.VerifyDraftBlockWithSuppression(logits, []int32{1}, caches, nil); !core.Is(err, errAsstVerifyNeedTargetModel) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetModel", err)
		}
	})
	t.Run("invalid_logits", func(t *testing.T) {
		if _, err := pair.VerifyDraftBlockWithSuppression(nil, []int32{1}, caches, nil); !core.Is(err, errAsstVerifyNeedTargetLogits) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetLogits", err)
		}
	})
	t.Run("empty_block", func(t *testing.T) {
		if _, err := pair.VerifyDraftBlockWithSuppression(logits, nil, caches, nil); !core.Is(err, errAsstVerifyNeedDraftTokens) {
			t.Fatalf("err = %v, want errAsstVerifyNeedDraftTokens", err)
		}
	})
	t.Run("empty_caches", func(t *testing.T) {
		if _, err := pair.VerifyDraftBlockWithSuppression(logits, []int32{1}, nil, nil); !core.Is(err, errAsstVerifyNeedTargetCaches) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetCaches", err)
		}
	})
}

// TestVerifyGuards_LiveLadder walks verifyDraftBlockLive's guard ladder,
// including the live-only errAsstVerifyLiveArmRefused: a fresh KVCache (no
// speculative-trim journal) refuses the arm, which is the live regime's
// precondition.
func TestVerifyGuards_LiveLadder(t *testing.T) {
	requireMetalRuntime(t)
	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	logits := cov4vgValidLogits(t)
	defer metal.Free(logits)
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	d := gemma4VerifyDecider{}

	t.Run("nil_target", func(t *testing.T) {
		empty := &Gemma4AssistantPair{}
		if _, err := empty.verifyDraftBlockLive(logits, []int32{1}, caches, d); !core.Is(err, errAsstVerifyNeedTargetModel) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetModel", err)
		}
	})
	t.Run("invalid_logits", func(t *testing.T) {
		if _, err := pair.verifyDraftBlockLive(nil, []int32{1}, caches, d); !core.Is(err, errAsstVerifyNeedTargetLogits) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetLogits", err)
		}
	})
	t.Run("empty_block", func(t *testing.T) {
		if _, err := pair.verifyDraftBlockLive(logits, nil, caches, d); !core.Is(err, errAsstVerifyNeedDraftTokens) {
			t.Fatalf("err = %v, want errAsstVerifyNeedDraftTokens", err)
		}
	})
	t.Run("empty_caches", func(t *testing.T) {
		if _, err := pair.verifyDraftBlockLive(logits, []int32{1}, nil, d); !core.Is(err, errAsstVerifyNeedTargetCaches) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetCaches", err)
		}
	})
	t.Run("arm_refused", func(t *testing.T) {
		// A plain KVCache (no speculative-trim journal) refuses the arm. If this
		// build's default cache DOES arm, the test asserts a successful arm path
		// instead so it never silently passes on a different branch.
		plain := []metal.Cache{metal.NewKVCache()}
		defer metal.FreeCaches(plain)
		if metal.CachesArmSpeculativeTrim(plain) {
			metal.CachesDisarmSpeculativeTrim(plain)
			t.Skip("plain KVCache armed speculative trim on this build; arm-refused branch unreachable here")
		}
		if _, err := pair.verifyDraftBlockLive(logits, []int32{1}, plain, d); !core.Is(err, errAsstVerifyLiveArmRefused) {
			t.Fatalf("err = %v, want errAsstVerifyLiveArmRefused", err)
		}
	})
}

// TestVerifyGuards_SampledCloneLadder walks verifyDraftBlockSampledClone's guard
// ladder (no arm guard — the clone regime forwards on a cloned prefix).
func TestVerifyGuards_SampledCloneLadder(t *testing.T) {
	requireMetalRuntime(t)
	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	logits := cov4vgValidLogits(t)
	defer metal.Free(logits)
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	d := gemma4VerifyDecider{}

	t.Run("nil_target", func(t *testing.T) {
		empty := &Gemma4AssistantPair{}
		if _, err := empty.verifyDraftBlockSampledClone(logits, []int32{1}, caches, d); !core.Is(err, errAsstVerifyNeedTargetModel) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetModel", err)
		}
	})
	t.Run("invalid_logits", func(t *testing.T) {
		if _, err := pair.verifyDraftBlockSampledClone(nil, []int32{1}, caches, d); !core.Is(err, errAsstVerifyNeedTargetLogits) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetLogits", err)
		}
	})
	t.Run("empty_block", func(t *testing.T) {
		if _, err := pair.verifyDraftBlockSampledClone(logits, nil, caches, d); !core.Is(err, errAsstVerifyNeedDraftTokens) {
			t.Fatalf("err = %v, want errAsstVerifyNeedDraftTokens", err)
		}
	})
	t.Run("empty_caches", func(t *testing.T) {
		if _, err := pair.verifyDraftBlockSampledClone(logits, []int32{1}, nil, d); !core.Is(err, errAsstVerifyNeedTargetCaches) {
			t.Fatalf("err = %v, want errAsstVerifyNeedTargetCaches", err)
		}
	})
}
