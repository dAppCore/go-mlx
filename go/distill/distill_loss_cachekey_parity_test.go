// SPDX-Licence-Identifier: EUPL-1.2

// Byte-parity guard for DistillBatchCacheKey, which now delegates its
// emission to the shared dappco.re/go/inference/distill engine
// (distillinf.BatchCacheKey — a byte-identical port of what this file
// hand-rolled before the delegation). The teacher-logit cache is keyed on
// the SHA256 of the emitted JSON, so any drift between the shared emitter
// and core.JSONMarshal (encoding/json) would silently invalidate every
// cached teacher entry. A parse-back round-trip would NOT catch format
// drift that still changes the hash, so this diffs DistillBatchCacheKey's
// hash against SHA256(core.JSONMarshal(...)) directly over adversarial
// inputs, and pins the NaN/Inf fallback contract (json.Marshal errors
// there, so the shared emitter must signal the same so the caller takes
// the identical Sprintf-based key) — all through the public API, since the
// hand-rolled byte emitter this test used to drive directly no longer has
// a local copy to call.
//
// Run:    go test -run='BatchCacheKeyParity' ./distill/

package distill

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// batchCacheKeyPayload mirrors the exact anonymous struct
// distillinf.BatchCacheKey marshals — same field order, same json tags —
// so core.JSONMarshal here is the same oracle the production path used.
type batchCacheKeyPayload struct {
	Tokens  [][]int     `json:"tokens"`
	Targets [][]int     `json:"targets"`
	Mask    [][]float32 `json:"mask"`
}

type batchCacheKeyFixture struct {
	name    string
	tokens  [][]int
	targets [][]int
	mask    [][]float32
}

// batchCacheKeyParityFixtures are the adversarial inputs the byte-parity
// test (DistillBatchCacheKey's hash == SHA256(core.JSONMarshal(...))) runs
// against. None contain NaN/Inf — those have their own fallback test.
func batchCacheKeyParityFixtures() []batchCacheKeyFixture {
	return []batchCacheKeyFixture{
		{
			// The zero batch — all three nil. encoding/json emits null for
			// each (nil slice). _Ugly already asserts a stable hash here.
			name: "all_nil_zero_batch",
		},
		{
			// nil-vs-empty at the OUTER level: non-nil empty slice -> [].
			name:    "empty_outer",
			tokens:  [][]int{},
			targets: [][]int{},
			mask:    [][]float32{},
		},
		{
			// nil-vs-empty at the INNER level: [][]int{nil} -> [null],
			// [][]int{{}} -> [[]]. Mask: [][]float32{nil} -> [null].
			name:    "inner_nil_and_empty",
			tokens:  [][]int{nil},
			targets: [][]int{{}},
			mask:    [][]float32{nil},
		},
		{
			name:    "inner_empty_float",
			tokens:  [][]int{{}},
			targets: [][]int{nil},
			mask:    [][]float32{{}},
		},
		{
			// The ordinary case — small batch, mask all 1.0.
			name:    "ordinary_batch",
			tokens:  [][]int{{0, 1, 2}, {3, 4, 5}},
			targets: [][]int{{1, 2, 3}, {4, 5, 6}},
			mask:    [][]float32{{1, 1, 1}, {1, 1, 1}},
		},
		{
			// Ragged rows of differing length (right-padding not yet applied).
			name:    "ragged_rows",
			tokens:  [][]int{{7}, {8, 9, 10, 11}, {}},
			targets: [][]int{{12, 13}, {14}, {15, 16, 17}},
			mask:    [][]float32{{1}, {0, 1}, {1, 1, 0}},
		},
		{
			// Negative targets are REAL: -100 is the cross-entropy ignore
			// index. Also pins the int emitter against strconv base-10.
			name:    "negative_ignore_index_targets",
			tokens:  [][]int{{0, 1}},
			targets: [][]int{{-100, -100}},
			mask:    [][]float32{{0, 0}},
		},
		{
			// int64 magnitude extremes — pins the int emitter incl. the
			// MinInt64 uint64(-v) wrap. (int is 64-bit on this target.)
			name:    "int_extremes",
			tokens:  [][]int{{0, math.MaxInt64, math.MinInt64}},
			targets: [][]int{{-1, 1, -9223372036854775807}},
			mask:    [][]float32{{0, 0, 0}},
		},
		{
			// Fractional mask values that PROVE shortest float32 round-trip
			// (bitSize 32): 0.1 must emit "0.1", not the float64 widening
			// "0.10000000149011612". 0.5/0.25 are exact; 1.0/0.0 trivial.
			name:    "fractional_mask_round_trip",
			tokens:  [][]int{{0, 0, 0, 0, 0}},
			targets: [][]int{{0, 0, 0, 0, 0}},
			mask:    [][]float32{{0.1, 0.5, 0.25, 1, 0}},
		},
		{
			// Negative zero: encoding/json emits "-0" for float32(-0.0).
			name:    "negative_zero_mask",
			tokens:  [][]int{{0}},
			targets: [][]int{{0}},
			mask:    [][]float32{{float32(math.Copysign(0, -1))}},
		},
		{
			// THE float discriminator: the 'e' (scientific) format path.
			// Values straddle the 1e-6 / 1e21 float32 cutoffs and exercise
			// the e-09 -> e-9 exponent cleanup. A mask of only {0,1} would
			// pass while this whole branch was silently wrong.
			name:    "scientific_notation_e_path",
			tokens:  [][]int{{0, 0, 0, 0}},
			targets: [][]int{{0, 0, 0, 0}},
			mask: [][]float32{
				{1e-7, 1e22, 1.5e-10, 3e25},      // small + large -> 'e'
				{1e-6, 1e21, 9.999999e-7, 1e-45}, // boundary + subnormal
			},
		},
		{
			// float32 dynamic-range extremes: largest finite (~3.4e38),
			// smallest normal (~1.18e-38), smallest subnormal (~1.4e-45),
			// each negated. All take the 'e' path.
			name:    "float32_range_extremes",
			tokens:  [][]int{{0, 0, 0}},
			targets: [][]int{{0, 0, 0}},
			mask: [][]float32{
				{math.MaxFloat32, math.SmallestNonzeroFloat32, -math.MaxFloat32},
				{1.1754944e-38, -1.1754944e-38, -math.SmallestNonzeroFloat32},
			},
		},
		{
			// A value just under 'f'/'e' boundary that stays 'f', next to one
			// that flips to 'e' — pins the exact cutoff side.
			name:    "f_path_boundary",
			tokens:  [][]int{{0, 0}},
			targets: [][]int{{0, 0}},
			mask:    [][]float32{{0.000001, 0.0000009}}, // 1e-6 ('f') vs <1e-6 ('e')
		},
	}
}

// TestDistillLoss_BatchCacheKeyParity diffs DistillBatchCacheKey's hash
// against SHA256(core.JSONMarshal(...)) over every adversarial fixture —
// the same contract the deleted hand-rolled emitter used to prove
// byte-for-byte, now proven through the public API since the emitter
// itself now lives in the shared dappco.re/go/inference/distill engine.
func TestDistillLoss_BatchCacheKeyParity(t *testing.T) {
	for _, tc := range batchCacheKeyParityFixtures() {
		t.Run(tc.name, func(t *testing.T) {
			payload := batchCacheKeyPayload{Tokens: tc.tokens, Targets: tc.targets, Mask: tc.mask}
			want := core.JSONMarshal(payload)
			batch := SFTBatch{
				Batch:   Batch{Tokens: tc.tokens, LossMask: tc.mask},
				Targets: tc.targets,
			}
			got := DistillBatchCacheKey(batch)
			if !want.OK {
				// json.Marshal failed (it cannot here — no NaN/Inf fixture in
				// this slice); if it ever does, this fixture belongs in the
				// NaN/Inf fallback test instead.
				t.Fatalf("core.JSONMarshal unexpectedly failed for fixture %q: %v", tc.name, want.Value)
			}
			wantKey := core.SHA256Hex(want.Value.([]byte))
			if got != wantKey {
				t.Fatalf("DistillBatchCacheKey = %q, want SHA256(core.JSONMarshal) = %q", got, wantKey)
			}
		})
	}
}

// TestDistillLoss_BatchCacheKeyParity_NaNInfFallback pins the NaN/Inf
// contract: encoding/json errors on those (core.JSONMarshal.OK == false),
// so DistillBatchCacheKey must still return a stable, deterministic,
// non-empty key — the Sprintf-based fallback the shared engine takes when
// its own byte emitter signals it cannot produce a JSON-exact encoding.
func TestDistillLoss_BatchCacheKeyParity_NaNInfFallback(t *testing.T) {
	cases := []struct {
		name string
		mask [][]float32
	}{
		{"nan", [][]float32{{float32(math.NaN())}}},
		{"pos_inf", [][]float32{{float32(math.Inf(1))}}},
		{"neg_inf", [][]float32{{float32(math.Inf(-1))}}},
		{"nan_among_finite", [][]float32{{1, 0, float32(math.NaN())}}},
		{"inf_second_row", [][]float32{{1, 1}, {0, float32(math.Inf(1))}}},
	}
	tokens := [][]int{{0}}
	targets := [][]int{{0}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Confirm the oracle really errors — otherwise the fallback
			// contract this test pins would be vacuous.
			payload := batchCacheKeyPayload{Tokens: tokens, Targets: targets, Mask: tc.mask}
			if core.JSONMarshal(payload).OK {
				t.Fatalf("expected core.JSONMarshal to error on NaN/Inf, but it succeeded")
			}
			batch := SFTBatch{
				Batch:   Batch{Tokens: tokens, LossMask: tc.mask},
				Targets: targets,
			}
			got := DistillBatchCacheKey(batch)
			if got == "" {
				t.Fatal("DistillBatchCacheKey(NaN/Inf) = empty, want a stable fallback hash")
			}
			if again := DistillBatchCacheKey(batch); again != got {
				t.Fatalf("DistillBatchCacheKey(NaN/Inf) = %q then %q, want deterministic", got, again)
			}
		})
	}
}

// TestDistillLoss_BatchCacheKeyParity_EndToEnd closes the loop on the
// public function: two SFTBatches that JSON-marshal to identical payload
// bytes must hash to the same key, and the key must equal the SHA256 of
// the core.JSONMarshal bytes (proving DistillBatchCacheKey still produces
// the pre-delegation hash, not just self-consistent ones).
func TestDistillLoss_BatchCacheKeyParity_EndToEnd(t *testing.T) {
	batch := SFTBatch{
		Batch: Batch{
			Tokens:   [][]int{{1, 2, 3}, {4, 5, 6}},
			LossMask: [][]float32{{1, 1, 0.5}, {1, 0, 1}},
		},
		Targets: [][]int{{2, 3, -100}, {5, 6, 7}},
	}
	payload := batchCacheKeyPayload{Tokens: batch.Batch.Tokens, Targets: batch.Targets, Mask: batch.Batch.LossMask}
	want := core.JSONMarshal(payload)
	if !want.OK {
		t.Fatalf("core.JSONMarshal failed: %v", want.Value)
	}
	wantKey := core.SHA256Hex(want.Value.([]byte))
	if got := DistillBatchCacheKey(batch); got != wantKey {
		t.Fatalf("DistillBatchCacheKey = %q, want SHA256 of core.JSONMarshal bytes %q", got, wantKey)
	}
}
