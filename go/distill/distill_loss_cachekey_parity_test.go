// SPDX-Licence-Identifier: EUPL-1.2

// Byte-parity guard for appendBatchCacheKeyJSON, the hand-rolled emitter
// behind DistillBatchCacheKey. The teacher-logit cache is keyed on the
// SHA256 of these bytes, so the emitted JSON MUST be byte-for-byte
// identical to what core.JSONMarshal (encoding/json) produced before the
// emitter replaced it — any drift silently invalidates every cached
// teacher entry. A parse-back round-trip would NOT catch format drift
// that still changes the hash, so this diffs the emitter's raw bytes
// against core.JSONMarshal directly over adversarial inputs, and pins the
// NaN/Inf -> ok==false fallback contract (json.Marshal errors there, so
// the emitter must signal the same so the caller takes the identical
// Sprintf-based key).
//
// Run:    go test -run='BatchCacheKeyParity' ./distill/

package distill

import (
	"bytes"
	"math"
	"testing"

	core "dappco.re/go"
)

// batchCacheKeyPayload mirrors the exact anonymous struct DistillBatchCacheKey
// marshals — same field order, same json tags — so core.JSONMarshal here is
// the same oracle the production path used.
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

// batchCacheKeyParityFixtures are the adversarial inputs shared by the
// byte-parity test (emitter bytes == core.JSONMarshal) and the
// length-exactness test (batchCacheKeyJSONLen == len(emitted)). None
// contain NaN/Inf — those have their own fallback test.
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
			// int64 magnitude extremes — pins appendCacheKeyInt64 incl. the
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

func TestDistillLoss_BatchCacheKeyParity(t *testing.T) {
	for _, tc := range batchCacheKeyParityFixtures() {
		t.Run(tc.name, func(t *testing.T) {
			payload := batchCacheKeyPayload{Tokens: tc.tokens, Targets: tc.targets, Mask: tc.mask}
			want := core.JSONMarshal(payload)
			got, ok := appendBatchCacheKeyJSON(nil, tc.tokens, tc.targets, tc.mask)
			if !want.OK {
				// json.Marshal failed (it cannot here — no NaN/Inf fixture in
				// this slice), so the emitter must also signal fallback.
				if ok {
					t.Fatalf("core.JSONMarshal failed but emitter returned ok=true: %s", got)
				}
				return
			}
			if !ok {
				t.Fatalf("emitter returned ok=false but core.JSONMarshal succeeded: %s", want.Value)
			}
			wantBytes := want.Value.([]byte)
			if !bytes.Equal(got, wantBytes) {
				t.Fatalf("byte mismatch:\n  json: %s\n  hand: %s", wantBytes, got)
			}
		})
	}
}

// TestDistillLoss_BatchCacheKeyLenExact pins batchCacheKeyJSONLen as the
// EXACT emitted byte count for every parity fixture — not just an upper
// bound. If it under-counts, appendBatchCacheKeyJSON's single make grows
// (silently restoring the extra alloc this change removed); if it
// over-counts, B/op regresses. Asserting equality also catches any drift
// between float32JSONLen's measure logic and appendCacheKeyFloat32's emit
// logic (they duplicate the format/cleanup branch) and any scaffold
// miscount.
func TestDistillLoss_BatchCacheKeyLenExact(t *testing.T) {
	for _, tc := range batchCacheKeyParityFixtures() {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := appendBatchCacheKeyJSON(nil, tc.tokens, tc.targets, tc.mask)
			if !ok {
				t.Fatalf("emitter returned ok=false on a finite fixture")
			}
			n, lenOK := batchCacheKeyJSONLen(tc.tokens, tc.targets, tc.mask)
			if !lenOK {
				t.Fatalf("batchCacheKeyJSONLen returned ok=false on a finite fixture")
			}
			if n != len(got) {
				t.Fatalf("batchCacheKeyJSONLen = %d, emitted %d bytes (%s)", n, len(got), got)
			}
		})
	}
}

// TestDistillLoss_BatchCacheKeyParity_NaNInfFallback pins the NaN/Inf
// contract directly: encoding/json errors on those (data.OK == false), so
// the emitter MUST return ok == false, sending DistillBatchCacheKey down
// the identical Sprintf-based fallback. Asserts both that json.Marshal
// rejects the payload AND that the emitter agrees.
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
			if _, ok := appendBatchCacheKeyJSON(nil, tokens, targets, tc.mask); ok {
				t.Fatal("emitter returned ok=true on NaN/Inf, want ok=false for the Sprintf fallback")
			}
		})
	}
}

// TestDistillLoss_BatchCacheKeyParity_EndToEnd closes the loop on the
// public function: two SFTBatches that JSON-marshal to identical payload
// bytes must hash to the same key, and the key must equal the SHA256 of
// the core.JSONMarshal bytes (proving DistillBatchCacheKey still produces
// the pre-emitter hash, not just self-consistent ones).
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
