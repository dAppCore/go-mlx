// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the pure config validators and the fixed-attention mask set. These
// are deterministic Go functions reachable without a model; helpers prefixed
// cov4cm to stay unique.

// TestConfig_ValidateQuantizationConfig_Good walks every quant mode branch of
// validateGemma4QuantizationConfig plus the size/bit error branches.
func TestConfig_ValidateQuantizationConfig_Good(t *testing.T) {
	cases := []struct {
		name    string
		q       *metal.QuantizationConfig
		wantErr bool
	}{
		{"nil config ok", nil, false},
		{"affine bits 4 ok", &metal.QuantizationConfig{Mode: "affine", Bits: 4, GroupSize: 64}, false},
		{"affine bits 0 ok (unset)", &metal.QuantizationConfig{Mode: "affine", Bits: 0}, false},
		{"affine bits 7 rejected", &metal.QuantizationConfig{Mode: "affine", Bits: 7}, true},
		{"negative group rejected", &metal.QuantizationConfig{Mode: "affine", GroupSize: -1}, true},
		{"negative bits rejected", &metal.QuantizationConfig{Mode: "affine", Bits: -1}, true},
		{"mxfp4 group 32 ok", &metal.QuantizationConfig{Mode: "mxfp4", GroupSize: 32, Bits: 4}, false},
		{"mxfp4 wrong group rejected", &metal.QuantizationConfig{Mode: "mxfp4", GroupSize: 64}, true},
		{"mxfp4 wrong bits rejected", &metal.QuantizationConfig{Mode: "mxfp4", Bits: 8}, true},
		{"mxfp8 group 32 bits 8 ok", &metal.QuantizationConfig{Mode: "mxfp8", GroupSize: 32, Bits: 8}, false},
		{"mxfp8 wrong group rejected", &metal.QuantizationConfig{Mode: "mxfp8", GroupSize: 64}, true},
		{"mxfp8 wrong bits rejected", &metal.QuantizationConfig{Mode: "mxfp8", Bits: 4}, true},
		{"nvfp4 group 16 ok", &metal.QuantizationConfig{Mode: "nvfp4", GroupSize: 16, Bits: 4}, false},
		{"nvfp4 wrong group rejected", &metal.QuantizationConfig{Mode: "nvfp4", GroupSize: 32}, true},
		{"nvfp4 wrong bits rejected", &metal.QuantizationConfig{Mode: "nvfp4", Bits: 8}, true},
		{"unknown mode rejected", &metal.QuantizationConfig{Mode: "weird"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateGemma4QuantizationConfig(tc.q)
			if (err != nil) != tc.wantErr {
				t.Fatalf("validateGemma4QuantizationConfig(%+v) err = %v, wantErr %v", tc.q, err, tc.wantErr)
			}
		})
	}
}

// TestConfig_RequiredAndNegativeFields_Good covers gemma4RequiredConfigField and
// gemma4NegativeConfigField: a complete config reports no missing/negative field,
// while a zeroed sizing field and a negative token id are each named.
func TestConfig_RequiredAndNegativeFields_Good(t *testing.T) {
	full := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:            32,
			NumHiddenLayers:       2,
			IntermediateSize:      64,
			NumAttentionHeads:     1,
			NumKeyValueHeads:      1,
			MaxPositionEmbeddings: 16,
			HeadDim:               32,
			VocabSize:             10,
		},
		SlidingWindow:        4,
		SlidingWindowPattern: 2,
	}
	if got := gemma4RequiredConfigField(full); got != "" {
		t.Fatalf("required field on complete config = %q, want \"\"", got)
	}
	if got := gemma4NegativeConfigField(full); got != "" {
		t.Fatalf("negative field on valid config = %q, want \"\"", got)
	}

	// each zeroed required field is reported (first in declaration order wins).
	missing := *full
	missing.IntermediateSize = 0
	missing.HiddenSize = 0
	if got := gemma4RequiredConfigField(&missing); got != "hidden_size" {
		t.Fatalf("required field = %q, want hidden_size (first zero)", got)
	}

	// a negative token id is named.
	neg := *full
	neg.AudioTokenID = -5
	if got := gemma4NegativeConfigField(&neg); got != "audio_token_id" {
		t.Fatalf("negative field = %q, want audio_token_id", got)
	}

	// a negative pointer-valued MoE field is named.
	bad := int32(-1)
	negPtr := *full
	negPtr.NumExperts = &bad
	if got := gemma4NegativeConfigField(&negPtr); got != "num_experts" {
		t.Fatalf("negative ptr field = %q, want num_experts", got)
	}
}

// TestMasks_FixedCapacityOffset_Good covers fixedGemma4AttentionMaskCapacityOffset
// across its three resolution paths: a sized FixedKVCache below capacity, a
// fixed shared-KV prev, and the decline cases (multi-token, over-capacity, no
// fixed source).
func TestMasks_FixedCapacityOffset_Good(t *testing.T) {
	requireMetalRuntime(t)

	// multi-token always declines.
	if _, _, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewFixedKVCache(8), sharedKV{}, 2); ok {
		t.Fatal("seqLen 2 must decline")
	}

	// FixedKVCache below capacity resolves to (maxSize, offset).
	cap0, off0, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewFixedKVCacheAtOffset(16, 5, 5), sharedKV{}, 1)
	if !ok || cap0 != 16 || off0 != 5 {
		t.Fatalf("fixed cache resolve = (%d,%d,%v), want (16,5,true)", cap0, off0, ok)
	}

	// FixedKVCache at capacity (offset+1 > maxSize) declines.
	if _, _, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewFixedKVCacheAtOffset(16, 16, 16), sharedKV{}, 1); ok {
		t.Fatal("offset at capacity must decline")
	}

	// fixed shared-KV prev resolves from the key tensor's capacity dim.
	keys := seqArray(0.1, 1, 1, 8, 4)
	defer metal.Free(keys)
	cap1, off1, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewKVCache(), sharedKV{Fixed: true, Keys: keys, Offset: 3}, 1)
	if !ok || cap1 != 8 || off1 != 3 {
		t.Fatalf("prev resolve = (%d,%d,%v), want (8,3,true)", cap1, off1, ok)
	}

	// a non-fixed cache with no fixed prev declines.
	if _, _, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewKVCache(), sharedKV{}, 1); ok {
		t.Fatal("non-fixed cache with no prev must decline")
	}
}

// TestMasks_FixedMaskSetForLayer_Good drives fixedGemma4AttentionMaskSet.ForLayer
// end to end with the shared-mask gate enabled: it builds a mask for a fixed
// cache, caches it (second call returns the same handle), and Free releases it. A
// disabled set returns nil.
func TestMasks_FixedMaskSetForLayer_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(metal.SetRuntimeGate(metal.GateFixedSharedMask, true))

	set := newFixedGemma4AttentionMaskSet(1, 1, nil)
	if set.disabled {
		t.Skip("fixed shared mask unavailable in this build")
	}
	cache := metal.NewFixedKVCacheAtOffset(16, 4, 4)

	mask1 := set.ForLayer(cache, sharedKV{})
	if mask1 == nil || !mask1.Valid() {
		t.Fatal("ForLayer returned no mask for a fixed cache below capacity")
	}
	mask2 := set.ForLayer(cache, sharedKV{})
	if mask2 != mask1 {
		t.Fatal("second ForLayer call should return the cached mask handle")
	}
	set.Free()

	// a disabled set returns nil.
	disabled := &fixedGemma4AttentionMaskSet{disabled: true}
	if disabled.ForLayer(cache, sharedKV{}) != nil {
		t.Fatal("disabled set must return nil")
	}
	disabled.Free() // no-op safe

	// a nil set is safe.
	var nilSet *fixedGemma4AttentionMaskSet
	if nilSet.ForLayer(cache, sharedKV{}) != nil {
		t.Fatal("nil set ForLayer must return nil")
	}
	nilSet.Free()

	// An enabled set whose cache declines capacity/offset (a fixed cache exactly
	// at capacity) returns nil — the ForLayer "not ok" branch.
	enabled := newFixedGemma4AttentionMaskSet(1, 1, nil)
	if !enabled.disabled {
		atCap := metal.NewFixedKVCacheAtOffset(16, 16, 16)
		if enabled.ForLayer(atCap, sharedKV{}) != nil {
			t.Fatal("at-capacity cache should decline → nil mask")
		}
		enabled.Free()
	}
}

// TestMasks_RuntimeMaskCache_Branches covers gemma4RuntimeMaskCache's
// nil-receiver fast paths and the runtime-cache hit/miss bookkeeping, plus the
// pure gemma4CanUseOffsetCausalAttention predicate.
func TestMasks_RuntimeMaskCache_Branches(t *testing.T) {
	requireMetalRuntime(t)

	// nil receiver: CachedAttentionMask builds directly (no caching), Free no-ops.
	var nilCache *gemma4RuntimeMaskCache
	direct := nilCache.CachedAttentionMask(1, 2, 4, 0, 0, 0)
	if direct == nil || !direct.Valid() {
		t.Fatal("nil-receiver CachedAttentionMask should still build a mask")
	}
	metal.Free(direct)
	nilCache.Free() // must not panic

	// populated cache: a miss builds + stores, a repeat hit returns the same
	// handle, and Free releases the owned masks.
	c := &gemma4RuntimeMaskCache{}
	m1 := c.CachedAttentionMask(1, 2, 4, 0, 0, 0)
	if m1 == nil || !m1.Valid() {
		t.Fatal("first CachedAttentionMask should build a mask")
	}
	m2 := c.CachedAttentionMask(1, 2, 4, 0, 0, 0)
	if m2 != m1 {
		t.Fatal("repeat key should return the cached mask handle")
	}
	c.Free()

	// gemma4CanUseOffsetCausalAttention predicate: the degenerate guards, the
	// no-window true, and the windowed in/out-of-range cases.
	if gemma4CanUseOffsetCausalAttention(1, 4, 0) {
		t.Fatal("queryLen<=1 must be false")
	}
	if gemma4CanUseOffsetCausalAttention(2, 0, 0) {
		t.Fatal("keyLen<=0 must be false")
	}
	if !gemma4CanUseOffsetCausalAttention(2, 4, 0) {
		t.Fatal("no window (window<=0) must be true")
	}
	if !gemma4CanUseOffsetCausalAttention(2, 4, 8) {
		t.Fatal("within window must be true")
	}
	if gemma4CanUseOffsetCausalAttention(10, 4, 2) {
		t.Fatal("queryLen beyond window must be false")
	}
}
