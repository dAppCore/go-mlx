// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
)

// TestToRootMetrics_FullyPopulated drives toRootMetrics with every nested
// pointer non-nil and multi-element slices so the non-nil / multi-element
// arms of toRootCacheProfile, toRootTurboQuantKVPayloadEstimate,
// toRootMTPMetrics, toRootTokenPhaseTraces and toRootNativePhaseTraces all
// execute. The previous coverage only ever passed nil sub-pointers, so the
// arena-allocation bodies were never reached.
func TestToRootMetrics_FullyPopulated(t *testing.T) {
	metrics := metal.Metrics{
		PromptTokens:    7,
		GeneratedTokens: 11,
		CacheProfile: &metal.CacheProfile{
			Architecture: "gemma4_text",
			TotalCaches:  4,
			LocalCaches:  2,
			GlobalCaches: 2,
		},
		TurboQuantKVPayload: &metal.TurboQuantKVCachePayloadEstimate{
			Pages:             3,
			PageVectors:       12,
			PayloadBytes:      4096,
			FP16BaselineBytes: 8192,
		},
		MTP: &metal.MTPMetrics{
			DraftTokenSchedule: []int{2, 2, 1},
			ProposedTokens:     5,
			AcceptedTokens:     4,
			AcceptanceRate:     0.8,
		},
		TokenPhases: []metal.TokenPhaseTrace{
			{
				Step:      0,
				TokenID:   100,
				TokenText: "a",
				NativeEvents: []metal.NativePhaseTrace{
					{Name: "materialize", Pages: 2, Tokens: 3},
					{Name: "detach", Pages: 1, Tokens: 1},
				},
			},
			{
				Step:       1,
				TokenID:    101,
				TokenText:  "b",
				FinalToken: true,
				// No NativeEvents — exercises the n==0 branch inside the loop.
			},
		},
	}

	root := toRootMetrics(metrics)

	if root.PromptTokens != 7 || root.GeneratedTokens != 11 {
		t.Fatalf("scalar metrics dropped: %+v", root)
	}
	if root.CacheProfile == nil || root.CacheProfile.Architecture != "gemma4_text" {
		t.Fatalf("CacheProfile not converted: %+v", root.CacheProfile)
	}
	if root.TurboQuantKVPayload == nil || root.TurboQuantKVPayload.Pages != 3 {
		t.Fatalf("TurboQuantKVPayload not converted: %+v", root.TurboQuantKVPayload)
	}
	if root.MTP == nil || root.MTP.AcceptedTokens != 4 {
		t.Fatalf("MTP not converted: %+v", root.MTP)
	}
	if len(root.TokenPhases) != 2 {
		t.Fatalf("TokenPhases len = %d, want 2", len(root.TokenPhases))
	}
	if len(root.TokenPhases[0].NativeEvents) != 2 {
		t.Fatalf("TokenPhases[0].NativeEvents len = %d, want 2", len(root.TokenPhases[0].NativeEvents))
	}
	if root.TokenPhases[0].NativeEvents[1].Name != "detach" {
		t.Fatalf("native event not converted: %+v", root.TokenPhases[0].NativeEvents)
	}
	if len(root.TokenPhases[1].NativeEvents) != 0 {
		t.Fatalf("empty NativeEvents should stay nil: %+v", root.TokenPhases[1].NativeEvents)
	}
}

// TestToRootMetrics_EmptySlices keeps the len==0 early-return arms covered
// alongside the populated case so the converters' nil-in/nil-out contract is
// pinned (no allocation, nil out).
func TestToRootMetrics_EmptySlices(t *testing.T) {
	root := toRootMetrics(metal.Metrics{})
	if root.TokenPhases != nil {
		t.Fatalf("empty TokenPhases should be nil, got %+v", root.TokenPhases)
	}
	if root.CacheProfile != nil || root.TurboQuantKVPayload != nil || root.MTP != nil {
		t.Fatalf("nil sub-pointers should stay nil: %+v", root)
	}
}

// TestToRootNativePhaseTraces_MultiElement exercises the standalone helper
// directly with a non-empty slice (the per-event field rebuild loop).
func TestToRootNativePhaseTraces_MultiElement(t *testing.T) {
	out := toRootNativePhaseTraces([]metal.NativePhaseTrace{
		{Name: "first", Duration: 5, Pages: 1, Tokens: 2, Error: "boom"},
		{Name: "second", Duration: 7},
	})
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2", len(out))
	}
	if out[0].Name != "first" || out[0].Error != "boom" || out[1].Name != "second" {
		t.Fatalf("conversion wrong: %+v", out)
	}
	if toRootNativePhaseTraces(nil) != nil {
		t.Fatalf("nil-in should be nil-out")
	}
}

// TestToRootClassifyResults_Branches covers the three Logits arms: nil,
// empty-non-nil, and populated (the arena copy).
func TestToRootClassifyResults_Branches(t *testing.T) {
	results := []metal.ClassifyResult{
		{Token: metal.Token{ID: 1, Text: "x"}, Logits: nil},
		{Token: metal.Token{ID: 2, Text: "y"}, Logits: []float32{}},
		{Token: metal.Token{ID: 3, Text: "z"}, Logits: []float32{0.1, 0.2, 0.3}},
	}
	out := toRootClassifyResults(results)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[0].Logits != nil {
		t.Fatalf("nil logits should stay nil, got %+v", out[0].Logits)
	}
	if out[1].Logits == nil || len(out[1].Logits) != 0 {
		t.Fatalf("empty logits should stay empty-non-nil, got %+v", out[1].Logits)
	}
	if len(out[2].Logits) != 3 || out[2].Logits[2] != 0.3 {
		t.Fatalf("populated logits not copied: %+v", out[2].Logits)
	}
	if toRootClassifyResults(nil) != nil {
		t.Fatalf("nil-in should be nil-out")
	}
}

// TestToRootBatchResults_MultiPrompt covers the arena layout across several
// prompts each with several tokens, plus the propagated Err.
func TestToRootBatchResults_MultiPrompt(t *testing.T) {
	wantErr := metalTestError("partial batch")
	results := []metal.BatchResult{
		{Tokens: []metal.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}},
		{Tokens: []metal.Token{{ID: 3, Text: "c"}}, Err: wantErr},
	}
	out := toRootBatchResults(results)
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2", len(out))
	}
	if len(out[0].Tokens) != 2 || out[0].Tokens[1].Text != "b" {
		t.Fatalf("first prompt tokens wrong: %+v", out[0].Tokens)
	}
	if len(out[1].Tokens) != 1 || out[1].Err != wantErr {
		t.Fatalf("second prompt wrong: %+v", out[1])
	}
	if toRootBatchResults(nil) != nil {
		t.Fatalf("nil-in should be nil-out")
	}
}

// TestToRootAttentionSnapshot covers the non-nil conversion arm.
func TestToRootAttentionSnapshot(t *testing.T) {
	if toRootAttentionSnapshot(nil) != nil {
		t.Fatalf("nil-in should be nil-out")
	}
	snap := toRootAttentionSnapshot(&metal.AttentionResult{
		NumLayers:     2,
		NumHeads:      4,
		SeqLen:        8,
		HeadDim:       16,
		NumQueryHeads: 4,
		Architecture:  "gemma4_text",
		Keys:          [][][]float32{{{1, 2}}},
		Queries:       [][][]float32{{{3, 4}}},
	})
	if snap == nil || snap.NumLayers != 2 || snap.Architecture != "gemma4_text" {
		t.Fatalf("attention snapshot conversion wrong: %+v", snap)
	}
}

// TestLoadOptionSetters_CoverAllArms walks every enum arm of the
// finite-domain option constructors so each switch case (and the default
// fall-through) is reached, then applies the returned closure to confirm it
// sets the intended field.
func TestLoadOptionSetters_CoverAllArms(t *testing.T) {
	policies := []memory.KVCachePolicy{
		memory.KVCacheDefault,
		memory.KVCacheRotating,
		memory.KVCacheFull,
		memory.KVCachePolicy("custom-policy"), // default fall-through
	}
	for _, policy := range policies {
		var cfg LoadConfig
		WithCachePolicy(policy)(&cfg)
		if cfg.CachePolicy != policy {
			t.Fatalf("WithCachePolicy(%q) set %q", policy, cfg.CachePolicy)
		}
	}

	modes := []memory.KVCacheMode{
		memory.KVCacheModeDefault,
		memory.KVCacheModeFP16,
		memory.KVCacheModeQ8,
		memory.KVCacheModeKQ8VQ4,
		memory.KVCacheModePaged,
		memory.KVCacheModeTurboQuant,
		memory.KVCacheMode("custom-mode"), // default fall-through
	}
	for _, mode := range modes {
		var cfg LoadConfig
		WithKVCacheMode(mode)(&cfg)
		if cfg.CacheMode != mode {
			t.Fatalf("WithKVCacheMode(%q) set %q", mode, cfg.CacheMode)
		}
	}

	// WithKVCacheStorageDType: "", native, default → "", fp16/bf16 → passthrough,
	// unknown → passthrough.
	for _, in := range []string{"", "native", "default"} {
		var cfg LoadConfig
		WithKVCacheStorageDType(in)(&cfg)
		if cfg.KVCacheStorageDType != "" {
			t.Fatalf("WithKVCacheStorageDType(%q) = %q, want empty", in, cfg.KVCacheStorageDType)
		}
	}
	for _, in := range []string{"fp16", "bf16", "q4-weird"} {
		var cfg LoadConfig
		WithKVCacheStorageDType(in)(&cfg)
		if cfg.KVCacheStorageDType != in {
			t.Fatalf("WithKVCacheStorageDType(%q) = %q, want passthrough", in, cfg.KVCacheStorageDType)
		}
	}
}

type metalTestErr string

func (e metalTestErr) Error() string { return string(e) }

func metalTestError(msg string) error { return metalTestErr(msg) }
