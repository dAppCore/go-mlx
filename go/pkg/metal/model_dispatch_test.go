// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// go-mlx #45 part B — these tests pin that model-specific behaviour dispatches
// through capability interfaces (model.(someCapability)) rather than concrete
// type-switches (case *Gemma4Model:). Interface dispatch is what lets a model
// type live in its own package (e.g. go/model/gemma/4) instead of package metal,
// breaking the metal⇄model import cycle.

// fakeCapModel is a minimal InternalModel that also implements the part-B
// capability interfaces, each capability's return value controlled by a field.
// It lets the dispatch tests exercise the metal-side dispatch helpers without
// loading real model weights.
type fakeCapModel struct {
	heads                 int
	loraLinear            *Linear
	cacheTopologySentinel int
	cacheLayout           []int
}

func (f *fakeCapModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeCapModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeCapModel) NewCache() []Cache                                  { return nil }
func (f *fakeCapModel) NumLayers() int                                     { return 0 }
func (f *fakeCapModel) Tokenizer() *Tokenizer                              { return nil }
func (f *fakeCapModel) ModelType() string                                  { return "fake" }
func (f *fakeCapModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }
func (f *fakeCapModel) numQueryHeads() int                                 { return f.heads }
func (f *fakeCapModel) resolveLoRALinear(_ int, _ string) *Linear          { return f.loraLinear }
func (f *fakeCapModel) recordCacheTopology(profile *CacheProfile, _ []Cache) {
	profile.SharedLayers = f.cacheTopologySentinel
}
func (f *fakeCapModel) attentionCacheLayout(_, _ int) []int { return f.cacheLayout }

// fakeNoCapModel implements InternalModel only — it reports no capabilities, so
// capability lookups must fall back to their default behaviour.
type fakeNoCapModel struct{}

func (fakeNoCapModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (fakeNoCapModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (fakeNoCapModel) NewCache() []Cache                                  { return nil }
func (fakeNoCapModel) NumLayers() int                                     { return 0 }
func (fakeNoCapModel) Tokenizer() *Tokenizer                              { return nil }
func (fakeNoCapModel) ModelType() string                                  { return "fake-nocap" }
func (fakeNoCapModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }

// --- attentionQueryHeads (queryHeadCounter) ---

// TestAttentionQueryHeads_DispatchesViaInterface_Good pins that attentionQueryHeads
// routes through the queryHeadCounter capability rather than a concrete type-switch.
func TestAttentionQueryHeads_DispatchesViaInterface_Good(t *testing.T) {
	if got := attentionQueryHeads(&fakeCapModel{heads: 8}); got != 8 {
		t.Fatalf("attentionQueryHeads(queryHeadCounter) = %d, want 8", got)
	}
}

// TestAttentionQueryHeads_UnknownModelZero_Bad pins the behaviour-preserving
// fallback: a model that reports no query-head count yields 0.
func TestAttentionQueryHeads_UnknownModelZero_Bad(t *testing.T) {
	if got := attentionQueryHeads(fakeNoCapModel{}); got != 0 {
		t.Fatalf("attentionQueryHeads(no capability) = %d, want 0", got)
	}
}

// --- resolveLinear (loRALinearResolver) ---

// TestResolveLinear_DispatchesViaInterface_Good pins that resolveLinear routes
// LoRA projection lookups through the loRALinearResolver capability rather than a
// concrete type-switch.
func TestResolveLinear_DispatchesViaInterface_Good(t *testing.T) {
	sentinel := &Linear{}
	if got := resolveLinear(&fakeCapModel{loraLinear: sentinel}, 0, "self_attn.q_proj"); got != sentinel {
		t.Fatalf("resolveLinear(loRALinearResolver) = %p, want sentinel %p", got, sentinel)
	}
}

// TestResolveLinear_UnknownModelNil_Bad pins the behaviour-preserving fallback: a
// model that resolves no projections yields nil, exactly as the old default arm.
func TestResolveLinear_UnknownModelNil_Bad(t *testing.T) {
	if got := resolveLinear(fakeNoCapModel{}, 0, "self_attn.q_proj"); got != nil {
		t.Fatalf("resolveLinear(no capability) = %p, want nil", got)
	}
}

// --- modelCacheProfile (cacheTopologyRecorder) ---

// TestModelCacheProfile_DispatchesViaInterface_Good pins that modelCacheProfile
// records architecture-specific cache topology through the cacheTopologyRecorder
// capability rather than a concrete *Gemma4Model type-switch.
func TestModelCacheProfile_DispatchesViaInterface_Good(t *testing.T) {
	got := modelCacheProfile(&fakeCapModel{cacheTopologySentinel: 7}, []Cache{nil})
	if got == nil {
		t.Fatal("modelCacheProfile returned nil profile")
	}
	if got.SharedLayers != 7 {
		t.Fatalf("recordCacheTopology not dispatched: SharedLayers = %d, want 7", got.SharedLayers)
	}
}

// TestModelCacheProfile_UnknownModelNoTopology_Bad pins the behaviour-preserving
// fallback: a model with no special topology leaves the profile as the generic
// per-cache pass recorded it.
func TestModelCacheProfile_UnknownModelNoTopology_Bad(t *testing.T) {
	got := modelCacheProfile(fakeNoCapModel{}, []Cache{nil})
	if got == nil {
		t.Fatal("modelCacheProfile returned nil profile")
	}
	if got.SharedLayers != 0 {
		t.Fatalf("unexpected topology recorded: SharedLayers = %d, want 0", got.SharedLayers)
	}
}

// --- attentionCacheIndexByLayer (attentionCacheLayouter) ---

// TestAttentionCacheIndexByLayer_DispatchesViaInterface_Good pins that the
// per-layer cache mapping comes from the attentionCacheLayouter capability rather
// than a concrete *Gemma4Model type-switch.
func TestAttentionCacheIndexByLayer_DispatchesViaInterface_Good(t *testing.T) {
	want := []int{7, 7, 7}
	got := attentionCacheIndexByLayer(&fakeCapModel{cacheLayout: want}, 3, 2)
	if len(got) != 3 || got[0] != 7 {
		t.Fatalf("attentionCacheLayout not dispatched: got %v, want %v", got, want)
	}
}

// TestAttentionCacheIndexByLayer_UnknownModelIdentity_Bad pins the behaviour-
// preserving fallback: a model with no custom layout gets the identity mapping
// (layer i → cache i, capped by cache count, rest -1), exactly as the old default.
func TestAttentionCacheIndexByLayer_UnknownModelIdentity_Bad(t *testing.T) {
	got := attentionCacheIndexByLayer(fakeNoCapModel{}, 3, 2)
	if len(got) != 3 || got[0] != 0 || got[1] != 1 || got[2] != -1 {
		t.Fatalf("identity fallback = %v, want [0 1 -1]", got)
	}
}
