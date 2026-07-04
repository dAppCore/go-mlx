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
	closed                bool
	prefillLimit          int
	vocabSize             int
}

func (f *fakeCapModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeCapModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeCapModel) NewCache() []Cache                                  { return nil }
func (f *fakeCapModel) NumLayers() int                                     { return 0 }
func (f *fakeCapModel) Tokenizer() *Tokenizer                              { return nil }
func (f *fakeCapModel) ModelType() string                                  { return "fake" }
func (f *fakeCapModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }
func (f *fakeCapModel) NumQueryHeads() int                                 { return f.heads }
func (f *fakeCapModel) ResolveLoRALinear(_ int, _ string) *Linear          { return f.loraLinear }
func (f *fakeCapModel) RecordCacheTopology(profile *CacheProfile, _ []Cache) {
	profile.SharedLayers = f.cacheTopologySentinel
}
func (f *fakeCapModel) AttentionCacheLayout(_, _ int) []int         { return f.cacheLayout }
func (f *fakeCapModel) CloseModel()                                 { f.closed = true }
func (f *fakeCapModel) FixedSlidingPrefillChunkLimit(_ []Cache) int { return f.prefillLimit }
func (f *fakeCapModel) FillModelInfo(info *ModelInfo)               { info.VocabSize = f.vocabSize }

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

// --- attentionQueryHeads (QueryHeadCounter) ---

// TestAttentionQueryHeads_DispatchesViaInterface_Good pins that attentionQueryHeads
// routes through the QueryHeadCounter capability rather than a concrete type-switch.
func TestAttentionQueryHeads_DispatchesViaInterface_Good(t *testing.T) {
	if got := attentionQueryHeads(&fakeCapModel{heads: 8}); got != 8 {
		t.Fatalf("attentionQueryHeads(QueryHeadCounter) = %d, want 8", got)
	}
}

// TestAttentionQueryHeads_UnknownModelZero_Bad pins the behaviour-preserving
// fallback: a model that reports no query-head count yields 0.
func TestAttentionQueryHeads_UnknownModelZero_Bad(t *testing.T) {
	if got := attentionQueryHeads(fakeNoCapModel{}); got != 0 {
		t.Fatalf("attentionQueryHeads(no capability) = %d, want 0", got)
	}
}

// --- resolveLinear (LoRALinearResolver) ---

// TestResolveLinear_DispatchesViaInterface_Good pins that resolveLinear routes
// LoRA projection lookups through the LoRALinearResolver capability rather than a
// concrete type-switch.
func TestResolveLinear_DispatchesViaInterface_Good(t *testing.T) {
	sentinel := &Linear{}
	if got := resolveLinear(&fakeCapModel{loraLinear: sentinel}, 0, "self_attn.q_proj"); got != sentinel {
		t.Fatalf("resolveLinear(LoRALinearResolver) = %p, want sentinel %p", got, sentinel)
	}
}

// TestResolveLinear_UnknownModelNil_Bad pins the behaviour-preserving fallback: a
// model that resolves no projections yields nil, exactly as the old default arm.
func TestResolveLinear_UnknownModelNil_Bad(t *testing.T) {
	if got := resolveLinear(fakeNoCapModel{}, 0, "self_attn.q_proj"); got != nil {
		t.Fatalf("resolveLinear(no capability) = %p, want nil", got)
	}
}

// --- modelCacheProfile (CacheTopologyRecorder) ---

// TestModelCacheProfile_DispatchesViaInterface_Good pins that modelCacheProfile
// records architecture-specific cache topology through the CacheTopologyRecorder
// capability rather than a concrete *Gemma4Model type-switch.
func TestModelCacheProfile_DispatchesViaInterface_Good(t *testing.T) {
	got := modelCacheProfile(&fakeCapModel{cacheTopologySentinel: 7}, []Cache{nil})
	if got == nil {
		t.Fatal("modelCacheProfile returned nil profile")
	}
	if got.SharedLayers != 7 {
		t.Fatalf("RecordCacheTopology not dispatched: SharedLayers = %d, want 7", got.SharedLayers)
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

// --- attentionCacheIndexByLayer (AttentionCacheLayouter) ---

// TestAttentionCacheIndexByLayer_DispatchesViaInterface_Good pins that the
// per-layer cache mapping comes from the AttentionCacheLayouter capability rather
// than a concrete *Gemma4Model type-switch.
func TestAttentionCacheIndexByLayer_DispatchesViaInterface_Good(t *testing.T) {
	want := []int{7, 7, 7}
	got := attentionCacheIndexByLayer(&fakeCapModel{cacheLayout: want}, 3, 2)
	if len(got) != 3 || got[0] != 7 {
		t.Fatalf("AttentionCacheLayout not dispatched: got %v, want %v", got, want)
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

// --- Model.Close (ModelCloser) ---

// TestModelClose_DispatchesViaInterface_Good pins that Close releases model
// weights through the ModelCloser capability rather than a concrete type-switch.
func TestModelClose_DispatchesViaInterface_Good(t *testing.T) {
	fake := &fakeCapModel{}
	m := &Model{model: fake}
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !fake.closed {
		t.Fatal("CloseModel not dispatched during Close")
	}
}

// TestModelClose_UnknownModelNoClose_Bad pins the behaviour-preserving fallback: a
// model with no closer still has its state cleared and returns no error.
func TestModelClose_UnknownModelNoClose_Bad(t *testing.T) {
	m := &Model{model: fakeNoCapModel{}}
	if err := m.Close(); err != nil {
		t.Fatalf("Close on non-closer: %v", err)
	}
	if m.model != nil {
		t.Fatal("Close did not clear model reference")
	}
}

// --- fixedSlidingPrefillChunkLimit (FixedSlidingPrefillLimiter) ---

// TestFixedSlidingPrefillChunkLimit_DispatchesViaInterface_Good pins that the
// fixed-sliding prefill chunk limit comes from the FixedSlidingPrefillLimiter
// capability rather than a concrete *Gemma4Model assertion.
func TestFixedSlidingPrefillChunkLimit_DispatchesViaInterface_Good(t *testing.T) {
	m := &Model{model: &fakeCapModel{prefillLimit: 9}}
	if got := fixedSlidingPrefillChunkLimit(m, []Cache{nil}); got != 9 {
		t.Fatalf("FixedSlidingPrefillChunkLimit not dispatched: got %d, want 9", got)
	}
}

// TestFixedSlidingPrefillChunkLimit_UnknownModelZero_Bad pins the behaviour-
// preserving fallback: a model without the capability yields 0.
func TestFixedSlidingPrefillChunkLimit_UnknownModelZero_Bad(t *testing.T) {
	m := &Model{model: fakeNoCapModel{}}
	if got := fixedSlidingPrefillChunkLimit(m, []Cache{nil}); got != 0 {
		t.Fatalf("FixedSlidingPrefillChunkLimit(no capability) = %d, want 0", got)
	}
}

// --- Model.Info (ModelInfoReporter) ---

// TestModelInfo_DispatchesViaInterface_Good pins that Info fills architecture
// metadata through the ModelInfoReporter capability rather than a concrete
// type-switch over every model type.
func TestModelInfo_DispatchesViaInterface_Good(t *testing.T) {
	m := &Model{model: &fakeCapModel{vocabSize: 4242}}
	if got := m.Info(); got.VocabSize != 4242 {
		t.Fatalf("FillModelInfo not dispatched: VocabSize = %d, want 4242", got.VocabSize)
	}
}

// TestModelInfo_UnknownModelBaseFieldsOnly_Bad pins the behaviour-preserving
// fallback: a model that reports no metadata leaves the architecture-specific
// fields at zero (only the base Architecture/NumLayers are set).
func TestModelInfo_UnknownModelBaseFieldsOnly_Bad(t *testing.T) {
	m := &Model{model: fakeNoCapModel{}}
	if got := m.Info(); got.VocabSize != 0 {
		t.Fatalf("unexpected metadata for no-reporter model: VocabSize = %d, want 0", got.VocabSize)
	}
}
