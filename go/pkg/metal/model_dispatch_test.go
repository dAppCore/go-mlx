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
	heads      int
	loraLinear *Linear
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
