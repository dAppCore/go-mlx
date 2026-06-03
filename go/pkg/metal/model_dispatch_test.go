// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// go-mlx #45 part B — these tests pin that model-specific behaviour dispatches
// through capability interfaces (model.(someCapability)) rather than concrete
// type-switches (case *Gemma4Model:). Interface dispatch is what lets a model
// type live in its own package (e.g. go/model/gemma/4) instead of package metal,
// breaking the metal⇄model import cycle.

// fakeQueryHeadModel is a minimal InternalModel that also reports a query-head
// count. It lets the dispatch tests exercise attentionQueryHeads without loading
// real model weights.
type fakeQueryHeadModel struct {
	heads int
}

func (f *fakeQueryHeadModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeQueryHeadModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeQueryHeadModel) NewCache() []Cache                                  { return nil }
func (f *fakeQueryHeadModel) NumLayers() int                                     { return 0 }
func (f *fakeQueryHeadModel) Tokenizer() *Tokenizer                              { return nil }
func (f *fakeQueryHeadModel) ModelType() string                                  { return "fake" }
func (f *fakeQueryHeadModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }
func (f *fakeQueryHeadModel) numQueryHeads() int                                 { return f.heads }

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

// TestAttentionQueryHeads_DispatchesViaInterface_Good pins that attentionQueryHeads
// routes through the queryHeadCounter capability interface rather than a concrete
// type-switch — the dispatch shape that lets model types live outside package metal.
func TestAttentionQueryHeads_DispatchesViaInterface_Good(t *testing.T) {
	if got := attentionQueryHeads(&fakeQueryHeadModel{heads: 8}); got != 8 {
		t.Fatalf("attentionQueryHeads(queryHeadCounter) = %d, want 8", got)
	}
}

// TestAttentionQueryHeads_UnknownModelZero_Bad pins the behaviour-preserving
// fallback: a model that reports no query-head count yields 0, exactly as the old
// type-switch default arm did.
func TestAttentionQueryHeads_UnknownModelZero_Bad(t *testing.T) {
	if got := attentionQueryHeads(fakeNoCapModel{}); got != 0 {
		t.Fatalf("attentionQueryHeads(no capability) = %d, want 0", got)
	}
}
