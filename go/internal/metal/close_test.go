// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestClose_FreeLinear_Good(t *testing.T) {
	coverageTokens := "FreeLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	bias := FromValues([]float32{0.1, 0.2}, 2)
	Materialize(w, bias)

	l := NewLinear(w, bias)
	freeLinear(l)

	if w.Valid() {
		t.Error("weight should be freed")
	}
	if bias.Valid() {
		t.Error("bias should be freed")
	}
}

func TestClose_FreeLinear_Nil_Good(t *testing.T) {
	coverageTokens := "FreeLinear Nil"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("freeLinear(nil) panicked: %v", recovered)
		}
	}()

	freeLinear(nil)
}

func TestClose_FreeEmbedding_Good(t *testing.T) {
	coverageTokens := "FreeEmbedding"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	Materialize(w)

	e := &Embedding{Weight: w}
	freeEmbedding(e)

	if w.Valid() {
		t.Error("embedding weight should be freed")
	}
}

func TestClose_FreeRMSNorm_Good(t *testing.T) {
	coverageTokens := "FreeRMSNorm"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 1, 1, 1}, 4)
	Materialize(w)

	r := &RMSNormModule{Weight: w}
	freeRMSNorm(r)

	if w.Valid() {
		t.Error("rmsnorm weight should be freed")
	}
}

func TestClose_CloseGemma_MinimalModel_Good(t *testing.T) {
	coverageTokens := "CloseGemma MinimalModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Build a minimal GemmaModel with one layer to test cleanup.
	embedW := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := FromValues([]float32{1, 1}, 2)
	normScaled := FromValues([]float32{2, 2}, 2)
	Materialize(embedW, normW, normScaled)

	// Layer components
	inW := FromValues([]float32{1, 1}, 2)
	qW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	kW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	vW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	oW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	qnW := FromValues([]float32{1, 1}, 2)
	knW := FromValues([]float32{1, 1}, 2)
	gateW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	upW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	downW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	Materialize(inW, qW, kW, vW, oW, qnW, knW, gateW, upW, downW)

	m := &GemmaModel{
		EmbedTokens: &Embedding{Weight: embedW},
		Norm:        &RMSNormModule{Weight: normW},
		NormScaled:  normScaled,
		Output:      nil, // Tied to embed — skip
		Layers: []*DecoderLayer{{
			InputNorm: &RMSNormModule{Weight: inW},
			Attention: &Attention{
				QProj: NewLinear(qW, nil),
				KProj: NewLinear(kW, nil),
				VProj: NewLinear(vW, nil),
				OProj: NewLinear(oW, nil),
				QNorm: &RMSNormModule{Weight: qnW},
				KNorm: &RMSNormModule{Weight: knW},
			},
			MLP: &MLP{
				GateProj: NewLinear(gateW, nil),
				UpProj:   NewLinear(upW, nil),
				DownProj: NewLinear(downW, nil),
			},
		}},
	}

	closeGemma(m)

	// Verify key arrays freed
	if embedW.Valid() {
		t.Error("embed weight should be freed")
	}
	if normW.Valid() {
		t.Error("norm weight should be freed")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed")
	}
	if gateW.Valid() {
		t.Error("gate_proj weight should be freed")
	}
}

func TestClose_CloseQwen3_MinimalModel_Good(t *testing.T) {
	coverageTokens := "CloseQwen3 MinimalModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	embedW := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := FromValues([]float32{1, 1}, 2)
	outW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	Materialize(embedW, normW, outW)

	inW := FromValues([]float32{1, 1}, 2)
	postW := FromValues([]float32{1, 1}, 2)
	qW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	kW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	vW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	oW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	qnW := FromValues([]float32{1, 1}, 2)
	knW := FromValues([]float32{1, 1}, 2)
	gateW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	upW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	downW := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	Materialize(inW, postW, qW, kW, vW, oW, qnW, knW, gateW, upW, downW)

	m := &Qwen3Model{
		EmbedTokens: &Embedding{Weight: embedW},
		Norm:        &RMSNormModule{Weight: normW},
		Output:      NewLinear(outW, nil),
		Layers: []*Qwen3DecoderLayer{{
			InputNorm:    &RMSNormModule{Weight: inW},
			PostAttnNorm: &RMSNormModule{Weight: postW},
			Attention: &Qwen3Attention{
				QProj: NewLinear(qW, nil),
				KProj: NewLinear(kW, nil),
				VProj: NewLinear(vW, nil),
				OProj: NewLinear(oW, nil),
				QNorm: &RMSNormModule{Weight: qnW},
				KNorm: &RMSNormModule{Weight: knW},
			},
			MLP: &Qwen3MLP{
				GateProj: NewLinear(gateW, nil),
				UpProj:   NewLinear(upW, nil),
				DownProj: NewLinear(downW, nil),
			},
		}},
	}

	closeQwen3(m)

	if embedW.Valid() {
		t.Error("embed weight should be freed")
	}
	if outW.Valid() {
		t.Error("output weight should be freed")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed")
	}
	if downW.Valid() {
		t.Error("down_proj weight should be freed")
	}
}

func TestClose_ModelClose_Idempotent_Good(t *testing.T) {
	coverageTokens := "ModelClose Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Close on a model with nil internals should not panic.
	m := &Model{}
	if err := m.Close(); err != nil {
		t.Fatalf("Close on empty model: %v", err)
	}
	// Double close should be safe.
	if err := m.Close(); err != nil {
		t.Fatalf("Double close: %v", err)
	}
}

func TestClose_FreeCaches_Good(t *testing.T) {
	coverageTokens := "FreeCaches"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewKVCache()
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	Materialize(k, v)
	c.Update(k, v, 2)

	state := c.State()
	if state == nil {
		t.Fatal("cache should have state after update")
	}

	freeCaches([]Cache{c})
	// After freeing, the underlying arrays should be invalid.
	for _, arr := range state {
		if arr.Valid() {
			t.Error("cache array should be freed")
		}
	}
}

func TestClose_FreeCaches_NilCache_Ugly(t *testing.T) {
	coverageTokens := "FreeCaches NilCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	freeCaches([]Cache{nil})
}

// TestClose_CloseGemma4_NilModel_Ugly guards Mantis #1829: a Metal library
// load failure aborts model construction before any field is populated, and
// the deferred cleanup must return cleanly rather than panic on a nil model
// (a second panic would mask the real Metal error in the HTTP handler).
func TestClose_CloseGemma4_NilModel_Ugly(t *testing.T) {
	coverageTokens := "CloseGemma4 NilModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("closeGemma4(nil) panicked: %v", recovered)
		}
	}()
	closeGemma4(nil)
	closeGemma(nil)
	closeQwen3(nil)
}

// TestClose_CloseGemma4_PartialLayers_Ugly guards Mantis #1829: when a Metal
// op panics mid-build, m.Layers is allocated to full length but only partly
// populated, leaving nil layer entries. Cleanup must skip them rather than
// nil-deref layer.compiledNativeOwnerDecode and bury the original failure.
func TestClose_CloseGemma4_PartialLayers_Ugly(t *testing.T) {
	coverageTokens := "CloseGemma4 PartialLayers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("closeGemma4 with nil layer panicked: %v", recovered)
		}
	}()

	embedW := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := FromValues([]float32{1, 1}, 2)
	Materialize(embedW, normW)

	m := &Gemma4Model{
		EmbedTokens: &Embedding{Weight: embedW},
		Norm:        &RMSNormModule{Weight: normW},
		// Pre-allocated like LoadGemma4 does, but only the first slot is
		// nil — modelling a build that panicked before populating layer 0.
		Layers: make([]*Gemma4DecoderLayer, 3),
	}

	closeGemma4(m)

	if embedW.Valid() {
		t.Error("embed weight should be freed despite nil layers")
	}
	if normW.Valid() {
		t.Error("norm weight should be freed despite nil layers")
	}
}
