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
	FreeLinear(l)

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
			t.Fatalf("FreeLinear(nil) panicked: %v", recovered)
		}
	}()

	FreeLinear(nil)
}

func TestClose_FreeEmbedding_Good(t *testing.T) {
	coverageTokens := "FreeEmbedding"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	Materialize(w)

	e := &Embedding{Weight: w}
	FreeEmbedding(e)

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
	FreeRMSNorm(r)

	if w.Valid() {
		t.Error("rmsnorm weight should be freed")
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

	FreeCaches([]Cache{c})
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
	FreeCaches([]Cache{nil})
}

// TestClose_CloseGemma4_NilModel_Ugly guards Mantis #1829: a Metal library
// load failure aborts model construction before any field is populated, and
// the deferred cleanup must return cleanly rather than panic on a nil model
// (a second panic would mask the real Metal error in the HTTP handler).
// TestClose_CloseArchitectures_NilModel_Ugly pins nil-safety for the
// metal-resident per-architecture close helpers. The Gemma 4 counterpart
// (closeGemma4 nil + partial-layers, Mantis #1829) moved to package gemma4's
// close_test.go with the model type.
func TestClose_CloseArchitectures_NilModel_Ugly(t *testing.T) {
	coverageTokens := "CloseArchitectures NilModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("close helper(nil) panicked: %v", recovered)
		}
	}()
	closeQwen3(nil)
}
