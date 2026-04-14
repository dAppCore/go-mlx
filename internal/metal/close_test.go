//go:build darwin && arm64 && !nomlx

package metal

import (
	"testing"
)

func TestClose_FreeLinear_Good(t *testing.T) {
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
	freeLinear(nil) // should not panic
}

func TestClose_FreeEmbedding_Good(t *testing.T) {
	w := FromValues([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	Materialize(w)

	e := &Embedding{Weight: w}
	freeEmbedding(e)

	if w.Valid() {
		t.Error("embedding weight should be freed")
	}
}

func TestClose_FreeRMSNorm_Good(t *testing.T) {
	w := FromValues([]float32{1, 1, 1, 1}, 4)
	Materialize(w)

	r := &RMSNormModule{Weight: w}
	freeRMSNorm(r)

	if w.Valid() {
		t.Error("rmsnorm weight should be freed")
	}
}

func TestClose_CloseGemma_MinimalModel_Good(t *testing.T) {
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
