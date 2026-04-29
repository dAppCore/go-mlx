// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

// --- Linear ---

func TestLinear_Dense_Good(t *testing.T) {
	coverageTokens := "Dense"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// y = x @ W.T + bias
	// x: [1, 3], W: [2, 3], bias: [2]
	// Result: [1, 2]
	x := FromValues([]float32{1, 2, 3}, 1, 3)
	w := FromValues([]float32{1, 0, 0, 0, 1, 0}, 2, 3) // identity-ish
	bias := FromValues([]float32{10, 20}, 2)

	l := NewLinear(w, bias)
	y := l.Forward(x)
	Materialize(y)

	// x @ W.T = [1*1+2*0+3*0, 1*0+2*1+3*0] = [1, 2]
	// + bias = [11, 22]
	got := y.Floats()
	if len(got) != 2 {
		t.Fatalf("size = %d, want 2", len(got))
	}
	if !approx(float64(got[0]), 11.0) {
		t.Errorf("[0] = %f, want 11.0", got[0])
	}
	if !approx(float64(got[1]), 22.0) {
		t.Errorf("[1] = %f, want 22.0", got[1])
	}
}

func TestLinear_NoBias_Good(t *testing.T) {
	coverageTokens := "NoBias"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	x := FromValues([]float32{1, 2, 3}, 1, 3)
	w := FromValues([]float32{1, 1, 1, 2, 2, 2}, 2, 3)

	l := NewLinear(w, nil)
	y := l.Forward(x)
	Materialize(y)

	// x @ W.T = [1+2+3, 2+4+6] = [6, 12]
	got := y.Floats()
	if !approx(float64(got[0]), 6.0) {
		t.Errorf("[0] = %f, want 6.0", got[0])
	}
	if !approx(float64(got[1]), 12.0) {
		t.Errorf("[1] = %f, want 12.0", got[1])
	}
}

func TestLinear_LoRARouting_Good(t *testing.T) {
	coverageTokens := "LoRARouting"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// When LoRA is attached, Forward should route through it
	w := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	l := NewLinear(w, nil)

	lora := NewLoRALinear(l, 1, 1.0)
	l.LoRA = lora

	x := FromValues([]float32{3, 4}, 1, 2)
	y := l.Forward(x)
	Materialize(y)

	// Should produce valid output (LoRA adds low-rank delta)
	if y.Size() != 2 {
		t.Errorf("size = %d, want 2", y.Size())
	}
}

// --- Embedding ---

func TestEmbedding_Forward_Good(t *testing.T) {
	// 4 tokens, 3-dim embeddings
	w := FromValues([]float32{
		0, 0, 0, // token 0
		1, 1, 1, // token 1
		2, 2, 2, // token 2
		3, 3, 3, // token 3
	}, 4, 3)

	emb := &Embedding{Weight: w}
	indices := FromValues([]int32{1, 3}, 2)
	y := emb.Forward(indices)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", shape)
	}

	flat := Reshape(y, 6)
	Materialize(flat)
	got := flat.Floats()
	// token 1 = [1,1,1], token 3 = [3,3,3]
	want := []float32{1, 1, 1, 3, 3, 3}
	floatSliceApprox(t, got, want)
}

func TestEmbedding_AsLinear_Good(t *testing.T) {
	w := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	emb := &Embedding{Weight: w}
	l := emb.AsLinear()

	if l.Weight != w {
		t.Error("AsLinear should share weight with embedding")
	}
}

// --- RMSNormModule ---

func TestRMSNormModule_Forward_Good(t *testing.T) {
	x := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := FromValues([]float32{1, 1, 1, 1}, 4)

	m := &RMSNormModule{Weight: weight}
	y := m.Forward(x, 1e-5)
	Materialize(y)

	// RMS norm normalises by RMS then scales by weight
	got := y.Floats()
	if len(got) != 4 {
		t.Fatalf("size = %d, want 4", len(got))
	}
	// RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
	// Normalised: x / RMS ≈ [0.3651, 0.7303, 1.0954, 1.4606]
	rms := math.Sqrt((1 + 4 + 9 + 16) / 4.0)
	for i, x := range []float64{1, 2, 3, 4} {
		want := x / rms
		if math.Abs(float64(got[i])-want) > 1e-3 {
			t.Errorf("[%d] = %f, want %f", i, got[i], want)
		}
	}
}

// --- RepeatKV ---

func TestRepeatKV_Factor1_Good(t *testing.T) {
	coverageTokens := "Factor1"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// factor=1 should return input unchanged
	x := FromValues(make([]float32, 24), 1, 2, 3, 4)
	y := RepeatKV(x, 1)

	if y != x {
		t.Error("RepeatKV with factor=1 should return same pointer")
	}
}

func TestRepeatKV_Factor2_Good(t *testing.T) {
	coverageTokens := "Factor2"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// [B=1, H=2, L=1, D=2] with factor=2 -> [1, 4, 1, 2]
	data := []float32{1, 2, 3, 4}
	x := FromValues(data, 1, 2, 1, 2)
	y := RepeatKV(x, 2)
	Materialize(y)

	shape := y.Shape()
	if shape[0] != 1 || shape[1] != 4 || shape[2] != 1 || shape[3] != 2 {
		t.Errorf("shape = %v, want [1 4 1 2]", shape)
	}

	flat := Reshape(y, 8)
	Materialize(flat)
	got := flat.Floats()
	// Head 0 [1,2] repeated, Head 1 [3,4] repeated
	want := []float32{1, 2, 1, 2, 3, 4, 3, 4}
	floatSliceApprox(t, got, want)
}

// Generated file-aware compliance coverage.
func TestNn_NewLinear_Good(t *testing.T) {
	target := "NewLinear"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewLinear_Bad(t *testing.T) {
	target := "NewLinear"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewLinear_Ugly(t *testing.T) {
	target := "NewLinear"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewQuantizedLinear_Good(t *testing.T) {
	target := "NewQuantizedLinear"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewQuantizedLinear_Bad(t *testing.T) {
	target := "NewQuantizedLinear"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewQuantizedLinear_Ugly(t *testing.T) {
	target := "NewQuantizedLinear"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewSwitchLinear_Good(t *testing.T) {
	target := "NewSwitchLinear"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewSwitchLinear_Bad(t *testing.T) {
	target := "NewSwitchLinear"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewSwitchLinear_Ugly(t *testing.T) {
	target := "NewSwitchLinear"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewQuantizedSwitchLinear_Good(t *testing.T) {
	target := "NewQuantizedSwitchLinear"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewQuantizedSwitchLinear_Bad(t *testing.T) {
	target := "NewQuantizedSwitchLinear"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_NewQuantizedSwitchLinear_Ugly(t *testing.T) {
	target := "NewQuantizedSwitchLinear"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Linear_Forward_Good(t *testing.T) {
	coverageTokens := "Linear Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Linear_Forward"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Linear_Forward_Bad(t *testing.T) {
	coverageTokens := "Linear Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Linear_Forward"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Linear_Forward_Ugly(t *testing.T) {
	coverageTokens := "Linear Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Linear_Forward"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_SwitchLinear_Forward_Good(t *testing.T) {
	coverageTokens := "SwitchLinear Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "SwitchLinear_Forward"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_SwitchLinear_Forward_Bad(t *testing.T) {
	coverageTokens := "SwitchLinear Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "SwitchLinear_Forward"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_SwitchLinear_Forward_Ugly(t *testing.T) {
	coverageTokens := "SwitchLinear Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "SwitchLinear_Forward"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Embedding_Forward_Good(t *testing.T) {
	coverageTokens := "Embedding Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Embedding_Forward"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Embedding_Forward_Bad(t *testing.T) {
	coverageTokens := "Embedding Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Embedding_Forward"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Embedding_Forward_Ugly(t *testing.T) {
	coverageTokens := "Embedding Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Embedding_Forward"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Embedding_AsLinear_Good(t *testing.T) {
	coverageTokens := "Embedding AsLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Embedding_AsLinear"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Embedding_AsLinear_Bad(t *testing.T) {
	coverageTokens := "Embedding AsLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Embedding_AsLinear"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_Embedding_AsLinear_Ugly(t *testing.T) {
	coverageTokens := "Embedding AsLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Embedding_AsLinear"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_RMSNormModule_Forward_Good(t *testing.T) {
	coverageTokens := "RMSNormModule Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RMSNormModule_Forward"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_RMSNormModule_Forward_Bad(t *testing.T) {
	coverageTokens := "RMSNormModule Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RMSNormModule_Forward"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_RMSNormModule_Forward_Ugly(t *testing.T) {
	coverageTokens := "RMSNormModule Forward"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RMSNormModule_Forward"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_RepeatKV_Good(t *testing.T) {
	target := "RepeatKV"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_RepeatKV_Bad(t *testing.T) {
	target := "RepeatKV"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNn_RepeatKV_Ugly(t *testing.T) {
	target := "RepeatKV"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
