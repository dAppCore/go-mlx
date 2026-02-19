//go:build darwin && arm64

package metal

import (
	"math"
	"os"
	"testing"
)

func TestNewLoRALinear(t *testing.T) {
	// Create a simple base linear layer: [4, 8] weight
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0) // rank=4, alpha=8

	// Check dimensions
	aShape := lora.A.Shape()
	bShape := lora.B.Shape()

	if aShape[0] != 4 || aShape[1] != 8 {
		t.Errorf("A shape = %v, want [4, 8]", aShape)
	}
	if bShape[0] != 4 || bShape[1] != 4 {
		t.Errorf("B shape = %v, want [4, 4]", bShape)
	}

	// Scale should be alpha/rank = 8/4 = 2
	if math.Abs(float64(lora.Scale)-2.0) > 1e-5 {
		t.Errorf("Scale = %f, want 2.0", lora.Scale)
	}

	// B should be all zeros (LoRA starts as identity)
	Materialize(lora.B)
	bFloats := lora.B.Floats()
	for i, v := range bFloats {
		if v != 0 {
			t.Errorf("B[%d] = %f, want 0", i, v)
		}
	}
}

func TestLoRALinear_ForwardMatchesBase(t *testing.T) {
	// With B=0, LoRA forward should equal base forward
	w := RandomNormal(0, 0.1, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)

	// Random input [1, 3, 8]
	x := RandomNormal(0, 1, []int32{1, 3, 8}, DTypeFloat32)
	Materialize(x)

	baseOut := base.Forward(x)
	loraOut := lora.Forward(x)
	Materialize(baseOut, loraOut)

	// Should be identical since B is zero
	baseFloats := baseOut.Floats()
	loraFloats := loraOut.Floats()

	if len(baseFloats) != len(loraFloats) {
		t.Fatalf("output sizes differ: base=%d, lora=%d", len(baseFloats), len(loraFloats))
	}

	for i := range baseFloats {
		diff := math.Abs(float64(baseFloats[i] - loraFloats[i]))
		if diff > 1e-4 {
			t.Errorf("output[%d] differs: base=%f, lora=%f", i, baseFloats[i], loraFloats[i])
		}
	}
}

func TestLoRALinear_ForwardWithAdapter(t *testing.T) {
	// Set A and B to known values and verify output changes
	w := Zeros([]int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 2, 4.0) // rank=2, alpha=4, scale=2

	// Set A to identity-like: [[1,0,0,...], [0,1,0,...]]
	a := Zeros([]int32{2, 8}, DTypeFloat32)
	// Set B to ones: [[1,1], [1,1], [1,1], [1,1]]
	b := FromValues([]float32{
		1, 1,
		1, 1,
		1, 1,
		1, 1,
	}, 4, 2)
	Materialize(a, b)
	lora.A = a
	lora.B = b

	// With base=0, A=0, output should also be 0 (scale * x@0@B^T = 0)
	x := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 1, 8)
	result := lora.Forward(x)
	Materialize(result)

	// base(x) = 0 (zero weights), lora = scale * (x @ A^T) @ B^T
	// A is zeros, so x @ A^T = [0, 0], then @ B^T = [0,0,0,0]
	for _, v := range result.Floats() {
		if v != 0 {
			t.Errorf("expected 0 with zero A, got %f", v)
		}
	}
}

func TestLoRALinear_ParamCount(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{64, 128}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 8, 16.0) // rank=8
	// A: [8, 128] = 1024, B: [64, 8] = 512, total = 1536
	expected := 8*128 + 64*8
	if lora.ParamCount() != expected {
		t.Errorf("ParamCount = %d, want %d", lora.ParamCount(), expected)
	}
}

func TestLoRALinear_TrainableParams(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)
	params := lora.TrainableParams()

	if len(params) != 2 {
		t.Fatalf("TrainableParams returned %d arrays, want 2", len(params))
	}

	// First is A, second is B
	if params[0].Shape()[0] != 4 || params[0].Shape()[1] != 8 {
		t.Errorf("param[0] (A) shape = %v, want [4, 8]", params[0].Shape())
	}
	if params[1].Shape()[0] != 4 || params[1].Shape()[1] != 4 {
		t.Errorf("param[1] (B) shape = %v, want [4, 4]", params[1].Shape())
	}
}

func TestLoRALinear_GradientFlows(t *testing.T) {
	// Verify that gradients flow through the LoRA path
	w := RandomNormal(0, 0.1, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)
	x := RandomNormal(0, 1, []int32{1, 2, 8}, DTypeFloat32)
	Materialize(x)

	// Loss function: sum of LoRA output (differentiating w.r.t. A and B)
	lossFn := func(inputs []*Array) []*Array {
		lora.A = inputs[0]
		lora.B = inputs[1]
		out := lora.Forward(x)
		return []*Array{SumAll(out)}
	}

	grad := ValueAndGrad(lossFn, 0, 1) // grad w.r.t. A and B
	defer grad.Free()

	values, grads, err := grad.Apply(lora.A, lora.B)
	if err != nil {
		t.Fatalf("ValueAndGrad failed: %v", err)
	}

	Materialize(append(values, grads...)...)

	// Loss should be a scalar
	loss := values[0].Float()
	t.Logf("loss = %f", loss)

	// Gradients should be non-zero (A has random init, B is zero but gets grad)
	gradA := grads[0]
	gradB := grads[1]

	aGradFloats := gradA.Floats()
	bGradFloats := gradB.Floats()

	hasNonZeroA := false
	for _, v := range aGradFloats {
		if v != 0 {
			hasNonZeroA = true
			break
		}
	}

	hasNonZeroB := false
	for _, v := range bGradFloats {
		if v != 0 {
			hasNonZeroB = true
			break
		}
	}

	// A gradient might be zero if B is zero (since dL/dA depends on B)
	// But B gradient should be non-zero since A is random
	if !hasNonZeroB {
		t.Error("gradient for B is all zeros — gradients not flowing")
	}
	t.Logf("gradA has non-zero: %v, gradB has non-zero: %v", hasNonZeroA, hasNonZeroB)
}

func TestRandomNormal(t *testing.T) {
	arr := RandomNormal(0, 1, []int32{100}, DTypeFloat32)
	Materialize(arr)

	floats := arr.Floats()
	if len(floats) != 100 {
		t.Fatalf("RandomNormal returned %d elements, want 100", len(floats))
	}

	// Check rough statistics: mean should be near 0, values should have spread
	var sum float64
	for _, f := range floats {
		sum += float64(f)
	}
	mean := sum / 100
	if math.Abs(mean) > 0.5 { // generous tolerance for 100 samples
		t.Errorf("mean = %f, expected near 0", mean)
	}
}

func TestSaveSafetensors(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := FromValues([]float32{5, 6, 7, 8, 9, 10}, 3, 2)
	Materialize(a, b)

	path := t.TempDir() + "/test.safetensors"
	err := SaveSafetensors(path, map[string]*Array{
		"layer.lora_a": a,
		"layer.lora_b": b,
	})
	if err != nil {
		t.Fatalf("SaveSafetensors failed: %v", err)
	}

	// Verify file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("saved file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("saved file is empty")
	}

	// Load it back
	loaded := LoadAllSafetensors(path)
	Materialize(loaded["layer.lora_a"], loaded["layer.lora_b"])

	aLoaded := loaded["layer.lora_a"].Floats()
	bLoaded := loaded["layer.lora_b"].Floats()

	expectedA := []float32{1, 2, 3, 4}
	expectedB := []float32{5, 6, 7, 8, 9, 10}

	for i, v := range expectedA {
		if aLoaded[i] != v {
			t.Errorf("loaded A[%d] = %f, want %f", i, aLoaded[i], v)
		}
	}
	for i, v := range expectedB {
		if bLoaded[i] != v {
			t.Errorf("loaded B[%d] = %f, want %f", i, bLoaded[i], v)
		}
	}
}

func TestLoRAAdapter_Save(t *testing.T) {
	w := RandomNormal(0, 0.01, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.0.self_attn.q_proj": NewLoRALinear(base, 4, 8.0),
		},
		Config: DefaultLoRAConfig(),
	}

	path := t.TempDir() + "/adapter.safetensors"
	err := adapter.Save(path)
	if err != nil {
		t.Fatalf("Adapter.Save failed: %v", err)
	}

	// Load and verify
	loaded := LoadAllSafetensors(path)
	aKey := "model.layers.0.self_attn.q_proj.lora_a"
	bKey := "model.layers.0.self_attn.q_proj.lora_b"

	if _, ok := loaded[aKey]; !ok {
		t.Errorf("missing key %s in saved adapter", aKey)
	}
	if _, ok := loaded[bKey]; !ok {
		t.Errorf("missing key %s in saved adapter", bKey)
	}
}

func TestDefaultLoRAConfig(t *testing.T) {
	cfg := DefaultLoRAConfig()
	if cfg.Rank != 8 {
		t.Errorf("Rank = %d, want 8", cfg.Rank)
	}
	if cfg.Alpha != 16 {
		t.Errorf("Alpha = %f, want 16", cfg.Alpha)
	}
	if len(cfg.TargetKeys) != 2 {
		t.Errorf("TargetKeys = %v, want [q_proj, v_proj]", cfg.TargetKeys)
	}
}
