// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestOptim_AdamW_BasicStep_Good(t *testing.T) {
	// Simple test: minimise f(x) = x^2, starting at x=10
	x := FromValue(float32(10.0))
	Materialize(x)

	opt := NewAdamW(0.1)

	for i := range 300 {
		// Gradient of x^2 is 2x
		lossFn := func(inputs []*Array) []*Array {
			p := inputs[0]
			return []*Array{Mul(p, p)}
		}

		grad := ValueAndGrad(lossFn)
		_, grads, err := grad.Apply(x)
		grad.Free()
		if err != nil {
			t.Fatalf("step %d: grad failed: %v", i, err)
		}

		updated := opt.Step([]*Array{x}, grads)
		x = updated[0]
		Materialize(x)
	}

	final := x.Float()
	if math.Abs(final) > 0.5 {
		t.Errorf("after 300 steps, x = %f, want near 0", final)
	}
	t.Logf("final x = %f (started at 10.0)", final)
}

func TestOptim_AdamW_MultiParam_Good(t *testing.T) {
	// Minimise f(x, y) = x^2 + y^2
	x := FromValue(float32(5.0))
	y := FromValue(float32(-3.0))
	Materialize(x, y)

	opt := NewAdamW(0.1)

	for i := range 100 {
		lossFn := func(inputs []*Array) []*Array {
			return []*Array{Add(Mul(inputs[0], inputs[0]), Mul(inputs[1], inputs[1]))}
		}

		grad := ValueAndGrad(lossFn, 0, 1)
		_, grads, err := grad.Apply(x, y)
		grad.Free()
		if err != nil {
			t.Fatalf("step %d failed: %v", i, err)
		}

		updated := opt.Step([]*Array{x, y}, grads)
		x = updated[0]
		y = updated[1]
		Materialize(x, y)
	}

	xFinal := x.Float()
	yFinal := y.Float()
	if math.Abs(xFinal) > 0.1 || math.Abs(yFinal) > 0.1 {
		t.Errorf("x=%f, y=%f, want both near 0", xFinal, yFinal)
	}
	t.Logf("final x=%f, y=%f", xFinal, yFinal)
}

func TestOptim_AdamW_WeightDecay_Good(t *testing.T) {
	// With large weight decay and zero gradient, param should decay toward 0
	x := FromValue(float32(10.0))
	Materialize(x)

	opt := NewAdamW(0.01)
	opt.WeightDecay = 0.5 // aggressive decay

	zeroGrad := FromValue(float32(0.0))
	Materialize(zeroGrad)

	for range 10 {
		updated := opt.Step([]*Array{x}, []*Array{zeroGrad})
		x = updated[0]
		Materialize(x)
	}

	final := x.Float()
	if final >= 10.0 {
		t.Errorf("x = %f, should have decayed from 10.0", final)
	}
	if final <= 0 {
		t.Errorf("x = %f, decayed too much", final)
	}
	t.Logf("after 10 steps with weight_decay=0.5: x = %f (started at 10.0)", final)
}

func TestOptim_AdamW_ConfigExplicitZero_Good(t *testing.T) {
	opt := NewAdamW(&AdamWConfig{
		LearningRate:   1e-4,
		WeightDecay:    0,
		WeightDecaySet: true,
	})
	if opt.LR != 1e-4 {
		t.Fatalf("LR = %f, want 1e-4", opt.LR)
	}
	if opt.WeightDecay != 0 {
		t.Fatalf("WeightDecay = %f, want explicit zero", opt.WeightDecay)
	}
	if opt.Beta1 != 0.9 || opt.Beta2 != 0.999 || opt.Eps != 1e-8 {
		t.Fatalf("defaults not preserved: beta1=%f beta2=%f eps=%f", opt.Beta1, opt.Beta2, opt.Eps)
	}
}

func TestOptim_AdamW_Reset_Good(t *testing.T) {
	opt := NewAdamW(0.01)

	x := FromValue(float32(5.0))
	grad := FromValue(float32(1.0))
	Materialize(x, grad)

	opt.Step([]*Array{x}, []*Array{grad})
	if opt.step != 1 {
		t.Errorf("step = %d, want 1", opt.step)
	}

	opt.Reset()
	if opt.step != 0 {
		t.Errorf("after reset, step = %d, want 0", opt.step)
	}
	if opt.m != nil {
		t.Error("after reset, moments should be nil")
	}
}

func TestOptim_AdamW_ReleasesSupersededMoments_Good(t *testing.T) {
	x := FromValue(float32(2.0))
	grad := FromValue(float32(1.0))
	Materialize(x, grad)

	opt := NewAdamW(0.01)

	first := opt.Step([]*Array{x}, []*Array{grad})
	x1 := first[0]
	firstM := opt.m[0]
	firstV := opt.v[0]
	Materialize(x1, firstM, firstV)

	second := opt.Step([]*Array{x1}, []*Array{grad})
	Materialize(second[0])
	defer Free(x, grad, x1, second[0])

	if firstM.Valid() {
		t.Fatal("first moment buffer should be freed after the next step replaces it")
	}
	if firstV.Valid() {
		t.Fatal("second moment buffer should be freed after the next step replaces it")
	}
}

func TestOptim_AdamW_Reset_ReleasesMoments_Good(t *testing.T) {
	x := FromValue(float32(3.0))
	grad := FromValue(float32(1.0))
	Materialize(x, grad)
	defer Free(x, grad)

	opt := NewAdamW(0.01)
	updated := opt.Step([]*Array{x}, []*Array{grad})
	defer Free(updated...)

	firstM := opt.m[0]
	firstV := opt.v[0]
	Materialize(firstM, firstV)

	opt.Reset()

	if firstM.Valid() {
		t.Fatal("Reset should free the first-moment buffer")
	}
	if firstV.Valid() {
		t.Fatal("Reset should free the second-moment buffer")
	}
}

func TestOptim_AdamW_WithLoRA_Good(t *testing.T) {
	// End-to-end: create LoRA layer, compute gradients, update with AdamW
	w := RandomNormal(0, 0.1, []int32{4, 8}, DTypeFloat32)
	Materialize(w)
	base := NewLinear(w, nil)

	lora := NewLoRALinear(base, 4, 8.0)
	opt := NewAdamW(0.001)

	x := RandomNormal(0, 1, []int32{1, 2, 8}, DTypeFloat32)
	target := RandomNormal(0, 1, []int32{1, 2, 4}, DTypeFloat32)
	Materialize(x, target)

	var initialLoss, finalLoss float64

	for step := range 50 {
		lossFn := func(inputs []*Array) []*Array {
			lora.A = inputs[0]
			lora.B = inputs[1]
			pred := lora.Forward(x)
			return []*Array{MSELoss(pred, target)}
		}

		grad := ValueAndGrad(lossFn, 0, 1)
		values, grads, err := grad.Apply(lora.A, lora.B)
		grad.Free()
		if err != nil {
			t.Fatalf("step %d failed: %v", step, err)
		}

		Materialize(append(values, grads...)...)

		loss := values[0].Float()
		if step == 0 {
			initialLoss = loss
		}
		if step == 49 {
			finalLoss = loss
		}

		updated := opt.Step([]*Array{lora.A, lora.B}, grads)
		lora.A = updated[0]
		lora.B = updated[1]
		Materialize(lora.A, lora.B)
	}

	t.Logf("loss: %.6f -> %.6f", initialLoss, finalLoss)
	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease: %f -> %f", initialLoss, finalLoss)
	}
}

func TestOptim_AdamW_ConfigCtor_Good(t *testing.T) {
	opt := NewAdamW(&AdamWConfig{
		LearningRate: 1e-3,
		Beta1:        0.8,
		Beta2:        0.95,
		Eps:          1e-6,
		WeightDecay:  0.05,
	})
	if opt.LR != 1e-3 {
		t.Fatalf("LR = %f, want 0.001", opt.LR)
	}
	if opt.Beta1 != 0.8 {
		t.Fatalf("Beta1 = %f, want 0.8", opt.Beta1)
	}
	if opt.Beta2 != 0.95 {
		t.Fatalf("Beta2 = %f, want 0.95", opt.Beta2)
	}
	if opt.Eps != 1e-6 {
		t.Fatalf("Eps = %f, want 1e-6", opt.Eps)
	}
	if opt.WeightDecay != 0.05 {
		t.Fatalf("WeightDecay = %f, want 0.05", opt.WeightDecay)
	}
}
