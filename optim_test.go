//go:build darwin && arm64

package mlx

import (
	"math"
	"testing"
)

func TestAdamW_BasicStep(t *testing.T) {
	// Simple test: minimise f(x) = x^2, starting at x=10
	x := FromValue(float32(10.0))
	Materialize(x)

	opt := NewAdamW(0.1)

	for i := 0; i < 300; i++ {
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

func TestAdamW_MultiParam(t *testing.T) {
	// Minimise f(x, y) = x^2 + y^2
	x := FromValue(float32(5.0))
	y := FromValue(float32(-3.0))
	Materialize(x, y)

	opt := NewAdamW(0.1)

	for i := 0; i < 100; i++ {
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

func TestAdamW_WeightDecay(t *testing.T) {
	// With large weight decay and zero gradient, param should decay toward 0
	x := FromValue(float32(10.0))
	Materialize(x)

	opt := NewAdamW(0.01)
	opt.WeightDecay = 0.5 // aggressive decay

	zeroGrad := FromValue(float32(0.0))
	Materialize(zeroGrad)

	for i := 0; i < 10; i++ {
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

func TestAdamW_Reset(t *testing.T) {
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

func TestAdamW_WithLoRA(t *testing.T) {
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

	for step := 0; step < 50; step++ {
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
