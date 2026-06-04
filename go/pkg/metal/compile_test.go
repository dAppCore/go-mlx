// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Generated file-aware compliance coverage.
func TestCompile_CompileShapeless_Good(t *testing.T) {
	x := FromValues([]float32{1, 2, 3}, 3)
	defer Free(x)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{AddScalar(inputs[0], 1)}
	}, true)
	if compiled == nil || !compiled.Valid() {
		t.Fatal("CompileShapeless returned an invalid compiled closure")
	}
	defer compiled.Free()
	y := compiled.Call(x)[0]
	defer Free(y)
	if err := Eval(y); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	floatSliceApprox(t, y.Floats(), []float32{2, 3, 4})
}

func TestCompile_CompiledFunc_Call_Good(t *testing.T) {
	x := FromValues([]float32{2, 4}, 2)
	defer Free(x)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{MulScalar(inputs[0], 0.5)}
	}, false)
	defer compiled.Free()
	y := compiled.Call(x)[0]
	defer Free(y)
	if err := Eval(y); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	floatSliceApprox(t, y.Floats(), []float32{1, 2})
}

func TestCompile_CompiledFunc_CallOne_Good(t *testing.T) {
	x := FromValues([]float32{2, 4}, 2)
	defer Free(x)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{MulScalar(inputs[0], 0.25)}
	}, false)
	defer compiled.Free()
	y := compiled.CallOne(x)
	defer Free(y)
	if err := Eval(y); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	floatSliceApprox(t, y.Floats(), []float32{0.5, 1})
}

func TestCompile_GELUGateMul_Good(t *testing.T) {
	gate := FromValues([]float32{0, 1}, 2)
	up := FromValues([]float32{2, 3}, 2)
	defer Free(gate, up)
	got := GeluGateMul(gate, up)
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	want := Mul(geluApprox(gate), up)
	defer Free(want)
	if err := Eval(want); err != nil {
		t.Fatalf("Eval want: %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestCompile_GELUGateMul_NativeGateGood(t *testing.T) {
	target := "GeluGateMul native gate"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	old := enableNativeGELUGateMul
	enableNativeGELUGateMul = true
	t.Cleanup(func() { enableNativeGELUGateMul = old })

	gate := FromValues([]float32{0, 1}, 2)
	up := FromValues([]float32{2, 3}, 2)
	defer Free(gate, up)
	got := GeluGateMul(gate, up)
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	want := Mul(geluApprox(gate), up)
	defer Free(want)
	if err := Eval(want); err != nil {
		t.Fatalf("Eval want: %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestCompile_SiLUGateMul_Good(t *testing.T) {
	gate := FromValues([]float32{0, 1}, 2)
	up := FromValues([]float32{2, 3}, 2)
	defer Free(gate, up)
	got := SiluGateMul(gate, up)
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	want := Mul(SiLU(gate), up)
	defer Free(want)
	if err := Eval(want); err != nil {
		t.Fatalf("Eval want: %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}
