// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"testing"
	"unsafe"

	mlxmetal "dappco.re/go/mlx/pkg/metal"
)

// TestQMVIntoReusesOutputBackingAndMatchesQMV and TestQMVMatchesMetalQuantizedMatmul need the real
// cgo metal package as their oracle (mlxmetal.Quantize / mlxmetal.QuantizedMatmul), so they're
// gated behind metal_runtime. The rest of qmv_test.go (kernel-name caching, allocation budgets,
// scratch-pool residency, output-buffer reuse) is hermetic and stays untagged.

func TestQMVIntoReusesOutputBackingAndMatchesQMV(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, groupSize, bits = 16, 64, 32, 4
	w := syntheticFloat32(outDim*inDim, 17)
	x := syntheticFloat32(inDim, 5)
	wArr := mlxmetal.FromValues(w, outDim, inDim)
	wq, scales, biases, err := mlxmetal.Quantize(wArr, groupSize, bits, "affine")
	if err != nil {
		mlxmetal.Free(wArr)
		t.Fatalf("Quantize: %v", err)
	}
	mlxmetal.Materialize(wq, scales, biases)
	defer mlxmetal.Free(wArr, wq, scales, biases)

	want, err := QMV(x, wq.RawBytes(), scales.RawBytes(), biases.RawBytes(), outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMV reference: %v", err)
	}
	out := make([]float32, outDim)
	outPtr := unsafe.Pointer(&out[0])

	got, err := QMVInto(out, x, wq.RawBytes(), scales.RawBytes(), biases.RawBytes(), outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("QMVInto did not reuse caller-owned output backing")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("QMVInto[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func TestQMVMatchesMetalQuantizedMatmul(t *testing.T) {
	requireNativeRuntime(t)
	tests := []struct {
		name                 string
		outDim, inDim, gs, b int
	}{
		{name: "regular", outDim: 16, inDim: 64, gs: 32, b: 4},
		{name: "fast", outDim: 8, inDim: 512, gs: 64, b: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := syntheticFloat32(tt.outDim*tt.inDim, 17)
			x := syntheticFloat32(tt.inDim, 23)
			wArr := mlxmetal.FromValues(w, tt.outDim, tt.inDim)
			wq, scales, biases, err := mlxmetal.Quantize(wArr, tt.gs, tt.b, "affine")
			if err != nil {
				mlxmetal.Free(wArr)
				t.Fatalf("Quantize: %v", err)
			}
			mlxmetal.Materialize(wq, scales, biases)
			xArr := mlxmetal.FromValues(x, 1, tt.inDim)
			res := mlxmetal.QuantizedMatmul(xArr, wq, scales, biases, true, tt.gs, tt.b)
			mlxmetal.Materialize(res)
			want := res.Floats()

			got, err := QMV(x, wq.RawBytes(), scales.RawBytes(), biases.RawBytes(), tt.outDim, tt.inDim, tt.gs, tt.b)
			mlxmetal.Free(wArr, wq, scales, biases, xArr, res)
			if err != nil {
				t.Fatalf("QMV: %v", err)
			}
			if len(got) != len(want) {
				t.Fatalf("QMV length = %d, want %d", len(got), len(want))
			}
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 2e-5 {
					t.Fatalf("QMV[%d] = %v, want %v (diff %.3g)", i, got[i], want[i], diff)
				}
			}
		})
	}
}
