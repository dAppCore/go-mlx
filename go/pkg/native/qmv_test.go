// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mlxmetal "dappco.re/go/mlx/pkg/metal"
)

func TestQMVZeroSizedProjection(t *testing.T) {
	requireNativeRuntime(t)

	got, err := QMV(nil, nil, nil, nil, 0, 0, 64, 4)
	if err != nil {
		t.Fatalf("QMV zero-sized projection: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("QMV zero-sized projection length = %d, want 0", len(got))
	}

	gotBF16, err := QMVBF16(nil, nil, nil, nil, 0, 0, 64, 4)
	if err != nil {
		t.Fatalf("QMVBF16 zero-sized projection: %v", err)
	}
	if len(gotBF16) != 0 {
		t.Fatalf("QMVBF16 zero-sized projection length = %d, want 0", len(gotBF16))
	}
}

func TestQMVRejectsInputShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := QMV([]float32{1}, nil, nil, nil, 0, 2, 64, 4); err == nil {
		t.Fatal("expected QMV to reject len(x) != inDim")
	}
	if _, err := QMVBF16([]byte{0}, nil, nil, nil, 0, 1, 64, 4); err == nil {
		t.Fatal("expected QMVBF16 to reject len(x) != inDim*2")
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
