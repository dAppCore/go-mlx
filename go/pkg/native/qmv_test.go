// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"

	mlxmetal "dappco.re/go/mlx/pkg/metal"
)

func TestQMVBF16KernelNameCachesGeometryString(t *testing.T) {
	names := []string{
		qmvBF16KernelName(32768, 128, 64, 4),
		qmvBF16KernelName(32768, 128, 64, 4),
	}
	if names[0] != names[1] {
		t.Fatalf("qmv kernel names differ: %q vs %q", names[0], names[1])
	}
	if unsafe.StringData(names[0]) != unsafe.StringData(names[1]) {
		t.Fatalf("qmv kernel name backing was not cached for repeated geometry")
	}
}

func TestQMVBF16KernelNameReusesEquivalentKernelString(t *testing.T) {
	tests := []struct {
		name                 string
		outA, inA, outB, inB int
	}{
		{name: "regular", outA: 64, inA: 128, outB: 128, inB: 128},
		{name: "fast", outA: 64, inA: 512, outB: 128, inB: 512},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := qmvBF16KernelName(tt.outA, tt.inA, 64, 4)
			b := qmvBF16KernelName(tt.outB, tt.inB, 64, 4)
			if a != b {
				t.Fatalf("qmv bf16 equivalent kernel names differ: %q vs %q", a, b)
			}
			if unsafe.StringData(a) != unsafe.StringData(b) {
				t.Fatalf("qmv bf16 equivalent kernel name backing was not shared")
			}
		})
	}
}

func TestQMVKernelNameCachesGeometryString(t *testing.T) {
	names := []string{
		qmvKernelName(32768, 128, 64, 4),
		qmvKernelName(32768, 128, 64, 4),
	}
	if names[0] != names[1] {
		t.Fatalf("qmv float kernel names differ: %q vs %q", names[0], names[1])
	}
	if unsafe.StringData(names[0]) != unsafe.StringData(names[1]) {
		t.Fatalf("qmv float kernel name backing was not cached for repeated geometry")
	}
}

func TestQMVKernelNameReusesEquivalentKernelString(t *testing.T) {
	tests := []struct {
		name                 string
		outA, inA, outB, inB int
	}{
		{name: "regular", outA: 64, inA: 128, outB: 128, inB: 128},
		{name: "fast", outA: 64, inA: 512, outB: 128, inB: 512},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := qmvKernelName(tt.outA, tt.inA, 64, 4)
			b := qmvKernelName(tt.outB, tt.inB, 64, 4)
			if a != b {
				t.Fatalf("qmv float equivalent kernel names differ: %q vs %q", a, b)
			}
			if unsafe.StringData(a) != unsafe.StringData(b) {
				t.Fatalf("qmv float equivalent kernel name backing was not shared")
			}
		})
	}
}

func TestQMVBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	if _, err := QMVBF16(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
		t.Fatalf("QMVBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := QMVBF16(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			t.Fatalf("QMVBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("QMVBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestQMVBF16IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	want, err := QMVBF16(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16 reference: %v", err)
	}
	out := make([]byte, outDim*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := QMVBF16Into(out, x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16Into: %v", err)
	}
	if len(got) != outDim*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("QMVBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "QMVBF16Into", got, want)

	scratch, err = getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("QMVBF16Into wrote through pooled scratch output instead of caller output")
	}
}

func TestQMVBF16ResidentIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	want, err := qmvBF16Resident(x, qw, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("qmvBF16Resident reference: %v", err)
	}
	out := make([]byte, outDim*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x3c}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := qmvBF16ResidentInto(out, x, qw, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("qmvBF16ResidentInto: %v", err)
	}
	if len(got) != outDim*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("qmvBF16ResidentInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "qmvBF16ResidentInto", got, want)

	scratch, err = getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("qmvBF16ResidentInto wrote through pooled scratch output instead of caller output")
	}
}

func TestQMVBF16ScratchPoolKeepsAlternatingDimensions(t *testing.T) {
	requireNativeRuntime(t)

	a, err := getQMVBF16Scratch(64, 64)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch a: %v", err)
	}
	putQMVBF16Scratch(a)
	b, err := getQMVBF16Scratch(8, 64)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch b: %v", err)
	}
	putQMVBF16Scratch(b)

	gotA, err := getQMVBF16Scratch(64, 64)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch gotA: %v", err)
	}
	defer putQMVBF16Scratch(gotA)
	if gotA != a {
		t.Fatal("QMV BF16 scratch pool did not preserve the 64x64 scratch across an alternating dimension")
	}
	gotB, err := getQMVBF16Scratch(8, 64)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch gotB: %v", err)
	}
	defer putQMVBF16Scratch(gotB)
	if gotB != b {
		t.Fatal("QMV BF16 scratch pool did not preserve the 8x64 scratch across an alternating dimension")
	}
}

func TestQMVAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 3)
	x := syntheticFloat32(inDim, 5)
	if _, err := QMV(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
		t.Fatalf("QMV warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := QMV(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			t.Fatalf("QMV: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("QMV allocations = %.0f, want <= 10", allocs)
	}
}

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
