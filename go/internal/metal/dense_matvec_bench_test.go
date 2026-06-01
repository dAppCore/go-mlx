// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// BenchmarkDenseMatVec_NativeLinear_Decode measures the single-token dense
// quantized matvec path used by Gemma 4 small-model q6 default projections.
func BenchmarkDenseMatVec_NativeLinear_Decode(b *testing.B) {
	requireMetalRuntime(b)

	for _, tc := range []struct {
		name string
		bits int
	}{
		{name: "Q4", bits: 4},
		{name: "Q6ProductDefault", bits: 6},
		{name: "Q8", bits: 8},
	} {
		b.Run(tc.name, func(b *testing.B) {
			const (
				inDim     = 320
				outDim    = 256
				groupSize = 64
			)
			inputValues := make([]float32, inDim)
			for i := range inputValues {
				inputValues[i] = -1.5 + 0.03125*float32((i*7)%97)
			}
			fixture := quantizedLinearDenseMatVecFixture(b, outDim, inDim, groupSize, tc.bits, 19)
			linear := fixture.linear
			denseMatVecSidecarsAsType(linear, DTypeBFloat16)
			defer freeLinear(linear)

			x := FromValues(inputValues, 1, 1, inDim)
			defer Free(x)
			Materialize(x, linear.Weight, linear.Scales, linear.Biases)

			warm, ok, err := quantizedDenseMatVec(x, linear)
			if err != nil {
				b.Fatalf("warmup quantizedDenseMatVec(q%d): %v", tc.bits, err)
			}
			if !ok {
				b.Fatalf("warmup quantizedDenseMatVec(q%d) ok = false", tc.bits)
			}
			Materialize(warm)
			Free(warm)

			packedWeightBytes := int64((outDim*inDim*tc.bits + 7) / 8)
			sidecarBytes := int64(2 * outDim * (inDim / groupSize) * 2)
			b.SetBytes(packedWeightBytes + sidecarBytes)
			b.ReportAllocs()
			for b.Loop() {
				out, ok, err := quantizedDenseMatVec(x, linear)
				if err != nil {
					b.Fatalf("quantizedDenseMatVec(q%d): %v", tc.bits, err)
				}
				if !ok {
					b.Fatalf("quantizedDenseMatVec(q%d) ok = false", tc.bits)
				}
				Materialize(out)
				Free(out)
			}
		})
	}
}
