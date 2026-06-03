// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// BenchmarkDenseMatVec_NativeLinear_Decode measures the single-token native
// dense quantized matvec path for q4/q6/q8 packed projection shapes.
func BenchmarkDenseMatVec_NativeLinear_Decode(b *testing.B) {
	requireMetalRuntime(b)

	for _, tc := range []struct {
		name      string
		bits      int
		bitstream bool
	}{
		{name: "Q4", bits: 4},
		{name: "Q6NativeBitstream", bits: 6, bitstream: true},
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
			defer FreeLinear(linear)
			if tc.bitstream {
				restoreQ6 := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "1")
				defer restoreQ6()
			}

			x := FromValues(inputValues, 1, 1, inDim)
			defer Free(x)
			Materialize(x, linear.Weight, linear.Scales, linear.Biases)

			warm, ok, err := QuantizedDenseMatVec(x, linear)
			if err != nil {
				b.Fatalf("warmup QuantizedDenseMatVec(q%d): %v", tc.bits, err)
			}
			if !ok {
				b.Fatalf("warmup QuantizedDenseMatVec(q%d) ok = false", tc.bits)
			}
			Materialize(warm)
			Free(warm)

			packedWeightBytes := int64((outDim*inDim*tc.bits + 7) / 8)
			sidecarBytes := int64(2 * outDim * (inDim / groupSize) * 2)
			b.SetBytes(packedWeightBytes + sidecarBytes)
			b.ReportAllocs()
			for b.Loop() {
				out, ok, err := QuantizedDenseMatVec(x, linear)
				if err != nil {
					b.Fatalf("QuantizedDenseMatVec(q%d): %v", tc.bits, err)
				}
				if !ok {
					b.Fatalf("QuantizedDenseMatVec(q%d) ok = false", tc.bits)
				}
				Materialize(out)
				Free(out)
			}
		})
	}
}

// BenchmarkDenseMatVec_NativeLinear_E2BOutputSlice measures the product-lane
// single-token output-projection shape on a bounded vocab slice. The full E2B
// tied output is [262144, 1536]; the 16k-row slice keeps the benchmark safe
// while preserving the q4/q6/q8 packed-row width and memory-access pattern.
func BenchmarkDenseMatVec_NativeLinear_E2BOutputSlice(b *testing.B) {
	requireMetalRuntime(b)

	for _, tc := range []struct {
		name      string
		bits      int
		bitstream bool
	}{
		{name: "Q4", bits: 4},
		{name: "Q6NativeBitstream", bits: 6, bitstream: true},
		{name: "Q8", bits: 8},
	} {
		b.Run(tc.name, func(b *testing.B) {
			const (
				inDim     = 1536
				outDim    = 16384
				groupSize = 64
			)
			inputValues := make([]float32, inDim)
			for i := range inputValues {
				inputValues[i] = -1.25 + 0.03125*float32((i*11)%89)
			}
			fixture := quantizedLinearDenseMatVecFixture(b, outDim, inDim, groupSize, tc.bits, 31)
			linear := fixture.linear
			denseMatVecSidecarsAsType(linear, DTypeBFloat16)
			defer FreeLinear(linear)
			if tc.bitstream {
				restoreQ6 := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "1")
				defer restoreQ6()
			}

			x := FromValues(inputValues, 1, 1, inDim)
			defer Free(x)
			Materialize(x, linear.Weight, linear.Scales, linear.Biases)

			warm, ok, err := QuantizedDenseMatVec(x, linear)
			if err != nil {
				b.Fatalf("warmup QuantizedDenseMatVec(q%d): %v", tc.bits, err)
			}
			if !ok {
				b.Fatalf("warmup QuantizedDenseMatVec(q%d) ok = false", tc.bits)
			}
			Materialize(warm)
			Free(warm)

			packedWeightBytes := int64(outDim * quantizedDenseMatVecPackedIn(inDim, tc.bits) * 4)
			sidecarBytes := int64(2 * outDim * (inDim / groupSize) * 2)
			b.SetBytes(packedWeightBytes + sidecarBytes)
			b.ReportAllocs()
			for b.Loop() {
				out, ok, err := QuantizedDenseMatVec(x, linear)
				if err != nil {
					b.Fatalf("QuantizedDenseMatVec(q%d): %v", tc.bits, err)
				}
				if !ok {
					b.Fatalf("QuantizedDenseMatVec(q%d) ok = false", tc.bits)
				}
				Materialize(out)
				Free(out)
			}
		})
	}
}

// BenchmarkDenseMatVec_Q6FallbackVsBitstream_E2BShapes compares the current
// q6 default fallback with the opt-in native q6 bitstream kernel on product
// E2B-sized single-token shapes. This keeps the q6 default decision tied to
// measured whole-run suspects: internal projections, MLP projections, and the
// large tied output head.
func BenchmarkDenseMatVec_Q6FallbackVsBitstream_E2BShapes(b *testing.B) {
	requireMetalRuntime(b)

	for _, shape := range []struct {
		name   string
		inDim  int
		outDim int
	}{
		{name: "HiddenProjection", inDim: 1536, outDim: 1536},
		{name: "MLPProjection", inDim: 1536, outDim: 6144},
		{name: "OutputHeadSlice", inDim: 1536, outDim: 16384},
	} {
		b.Run(shape.name, func(b *testing.B) {
			for _, mode := range []struct {
				name      string
				bitstream string
			}{
				{name: "Fallback", bitstream: "0"},
				{name: "Bitstream", bitstream: "1"},
			} {
				b.Run(mode.name, func(b *testing.B) {
					const (
						bits      = 6
						groupSize = 64
					)
					inputValues := make([]float32, shape.inDim)
					for i := range inputValues {
						inputValues[i] = -1.25 + 0.03125*float32((i*11)%89)
					}
					fixture := quantizedLinearDenseMatVecFixture(b, shape.outDim, shape.inDim, groupSize, bits, 41)
					linear := fixture.linear
					denseMatVecSidecarsAsType(linear, DTypeBFloat16)
					defer FreeLinear(linear)

					x := FromValues(inputValues, 1, 1, shape.inDim)
					defer Free(x)
					Materialize(x, linear.Weight, linear.Scales, linear.Biases)

					restoreNative := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC", "1")
					restoreQ6 := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", mode.bitstream)
					defer restoreQ6()
					defer restoreNative()

					warm := linear.baseForward(x)
					Materialize(warm)
					Free(warm)

					packedWeightBytes := int64(shape.outDim * quantizedDenseMatVecPackedIn(shape.inDim, bits) * 4)
					sidecarBytes := int64(2 * shape.outDim * (shape.inDim / groupSize) * 2)
					b.SetBytes(packedWeightBytes + sidecarBytes)
					b.ReportAllocs()
					for b.Loop() {
						out := linear.baseForward(x)
						Materialize(out)
						Free(out)
					}
				})
			}
		})
	}
}
