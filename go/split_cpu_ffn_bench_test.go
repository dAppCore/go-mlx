// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the CPU split FFN dequant inner loop. Per Wave 10 lane
// W10-I — `cpuSplitPackedDot` is the FFN projection row dispatcher that
// fires `intermediate` rows per token (MiniMax M2: 1536 rows × 3 projections
// × 62 layers per decoded token). The inner walk through
// `cpuSplitUnpackPackedValue` runs hundreds of millions of times per layer
// for routed-expert 2-bit weights.
//
// Run: go test -bench='BenchmarkCPUSplit' -benchmem -run='^$' ./go
//
//revive:disable-next-line:file-length-limit -- bench file groups closely related fixtures.

package mlx

import (
	"testing"

	infjang "dappco.re/go/inference/quant/jang"
)

// Sinks defeat compiler DCE on bench hot loops.
var (
	cpuSplitBenchUnpackSink uint8
	cpuSplitBenchDotSink    float32
)

// buildCPUSplitPackedMatrix builds a realistic packed weight matrix for the
// given bit width and row/col shape. Scales/biases are non-trivial so the
// inner dequant arithmetic stays representative.
func buildCPUSplitPackedMatrix(b *testing.B, rows, cols, bits, groupSize int) *cpuSplitPackedMatrix {
	b.Helper()
	desc := infjang.PackedTensorDescriptor{
		Name:        "bench.weight",
		Type:        "jangtq",
		Format:      "mxtq",
		Role:        infjang.TensorRoleRoutedExpert,
		Shape:       []uint64{uint64(rows), uint64(cols)},
		Elements:    uint64(rows * cols),
		Bits:        bits,
		GroupSize:   groupSize,
		Groups:      (rows*cols + groupSize - 1) / groupSize,
		PackedBytes: (rows*cols*bits + 7) / 8,
		ScaleCount:  (rows*cols + groupSize - 1) / groupSize,
		BiasCount:   (rows*cols + groupSize - 1) / groupSize,
		BitOrder:    infjang.BitOrderLSB0,
		Encoding:    infjang.EncodingAffine,
	}
	values := make([]uint8, rows*cols)
	mask := uint8((1 << bits) - 1)
	for i := range values {
		values[i] = uint8(i) & mask
	}
	packed, err := infjang.PackQuantizedValues(desc, values)
	if err != nil {
		b.Fatalf("PackQuantizedValues: %v", err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = float32(0.125) + float32(i%7)*float32(0.0078125)
		biases[i] = float32(-1) + float32(i%5)*float32(0.25)
	}
	return &cpuSplitPackedMatrix{
		desc:      desc,
		packed:    packed,
		scales:    scales,
		biases:    biases,
		rows:      rows,
		cols:      cols,
		groupSize: groupSize,
		bits:      bits,
		elements:  uint64(rows * cols),
	}
}

func buildCPUSplitInput(cols int) []float32 {
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32(0.5) + float32(i%17)*float32(0.0625)
	}
	return input
}

// --- cpuSplitUnpackPackedValue: element-by-element bit extraction ---
// MiniMax M2 routed-expert dominant width is 2-bit; attention/shared
// expert wide widths are 8-bit. 4-bit is the JANG_4 family.

func BenchmarkCPUSplitUnpackPackedValue_2bit(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 1, 4096, 2, 64)
	packed := matrix.packed
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchUnpackSink = cpuSplitUnpackPackedValue(packed, i&4095, 2)
	}
}

func BenchmarkCPUSplitUnpackPackedValue_4bit(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 1, 4096, 4, 64)
	packed := matrix.packed
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchUnpackSink = cpuSplitUnpackPackedValue(packed, i&4095, 4)
	}
}

func BenchmarkCPUSplitUnpackPackedValue_8bit(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 1, 4096, 8, 64)
	packed := matrix.packed
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchUnpackSink = cpuSplitUnpackPackedValue(packed, i&4095, 8)
	}
}

// --- cpuSplitPackedDot: fused dequant + dot product over one row ---
// MiniMax M2 row size: hidden=3072 (gate/up out) or intermediate=1536
// (down out). Routed expert weights are 2-bit, attention is 8-bit. Group
// size from JANGTQ profile is 64.

func BenchmarkCPUSplitPackedDot_2bit_Row3072(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 1536, 3072, 2, 64)
	input := buildCPUSplitInput(3072)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchDotSink = cpuSplitPackedDot(input, matrix, i%matrix.rows)
	}
}

func BenchmarkCPUSplitPackedDot_4bit_Row3072(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 1536, 3072, 4, 64)
	input := buildCPUSplitInput(3072)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchDotSink = cpuSplitPackedDot(input, matrix, i%matrix.rows)
	}
}

func BenchmarkCPUSplitPackedDot_8bit_Row3072(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 1536, 3072, 8, 64)
	input := buildCPUSplitInput(3072)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchDotSink = cpuSplitPackedDot(input, matrix, i%matrix.rows)
	}
}

func BenchmarkCPUSplitPackedDot_2bit_Row1536(b *testing.B) {
	matrix := buildCPUSplitPackedMatrix(b, 3072, 1536, 2, 64)
	input := buildCPUSplitInput(1536)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuSplitBenchDotSink = cpuSplitPackedDot(input, matrix, i%matrix.rows)
	}
}
