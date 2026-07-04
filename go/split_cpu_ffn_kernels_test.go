// SPDX-Licence-Identifier: EUPL-1.2

// Pure-CPU tests for split_cpu_ffn_kernels.go — the packed-quant math kernels
// for the CPU split-FFN path. None of these load a model: they build tiny
// synthetic cpuSplitFFNLayer / cpuSplitPackedMatrix fixtures and check the
// dense forward pass, the per-bit-width packed dot products, value unpacking,
// and the numeric helpers. All fixtures use scale=1 / bias=0 over a single
// quant group so the dequantised weight equals the raw quantised code and the
// expected dot products are hand-computable.

package mlx

import (
	"math"
	"testing"
)

const kernelEps32 = 1e-4

func approxEqual32(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return float64(d) <= kernelEps32
}

// packMatrix builds a cpuSplitPackedMatrix with the given codes packed at the
// requested bit width, identity dequant (scale=1, bias=0) over one group.
func packMatrix(t *testing.T, rows, cols, bits int, codes []uint8) *cpuSplitPackedMatrix {
	t.Helper()
	n := rows * cols
	if len(codes) != n {
		t.Fatalf("packMatrix: got %d codes, want rows*cols=%d", len(codes), n)
	}
	var packed []byte
	switch bits {
	case 8:
		packed = make([]byte, n)
		for i, c := range codes {
			packed[i] = c
		}
	case 4:
		packed = make([]byte, (n+1)/2)
		for i, c := range codes {
			if i&1 == 0 {
				packed[i>>1] |= c & 0x0F
			} else {
				packed[i>>1] |= (c & 0x0F) << 4
			}
		}
	case 2:
		packed = make([]byte, (n+3)/4)
		for i, c := range codes {
			packed[i>>2] |= (c & 0x03) << uint((i&3)<<1)
		}
	case 1:
		packed = make([]byte, (n+7)/8)
		for i, c := range codes {
			packed[i>>3] |= (c & 0x01) << uint(i&7)
		}
	case 3:
		// Generic LSB-first bit packing for the non-aligned slow path.
		packed = make([]byte, (n*bits+7)/8)
		bitOffset := 0
		for _, c := range codes {
			v := uint16(c) & uint16((1<<bits)-1)
			rem := bits
			for rem > 0 {
				byteIdx := bitOffset / 8
				shiftIn := bitOffset % 8
				take := 8 - shiftIn
				if rem < take {
					take = rem
				}
				mask := uint16((1 << take) - 1)
				packed[byteIdx] |= byte((v & mask) << uint(shiftIn))
				v >>= take
				rem -= take
				bitOffset += take
			}
		}
	default:
		t.Fatalf("packMatrix: unsupported bits=%d", bits)
	}
	scales := make([]float32, 1)
	biases := make([]float32, 1)
	scales[0] = 1
	biases[0] = 0
	return &cpuSplitPackedMatrix{
		packed:    packed,
		scales:    scales,
		biases:    biases,
		rows:      rows,
		cols:      cols,
		groupSize: n, // one group spans the whole matrix
		bits:      bits,
		elements:  uint64(n),
	}
}

func TestCPUSplitDot_Good(t *testing.T) {
	if got := cpuSplitDot([]float32{1, 2, 3}, []float32{4, 5, 6}); !approxEqual32(got, 32) {
		t.Fatalf("cpuSplitDot = %v, want 32", got)
	}
}

func TestCPUSplitDot_MismatchedLengthsTakeMin_Ugly(t *testing.T) {
	// Longer b is truncated to len(a).
	if got := cpuSplitDot([]float32{2, 3}, []float32{4, 5, 99}); !approxEqual32(got, 23) {
		t.Fatalf("cpuSplitDot(min) = %v, want 23", got)
	}
	if got := cpuSplitDot(nil, []float32{1}); got != 0 {
		t.Fatalf("cpuSplitDot(nil) = %v, want 0", got)
	}
}

func TestCPUSplitProjectRow_Dense_Good(t *testing.T) {
	// 2x3 dense matrix, row 1 = [4,5,6] · input [1,1,1] = 15.
	dense := []float32{1, 2, 3, 4, 5, 6}
	if got := cpuSplitProjectRow([]float32{1, 1, 1}, dense, nil, 1, 3); !approxEqual32(got, 15) {
		t.Fatalf("cpuSplitProjectRow dense = %v, want 15", got)
	}
}

func TestCPUSplitProjectRow_Packed8_Good(t *testing.T) {
	m := packMatrix(t, 2, 3, 8, []uint8{1, 2, 3, 4, 5, 6})
	if got := cpuSplitProjectRow([]float32{1, 1, 1}, nil, m, 1, 3); !approxEqual32(got, 15) {
		t.Fatalf("cpuSplitProjectRow packed8 = %v, want 15", got)
	}
}

func TestCPUSplitPackedDot_AllBitWidths_Good(t *testing.T) {
	input := []float32{1, 1, 1, 1}
	for _, bits := range []int{8, 4, 2, 1, 3} {
		codes := []uint8{1, 0, 1, 1} // sum where present
		want := float32(3)
		if bits == 1 {
			// codes clamp to {0,1}; same here → 3.
			want = 3
		}
		m := packMatrix(t, 1, 4, bits, codes)
		got := cpuSplitPackedDot(input, m, 0)
		if !approxEqual32(got, want) {
			t.Fatalf("cpuSplitPackedDot bits=%d = %v, want %v", bits, got, want)
		}
	}
}

func TestCPUSplitPackedDot_ScaleBias_Good(t *testing.T) {
	// dequant(code) = code*scale + bias. scale=2, bias=1 over one group.
	m := packMatrix(t, 1, 3, 8, []uint8{1, 2, 3})
	m.scales[0] = 2
	m.biases[0] = 1
	// input·dequant = 1*(1*2+1) + 1*(2*2+1) + 1*(3*2+1) = 3+5+7 = 15.
	if got := cpuSplitPackedDot([]float32{1, 1, 1}, m, 0); !approxEqual32(got, 15) {
		t.Fatalf("cpuSplitPackedDot scale/bias = %v, want 15", got)
	}
}

func TestCPUSplitPackedDot_NilOrOutOfRange_Ugly(t *testing.T) {
	if got := cpuSplitPackedDot([]float32{1}, nil, 0); got != 0 {
		t.Fatalf("cpuSplitPackedDot(nil) = %v, want 0", got)
	}
	m := packMatrix(t, 1, 2, 8, []uint8{1, 2})
	if got := cpuSplitPackedDot([]float32{1, 1}, m, 5); got != 0 {
		t.Fatalf("cpuSplitPackedDot(row OOB) = %v, want 0", got)
	}
	if got := cpuSplitPackedDot([]float32{1, 1}, m, -1); got != 0 {
		t.Fatalf("cpuSplitPackedDot(row<0) = %v, want 0", got)
	}
}

func TestCPUSplitPackedDot_ShorterInputTruncates_Ugly(t *testing.T) {
	// Input shorter than cols: only the first len(input) columns contribute.
	m := packMatrix(t, 1, 4, 8, []uint8{10, 20, 30, 40})
	if got := cpuSplitPackedDot([]float32{1, 1}, m, 0); !approxEqual32(got, 30) {
		t.Fatalf("cpuSplitPackedDot(short input) = %v, want 30", got)
	}
}

func TestCPUSplitPackedMatrixValue_Good(t *testing.T) {
	m := packMatrix(t, 1, 3, 8, []uint8{5, 6, 7})
	m.scales[0] = 2
	m.biases[0] = 1
	if got := m.value(1); !approxEqual32(got, 13) { // 6*2+1
		t.Fatalf("value(1) = %v, want 13", got)
	}
}

func TestCPUSplitPackedMatrixValue_Bounds_Ugly(t *testing.T) {
	m := packMatrix(t, 1, 2, 8, []uint8{1, 2})
	if got := m.value(-1); got != 0 {
		t.Fatalf("value(-1) = %v, want 0", got)
	}
	if got := m.value(99); got != 0 {
		t.Fatalf("value(OOB) = %v, want 0", got)
	}
	var nilM *cpuSplitPackedMatrix
	if got := nilM.value(0); got != 0 {
		t.Fatalf("nil.value = %v, want 0", got)
	}
}

func TestCPUSplitUnpackPackedValue_AllPaths_Good(t *testing.T) {
	// 8-bit
	if got := cpuSplitUnpackPackedValue([]byte{0xAB}, 0, 8); got != 0xAB {
		t.Fatalf("unpack 8-bit = %#x, want 0xAB", got)
	}
	// 4-bit: low nibble then high nibble
	if got := cpuSplitUnpackPackedValue([]byte{0x21}, 0, 4); got != 0x1 {
		t.Fatalf("unpack 4-bit even = %#x, want 0x1", got)
	}
	if got := cpuSplitUnpackPackedValue([]byte{0x21}, 1, 4); got != 0x2 {
		t.Fatalf("unpack 4-bit odd = %#x, want 0x2", got)
	}
	// 2-bit: four lanes of 0b11_10_01_00 = 0xE4
	for i, want := range []uint8{0, 1, 2, 3} {
		if got := cpuSplitUnpackPackedValue([]byte{0xE4}, i, 2); got != want {
			t.Fatalf("unpack 2-bit lane %d = %d, want %d", i, got, want)
		}
	}
	// 1-bit: byte 0b10110001 lanes
	want1 := []uint8{1, 0, 0, 0, 1, 1, 0, 1}
	for i, want := range want1 {
		if got := cpuSplitUnpackPackedValue([]byte{0b10110001}, i, 1); got != want {
			t.Fatalf("unpack 1-bit lane %d = %d, want %d", i, got, want)
		}
	}
}

func TestCPUSplitUnpackPackedValue_GenericBitWidth_Ugly(t *testing.T) {
	// 3-bit generic slow path: pack codes 5,2 LSB-first → byte bits
	// 010_101 = 0b00010101 = 0x15; index 0 = 5, index 1 = 2.
	packed := []byte{0x15}
	if got := cpuSplitUnpackPackedValue(packed, 0, 3); got != 5 {
		t.Fatalf("unpack 3-bit idx0 = %d, want 5", got)
	}
	if got := cpuSplitUnpackPackedValue(packed, 1, 3); got != 2 {
		t.Fatalf("unpack 3-bit idx1 = %d, want 2", got)
	}
}

func TestCPUSplitMinInt_Good(t *testing.T) {
	if got := cpuSplitMinInt(3, 7); got != 3 {
		t.Fatalf("cpuSplitMinInt(3,7) = %d, want 3", got)
	}
	if got := cpuSplitMinInt(9, 4); got != 4 {
		t.Fatalf("cpuSplitMinInt(9,4) = %d, want 4", got)
	}
}

func TestCPUSplitSiLU_Good(t *testing.T) {
	// SiLU(0) = 0; SiLU(x) = x*sigmoid(x).
	if got := cpuSplitSiLU(0); got != 0 {
		t.Fatalf("cpuSplitSiLU(0) = %v, want 0", got)
	}
	x := float32(1.5)
	want := x / (1 + float32(math.Exp(float64(-x))))
	if got := cpuSplitSiLU(x); !approxEqual32(got, want) {
		t.Fatalf("cpuSplitSiLU(1.5) = %v, want %v", got, want)
	}
}

func TestCPUSplitFirstPositive_Good(t *testing.T) {
	if got := cpuSplitFirstPositive(0, -1, 8, 2); got != 8 {
		t.Fatalf("cpuSplitFirstPositive = %d, want 8", got)
	}
	if got := cpuSplitFirstPositive(0, -1); got != 0 {
		t.Fatalf("cpuSplitFirstPositive(none) = %d, want 0", got)
	}
}

func TestCPUSplitForwardDenseRow_Good(t *testing.T) {
	// hidden=2, intermediate=2. norm=ones, eps small → RMSNorm scales input
	// by 1/sqrt(mean(h^2)). Use hidden=[3,4] → mean sq = 12.5 → scale ≈ 0.2828.
	hidden := []float32{3, 4}
	layer := cpuSplitFFNLayer{
		norm:         []float32{1, 1},
		gate:         []float32{1, 0, 0, 1}, // identity 2x2
		up:           []float32{1, 1, 1, 1}, // sum 2x2
		down:         []float32{1, 0, 0, 1}, // identity 2x2
		hidden:       2,
		intermediate: 2,
	}
	out := make([]float32, 2)
	normed := make([]float32, 2)
	activated := make([]float32, 2)
	cpuSplitForwardDenseRow(hidden, out, layer, 1e-6, normed, activated)

	// Recompute the reference value independently.
	var sq float64
	for _, v := range hidden {
		sq += float64(v * v)
	}
	scale := float32(1 / math.Sqrt(sq/2+1e-6))
	n0, n1 := hidden[0]*scale, hidden[1]*scale
	gate0, gate1 := n0, n1   // identity
	up0, up1 := n0+n1, n0+n1 // sum rows
	act0 := cpuSplitSiLU(gate0) * up0
	act1 := cpuSplitSiLU(gate1) * up1
	down0 := act0 // identity → row0 picks act0
	down1 := act1
	want0 := hidden[0] + down0
	want1 := hidden[1] + down1
	if !approxEqual32(out[0], want0) || !approxEqual32(out[1], want1) {
		t.Fatalf("cpuSplitForwardDenseRow out = %v, want [%v %v]", out, want0, want1)
	}
}

func TestCPUSplitForwardDenseRow_WithBias_Good(t *testing.T) {
	// Exercises the gate/up/down bias-present branches.
	hidden := []float32{1, 1}
	layer := cpuSplitFFNLayer{
		norm:         []float32{1, 1},
		gate:         []float32{1, 0, 0, 1},
		gateBias:     []float32{0.5, 0.5},
		up:           []float32{1, 0, 0, 1},
		upBias:       []float32{1, 1},
		down:         []float32{1, 0, 0, 1},
		downBias:     []float32{2, 2},
		hidden:       2,
		intermediate: 2,
	}
	out := make([]float32, 2)
	normed := make([]float32, 2)
	activated := make([]float32, 2)
	cpuSplitForwardDenseRow(hidden, out, layer, 1e-6, normed, activated)
	// Just assert the bias path ran and produced finite, residual-added output.
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("cpuSplitForwardDenseRow with bias out[%d] = %v, want finite", i, v)
		}
		if v <= hidden[i] { // downBias=2 guarantees out > residual input
			t.Fatalf("cpuSplitForwardDenseRow with bias out[%d] = %v, want > residual %v", i, v, hidden[i])
		}
	}
}
