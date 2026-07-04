// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"math"
)

// split_cpu_ffn_kernels.go: the CPU packed-quant math kernels for split FFN — the
// dense-row forward pass, the dot-product variants (packed 8/4/2/1-bit), packed-value
// unpacking, and the small numeric helpers (SiLU, minInt, firstPositive).

func cpuSplitForwardDenseRow(hidden, out []float32, layer cpuSplitFFNLayer, eps float32, normed, activated []float32) {
	// Cache loop bounds + bias-presence checks before the inner loops. The
	// intermediate loop typically runs ~14336 iterations per token; re-doing
	// the len(layer.*Bias) > 0 check each pass shows up under perf.
	hiddenLen := layer.hidden
	intermediateLen := layer.intermediate
	hasGateBias := len(layer.gateBias) > 0
	hasUpBias := len(layer.upBias) > 0
	hasDownBias := len(layer.downBias) > 0

	var squares float64
	for _, value := range hidden {
		squares += float64(value * value)
	}
	scale := float32(1 / math.Sqrt(squares/float64(hiddenLen)+float64(eps)))
	// Re-slice all three views to hiddenLen up-front so the per-element
	// indexing has its bounds proved at the slice header — the compiler
	// can then drop the bounds checks on normed/hidden/layer.norm reads
	// in the inner loop.
	normedView := normed[:hiddenLen]
	hiddenView := hidden[:hiddenLen]
	normView := layer.norm[:hiddenLen]
	for i := range hiddenLen {
		normedView[i] = hiddenView[i] * scale * normView[i]
	}

	// Hoist the projection-weight slice headers + packed-matrix pointers
	// into locals before the row walks. The row loop ran ~intermediate
	// passes per token and each pass re-loaded gate/up/down slice headers
	// (and their packed-matrix counterparts) off the cpuSplitFFNLayer
	// struct in argument position; pulling them to registers up-front lets
	// the per-row call use a local instead.
	gateDense := layer.gate
	upDense := layer.up
	downDense := layer.down
	gatePacked := layer.gatePacked
	upPacked := layer.upPacked
	downPacked := layer.downPacked

	// Re-slice bias arrays + activated buffer to the loop bounds so the
	// per-row indexing in the projection-and-bias-fold loops compiles
	// without per-iter bounds checks. Loader keeps these matched to
	// intermediate/hidden sizes already, so the slice is exactly correct.
	activatedView := activated[:intermediateLen]
	var gateBiasView, upBiasView []float32
	if hasGateBias {
		gateBiasView = layer.gateBias[:intermediateLen]
	}
	if hasUpBias {
		upBiasView = layer.upBias[:intermediateLen]
	}
	for row := range intermediateLen {
		gate := cpuSplitProjectRow(normed, gateDense, gatePacked, row, hiddenLen)
		up := cpuSplitProjectRow(normed, upDense, upPacked, row, hiddenLen)
		if hasGateBias {
			gate += gateBiasView[row]
		}
		if hasUpBias {
			up += upBiasView[row]
		}
		activatedView[row] = cpuSplitSiLU(gate) * up
	}

	outView := out[:hiddenLen]
	hiddenViewRes := hidden[:hiddenLen]
	var downBiasView []float32
	if hasDownBias {
		downBiasView = layer.downBias[:hiddenLen]
	}
	for row := range hiddenLen {
		mlp := cpuSplitProjectRow(activated, downDense, downPacked, row, intermediateLen)
		if hasDownBias {
			mlp += downBiasView[row]
		}
		outView[row] = hiddenViewRes[row] + mlp
	}
}

func cpuSplitDot(a, b []float32) float32 {
	// Re-slice b to len(a) so the compiler can prove every b[i] is in
	// bounds when walking the indexed loop. Without the hint, each b[i]
	// triggers a per-iteration bounds check that dominates the inner dot
	// when len(a) is in the thousands (the projection row size).
	n := min(len(b), len(a))
	a = a[:n]
	b = b[:n]
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func cpuSplitProjectRow(input, dense []float32, packed *cpuSplitPackedMatrix, row, cols int) float32 {
	if packed != nil {
		return cpuSplitPackedDot(input, packed, row)
	}
	offset := row * cols
	return cpuSplitDot(input, dense[offset:offset+cols])
}

func cpuSplitPackedDot(input []float32, matrix *cpuSplitPackedMatrix, row int) float32 {
	if matrix == nil || row < 0 || row >= matrix.rows {
		return 0
	}
	// Hoist the loop bound: the original double-condition (col < matrix.cols
	// && col < len(input)) re-read both sources every iteration. min() once,
	// then a single-bound loop lets the compiler elide bounds checks on the
	// input slice when col stays under len(input).
	cols := matrix.cols
	if n := len(input); n < cols {
		cols = n
	}
	offset := row * matrix.cols
	in := input[:cols]
	// Hoist hot fields from matrix once — the per-element value() call
	// would chase each of these through the struct (and through the desc
	// for groupSize/bits/elements) on every element of every projection
	// row. With ~hidden_size elements per row and ~intermediate rows per
	// token, that ran into the billions per layer.
	//
	// matrix.elements equals matrix.rows * matrix.cols by construction
	// (PackedTensorDescriptor.Elements is the product of shape dims set in
	// NewPackedTensorDescriptor from []uint64{rows, cols}). With the row
	// bound check at the top of the function and col < cols <= matrix.cols
	// inside the loop, every idx is provably under elements, so the per-
	// element guard from the original (*cpuSplitPackedMatrix).value path
	// drops out entirely.
	packed := matrix.packed
	scales := matrix.scales
	biases := matrix.biases
	groupSize := matrix.groupSize
	bits := matrix.bits
	// Hoist scale/bias per group rather than re-indexing scales[idx/groupSize]
	// each iteration. The group boundary changes once every groupSize
	// elements; the inner loop runs `groupSize` elements with two constants.
	// This trades one integer division + two slice reads per element for one
	// integer division + two slice reads per group. With groupSize=64
	// (JANGTQ default), that is a 64x reduction in division work.
	//
	// Dispatch by bit-width once outside the loop so the inner unpack
	// becomes a single shift+mask the Go compiler can keep in registers,
	// instead of paying the un-inlinable cpuSplitUnpackPackedValue call
	// (cost 161 > inline budget 80) every element.
	switch bits {
	case 8:
		return cpuSplitPackedDot8(in, packed, scales, biases, offset, cols, groupSize)
	case 4:
		return cpuSplitPackedDot4(in, packed, scales, biases, offset, cols, groupSize)
	case 2:
		return cpuSplitPackedDot2(in, packed, scales, biases, offset, cols, groupSize)
	case 1:
		return cpuSplitPackedDot1(in, packed, scales, biases, offset, cols, groupSize)
	}
	var sum float32
	col := 0
	for col < cols {
		idx := offset + col
		group := idx / groupSize
		groupEnd := (group + 1) * groupSize
		end := min(groupEnd-offset, cols)
		scale := scales[group]
		bias := biases[group]
		for ; col < end; col++ {
			q := cpuSplitUnpackPackedValue(packed, offset+col, bits)
			sum += in[col] * (float32(q)*scale + bias)
		}
	}
	return sum
}

// cpuSplitPackedDot8 walks the 8-bit-aligned packed weight path with the
// unpack inlined. One byte per element, no shift required.
func cpuSplitPackedDot8(in []float32, packed []byte, scales, biases []float32, offset, cols, groupSize int) float32 {
	var sum float32
	col := 0
	for col < cols {
		idx := offset + col
		group := idx / groupSize
		groupEnd := (group + 1) * groupSize
		end := min(groupEnd-offset, cols)
		scale := scales[group]
		bias := biases[group]
		for ; col < end; col++ {
			sum += in[col] * (float32(packed[offset+col])*scale + bias)
		}
	}
	return sum
}

// cpuSplitPackedDot4 walks the 4-bit-nibble-packed weight path with the
// unpack inlined. Two values per byte; low nibble for even indices, high
// nibble for odd indices.
func cpuSplitPackedDot4(in []float32, packed []byte, scales, biases []float32, offset, cols, groupSize int) float32 {
	var sum float32
	col := 0
	for col < cols {
		idx := offset + col
		group := idx / groupSize
		groupEnd := (group + 1) * groupSize
		end := min(groupEnd-offset, cols)
		scale := scales[group]
		bias := biases[group]
		for ; col < end; col++ {
			b := packed[(offset+col)>>1]
			var q uint8
			if (offset+col)&1 == 0 {
				q = b & 0x0F
			} else {
				q = b >> 4
			}
			sum += in[col] * (float32(q)*scale + bias)
		}
	}
	return sum
}

// cpuSplitPackedDot2 walks the 2-bit-packed weight path with the unpack
// inlined. Four values per byte; the shift is `((index)&3)<<1`. This is
// the dominant MiniMax M2 routed-expert weight path.
//
// When the per-group walk lands on a byte boundary we batch 4 elements
// per byte read — amortises the packed-slice load across the four 2-bit
// lanes. JANGTQ's groupSize=64 (== 16 bytes at 2-bit) lands on a byte
// boundary at every group start, so the fast path covers the full group
// body. The single-element tail handles the (rare) case where the row's
// start offset is mid-byte or the group runs short at the row tail.
func cpuSplitPackedDot2(in []float32, packed []byte, scales, biases []float32, offset, cols, groupSize int) float32 {
	var sum float32
	col := 0
	for col < cols {
		idx := offset + col
		group := idx / groupSize
		groupEnd := (group + 1) * groupSize
		end := min(groupEnd-offset, cols)
		scale := scales[group]
		bias := biases[group]
		// Drain prefix elements until (offset+col) is byte-aligned.
		for ; col < end && ((offset+col)&3) != 0; col++ {
			i := offset + col
			q := (packed[i>>2] >> uint((i&3)<<1)) & 0x03
			sum += in[col] * (float32(q)*scale + bias)
		}
		// Walk 4-at-a-time on byte-aligned boundaries.
		for col+4 <= end {
			b := packed[(offset+col)>>2]
			sum += in[col] * (float32(b&0x03)*scale + bias)
			sum += in[col+1] * (float32((b>>2)&0x03)*scale + bias)
			sum += in[col+2] * (float32((b>>4)&0x03)*scale + bias)
			sum += in[col+3] * (float32((b>>6)&0x03)*scale + bias)
			col += 4
		}
		// Drain suffix.
		for ; col < end; col++ {
			i := offset + col
			q := (packed[i>>2] >> uint((i&3)<<1)) & 0x03
			sum += in[col] * (float32(q)*scale + bias)
		}
	}
	return sum
}

// cpuSplitPackedDot1 walks the 1-bit-packed weight path with the unpack
// inlined. Eight values per byte; mask + shift only.
func cpuSplitPackedDot1(in []float32, packed []byte, scales, biases []float32, offset, cols, groupSize int) float32 {
	var sum float32
	col := 0
	for col < cols {
		idx := offset + col
		group := idx / groupSize
		groupEnd := (group + 1) * groupSize
		end := min(groupEnd-offset, cols)
		scale := scales[group]
		bias := biases[group]
		for ; col < end; col++ {
			i := offset + col
			q := (packed[i>>3] >> uint(i&7)) & 0x01
			sum += in[col] * (float32(q)*scale + bias)
		}
	}
	return sum
}

func (matrix *cpuSplitPackedMatrix) value(index int) float32 {
	if matrix == nil || index < 0 || uint64(index) >= matrix.elements {
		return 0
	}
	group := index / matrix.groupSize
	q := cpuSplitUnpackPackedValue(matrix.packed, index, matrix.bits)
	return float32(q)*matrix.scales[group] + matrix.biases[group]
}

func cpuSplitUnpackPackedValue(packed []byte, index, bits int) uint8 {
	// Fast paths for the byte-aligned bit widths actually emitted by the
	// JANG packers (8-bit dense, 4-bit nibble-packed, 2-bit MiniMax M2
	// routed-expert, 1-bit binary). These cover the overwhelmingly common
	// cases and skip the per-bit walk loop, which is hit hundreds of
	// millions of times per layer otherwise.
	switch bits {
	case 8:
		return packed[index]
	case 4:
		b := packed[index>>1]
		if index&1 == 0 {
			return b & 0x0F
		}
		return b >> 4
	case 2:
		return (packed[index>>2] >> uint(((index)&3)<<1)) & 0x03
	case 1:
		return (packed[index>>3] >> uint(index&7)) & 0x01
	}
	bitOffset := index * bits
	remaining := bits
	shiftOut := 0
	value := uint16(0)
	for remaining > 0 {
		byteIndex := bitOffset / 8
		shiftIn := bitOffset % 8
		take := cpuSplitMinInt(remaining, 8-shiftIn)
		mask := uint16((1 << take) - 1)
		chunk := (uint16(packed[byteIndex]) >> shiftIn) & mask
		value |= chunk << shiftOut
		remaining -= take
		bitOffset += take
		shiftOut += take
	}
	return uint8(value)
}

func cpuSplitMinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func cpuSplitSiLU(value float32) float32 {
	return value / (1 + float32(math.Exp(float64(-value))))
}

func cpuSplitFirstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
