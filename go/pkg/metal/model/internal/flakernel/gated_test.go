// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package flakernel

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// gated_test.go drives the gated-family helpers (gated.go) on the real Metal
// runtime: the softplus/−exp discretisations, the causal depthwise conv with its
// ring across both the seeded-ring/zero-pad and bias/no-bias branches, and the
// gated RMSNorm across its norm/no-norm branches. Each output is checked against
// a CPU reference.

// TestSoftplus_Good proves log(1+exp(x)) element-wise.
func TestSoftplus_Good(t *testing.T) {
	requireMetalRuntime(t)
	xv := []float32{-2, -0.5, 0, 0.5, 2}
	x := metal.FromValues(xv, len(xv))
	defer metal.Free(x)
	out := Softplus(x)
	defer metal.Free(out)
	want := make([]float32, len(xv))
	for i, v := range xv {
		want[i] = float32(math.Log(1 + math.Exp(float64(v))))
	}
	closeFloats(t, "Softplus", out.Floats(), want, 1e-5)
}

// TestNegExp_Good proves −exp(x) element-wise.
func TestNegExp_Good(t *testing.T) {
	requireMetalRuntime(t)
	xv := []float32{-1, 0, 0.5, 1}
	x := metal.FromValues(xv, len(xv))
	defer metal.Free(x)
	out := NegExp(x)
	defer metal.Free(out)
	want := make([]float32, len(xv))
	for i, v := range xv {
		want[i] = float32(-math.Exp(float64(v)))
	}
	closeFloats(t, "NegExp", out.Floats(), want, 1e-5)
}

// causalConvRef is the CPU reference for CausalDepthwiseConv: for each batch and
// channel, output[t] = bias + Σ_{k} kernel[k] · in[t-(K-1)+k], where in is the
// per-channel signal left-padded by the prior ring (channels-first [B,C,pad]) or
// zeros. convWeight is the PyTorch Conv1d layout [C,1,K] (k-major within channel).
func causalConvRef(xBC, ring, w, bias []float32, b, l, convDim, kernel int) []float32 {
	pad := kernel - 1
	out := make([]float32, b*l*convDim)
	for bi := 0; bi < b; bi++ {
		for c := 0; c < convDim; c++ {
			// Build the padded per-channel signal: [ring(pad) | x(l)].
			sig := make([]float32, pad+l)
			for p := 0; p < pad; p++ {
				if ring != nil {
					sig[p] = ring[bi*convDim*pad+c*pad+p] // [B,C,pad]
				}
			}
			for t := 0; t < l; t++ {
				sig[pad+t] = xBC[bi*l*convDim+t*convDim+c] // [B,L,C]
			}
			for t := 0; t < l; t++ {
				var acc float64
				for k := 0; k < kernel; k++ {
					acc += float64(w[c*kernel+k]) * float64(sig[t+k])
				}
				if bias != nil {
					acc += float64(bias[c])
				}
				out[bi*l*convDim+t*convDim+c] = float32(acc)
			}
		}
	}
	return out
}

// ringTailRef is the CPU reference for the advanced ring the production code
// returns: SliceAxis(padded, axis=1, l, l+pad) on the NLC padded signal, i.e. the
// last `pad` tokens of [ring | x] kept in NLC token-major order. The production
// relabels its shape to [B,convDim,pad] but preserves this contiguous buffer, so
// the reference flattens the tail token-major (token outer, channel inner).
//
// Per channel the padded signal is [ring(pad) | x(l)] of length pad+l; the kept
// columns are indices [l, l+pad). Equivalently they are the last `pad` *input*
// positions of the combined ring+x stream: for column index l+p (0≤p<pad), that
// is x[l-pad+p] when l-pad+p≥0, else the prior ring at that position.
func ringTailRef(xBC, ring []float32, b, l, convDim, kernel int) []float32 {
	pad := kernel - 1
	newRing := make([]float32, b*convDim*pad)
	for bi := 0; bi < b; bi++ {
		// Token-major over the kept columns: out[bi][p][c], flattened p-outer.
		for p := 0; p < pad; p++ {
			col := l + p // index into the per-channel padded signal of length pad+l
			for c := 0; c < convDim; c++ {
				var v float32
				if col < pad {
					if ring != nil {
						v = ring[bi*convDim*pad+c*pad+col] // [B,C,pad]
					}
				} else {
					v = xBC[bi*l*convDim+(col-pad)*convDim+c] // [B,L,C]
				}
				newRing[bi*convDim*pad+p*convDim+c] = v
			}
		}
	}
	return newRing
}

// TestCausalDepthwiseConv_ZeroPadNoBias covers the first-chunk branch: a nil ring
// (zero left-pad) and a nil bias.
func TestCausalDepthwiseConv_ZeroPadNoBias(t *testing.T) {
	requireMetalRuntime(t)
	const (
		b       = 1
		l       = 4
		convDim = 2
		kernel  = 3
	)
	xv := make([]float32, b*l*convDim) // [B,L,C]
	for i := range xv {
		xv[i] = float32(i) + 1
	}
	wv := make([]float32, convDim*1*kernel) // [C,1,K]
	for i := range wv {
		wv[i] = 0.1 * float32(i+1)
	}
	x := metal.FromValues(xv, b, l, convDim)
	w := metal.FromValues(wv, convDim, 1, kernel)
	defer metal.Free(x, w)

	conv, ring := CausalDepthwiseConv(x, nil, w, nil, convDim, kernel, b, l)
	defer metal.Free(conv, ring)
	if conv.Dim(0) != b || conv.Dim(1) != l || conv.Dim(2) != convDim {
		t.Fatalf("conv shape = %v, want [%d,%d,%d]", conv.Shape(), b, l, convDim)
	}
	if ring.Dim(0) != b || ring.Dim(1) != convDim || ring.Dim(2) != kernel-1 {
		t.Fatalf("ring shape = %v, want [%d,%d,%d]", ring.Shape(), b, convDim, kernel-1)
	}
	closeFloats(t, "conv(zeropad,nobias)", conv.Floats(),
		causalConvRef(xv, nil, wv, nil, b, l, convDim, kernel), 1e-4)
	closeFloats(t, "ring(zeropad)", ring.Floats(),
		ringTailRef(xv, nil, b, l, convDim, kernel), 1e-4)
}

// TestCausalDepthwiseConv_SeededRingWithBias covers the streamed branch: a valid
// prior ring [B,C,pad] seeding the left pad, plus a bias add.
func TestCausalDepthwiseConv_SeededRingWithBias(t *testing.T) {
	requireMetalRuntime(t)
	const (
		b       = 1
		l       = 3
		convDim = 2
		kernel  = 3
	)
	pad := kernel - 1
	xv := make([]float32, b*l*convDim)
	for i := range xv {
		xv[i] = 0.2*float32(i) - 0.3
	}
	wv := make([]float32, convDim*kernel)
	for i := range wv {
		wv[i] = 0.05*float32(i) + 0.1
	}
	rv := make([]float32, b*convDim*pad) // [B,C,pad]
	for i := range rv {
		rv[i] = 0.5 - 0.1*float32(i)
	}
	bv := []float32{0.25, -0.5} // [C]
	x := metal.FromValues(xv, b, l, convDim)
	w := metal.FromValues(wv, convDim, 1, kernel)
	ringIn := metal.FromValues(rv, b, convDim, pad)
	bias := metal.FromValues(bv, convDim)
	defer metal.Free(x, w, ringIn, bias)

	conv, ring := CausalDepthwiseConv(x, ringIn, w, bias, convDim, kernel, b, l)
	defer metal.Free(conv, ring)
	if conv.Dim(0) != b || conv.Dim(1) != l || conv.Dim(2) != convDim {
		t.Fatalf("conv shape = %v, want [%d,%d,%d]", conv.Shape(), b, l, convDim)
	}
	closeFloats(t, "conv(ring,bias)", conv.Floats(),
		causalConvRef(xv, rv, wv, bv, b, l, convDim, kernel), 1e-4)
	closeFloats(t, "ring(ring,bias)", ring.Floats(),
		ringTailRef(xv, rv, b, l, convDim, kernel), 1e-4)
}

// TestCausalDepthwiseConv_MultiBatchShape drives the seeded-ring + bias path with
// B>1 to exercise the per-batch transpose/concat across the batch axis. The exact
// advanced-ring flat order under the production transpose is batch-interleaved, so
// this asserts the conv values (independent of that relabel) plus both output
// shapes, rather than the ring's element order.
func TestCausalDepthwiseConv_MultiBatchShape(t *testing.T) {
	requireMetalRuntime(t)
	const (
		b       = 2
		l       = 3
		convDim = 2
		kernel  = 3
	)
	pad := kernel - 1
	xv := make([]float32, b*l*convDim)
	for i := range xv {
		xv[i] = 0.2*float32(i) - 0.3
	}
	wv := make([]float32, convDim*kernel)
	for i := range wv {
		wv[i] = 0.05*float32(i) + 0.1
	}
	rv := make([]float32, b*convDim*pad)
	for i := range rv {
		rv[i] = 0.5 - 0.1*float32(i)
	}
	bv := []float32{0.25, -0.5}
	x := metal.FromValues(xv, b, l, convDim)
	w := metal.FromValues(wv, convDim, 1, kernel)
	ringIn := metal.FromValues(rv, b, convDim, pad)
	bias := metal.FromValues(bv, convDim)
	defer metal.Free(x, w, ringIn, bias)

	conv, ring := CausalDepthwiseConv(x, ringIn, w, bias, convDim, kernel, b, l)
	defer metal.Free(conv, ring)
	if conv.Dim(0) != b || conv.Dim(1) != l || conv.Dim(2) != convDim {
		t.Fatalf("conv shape = %v, want [%d,%d,%d]", conv.Shape(), b, l, convDim)
	}
	if ring.Dim(0) != b || ring.Dim(1) != convDim || ring.Dim(2) != pad {
		t.Fatalf("ring shape = %v, want [%d,%d,%d]", ring.Shape(), b, convDim, pad)
	}
	closeFloats(t, "conv(multibatch)", conv.Floats(),
		causalConvRef(xv, rv, wv, bv, b, l, convDim, kernel), 1e-4)
}

// TestCausalDepthwiseConv_StreamedEqualsOneShot proves the helper's core contract
// (gated.go: "the last (kernel-1) input columns … seed the next chunk's left
// padding so a streamed decode is identical to a single-shot prefill"): a single
// L-step prefill equals two chunks threaded through the advanced ring. kernel=3
// (pad=2) is the case where a token-major vs channel-major ring would diverge, so
// this is the discriminating check that the ring carries the right state — not
// just the right shape.
func TestCausalDepthwiseConv_StreamedEqualsOneShot(t *testing.T) {
	requireMetalRuntime(t)
	const (
		b       = 1
		l       = 4
		convDim = 2
		kernel  = 3
	)
	xv := make([]float32, b*l*convDim)
	for i := range xv {
		xv[i] = 0.3*float32(i) - 0.7
	}
	wv := make([]float32, convDim*kernel)
	for i := range wv {
		wv[i] = 0.05*float32(i) + 0.1
	}
	bv := []float32{0.25, -0.5}
	x := metal.FromValues(xv, b, l, convDim)
	w := metal.FromValues(wv, convDim, 1, kernel)
	bias := metal.FromValues(bv, convDim)
	defer metal.Free(x, w, bias)

	// Single-shot prefill over the whole sequence.
	convFull, ringFull := CausalDepthwiseConv(x, nil, w, bias, convDim, kernel, b, l)
	defer metal.Free(convFull, ringFull)

	// Streamed: chunk 0 = x[:,0:2,:] from the zero ring, chunk 1 = x[:,2:4,:]
	// seeded by chunk 0's advanced ring.
	x0 := metal.SliceAxis(x, 1, 0, 2)
	x1 := metal.SliceAxis(x, 1, 2, 4)
	defer metal.Free(x0, x1)
	conv0, ring0 := CausalDepthwiseConv(x0, nil, w, bias, convDim, kernel, b, 2)
	defer metal.Free(conv0, ring0)
	conv1, ring1 := CausalDepthwiseConv(x1, ring0, w, bias, convDim, kernel, b, 2)
	defer metal.Free(conv1, ring1)

	streamed := metal.Concatenate([]*metal.Array{conv0, conv1}, 1) // [B,L,C]
	defer metal.Free(streamed)
	if streamed.Dim(1) != l {
		t.Fatalf("streamed L = %d, want %d", streamed.Dim(1), l)
	}
	closeFloats(t, "streamed==oneShot", streamed.Floats(), convFull.Floats(), 1e-4)

	// The final advanced ring must also agree (the next chunk after either path
	// starts from the same state).
	closeFloats(t, "ring streamed==oneShot", ring1.Floats(), ringFull.Floats(), 1e-4)
}

// TestGatedRMSNorm_WithNorm covers the norm-weight branch: RMSNorm(y)·SiLU(z).
func TestGatedRMSNorm_WithNorm(t *testing.T) {
	requireMetalRuntime(t)
	const (
		rows = 2
		dim  = 3
	)
	const eps = 1e-6
	yv := []float32{1, 2, 3, -1, 0.5, 2}
	zv := []float32{0.5, -0.5, 1, 2, 0, -1}
	nv := []float32{1.5, 0.5, 2}
	y := metal.FromValues(yv, rows, dim)
	z := metal.FromValues(zv, rows, dim)
	norm := metal.FromValues(nv, dim)
	defer metal.Free(y, z, norm)

	out := GatedRMSNorm(y, z, norm, eps)
	defer metal.Free(out)
	if out.Dim(0) != rows || out.Dim(1) != dim {
		t.Fatalf("shape = %v, want [%d,%d]", out.Shape(), rows, dim)
	}
	want := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		var ss float64
		for c := 0; c < dim; c++ {
			ss += float64(yv[r*dim+c]) * float64(yv[r*dim+c])
		}
		inv := 1.0 / math.Sqrt(ss/float64(dim)+eps)
		for c := 0; c < dim; c++ {
			normed := float64(yv[r*dim+c]) * inv * float64(nv[c])
			zz := float64(zv[r*dim+c])
			silu := zz / (1 + math.Exp(-zz))
			want[r*dim+c] = float32(normed * silu)
		}
	}
	closeFloats(t, "GatedRMSNorm(withNorm)", out.Floats(), want, 1e-4)
}

// TestGatedRMSNorm_NoNorm covers the no-weight fallback branch: y·SiLU(z).
func TestGatedRMSNorm_NoNorm(t *testing.T) {
	requireMetalRuntime(t)
	const (
		rows = 2
		dim  = 3
	)
	yv := []float32{1, 2, 3, -1, 0.5, 2}
	zv := []float32{0.5, -0.5, 1, 2, 0, -1}
	y := metal.FromValues(yv, rows, dim)
	z := metal.FromValues(zv, rows, dim)
	defer metal.Free(y, z)

	out := GatedRMSNorm(y, z, nil, 1e-6)
	defer metal.Free(out)
	want := make([]float32, rows*dim)
	for i := range yv {
		zz := float64(zv[i])
		silu := zz / (1 + math.Exp(-zz))
		want[i] = float32(float64(yv[i]) * silu)
	}
	closeFloats(t, "GatedRMSNorm(noNorm)", out.Floats(), want, 1e-4)
}
