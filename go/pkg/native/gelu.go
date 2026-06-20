// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "sync"

// constVecKey identifies a materialised broadcast-scalar operand by length and
// value, so identical (n, v) requests share one immutable backing slice.
type constVecKey struct {
	n int
	v float32
}

// constVecCache memoises the dense scalar operands fillConst produces. The
// composed Gelu fires the same four compile-time constants (0.044715,
// 0.7978…, 1.0, 0.5) at a fixed decode width every call; caching collapses the
// per-call make([]float32, n) (the dominant B/op of the float32 Gelu path) to a
// one-time fill. Entries are never mutated — they feed the vv_ kernels purely as
// read-only operands, so the cached slice yields byte-identical kernel input.
var (
	constVecMu    sync.Mutex
	constVecCache = map[constVecKey][]float32{}
)

// fillConst returns n copies of v — a broadcast scalar materialised as a dense
// operand for the elementwise kernels. MLX broadcasts a 0-dim scalar; an
// all-v vector multiplies/adds to the identical per-element result. The result
// is cached and shared across calls: callers treat it as read-only (it is only
// ever passed as a kernel operand, which copies into a fresh output), so the
// shared slice is safe and the bytes are identical to a freshly filled one.
func fillConst(n int, v float32) []float32 {
	if n == 0 {
		return nil
	}
	key := constVecKey{n: n, v: v}
	constVecMu.Lock()
	defer constVecMu.Unlock()
	if s, ok := constVecCache[key]; ok {
		return s
	}
	s := make([]float32, n)
	for i := range s {
		s[i] = v
	}
	constVecCache[key] = s
	return s
}

// Gelu computes the tanh-approximation GELU element-wise, composed from the
// native primitives exactly as MLX's gelu_approx does (the graph mlx_compile
// fuses for gemma's MLP):
//
//	x2     = x · x
//	x3     = x2 · x
//	inner  = x + 0.044715 · x3
//	t      = tanh(0.7978845608028654 · inner)
//	gelu   = 0.5 · x · (1 + t)
//
// Unlike the single-kernel ops, GELU is not a metallib kernel — it is the first
// native op built by COMPOSING primitives rather than driving one kernel, which
// is the shape every mlx-compiled fused op takes on the native path. float32.
func Gelu(x []float32) ([]float32, error) {
	n := len(x)
	x2, err := Mul(x, x)
	if err != nil {
		return nil, err
	}
	x3, err := Mul(x2, x)
	if err != nil {
		return nil, err
	}
	x3scaled, err := Mul(x3, fillConst(n, 0.044715))
	if err != nil {
		return nil, err
	}
	inner, err := Add(x, x3scaled)
	if err != nil {
		return nil, err
	}
	scaled, err := Mul(inner, fillConst(n, 0.7978845608028654))
	if err != nil {
		return nil, err
	}
	t, err := Tanh(scaled)
	if err != nil {
		return nil, err
	}
	onePlus, err := Add(t, fillConst(n, 1.0))
	if err != nil {
		return nil, err
	}
	halfX, err := Mul(x, fillConst(n, 0.5))
	if err != nil {
		return nil, err
	}
	return Mul(halfX, onePlus)
}

// GeluGateMul computes gelu(gate)·up — gemma's MLP gate. It is the native
// composition of mlx-c's fused GELUGateMul. Parity (within fp tolerance, since
// native runs the ops separately while mlx fuses them) is gated in parity_test.go.
func GeluGateMul(gate, up []float32) ([]float32, error) {
	g, err := Gelu(gate)
	if err != nil {
		return nil, err
	}
	return Mul(g, up)
}
