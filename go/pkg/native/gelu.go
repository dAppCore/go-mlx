// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

// fillConst returns n copies of v — a broadcast scalar materialised as a dense
// operand for the elementwise kernels. MLX broadcasts a 0-dim scalar; an
// all-v vector multiplies/adds to the identical per-element result.
func fillConst(n int, v float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = v
	}
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
