// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestLinearBackwardF32 verifies the linear VJP against finite differences — the standard correctness
// bar for a gradient. With a fixed random cotangent dy, the scalar loss is L = Σ y·dy, so ∂L/∂x and
// ∂L/∂W are exactly what LinearBackwardF32 returns; we check a sample of entries against the central
// finite difference (L(θ+ε) − L(θ−ε)) / 2ε of the real forward y = x·Wᵀ.
func TestLinearBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const M, K, N = 3, 4, 5
	x := syntheticFloat32(M*K, 1)
	w := syntheticFloat32(N*K, 2)
	dy := syntheticFloat32(M*N, 3)

	forward := func(x, w []float32) []float32 {
		y, err := MatMulF32NT(x, w, M, K, N) // x[M,K] · w[N,K]ᵀ = y[M,N]
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		return y
	}
	loss := func(x, w []float32) float64 { // L = Σ y·dy
		y := forward(x, w)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}

	dx, dw, err := LinearBackwardF32(dy, x, w, M, K, N)
	if err != nil {
		t.Fatalf("LinearBackwardF32: %v", err)
	}

	const eps = 1.0 / 256 // bf16-free f32 forward; a coarse step keeps finite-diff noise low
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss(x, w)
			params[i] = orig - eps
			lm := loss(x, w)
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dx", x, dx)
	check("dw", w, dw)
	t.Logf("linear VJP matches finite differences: dx[%d] dw[%d] all within tol", len(dx), len(dw))
}

// TestRMSNormBackwardF32 verifies the RMSNorm VJP against central finite differences of the plain
// (no +1) RMSNorm forward y_i = g_i·x_i/sqrt(mean(x²)+eps), with L = Σ y·dy.
func TestRMSNormBackwardF32(t *testing.T) {
	const rows, n = 3, 8
	eps := float32(1e-5)
	x := syntheticFloat32(rows*n, 1)
	g := syntheticFloat32(n, 2)
	dy := syntheticFloat32(rows*n, 3)

	forward := func(x, g []float32) []float32 {
		y := make([]float32, rows*n)
		for r := 0; r < rows; r++ {
			var ss float64
			for i := 0; i < n; i++ {
				ss += float64(x[r*n+i]) * float64(x[r*n+i])
			}
			rms := math.Sqrt(ss/float64(n) + float64(eps))
			for i := 0; i < n; i++ {
				y[r*n+i] = float32(float64(g[i]) * float64(x[r*n+i]) / rms)
			}
		}
		return y
	}
	loss := func(x, g []float32) float64 {
		y := forward(x, g)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}

	dx, dg, err := RMSNormBackwardF32(dy, x, g, rows, n, eps)
	if err != nil {
		t.Fatalf("RMSNormBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 512
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss(x, g)
			params[i] = orig - eps2
			lm := loss(x, g)
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dx", x, dx)
	check("dg", g, dg)
	t.Logf("RMSNorm VJP matches finite differences: dx[%d] dg[%d] all within tol", len(dx), len(dg))
}

// TestGeluGateMulBackwardF32 verifies the MLP activation VJP against finite differences of the forward
// gated_i = gelu_tanh(gate_i)·up_i, with L = Σ gated·dgated.
func TestGeluGateMulBackwardF32(t *testing.T) {
	const n = 12
	gate := syntheticFloat32(n, 1)
	up := syntheticFloat32(n, 2)
	dgated := syntheticFloat32(n, 3)

	loss := func(gate, up []float32) float64 {
		var s float64
		for i := 0; i < n; i++ {
			s += geluTanh(float64(gate[i])) * float64(up[i]) * float64(dgated[i])
		}
		return s
	}
	dgate, dup, err := GeluGateMulBackwardF32(dgated, gate, up, n)
	if err != nil {
		t.Fatalf("GeluGateMulBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss(gate, up)
			params[i] = orig - eps
			lm := loss(gate, up)
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dgate", gate, dgate)
	check("dup", up, dup)
	t.Logf("gelu·up VJP matches finite differences: dgate[%d] dup[%d] all within tol", len(dgate), len(dup))
}

// TestMLPBlockBackwardF32 gradient-checks the COMPOSED MLP-block backward end to end — proving the
// linear, gelu·up and RMSNorm VJPs chain correctly (including the rms→gate+up branch sum and the
// residual) — against finite differences of the full block forward out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h))).
func TestMLPBlockBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const M, dModel, dFF = 2, 8, 16
	eps := float32(1e-5)
	h := syntheticFloat32(M*dModel, 1)
	normW := syntheticFloat32(dModel, 2)
	wGate := syntheticFloat32(dFF*dModel, 3)
	wUp := syntheticFloat32(dFF*dModel, 4)
	wDown := syntheticFloat32(dModel*dFF, 5)
	dout := syntheticFloat32(M*dModel, 6)

	forward := func() []float32 {
		normed := rmsNormForwardF32(h, normW, M, dModel, eps)
		gate, err := MatMulF32NT(normed, wGate, M, dModel, dFF)
		if err != nil {
			t.Fatal(err)
		}
		up, err := MatMulF32NT(normed, wUp, M, dModel, dFF)
		if err != nil {
			t.Fatal(err)
		}
		gated := make([]float32, M*dFF)
		for i := range gated {
			gated[i] = float32(geluTanh(float64(gate[i])) * float64(up[i]))
		}
		down, err := MatMulF32NT(gated, wDown, M, dFF, dModel)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, M*dModel)
		for i := range out {
			out[i] = h[i] + down[i]
		}
		return out
	}
	loss := func() float64 {
		out := forward()
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dout[i])
		}
		return s
	}

	g, err := MLPBlockBackwardF32(dout, h, normW, wGate, wUp, wDown, M, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MLPBlockBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 512
	// check a strided sample of each gradient (full finite-diff over every weight is needlessly slow).
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 12 {
			step = len(params) / 12
		}
		for i := 0; i < len(params); i += step {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss()
			params[i] = orig - eps2
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dH", h, g.DH)
	check("dNormW", normW, g.DNormW)
	check("dWGate", wGate, g.DWGate)
	check("dWUp", wUp, g.DWUp)
	check("dWDown", wDown, g.DWDown)
	t.Logf("MLP-block backward chains correctly: dH/dNormW/dWGate/dWUp/dWDown all match finite differences")
}

// TestSoftmaxBackwardF32 verifies the softmax VJP against finite differences of the row-wise softmax
// forward, with L = Σ y·dy.
func TestSoftmaxBackwardF32(t *testing.T) {
	const rows, n = 3, 7
	x := syntheticFloat32(rows*n, 1)
	dy := syntheticFloat32(rows*n, 2)

	softmax := func(x []float32) []float32 {
		y := make([]float32, rows*n)
		for r := 0; r < rows; r++ {
			xr, yr := x[r*n:(r+1)*n], y[r*n:(r+1)*n]
			mx := xr[0]
			for _, v := range xr {
				if v > mx {
					mx = v
				}
			}
			var sum float64
			for i, v := range xr {
				e := math.Exp(float64(v - mx))
				yr[i] = float32(e)
				sum += e
			}
			for i := range yr {
				yr[i] = float32(float64(yr[i]) / sum)
			}
		}
		return y
	}
	loss := func(x []float32) float64 {
		y := softmax(x)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}
	y := softmax(x)
	dx, err := SoftmaxBackwardF32(dy, y, rows, n)
	if err != nil {
		t.Fatalf("SoftmaxBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	for i := range x {
		orig := x[i]
		x[i] = orig + eps
		lp := loss(x)
		x[i] = orig - eps
		lm := loss(x)
		x[i] = orig
		fd := (lp - lm) / (2 * eps)
		if math.Abs(fd-float64(dx[i])) > 1e-2*(1+math.Abs(fd)) {
			t.Errorf("dx[%d]: analytic %.5f vs finite-diff %.5f", i, dx[i], fd)
		}
	}
	t.Logf("softmax VJP matches finite differences: dx[%d] within tol", len(dx))
}
