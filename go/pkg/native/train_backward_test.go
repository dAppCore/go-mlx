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
