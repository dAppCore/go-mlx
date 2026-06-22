// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// train_backward.go opens native training (12-14): the no-cgo path has only a forward, so unlike
// pkg/metal — which calls mlx's C autodiff (grad.go: mlx_closure / VJP) — native must build its own
// reverse-mode gradients, op by op, and chain them in reverse of the forward. These are the load-bearing
// VJPs the rest compose on: the linear layer (every projection + the LoRA A/B factors) and RMSNorm
// (every block's normalisation). Gradients are f32 (the precision metal's optimiser accumulates in) and
// the matmuls run through the steel GEMM (MatMulF32), so they match metal numerically. Each is verified
// by central finite differences (train_backward_test.go).

// LinearBackwardF32 is the vector-Jacobian product of the linear y = x · Wᵀ, where x is [M,K], W is
// [N,K] row-major (the way every projection weight is stored — out_features × in_features), and the
// forward output y is [M,N]. Given the upstream gradient dy [M,N] it returns:
//
//	dx [M,K] = dy · W        (∂L/∂x — flows to the previous layer)
//	dW [N,K] = dyᵀ · x       (∂L/∂W — the weight's gradient the optimiser steps)
//
// Both are computed in f32 via the fused steel GEMM, so they are byte-for-byte what metal's autodiff
// would produce for the same matmul. This is the backward half of a trainable Linear; a LoRA adapter
// composes two of these (the down-projection A and the up-projection B).
func LinearBackwardF32(dy, x, w []float32, M, K, N int) (dx, dw []float32, err error) {
	if len(dy) != M*N || len(x) != M*K || len(w) != N*K {
		return nil, nil, core.NewError("native.LinearBackwardF32: dy[M,N]/x[M,K]/w[N,K] size mismatch")
	}
	// dx = dy[M,N] · W[N,K]  → [M,K]  (nn GEMM: contract over N)
	dx, err = MatMulF32(dy, w, M, N, K)
	if err != nil {
		return nil, nil, err
	}
	// dW = dyᵀ[N,M] · x[M,K] → [N,K]  (nn GEMM: contract over M). Transpose dy host-side first.
	dyT := make([]float32, N*M)
	for m := 0; m < M; m++ {
		row := dy[m*N : (m+1)*N]
		for n := 0; n < N; n++ {
			dyT[n*M+m] = row[n]
		}
	}
	dw, err = MatMulF32(dyT, x, N, M, K)
	if err != nil {
		return nil, nil, err
	}
	return dx, dw, nil
}

// RMSNormBackwardF32 is the VJP of the (plain, no +1) RMSNorm over the last axis: for each of the rows
// rows×n, y_i = g_i · x_i / r with r = sqrt(mean(x²) + eps). Given dy it returns dx and the weight
// gradient dg (summed across rows, the shape of g). Per row:
//
//	dx_i = (g_i·dy_i)/r − x_i·(Σ_k g_k·dy_k·x_k)/(n·r³)
//	dg_i += dy_i·x_i/r
//
// f32. This is the normalisation backward every transformer block needs; it composes with the linear
// VJP into a full MLP/attention-block backward.
func RMSNormBackwardF32(dy, x, g []float32, rows, n int, eps float32) (dx, dg []float32, err error) {
	if len(dy) != rows*n || len(x) != rows*n || len(g) != n {
		return nil, nil, core.NewError("native.RMSNormBackwardF32: dy/x must be [rows,n] and g [n]")
	}
	dx = make([]float32, rows*n)
	dg = make([]float32, n)
	for r := 0; r < rows; r++ {
		xr, dyr, dxr := x[r*n:(r+1)*n], dy[r*n:(r+1)*n], dx[r*n:(r+1)*n]
		var ss float64
		for i := 0; i < n; i++ {
			ss += float64(xr[i]) * float64(xr[i])
		}
		rms := math.Sqrt(ss/float64(n) + float64(eps))
		var dot float64 // Σ_k g_k·dy_k·x_k
		for k := 0; k < n; k++ {
			dot += float64(g[k]) * float64(dyr[k]) * float64(xr[k])
		}
		coef := dot / (float64(n) * rms * rms * rms)
		for i := 0; i < n; i++ {
			dxr[i] = float32(float64(g[i])*float64(dyr[i])/rms - float64(xr[i])*coef)
			dg[i] += float32(float64(dyr[i]) * float64(xr[i]) / rms)
		}
	}
	return dx, dg, nil
}

const (
	geluC = 0.7978845608028654 // sqrt(2/π)
	geluA = 0.044715
)

// geluTanh is the tanh-approx GELU gemma's MLP uses: 0.5·z·(1+tanh(c·(z+a·z³))).
func geluTanh(z float64) float64 {
	u := geluC * (z + geluA*z*z*z)
	return 0.5 * z * (1 + math.Tanh(u))
}

// GeluGateMulBackwardF32 is the VJP of the gemma MLP activation gated = gelu_tanh(gate) · up (the
// elementwise product of the GELU'd gate branch with the up branch). Given the upstream gradient
// dgated it returns dgate and dup:
//
//	dup_i   = dgated_i · gelu(gate_i)
//	dgate_i = dgated_i · up_i · gelu'(gate_i)
//
// with gelu'(z) = 0.5(1+tanh u) + 0.5·z·(1−tanh²u)·c·(1+3a·z²), u = c(z+a·z³). f32. With the linear and
// RMSNorm VJPs this completes a full MLP-block backward (rms → gate/up linears → this → down linear).
func GeluGateMulBackwardF32(dgated, gate, up []float32, n int) (dgate, dup []float32, err error) {
	if len(dgated) != n || len(gate) != n || len(up) != n {
		return nil, nil, core.NewError("native.GeluGateMulBackwardF32: dgated/gate/up must be length n")
	}
	dgate = make([]float32, n)
	dup = make([]float32, n)
	for i := 0; i < n; i++ {
		z := float64(gate[i])
		u := geluC * (z + geluA*z*z*z)
		th := math.Tanh(u)
		gz := 0.5 * z * (1 + th)
		dgelu := 0.5*(1+th) + 0.5*z*(1-th*th)*geluC*(1+3*geluA*z*z)
		dup[i] = float32(float64(dgated[i]) * gz)
		dgate[i] = float32(float64(dgated[i]) * float64(up[i]) * dgelu)
	}
	return dgate, dup, nil
}

// rmsNormForwardF32 is the plain (no +1) RMSNorm forward over rows of width n, returning the normed
// rows (the backward recomputes this to feed the projection VJPs).
func rmsNormForwardF32(h, g []float32, rows, n int, eps float32) []float32 {
	out := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		hr, or := h[r*n:(r+1)*n], out[r*n:(r+1)*n]
		var ss float64
		for i := 0; i < n; i++ {
			ss += float64(hr[i]) * float64(hr[i])
		}
		rms := math.Sqrt(ss/float64(n) + float64(eps))
		for i := 0; i < n; i++ {
			or[i] = float32(float64(g[i]) * float64(hr[i]) / rms)
		}
	}
	return out
}

// MLPBlockGrads holds the parameter gradients of one gemma MLP block (the norm weight + the three
// projection weights), plus dh — the gradient w.r.t. the block input that flows to the previous layer.
type MLPBlockGrads struct {
	DH           []float32 // [M,dModel] gradient to the previous op (includes the residual)
	DNormW       []float32 // [dModel]
	DWGate, DWUp []float32 // [dFF,dModel]
	DWDown       []float32 // [dModel,dFF]
}

// MLPBlockBackwardF32 is the VJP of a full gemma MLP block — out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h)))
// — composed from the linear, RMSNorm and gelu·up VJPs, proving they chain. Given dout [M,dModel] it
// recomputes the forward (normed, gate, up, gated) and backpropagates: through the down projection, the
// gelu·up activation, the gate/up projections (summing the two gradients into rms's output since rms
// feeds both branches), the RMSNorm, and the residual (dh = dout + dh_through_norm). All f32. This is a
// real multi-op backward graph on the no-cgo path, gradient-checked end to end.
func MLPBlockBackwardF32(dout, h, normW, wGate, wUp, wDown []float32, M, dModel, dFF int, eps float32) (*MLPBlockGrads, error) {
	if len(dout) != M*dModel || len(h) != M*dModel || len(normW) != dModel {
		return nil, core.NewError("native.MLPBlockBackwardF32: dout/h must be [M,dModel] and normW [dModel]")
	}
	if len(wGate) != dFF*dModel || len(wUp) != dFF*dModel || len(wDown) != dModel*dFF {
		return nil, core.NewError("native.MLPBlockBackwardF32: projection weight size mismatch")
	}
	// recompute forward intermediates needed by the backward.
	normed := rmsNormForwardF32(h, normW, M, dModel, eps)
	gate, err := MatMulF32NT(normed, wGate, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	up, err := MatMulF32NT(normed, wUp, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	gated := make([]float32, M*dFF)
	for i := range gated {
		gated[i] = float32(geluTanh(float64(gate[i])) * float64(up[i]))
	}
	// backward: down projection (gated @ wDownᵀ → down; out = h + down).
	dGated, dWDown, err := LinearBackwardF32(dout, gated, wDown, M, dFF, dModel)
	if err != nil {
		return nil, err
	}
	// activation gelu(gate)·up (elementwise over all M·dFF).
	dGate, dUp, err := GeluGateMulBackwardF32(dGated, gate, up, M*dFF)
	if err != nil {
		return nil, err
	}
	// gate/up projections (normed @ Wᵀ); rms's output feeds BOTH, so sum the two input gradients.
	dNormedG, dWGate, err := LinearBackwardF32(dGate, normed, wGate, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	dNormedU, dWUp, err := LinearBackwardF32(dUp, normed, wUp, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	dNormed := make([]float32, M*dModel)
	for i := range dNormed {
		dNormed[i] = dNormedG[i] + dNormedU[i]
	}
	// RMSNorm, then the residual: dh = dout + (gradient through the norm path).
	dHNorm, dNormW, err := RMSNormBackwardF32(dNormed, h, normW, M, dModel, eps)
	if err != nil {
		return nil, err
	}
	dH := make([]float32, M*dModel)
	for i := range dH {
		dH[i] = dout[i] + dHNorm[i]
	}
	return &MLPBlockGrads{DH: dH, DNormW: dNormW, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}

// SoftmaxBackwardF32 is the VJP of a row-wise softmax y = softmax(x) over the last axis (rows×n) — the
// attention backward's key new op (the QKᵀ and ·V steps are matmuls, already covered). Given dy and the
// softmax OUTPUT y (cheaper to pass than recomputing), per row:
//
//	dx_i = y_i · (dy_i − Σ_j y_j·dy_j)
//
// f32. Composed with the matmul VJP (for QKᵀ and probs·V) and the RoPE VJP this gives the attention
// block backward; the softmax is the only non-matmul/non-elementwise piece, so it is the gate to it.
func SoftmaxBackwardF32(dy, y []float32, rows, n int) (dx []float32, err error) {
	if len(dy) != rows*n || len(y) != rows*n {
		return nil, core.NewError("native.SoftmaxBackwardF32: dy and y must be [rows,n]")
	}
	dx = make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		yr, dyr, dxr := y[r*n:(r+1)*n], dy[r*n:(r+1)*n], dx[r*n:(r+1)*n]
		var dot float64 // Σ_j y_j·dy_j
		for j := 0; j < n; j++ {
			dot += float64(yr[j]) * float64(dyr[j])
		}
		for i := 0; i < n; i++ {
			dxr[i] = float32(float64(yr[i]) * (float64(dyr[i]) - dot))
		}
	}
	return dx, nil
}
