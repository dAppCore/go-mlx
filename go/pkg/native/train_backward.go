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
