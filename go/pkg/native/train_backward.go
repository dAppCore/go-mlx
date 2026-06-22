// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// train_backward.go opens native training (12-14): the no-cgo path has only a forward, so unlike
// pkg/metal — which calls mlx's C autodiff (grad.go: mlx_closure / VJP) — native must build its own
// reverse-mode gradients, op by op. This is the first and most load-bearing one: the linear layer's
// VJP. Every trainable weight in a LoRA SFT (the q/k/v/o/gate/up/down projections, and the LoRA A/B
// factors) is a linear, so its backward is the spine of the whole training graph; the optimiser
// (AdamW) and the LoRA wiring compose on top. Gradients are f32 (the precision metal's optimiser
// accumulates in) and run through the steel GEMM (MatMulF32), so they match metal numerically.

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
