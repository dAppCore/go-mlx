// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import core "dappco.re/go"

// MLX linalg ops run on the CPU backend (LAPACK via Accelerate) only — mlx core
// guards each with check_cpu_stream and throws std::invalid_argument on a GPU
// stream. So every binding here dispatches on DefaultCPUStream(), never the
// default (GPU) stream the rest of ops.go uses. MLX inserts the cross-stream
// copy automatically when an input was produced on the GPU stream, so callers
// can build the matrices with the normal GPU ops and hand them straight in.
//
// They also require float32 or float64 inputs (LAPACK has no bf16 path); a bf16
// cache must AsType(a, DTypeFloat32) before calling and cast the result back.

// Solve returns x solving the batched linear system a·x = b, backed by the
// mlx_linalg_solve C op (LU factorisation through LAPACK). a is a stack of
// square matrices [..., n, n] and b is [..., n, k] (or [..., n] for a single
// right-hand side); the leading dims broadcast, so one call solves a whole
// batch. This is the solver behind the full-Attention-Matching compaction's
// normal-equations value fit — see compactAttentionMatchingFull.
//
//	x := Solve(gram, rhs)          // (AᵀA)·x = AᵀB, gram square [B,H,t,t]
//	_ = Eval(x)
func Solve(a, b *Array) (*Array, error) {
	out := NewArray("LINALG_SOLVE", a, b)
	if rc := C.mlx_linalg_solve(&out.ctx, a.ctx, b.ctx, DefaultCPUStream().ctx); rc != 0 {
		Free(out)
		if e := LastError(); e != nil {
			return nil, e
		}
		return nil, core.E("metal.Solve", core.Sprintf("mlx_linalg_solve failed (rc=%d)", int(rc)), nil)
	}
	return out, nil
}

// Pinv returns the Moore-Penrose pseudo-inverse of the batched matrix a
// [..., m, n] → [..., n, m], backed by mlx_linalg_pinv (SVD through LAPACK). It
// solves an over-determined least-squares fit directly (x = pinv(A)·b) without
// forming the normal equations — the lstsq alternative to the Solve route, kept
// for the better conditioning SVD gives on near-rank-deficient designs.
//
//	ainv := Pinv(design)           // design [..., m, n], m ≥ n
//	_ = Eval(ainv)
func Pinv(a *Array) (*Array, error) {
	out := NewArray("LINALG_PINV", a)
	if rc := C.mlx_linalg_pinv(&out.ctx, a.ctx, DefaultCPUStream().ctx); rc != 0 {
		Free(out)
		if e := LastError(); e != nil {
			return nil, e
		}
		return nil, core.E("metal.Pinv", core.Sprintf("mlx_linalg_pinv failed (rc=%d)", int(rc)), nil)
	}
	return out, nil
}

// Cholesky returns the Cholesky factor of the batched symmetric
// positive-definite matrix a [..., n, n], backed by mlx_linalg_cholesky. upper
// selects the upper-triangular factor U (a = UᵀU) over the lower-triangular L
// (a = LLᵀ). The Gram matrix AᵀA of the normal equations is SPD once ridge-
// regularised, so a Cholesky solve is the cheaper factorisation when Solve's
// general LU is more than the problem needs.
//
//	L := Cholesky(gram, false)     // gram = L·Lᵀ, lower factor
//	_ = Eval(L)
func Cholesky(a *Array, upper bool) (*Array, error) {
	out := NewArray("LINALG_CHOLESKY", a)
	if rc := C.mlx_linalg_cholesky(&out.ctx, a.ctx, C._Bool(upper), DefaultCPUStream().ctx); rc != 0 {
		Free(out)
		if e := LastError(); e != nil {
			return nil, e
		}
		return nil, core.E("metal.Cholesky", core.Sprintf("mlx_linalg_cholesky failed (rc=%d)", int(rc)), nil)
	}
	return out, nil
}
