// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

// solveAbsClose fails when any element of got is further than tol from want.
func solveAbsClose(t *testing.T, got, want []float32, tol float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Errorf("%s[%d] = %g, want %g (|Δ|=%g > %g)", label, i, got[i], want[i], d, tol)
		}
	}
}

// TestLinalgOp_Solve_Good — mlx_linalg_solve recovers the exact x of a small
// 2×2 system against a hand-computed reference, proving the binding (and the
// CPU-stream dispatch) round-trips. System:
//
//	[2 1][x0]   [3]        x0 = 1
//	[1 3][x1] = [5]   ⇒    x1 = 1.333…  (Cramer: det=5, x0=(9-5)/5, x1=(10-3)/5)
func TestLinalgOp_Solve_Good(t *testing.T) {
	a := FromValues([]float32{2, 1, 1, 3}, 2, 2)
	b := FromValues([]float32{3, 5}, 2, 1)
	defer Free(a, b)

	x, err := Solve(a, b)
	if err != nil {
		t.Fatalf("Solve: %v", err)
	}
	defer Free(x)
	if err := Eval(x); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if got := x.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 1 {
		t.Fatalf("x shape = %v, want [2 1]", got)
	}
	solveAbsClose(t, x.Floats(), []float32{0.8, 1.4}, 1e-4, "x")

	// Residual a·x − b must be ~0: an independent check the solve actually
	// satisfies the system rather than just matching a precomputed answer.
	resid := Subtract(Matmul(a, x), b)
	defer Free(resid)
	if err := Eval(resid); err != nil {
		t.Fatalf("Eval residual: %v", err)
	}
	for i, r := range resid.Floats() {
		if math.Abs(float64(r)) > 1e-4 {
			t.Errorf("residual[%d] = %g, want ~0", i, r)
		}
	}
}

// TestLinalgOp_SolveBatched_Ugly — the leading dims broadcast, so one call
// solves a stack of independent systems. Two stacked diagonal systems with
// different scales must each invert independently.
func TestLinalgOp_SolveBatched_Ugly(t *testing.T) {
	// Batch of 2 systems, each 2×2 diagonal: diag(2,4) and diag(5,10).
	a := FromValues([]float32{
		2, 0, 0, 4,
		5, 0, 0, 10,
	}, 2, 2, 2)
	// RHS [2,2,1]: solving diag·x = b ⇒ x = b/diag.
	b := FromValues([]float32{
		8, 8,
		5, 5,
	}, 2, 2, 1)
	defer Free(a, b)

	x, err := Solve(a, b)
	if err != nil {
		t.Fatalf("Solve batched: %v", err)
	}
	defer Free(x)
	if err := Eval(x); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	// Expected: [8/2, 8/4] = [4, 2]; [5/5, 5/10] = [1, 0.5].
	solveAbsClose(t, x.Floats(), []float32{4, 2, 1, 0.5}, 1e-4, "x")
}

// TestLinalgOp_Pinv_LeastSquares_Good — pinv solves an over-determined system
// in the least-squares sense. Fit y = m·t through three points (0,1),(1,3),
// (2,5) which lie exactly on y = 2t + 1, so the LS fit is exact: [m,c]=[2,1].
func TestLinalgOp_Pinv_LeastSquares_Good(t *testing.T) {
	// Design A [3,2] columns = [t, 1]; target y [3,1].
	a := FromValues([]float32{
		0, 1,
		1, 1,
		2, 1,
	}, 3, 2)
	y := FromValues([]float32{1, 3, 5}, 3, 1)
	defer Free(a, y)

	ainv, err := Pinv(a) // [2,3]
	if err != nil {
		t.Fatalf("Pinv: %v", err)
	}
	defer Free(ainv)
	coeffs := Matmul(ainv, y) // [2,1] = [m, c]
	defer Free(coeffs)
	if err := Eval(coeffs); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	solveAbsClose(t, coeffs.Floats(), []float32{2, 1}, 1e-3, "coeffs")
}

// TestLinalgOp_Cholesky_Good — the Cholesky factor reconstructs the SPD input:
// for L = Cholesky(A, upper=false), L·Lᵀ == A. A = [[4,2],[2,3]] is SPD.
func TestLinalgOp_Cholesky_Good(t *testing.T) {
	a := FromValues([]float32{4, 2, 2, 3}, 2, 2)
	defer Free(a)

	l, err := Cholesky(a, false)
	if err != nil {
		t.Fatalf("Cholesky: %v", err)
	}
	defer Free(l)
	// Reconstruct L·Lᵀ and compare to A.
	lT := Transpose(l, 1, 0)
	defer Free(lT)
	recon := Matmul(l, lT)
	defer Free(recon)
	if err := Eval(recon); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	solveAbsClose(t, recon.Floats(), []float32{4, 2, 2, 3}, 1e-4, "L·Lᵀ")
}

// TestLinalgOp_SolveNonSquare_Bad — mlx_linalg_solve requires a square a; a
// non-square first argument must surface a wrapped error, not a crash.
func TestLinalgOp_SolveNonSquare_Bad(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3) // not square
	b := FromValues([]float32{1, 2}, 2, 1)
	defer Free(a, b)

	x, err := Solve(a, b)
	if err == nil {
		if x != nil {
			Free(x)
		}
		t.Fatal("Solve on a non-square matrix returned nil error, want a failure")
	}
}
