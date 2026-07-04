// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package qwen3

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

func qwenEval(t *testing.T, a *metal.Array) []float32 {
	t.Helper()
	if a == nil || !a.Valid() {
		t.Fatal("resolve returned an invalid array")
	}
	if err := metal.Eval(a); err != nil {
		t.Fatalf("eval: %v", err)
	}
	return a.Floats()
}

func softplusRef(x float64) float64 { return math.Log(1 + math.Exp(x)) }

// TestQwen3_ResolveDecay_Good pins the gated-delta decay discretisation
// α = exp(−exp(A_log)·softplus(a+dt_bias)) and the [B,L,H]→[B,H,L,1] layout against
// a plain-Go reference. The cross-indexing (input l·H+h, output h·L+l) also proves
// the L↔H transpose.
func TestQwen3_ResolveDecay_Good(t *testing.T) {
	const B, L, H = 1, 3, 2
	aVals := []float32{0.2, -0.5, 0.1, 0.3, -0.2, 0.4} // [B,L,H]
	aLogVals := []float32{-1.0, 0.5}                   // [H]
	dtBiasVals := []float32{0.1, -0.3}                 // [H]
	a := metal.FromValues(aVals, B, L, H)
	aLog := metal.FromValues(aLogVals, H)
	dtBias := metal.FromValues(dtBiasVals, H)
	defer metal.Free(a, aLog, dtBias)

	alpha := resolveDecay(a, aLog, dtBias, B, L, H) // [B,H,L,1]
	defer metal.Free(alpha)
	got := qwenEval(t, alpha)

	const tol = 1e-4
	for h := 0; h < H; h++ {
		for l := 0; l < L; l++ {
			dt := softplusRef(float64(aVals[l*H+h]) + float64(dtBiasVals[h]))
			want := math.Exp(-math.Exp(float64(aLogVals[h])) * dt)
			g := float64(got[h*L+l]) // [B,H,L,1], b=0 → h·L+l
			if math.Abs(g-want) > tol {
				t.Fatalf("α[h=%d,l=%d] = %.6f, want %.6f", h, l, g, want)
			}
			if g <= 0 || g > 1.0+tol {
				t.Fatalf("α[h=%d,l=%d] = %.6f outside the (0,1] decay band", h, l, g)
			}
		}
	}
}

// TestQwen3_ResolveBeta_Good pins β = sigmoid(b) and the kernel layout.
func TestQwen3_ResolveBeta_Good(t *testing.T) {
	const B, L, H = 1, 3, 2
	bVals := []float32{0.5, -1.0, 0.2, 1.5, -0.3, 0.8} // [B,L,H]
	bp := metal.FromValues(bVals, B, L, H)
	defer metal.Free(bp)

	beta := resolveBeta(bp, B, L, H) // [B,H,L,1]
	defer metal.Free(beta)
	got := qwenEval(t, beta)

	const tol = 1e-4
	for h := 0; h < H; h++ {
		for l := 0; l < L; l++ {
			want := 1.0 / (1.0 + math.Exp(-float64(bVals[l*H+h])))
			g := float64(got[h*L+l])
			if math.Abs(g-want) > tol {
				t.Fatalf("β[h=%d,l=%d] = %.6f, want %.6f", h, l, g, want)
			}
		}
	}
}

// TestQwen3_ResolveDecay_NoBias_Good proves the dt-bias is optional (nil ⇒ plain
// softplus(a)).
func TestQwen3_ResolveDecay_NoBias_Good(t *testing.T) {
	const B, L, H = 1, 2, 2
	aVals := []float32{0.3, -0.1, 0.2, 0.5}
	aLogVals := []float32{0.0, -0.5}
	a := metal.FromValues(aVals, B, L, H)
	aLog := metal.FromValues(aLogVals, H)
	defer metal.Free(a, aLog)

	alpha := resolveDecay(a, aLog, nil, B, L, H)
	defer metal.Free(alpha)
	got := qwenEval(t, alpha)

	const tol = 1e-4
	for h := 0; h < H; h++ {
		for l := 0; l < L; l++ {
			dt := softplusRef(float64(aVals[l*H+h])) // no bias
			want := math.Exp(-math.Exp(float64(aLogVals[h])) * dt)
			if g := float64(got[h*L+l]); math.Abs(g-want) > tol {
				t.Fatalf("α[h=%d,l=%d] = %.6f, want %.6f", h, l, g, want)
			}
		}
	}
}
