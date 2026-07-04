// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestGeluProductionRef is the standing guard + the data behind the fused-gelu decision. The GUARD:
// the native composed gelu must equal production (metal.GeluGateMul, enableNativeGELUGateMul=false →
// geluApprox on bf16, each op rounded) byte-for-byte — the "0.999993 drift" was a parity-test
// artifact (mlx-c fed fp32 while the engine runs bf16), not a real divergence. The DATA: the fused
// fp32-internal kernel differs from production by ~34% at the 1-ulp level (1 rounding vs 10) — why
// wiring it into serve is a both-engines "compute fp32" decision, not a native-only swap.
func TestGeluProductionRef(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const n = 4096
	gb := make([]byte, n*bf16Size)
	ub := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		g := f32ToBF16(float32((i*37)%160-80) * 0.1)
		u := f32ToBF16(float32((i*53)%97-48) * 0.04)
		gb[i*2], gb[i*2+1] = byte(g), byte(g>>8)
		ub[i*2], ub[i*2+1] = byte(u), byte(u>>8)
	}

	gC, err := GeluBF16(gb) // native composed gelu = the production path
	if err != nil {
		t.Fatal(err)
	}
	composed, err := MulBF16(gC, ub)
	if err != nil {
		t.Fatal(err)
	}

	gA := mc.FromRawBytes(gb, []int{n}, mc.DTypeBFloat16)
	uA := mc.FromRawBytes(ub, []int{n}, mc.DTypeBFloat16)
	prodArr := mc.GeluGateMul(gA, uA) // production: composed bf16 (flag off)
	mc.Materialize(prodArr)
	prod := append([]byte(nil), prodArr.RawBytes()...)
	mc.Free(gA, uA, prodArr)

	ex := func(a, b []byte) int {
		c := 0
		for i := 0; i+1 < len(a) && i+1 < len(b); i += 2 {
			if a[i] == b[i] && a[i+1] == b[i+1] {
				c++
			}
		}
		return c
	}

	if e := ex(composed, prod); e != n { // GUARD
		t.Errorf("native composed gelu != production GeluGateMul: %d/%d exact — byte-faithfulness broken", e, n)
	}
	if gpuHasGeluKernel() { // DATA
		fused, err := geluGateMulFused(gb, ub, n)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("fused(fp32-internal) vs production(composed bf16): %d/%d exact", ex(fused, prod), n)
	}
}
