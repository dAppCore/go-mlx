// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// TestRMSQMVFastBF16ParityComposed is the NUMERICAL gate for the fused rms-norm + affine_qmv_fast
// kernel — the matmul-fusion tier. RMSQMVFastBF16(x, normW, W) must track the composed
// QMVBF16(RMSNormBF16(x, normW), W) at cosine ~1.0. The qmv arithmetic is byte-identical (bfloat16_t ==
// native bfloat); only the rms reduction order differs (~1 ULP). A real bug in the rms prologue or the
// per-element normalise collapses the cosine, so this proves the fused matmul in isolation before any
// decode wiring. Random quant weights (both paths share them, so cosine isolates the fusion).
func TestRMSQMVFastBF16ParityComposed(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded — run `task build:kernels`")
	}
	const eps = float32(1e-6)
	cases := []struct{ inDim, outDim, gs int }{
		{1536, 256, 64}, // e2b dModel → a KV-ish out, gs 64
		{1536, 2048, 32}, // e2b dModel → Q-proj width, gs 32
		{512, 1024, 64},  // smaller, single block
	}
	const bits = 4
	for _, c := range cases {
		x := toBF16Bytes(syntheticFloat32(c.inDim, c.inDim+1))
		normW := toBF16Bytes(syntheticFloat32(c.inDim, c.inDim+7))
		wq := make([]byte, c.outDim*c.inDim*bits/8)
		for i := range wq {
			wq[i] = byte((i*131 + 17) % 256) // deterministic packed 4-bit weights
		}
		nSB := c.outDim * (c.inDim / c.gs)
		scales := toBF16Bytes(syntheticFloat32(nSB, c.gs+3))
		biases := toBF16Bytes(syntheticFloat32(nSB, c.gs+5))

		normed, err := RMSNormBF16(x, normW, 1, c.inDim, eps)
		if err != nil {
			t.Fatalf("in=%d out=%d: RMSNormBF16: %v", c.inDim, c.outDim, err)
		}
		ref, err := QMVBF16(normed, wq, scales, biases, c.outDim, c.inDim, c.gs, bits)
		if err != nil {
			t.Fatalf("in=%d out=%d: QMVBF16: %v", c.inDim, c.outDim, err)
		}
		got, err := RMSQMVFastBF16(x, normW, wq, scales, biases, c.outDim, c.inDim, c.gs, bits, eps)
		if err != nil {
			t.Fatalf("in=%d out=%d: RMSQMVFastBF16: %v", c.inDim, c.outDim, err)
		}

		cos := cosineBF16(got, ref)
		t.Logf("in=%-4d out=%-4d gs=%-3d  cosine=%.7f", c.inDim, c.outDim, c.gs, cos)
		if cos < 0.999 {
			t.Fatalf("in=%d out=%d: fused rms+qmv cosine=%.7f < 0.999 — rms prologue / normalise wrong", c.inDim, c.outDim, cos)
		}
	}
}
