// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// TestVProjHeadRMSBF16ParityComposed gates the fused V-path kernel (input-rms → V-proj → value-norm)
// against the composed RMSNormBF16(QMVBF16(RMSNormBF16(x, inNormW)), ones, nKVHeads, headDim). Cosine
// ~1.0 (lockstep: the per-thread dot + the two rms reductions differ in summation order, ~1 ULP). A
// real wiring bug — wrong head slicing, missing a norm — collapses the cosine.
func TestVProjHeadRMSBF16ParityComposed(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded — run `task build:kernels`")
	}
	const eps = float32(1e-6)
	const bits = 4
	cases := []struct{ nKVHeads, headDim, inDim, gs int }{
		{1, 256, 1536, 64}, // e2b: 1 KV head, headDim 256
		{2, 128, 1536, 64}, // 2 KV heads, headDim 128 (outDim 256)
		{1, 512, 1536, 32}, // wider head
	}
	for _, c := range cases {
		outDim := c.nKVHeads * c.headDim
		x := toBF16Bytes(syntheticFloat32(c.inDim, c.inDim+1))
		inNormW := toBF16Bytes(syntheticFloat32(c.inDim, c.inDim+7))
		wq := make([]byte, outDim*c.inDim*bits/8)
		for i := range wq {
			wq[i] = byte((i*131 + 17) % 256)
		}
		nSB := outDim * (c.inDim / c.gs)
		scales := toBF16Bytes(syntheticFloat32(nSB, c.gs+3))
		biases := toBF16Bytes(syntheticFloat32(nSB, c.gs+5))
		onesF := make([]float32, c.headDim)
		for i := range onesF {
			onesF[i] = 1
		}
		ones := toBF16Bytes(onesF)

		normed, err := RMSNormBF16(x, inNormW, 1, c.inDim, eps)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: input RMSNorm: %v", c.nKVHeads, c.headDim, err)
		}
		vproj, err := QMVBF16(normed, wq, scales, biases, outDim, c.inDim, c.gs, bits)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: QMV: %v", c.nKVHeads, c.headDim, err)
		}
		ref, err := RMSNormBF16(vproj, ones, c.nKVHeads, c.headDim, eps)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: value-norm: %v", c.nKVHeads, c.headDim, err)
		}
		got, err := VProjHeadRMSBF16(x, inNormW, wq, scales, biases, c.nKVHeads, c.headDim, c.inDim, c.gs, bits, eps)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: VProjHeadRMSBF16: %v", c.nKVHeads, c.headDim, err)
		}

		cos := cosineBF16(got, ref)
		t.Logf("nkv=%-2d hd=%-4d inDim=%-4d gs=%-3d  cosine=%.7f", c.nKVHeads, c.headDim, c.inDim, c.gs, cos)
		if cos < 0.999 {
			t.Fatalf("nkv=%d hd=%d: fused V-path cosine=%.7f < 0.999 — wrong", c.nKVHeads, c.headDim, cos)
		}
	}
}
