// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// TestEmbedGatherQuantParity gates the GPU embed-gather: EmbedGatherQuantBF16 must reproduce the host
// embedTokenQuant for a token's 4-bit embedding row (same f32 affine arithmetic, same bf16 round). This
// is the seam that lets the chained decode step compute the next input on-GPU (the submit-ahead pipeline).
func TestEmbedGatherQuantParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}
	const vocab, dModel, gs, bits = 256, 1536, 64, 4
	const scale = float32(0.5)
	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*131 + 17) % 256)
	}
	nSB := vocab * (dModel / gs)
	scales := toBF16Bytes(syntheticFloat32(nSB, 11))
	biases := toBF16Bytes(syntheticFloat32(nSB, 13))

	for _, tok := range []int32{0, 5, 42, 255} {
		ref, err := embedTokenQuant(packed, scales, biases, tok, vocab, dModel, gs, bits, scale)
		if err != nil {
			t.Fatalf("tok %d: embedTokenQuant: %v", tok, err)
		}
		got, err := EmbedGatherQuantBF16(tok, packed, scales, biases, dModel, gs, bits, scale)
		if err != nil {
			t.Fatalf("tok %d: EmbedGatherQuantBF16: %v", tok, err)
		}
		if cos := cosineBF16(got, ref); cos < 0.99999 {
			t.Fatalf("tok %d: GPU embed-gather cosine=%.7f vs host embedTokenQuant", tok, cos)
		}
	}
	t.Logf("GPU embed-gather matches host embedTokenQuant")
}
