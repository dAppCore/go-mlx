// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// TestPerLayerInputsGPUParity gates the on-GPU PLE: PerLayerInputsGPU (per-layer embed-gather + projection
// + norm + combine, all on the GPU from a token id) must reproduce the host PerLayerInputs. This is the
// gate the submit-ahead decode pipeline needs for e2b — the PLE tensor computed on-GPU so the next step
// can be submitted before the token is read back.
func TestPerLayerInputsGPUParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	plDim := numLayers * pliDim

	// 4-bit per-layer embedding table [vocabPLI × plDim], bf16 projection [plDim × dModel] + projNorm [pliDim].
	embedPacked := make([]byte, vocabPLI*plDim*embBits/8)
	for i := range embedPacked {
		embedPacked[i] = byte((i*131 + 17) % 256)
	}
	embedScales := toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 11))
	embedBiases := toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 13))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 7))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 19))
	emb := toBF16Bytes(syntheticFloat32(dModel, 23))

	for _, tok := range []int32{0, 5, 17, 31} {
		ref, err := PerLayerInputs(embedPacked, embedScales, embedBiases, projW, nil, nil, projNormW, tok, emb, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, 0, 0, eps, bufView{})
		if err != nil {
			t.Fatalf("tok %d: host PerLayerInputs: %v", tok, err)
		}
		got, err := PerLayerInputsGPU(tok, emb, embedPacked, embedScales, embedBiases, projW, projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
		if err != nil {
			t.Fatalf("tok %d: PerLayerInputsGPU: %v", tok, err)
		}
		if cos := cosineBF16(got, ref); cos < 0.9999 {
			t.Fatalf("tok %d: GPU PLE cosine=%.6f vs host PerLayerInputs", tok, cos)
		}
	}
	t.Logf("GPU PLE matches host PerLayerInputs")
}
