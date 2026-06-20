// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestNormProjectICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	x := syntheticFloat32(64, 3)
	normW := syntheticFloat32(64, 5)
	projW := syntheticFloat32(128*64, 7)
	want, err := NormProject(x, normW, projW, 64, 128, 1e-5)
	if err != nil {
		t.Fatalf("NormProject: %v", err)
	}
	got, err := NormProjectICB(x, normW, projW, 64, 128, 1e-5, 1)
	if err != nil {
		t.Fatalf("NormProjectICB: %v", err)
	}
	assertFloat32Near(t, "NormProjectICB", got, want, 0)
}

func TestAttentionBlockICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	got, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("AttentionBlockICB: %v", err)
	}
	eqBytes(t, "AttentionBlockICB", got, want)
}
