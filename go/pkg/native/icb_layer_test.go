// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestDecodeLayerICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}
	got, err := DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeLayerICB: %v", err)
	}
	eqBytes(t, "DecodeLayerICB", got, want)
}

func TestDecodeTokenICBOneLayerMatchesDecodeLayer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}
	got, err := DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeTokenICB: %v", err)
	}
	eqBytes(t, "DecodeTokenICB", got, want)
}
