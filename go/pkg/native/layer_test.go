// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestDecodeLayerMatchesAttentionThenMLP(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 29))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 31))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 37))

	got, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}
	h, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	want, err := MLPBlockBF16(h, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("DecodeLayer = %v, want AttentionBlock+MLPBlockBF16 %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestDecodeLayerRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := DecodeLayer(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 64, 1, 1, 64, 1, 128, 10000, 0.125, 0, 1e-5); err == nil {
		t.Fatal("expected DecodeLayer to reject missing inputs and weights")
	}
}
