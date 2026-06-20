// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestAttentionBlockMatchesComposedPrimitives(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))

	got, err := AttentionBlock(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	normed, err := RMSNormBF16(x, normW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	q, err := MatVecBF16(wQ, normed, qDim, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16 q: %v", err)
	}
	qr, err := RoPEBF16(q, 1, nHeads, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	attn, err := SDPA(qr, kCache, vCache, 1, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	attnOut, err := MatVecBF16(wO, attn, dModel, qDim)
	if err != nil {
		t.Fatalf("MatVecBF16 o: %v", err)
	}
	want, err := AddBF16(x, attnOut)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("AttentionBlock = %v, want composed primitives %v", bf16Floats(got), bf16Floats(want))
	}
}
