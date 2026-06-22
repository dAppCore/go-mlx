// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestDecodeLayerBatchedKV asserts the MTP batched verify forward is BYTE-IDENTICAL to K sequential
// DecodeStepKV calls over the same growing KV cache: same K output hiddens, same final cache. This is
// the correctness bar for the batched verify — it must produce exactly what stepping the K draft
// tokens one at a time produces, so wiring it into MTPDecode keeps the token stream identical.
func TestDecodeLayerBatchedKV(t *testing.T) {
	requireNativeRuntime(t)
	const (
		dModel   = 256
		nHeads   = 4
		nKVHeads = 2
		headDim  = 64
		maxLen   = 32
		dFF      = 512
		basePos  = 5
		K        = 4
	)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	base, scale, eps := float32(10000), float32(1.0/8.0), float32(1e-6) // 1/sqrt(64)=1/8

	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))

	// a non-empty resident prefix: basePos rows of K/V already in the cache.
	kCache0 := make([]byte, maxLen*kvDim*bf16Size)
	vCache0 := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache0, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache0, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))

	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))
	rowBytes := dModel * bf16Size

	// sequential reference: K DecodeStepKV calls over a copy of the cache.
	kSeq := append([]byte(nil), kCache0...)
	vSeq := append([]byte(nil), vCache0...)
	seqOut := make([]byte, K*rowBytes)
	for i := 0; i < K; i++ {
		h, err := DecodeStepKV(xs[i*rowBytes:(i+1)*rowBytes], attnNormW, wQ, wK, wV, wO, kSeq, vSeq, mlpNormW, wGate, wUp, wDown,
			dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos+i, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeStepKV row %d: %v", i, err)
		}
		copy(seqOut[i*rowBytes:(i+1)*rowBytes], h)
	}

	// batched: one DecodeLayerBatchedKV over a fresh copy of the same prefix.
	kBat := append([]byte(nil), kCache0...)
	vBat := append([]byte(nil), vCache0...)
	batOut, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kBat, vBat, mlpNormW, wGate, wUp, wDown,
		dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeLayerBatchedKV: %v", err)
	}

	eqBytes(t, "batched verify output vs K sequential DecodeStepKV", batOut, seqOut)
	eqBytes(t, "batched verify kCache vs sequential", kBat, kSeq)
	eqBytes(t, "batched verify vCache vs sequential", vBat, vSeq)
}
