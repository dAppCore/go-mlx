// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestSDPASingleValueReturnsV(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 2, 1, 64, 1
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	got, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 1)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	want := append(append([]byte(nil), v...), v...)
	if !bytes.Equal(got, want) {
		t.Fatalf("single-value SDPA = %v, want repeated V %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestSDPARejectsInvalidGQA(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(64, 3))
	if _, err := SDPA(x, x, x, 1, 3, 2, 64, 1, 1); err == nil {
		t.Fatal("expected SDPA to reject nHeads not divisible by nKVHeads")
	}
}
