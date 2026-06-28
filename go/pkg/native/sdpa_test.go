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

func TestSDPABF16ScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getSDPABF16Scratch(128, 256, 256, 128)
	if err != nil {
		t.Fatalf("get small SDPA scratch: %v", err)
	}
	putSDPABF16Scratch(small)

	large, err := getSDPABF16Scratch(256, 512, 512, 256)
	if err != nil {
		t.Fatalf("get large SDPA scratch: %v", err)
	}
	putSDPABF16Scratch(large)

	gotSmall, err := getSDPABF16Scratch(128, 256, 256, 128)
	if err != nil {
		t.Fatalf("get small SDPA scratch again: %v", err)
	}
	defer putSDPABF16Scratch(gotSmall)
	if gotSmall != small {
		t.Fatal("SDPA scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getSDPABF16Scratch(256, 512, 512, 256)
	if err != nil {
		t.Fatalf("get large SDPA scratch again: %v", err)
	}
	defer putSDPABF16Scratch(gotLarge)
	if gotLarge != large {
		t.Fatal("SDPA scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestSDPAAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	if _, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
		t.Fatalf("SDPA warmup: %v", err)
	}

	var sdpaErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, sdpaErr = SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	})
	if sdpaErr != nil {
		t.Fatalf("SDPA: %v", sdpaErr)
	}
	if allocs > 10 {
		t.Fatalf("SDPA allocations = %.0f, want <= 10", allocs)
	}
}
