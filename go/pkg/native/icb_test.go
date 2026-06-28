// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"testing"
	"unsafe"
)

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

func TestNormProjectICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dIn, dOut = 128, 256
	x := syntheticFloat32(dIn, 3)
	normW := syntheticFloat32(dIn, 5)
	projW := syntheticFloat32(dOut*dIn, 7)
	if _, err := NormProjectICB(x, normW, projW, dIn, dOut, 1e-5, 1); err != nil {
		t.Fatalf("NormProjectICB warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := NormProjectICB(x, normW, projW, dIn, dOut, 1e-5, 1); err != nil {
			t.Fatalf("NormProjectICB: %v", err)
		}
	})
	if allocs > 205 {
		t.Fatalf("NormProjectICB allocations = %.0f, want <= 205", allocs)
	}
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

func TestAttentionBlockICBKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))

	if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("AttentionBlockICB: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasNorm := residentBufs[key(layer.AttnNormW)]
	_, hasQ := residentBufs[key(layer.WQ)]
	_, hasO := residentBufs[key(layer.WO)]
	residentBufMu.Unlock()

	if !hasNorm || !hasQ || !hasO {
		t.Fatalf("AttentionBlockICB did not keep fixed weights resident (norm=%v q=%v o=%v resident=%d want>=3)", hasNorm, hasQ, hasO, got)
	}
}

func TestAttentionBlockICBScratchPoolKeepsShapesResident(t *testing.T) {
	attentionBlockICBScratchPools = sync.Map{}
	t.Cleanup(func() { attentionBlockICBScratchPools = sync.Map{} })

	small := &attentionBlockICBScratch{dModel: 64, qDim: 64, nHeads: 1, nKVHeads: 1, headDim: 64, kvLen: 2}
	large := &attentionBlockICBScratch{dModel: 128, qDim: 128, nHeads: 2, nKVHeads: 1, headDim: 64, kvLen: 4}
	smallPool := attentionBlockICBScratchPoolFor(small.dModel, small.qDim, small.nHeads, small.nKVHeads, small.headDim, small.kvLen)
	largePool := attentionBlockICBScratchPoolFor(large.dModel, large.qDim, large.nHeads, large.nKVHeads, large.headDim, large.kvLen)
	if smallPool == largePool {
		t.Fatal("AttentionBlock ICB scratch reused one pool for distinct attention shapes")
	}

	putAttentionBlockICBScratch(small)
	putAttentionBlockICBScratch(large)

	if got := smallPool.Get(); got != small {
		t.Fatal("AttentionBlock ICB scratch pool evicted the small shape after using the larger shape")
	}
	if got := largePool.Get(); got != large {
		t.Fatal("AttentionBlock ICB scratch pool evicted the larger shape after reusing the small shape")
	}
}

func TestAttentionBlockICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("AttentionBlockICB warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, blockErr = AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	})
	if blockErr != nil {
		t.Fatalf("AttentionBlockICB: %v", blockErr)
	}
	if allocs > 80 {
		t.Fatalf("AttentionBlockICB allocations = %.0f, want <= 80", allocs)
	}
}
