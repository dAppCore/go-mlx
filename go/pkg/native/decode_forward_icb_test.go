// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"testing"
)

func TestDecodeForwardICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}

	want, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	got, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICB: %v", err)
	}
	for i := range want {
		eqBytes(t, "DecodeForwardICB token", got[i], want[i])
	}
}

func TestDecodeForwardICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	if _, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardICB warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardICB: %v", forwardErr)
	}
	if allocs > 540 {
		t.Fatalf("DecodeForwardICB allocations = %.0f, want <= 540", allocs)
	}
}

func TestDecodeForwardICBCoreScratchPoolKeepsShapesResident(t *testing.T) {
	decodeForwardICBCoreScratchPools = sync.Map{}
	t.Cleanup(func() { decodeForwardICBCoreScratchPools = sync.Map{} })

	small := &decodeForwardICBCoreScratch{dModel: 64, qDim: 64, kvDim: 64, dFF: 128, nLayers: 1}
	large := &decodeForwardICBCoreScratch{dModel: 128, qDim: 128, kvDim: 64, dFF: 256, nLayers: 2}
	smallPool := decodeForwardICBCoreScratchPoolFor(small.dModel, small.qDim, small.kvDim, small.dFF, small.nLayers)
	largePool := decodeForwardICBCoreScratchPoolFor(large.dModel, large.qDim, large.kvDim, large.dFF, large.nLayers)
	if smallPool == largePool {
		t.Fatal("DecodeForward ICB core scratch reused one pool for distinct core shapes")
	}

	putDecodeForwardICBCoreScratch(small)
	putDecodeForwardICBCoreScratch(large)

	if got := smallPool.Get(); got != small {
		t.Fatal("DecodeForward ICB core scratch pool evicted the small shape after using the larger shape")
	}
	if got := largePool.Get(); got != large {
		t.Fatal("DecodeForward ICB core scratch pool evicted the larger shape after reusing the small shape")
	}
}
