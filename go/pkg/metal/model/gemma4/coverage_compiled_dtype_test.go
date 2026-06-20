// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the storage-dtype-follow cast branches inside the compiled layer
// closure (gemma4CompiledLayerStep): when the FixedKVCache stores K/V in a
// half-precision dtype, the closure converts the freshly projected K/V — and the
// query the SDPA reads against them — to the cache dtype before the write. The
// fp32-cache tests in coverage_compiled_layer_test.go skip these casts because
// their storage dtype already equals the compute dtype; a bf16-storage cache
// forces them.

// cov4clPrimeFixedBF16 mirrors cov4clPrimeFixed but allocates bf16 storage, so
// the compiled closure sees cacheDType == bf16 != fp32 (the RMSNorm stream).
func cov4clPrimeFixedBF16(t *testing.T, maxSize, headDim, fill int) *metal.FixedKVCache {
	t.Helper()
	cache := metal.NewFixedKVCacheWithDType(maxSize, metal.DTypeBFloat16)
	for i := 0; i < fill; i++ {
		k := metal.AsType(seqArray(0.1+0.01*float32(i), 1, 1, 1, headDim), metal.DTypeBFloat16)
		v := metal.AsType(seqArray(0.2+0.01*float32(i), 1, 1, 1, headDim), metal.DTypeBFloat16)
		gotK, gotV := cache.Update(k, v, 1)
		if err := metal.Eval(gotK, gotV); err != nil {
			t.Fatalf("bf16 prime Update eval: %v", err)
		}
		metal.Free(k, v)
	}
	return cache
}

// TestCompiledLayer_OwnerBF16Cast drives the owner pre-cap regime against a bf16
// cache so the K/V (and query) dtype-cast branch executes. The hit counter
// confirms the closure served the step rather than declining.
func TestCompiledLayer_OwnerBF16Cast(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	layer := m.Layers[1] // full attention owner
	if layer.LayerType != "full_attention" {
		t.Fatalf("layer 1 type = %s, want full_attention", layer.LayerType)
	}
	cache := cov4clPrimeFixedBF16(t, 16, 32, 1)

	before := CompiledLayerDecodeHits()
	x := seqArray(0.02, 1, 1, 32)
	defer metal.Free(x)
	out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
	if !ok {
		t.Fatal("owner declined the compiled path against a bf16 cache")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval bf16-cast owner output: %v", err)
	}
	if CompiledLayerDecodeHits() == before {
		t.Fatal("bf16-cast owner compiled closure did not serve")
	}
}

// TestCompiledLayer_ConsumerBF16Cast drives the consumer regime against a bf16
// shared-KV state so the consumer-side query dtype-cast branch executes (the
// arm that converts qr to the cache dtype before the masked SDPA over the
// shared owner cache).
func TestCompiledLayer_ConsumerBF16Cast(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	// numShared=1 makes the last layer a KV-shared consumer.
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 3, sliding: 4, pattern: 2, numShared: 1,
	})
	ownerCache := cov4clPrimeFixedBF16(t, 16, 32, 1)

	// Seed the owner step so its borrowed bf16 KV becomes the consumer's shared
	// state.
	owner := m.Layers[1]
	xOwner := seqArray(0.02, 1, 1, 32)
	defer metal.Free(xOwner)
	_, ownerKV, ok := owner.compiledDecodeForward(xOwner, ownerCache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
	if !ok {
		t.Skip("owner declined; cannot stage a shared bf16 consumer state on this build")
	}
	if ownerKV.Keys != nil {
		if err := metal.Eval(ownerKV.Keys, ownerKV.Values); err != nil {
			t.Fatalf("eval owner KV: %v", err)
		}
	}

	consumer := m.Layers[len(m.Layers)-1]
	before := CompiledLayerDecodeHits()
	xCons := seqArray(0.03, 1, 1, 32)
	defer metal.Free(xCons)
	out, _, ok := consumer.compiledDecodeForward(xCons, ownerCache, 1, 1, nil, nil, ownerKV, m.Cfg)
	if !ok {
		t.Skip("consumer declined the shared bf16 state on this build")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval bf16-cast consumer output: %v", err)
	}
	if CompiledLayerDecodeHits() == before {
		t.Fatal("bf16-cast consumer compiled closure did not serve")
	}
}
