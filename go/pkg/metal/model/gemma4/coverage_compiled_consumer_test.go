// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the consumer (KV-shared) regime of the compiled layer and stack
// closures. A 3-layer quantised model with num_kv_shared_layers=1 makes the last
// layer a consumer that reads the prior owner's fixed K/V; driving its compiled
// decode covers the consumer arms in compiled_layer.go and compiled_stack.go that
// the owner-only tests leave untouched. Helpers prefixed cov4cc.

// TestCompiledLayer_ConsumerDecode_Good drives the consumer layer's compiled
// closure with a borrowed fixed K/V from an owner, exercising the consumer
// branch of gemma4CompiledLayerStep (no K/V projection, attends the shared cache).
func TestCompiledLayer_ConsumerDecode_Good(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 3, sliding: 16, pattern: 2, numShared: 1,
	})
	if m.PreviousKVs[2] != 1 {
		t.Fatalf("layer 2 should share layer 1's KV, PreviousKVs=%v", m.PreviousKVs)
	}
	owner := m.Layers[1]
	consumer := m.Layers[2]

	// Build the owner's fixed cache and run one compiled owner step to produce a
	// borrowed K/V state the consumer can read.
	ownerCache := cov4clPrimeFixed(t, 16, 32, 1)
	xOwner := seqArray(0.02, 1, 1, 32)
	_, ownerKV, ok := owner.compiledDecodeForward(xOwner, ownerCache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
	metal.Free(xOwner)
	if !ok || !ownerKV.HasState() {
		t.Fatalf("owner step did not produce shared K/V state (ok=%v)", ok)
	}

	before := CompiledLayerDecodeHits()
	xCons := seqArray(0.03, 1, 1, 32)
	// the consumer reads the owner's K/V via prev; its own cache arg is unused.
	out, kv, ok := consumer.compiledDecodeForward(xCons, ownerCache, 1, 1, nil, nil, ownerKV, m.Cfg)
	if !ok {
		t.Fatal("consumer layer declined the compiled path")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval consumer hidden: %v", err)
	}
	// the consumer returns the same shared K/V it was handed (it owns no cache).
	if !kv.HasState() {
		t.Fatal("consumer should pass the shared K/V through")
	}
	metal.Free(out, xCons)
	if CompiledLayerDecodeHits() == before {
		t.Fatal("consumer compiled closure did not serve")
	}
}

// TestCompiledLayer_ConsumerDeclinesNonFixedShare_Bad covers the consumer decline
// branch: a prev with state that is not a fixed borrowed K/V is refused.
func TestCompiledLayer_ConsumerDeclinesNonFixedShare_Bad(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 3, sliding: 16, pattern: 2, numShared: 1,
	})
	consumer := m.Layers[2]

	// a prev that HasState but is not Fixed declines ("shared KV state is not fixed").
	k := seqArray(0.1, 1, 1, 4, 32)
	v := seqArray(0.2, 1, 1, 4, 32)
	defer metal.Free(k, v)
	notFixed := sharedKV{Keys: k, Values: v, Offset: 0, Fixed: false}

	x := seqArray(0.03, 1, 1, 32)
	defer metal.Free(x)
	if _, _, ok := consumer.compiledDecodeForward(x, metal.NewFixedKVCache(16), 1, 1, nil, nil, notFixed, m.Cfg); ok {
		t.Fatal("consumer must decline a non-fixed shared K/V")
	}
}

// TestCompiledStack_ConsumerChain_Good drives the whole-stack closure on the
// shared-KV model: layer 2 chains off layer 1's freshly-written K/V inside the
// trace, covering the consumer arm of gemma4CompiledStackStep and buildStackPlan.
func TestCompiledStack_ConsumerChain_Good(t *testing.T) {
	t.Setenv("MLX_COMPILED_STACK", "1")
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 3, sliding: 16, pattern: 2, numShared: 1,
	})
	// one fixed cache per owner layer (layers 0 and 1; layer 2 is a consumer).
	caches := cov4csFixedCaches(t, m, 16, 32, 1)
	defer metal.FreeCaches(caches)

	hit := false
	for step := 0; step < 2; step++ {
		h := seqArray(0.01*float32(step+1), 1, 1, 32)
		out, ok := m.compiledStackHidden(h, caches, nil, 1, 1)
		if ok {
			hit = true
			if err := metal.Eval(out); err != nil {
				t.Fatalf("step %d eval: %v", step, err)
			}
			metal.Free(out)
		}
		metal.Free(h)
	}
	if !hit {
		t.Fatal("consumer-chain stack never served the composed closure")
	}
}
