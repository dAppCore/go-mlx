// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for whole-stack compiled decode (compiled_stack.go). The stack closure
// chains every layer's gemma4CompiledLayerStep into one trace; it is reachable by
// building FixedKVCaches for each owner layer, enabling MLX_COMPILED_STACK + the
// runtime gate, and calling compiledStackHidden directly with a decode-shaped
// hidden. Helpers are prefixed cov4cs to stay unique in the package test scope.

// cov4csFixedCaches builds and primes one FixedKVCache per owner layer of the
// model, matching the model's CacheIndexByLayer slot count, each filled to `fill`
// tokens (kept below maxSize so every layer stays a pre-cap owner).
func cov4csFixedCaches(t *testing.T, m *Gemma4Model, maxSize, headDim, fill int) []metal.Cache {
	t.Helper()
	numCaches := 0
	for _, idx := range m.CacheIndexByLayer {
		if idx >= 0 {
			numCaches++
		}
	}
	caches := make([]metal.Cache, numCaches)
	for i := range caches {
		caches[i] = cov4clPrimeFixed(t, maxSize, headDim, fill)
	}
	return caches
}

// TestCompiledStack_Decode_Good drives compiledStackHidden through a 2-owner
// dense quantised model: both layers are pre-cap owners, so the stack plan
// resolves and the composed closure returns the last-layer hidden.
func TestCompiledStack_Decode_Good(t *testing.T) {
	t.Setenv("MLX_COMPILED_STACK", "1")
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))

	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 16, pattern: 2,
	})
	// sliding window 16 == maxSize keeps layer 0 (sliding) below capacity at fill 1.
	caches := cov4csFixedCaches(t, m, 16, 32, 1)
	defer metal.FreeCaches(caches)

	hit := false
	for step := 0; step < 3; step++ {
		h := seqArray(0.01*float32(step+1), 1, 1, 32)
		out, ok := m.compiledStackHidden(h, caches, nil, 1, 1)
		if ok {
			hit = true
			if err := metal.Eval(out); err != nil {
				t.Fatalf("step %d eval: %v", step, err)
			}
			shape := out.Shape()
			if len(shape) != 3 || shape[2] != 32 {
				t.Fatalf("step %d stack hidden shape = %v, want [...32]", step, shape)
			}
			metal.Free(out)
		}
		metal.Free(h)
	}
	if !hit {
		t.Fatal("compiledStackHidden never served the composed closure")
	}
}

// TestCompiledStack_Decline_Bad covers the decline branches of compiledStackHidden
// and buildStackPlan: the env gate off, the wrong shape, and a non-Fixed cache
// slot all return ok=false without panicking.
func TestCompiledStack_Decline_Bad(t *testing.T) {
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 16, pattern: 2,
	})

	// env gate off: declines regardless.
	h := seqArray(0.01, 1, 1, 32)
	if _, ok := m.compiledStackHidden(h, nil, nil, 1, 1); ok {
		t.Fatal("stack must decline with MLX_COMPILED_STACK unset")
	}
	metal.Free(h)

	t.Setenv("MLX_COMPILED_STACK", "1")
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))

	// wrong shape (L != 1) declines before plan building.
	hL := seqArray(0.01, 1, 2, 32)
	if _, ok := m.compiledStackHidden(hL, nil, nil, 1, 2); ok {
		t.Fatal("stack must decline for L=2")
	}
	metal.Free(hL)

	// non-Fixed caches: buildStackPlan declines on the cache-type check.
	plainCaches := []metal.Cache{metal.NewKVCache(), metal.NewKVCache()}
	defer metal.FreeCaches(plainCaches)
	hP := seqArray(0.01, 1, 1, 32)
	if _, ok := m.compiledStackHidden(hP, plainCaches, nil, 1, 1); ok {
		t.Fatal("stack must decline when caches are not FixedKVCache")
	}
	metal.Free(hP)
}

// TestCompiledStack_BuildPlanDeclinesUnquantized_Bad covers buildStackPlan's
// eligibility decline: an unquantised model's layers are not trace-eligible, so
// the plan refuses even with Fixed caches and the gate on.
func TestCompiledStack_BuildPlanDeclinesUnquantized_Bad(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := loadGemma4DenseTestModel(t) // bf16, not quantised
	caches := cov4csFixedCaches(t, m, 16, 8, 1)
	defer metal.FreeCaches(caches)

	_, _, ok := m.buildStackPlan(caches)
	if ok {
		t.Fatal("buildStackPlan must decline an unquantised (non-trace-eligible) model")
	}
}

// TestCompiledStack_MakeStackTraceKey_Good covers makeStackTraceKey: the key
// records the layer count and copies each per-layer key, and stops at the 128
// signature cap.
func TestCompiledStack_MakeStackTraceKey_Good(t *testing.T) {
	keys := []gemma4CompiledLayerKey{
		{regime: gemma4LayerOwnerPreCap, hidden: 32, seqLen: 1},
		{regime: gemma4LayerConsumer, hidden: 32, seqLen: 1},
	}
	k := makeStackTraceKey(keys)
	if k.n != 2 {
		t.Fatalf("trace key n = %d, want 2", k.n)
	}
	if k.sig[0].regime != gemma4LayerOwnerPreCap || k.sig[1].regime != gemma4LayerConsumer {
		t.Fatalf("trace key sig not copied: %+v", k.sig[:2])
	}

	// more than 128 keys: the loop stops at the cap without panicking.
	big := make([]gemma4CompiledLayerKey, 200)
	for i := range big {
		big[i] = gemma4CompiledLayerKey{hidden: int32(i)}
	}
	kb := makeStackTraceKey(big)
	if kb.n != 200 {
		t.Fatalf("trace key n = %d, want 200 (count is unbounded; only the sig array caps)", kb.n)
	}
}
