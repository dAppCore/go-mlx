// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

func TestModel_Generate_AsyncDecodePrefetch_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate(GateAsyncDecodePrefetch, true))

	out := Zeros([]int32{1, 1, 2}, DTypeFloat32)
	defer Free(out)
	if err := asyncDecodePrefetch(0, "test", out); err != nil {
		t.Fatalf("asyncDecodePrefetch() error = %v", err)
	}
	if err := Eval(out); err != nil {
		t.Fatalf("Eval after asyncDecodePrefetch() error = %v", err)
	}

	cache := NewPagedKVCache(0, 2)
	defer cache.Reset()
	k, v := makeSingleTokenKV(1)
	defer Free(k, v)
	state := cache.UpdateBorrowedPages(k, v, 1)
	state.Free()
	timings, err := asyncDecodePrefetchWithCachesTrace("Model.Generate", 0, "test split", out, []Cache{cache})
	if err != nil {
		t.Fatalf("asyncDecodePrefetchWithCachesTrace() error = %v", err)
	}
	if timings.Logits <= 0 || timings.Cache != 0 {
		t.Fatalf("async prefetch timings = %+v, want production-shaped combined logits timing", timings)
	}
	splitTimings, err := asyncDecodePrefetchWithCachesTraceSplit("Model.Generate", 0, "test split", out, []Cache{cache})
	if err != nil {
		t.Fatalf("asyncDecodePrefetchWithCachesTraceSplit() error = %v", err)
	}
	if splitTimings.Logits <= 0 || splitTimings.Cache <= 0 {
		t.Fatalf("async split prefetch timings = %+v, want diagnostic logits and dirty-cache timing", splitTimings)
	}

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: NewForDecode(map[int32]string{0: "x"}),
	}
	for range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, TraceTokenPhases: true}) {
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 2 || phases[0].PrefetchDuration <= 0 {
		t.Fatalf("TokenPhases = %+v, want async next-token prefetch duration", phases)
	}
	if phases[0].PrefetchLogitsDuration <= 0 || phases[0].PrefetchCacheDuration != 0 {
		t.Fatalf("first phase prefetch split = %+v, want logits-only split for cacheless model", phases[0])
	}
}

func TestModel_Generate_AsyncDecodePrefetchRuntimeGate_Good(t *testing.T) {
	restoreOff := SetRuntimeGate(GateAsyncDecodePrefetch, false)
	t.Cleanup(restoreOff)
	if asyncDecodePrefetchEnabled() {
		t.Fatal("asyncDecodePrefetchEnabled() = true, want runtime gate off")
	}
	restoreOn := SetRuntimeGate(GateAsyncDecodePrefetch, true)
	t.Cleanup(restoreOn)
	if !asyncDecodePrefetchEnabled() {
		t.Fatal("asyncDecodePrefetchEnabled() = false, want runtime gate on")
	}
}

func TestModel_Generate_AsyncDecodePrefetch_Bad(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateAsyncDecodePrefetch, true))

	if err := asyncDecodePrefetch(0, "nil", nil); err != nil {
		t.Fatalf("asyncDecodePrefetch(nil) error = %v", err)
	}
}
