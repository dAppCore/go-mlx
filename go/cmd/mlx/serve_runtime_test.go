// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// serve_runtime_test.go holds the Metal-runtime serve + hot-swap-resolver
// tests that load a TINY synthetic model through the real path (the portable,
// no-device logic tests live in serve_test.go / serve_resolver_test.go). Split
// by build tag per the repo's *_runtime_test.go convention (see
// pkg/metal/metal_runtime_test.go, stream_runtime_test.go).
package main

import (
	"context"
	"sync"
	"testing"

	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// ResolveModel lazily loads the staged boot model on first call and returns the
// same instance on the second (the active-model fast path). Driven over a TINY
// synthetic gemma3 so the genuine lazy-load (initial.Do → LoadModelAsTextModel)
// runs, not a stub.
func TestHotSwapResolver_ResolveModel_LazyLoad_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	var loaded int
	r := newHotSwapResolver(model, "", 0, nil)
	r.setOnLoad(func(inference.TextModel) { loaded++ })

	tm, err := r.ResolveModel(context.Background(), "")
	if err != nil {
		t.Fatalf("first ResolveModel: %v", err)
	}
	if tm == nil {
		t.Fatal("ResolveModel returned a nil model")
	}
	// Second call returns the already-active model without reloading.
	tm2, err := r.ResolveModel(context.Background(), "")
	if err != nil {
		t.Fatalf("second ResolveModel: %v", err)
	}
	if tm2 != tm {
		t.Fatal("second ResolveModel returned a different instance — the active model must be reused")
	}
	if loaded != 1 {
		t.Fatalf("onLoad fired %d times, want exactly 1 (lazy single load)", loaded)
	}
	if r.CurrentPath() != model {
		t.Fatalf("CurrentPath = %q, want the loaded model path", r.CurrentPath())
	}
}

// A model-less resolver (empty boot path) returns errNoModelLoaded from
// ResolveModel without attempting any load — the fleet boot posture.
func TestHotSwapResolver_ResolveModel_ModellessError_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	r := newHotSwapResolver("", "", 0, nil)
	if _, err := r.ResolveModel(context.Background(), ""); err == nil {
		t.Fatal("model-less ResolveModel returned nil error, want errNoModelLoaded")
	}
}

// Replace loads a new synthetic model and atomically swaps it in, returning the
// previous loadedModel. The onLoad hook re-fires for the swapped-in model.
func TestHotSwapResolver_Replace_SwapsModel_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	first := writeSyntheticGemma3Model(t, t.TempDir())
	second := writeSyntheticGemma3Model(t, t.TempDir())
	var mu sync.Mutex
	loads := 0
	r := newHotSwapResolver(first, "", 0, nil)
	r.setOnLoad(func(inference.TextModel) { mu.Lock(); loads++; mu.Unlock() })

	// Force the boot load so there's a prior active model to return.
	if _, err := r.ResolveModel(context.Background(), ""); err != nil {
		t.Fatalf("boot load: %v", err)
	}
	prev, newActive, err := r.Replace(second, []mlx.LoadOption{})
	if err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if newActive != second {
		t.Fatalf("new active path = %q, want %q", newActive, second)
	}
	if prev == nil || prev.modelPath != first {
		t.Fatalf("prev = %+v, want the first model as the previous active", prev)
	}
	if r.CurrentPath() != second {
		t.Fatalf("CurrentPath after swap = %q, want %q", r.CurrentPath(), second)
	}
	mu.Lock()
	defer mu.Unlock()
	if loads != 2 {
		t.Fatalf("onLoad fired %d times, want 2 (boot + swap)", loads)
	}
}
