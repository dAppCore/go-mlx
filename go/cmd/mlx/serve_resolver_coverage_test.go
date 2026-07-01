// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// setLoader swaps the model loader before the first ResolveModel; the boot
// load then runs through the injected loader rather than the default cgo
// engine. Verified with a recording fake loader (no real model).
func TestHotSwapResolver_SetLoader_UsesInjectedLoader_Good(t *testing.T) {
	r := newHotSwapResolver("/boot/path", "", 0, nil)
	called := 0
	var gotPath string
	r.setLoader(func(path string, _ ...mlx.LoadOption) (inference.TextModel, error) {
		called++
		gotPath = path
		return nil, core.NewError("loader sentinel")
	})

	// ResolveModel triggers the lazy boot load → the injected loader runs.
	_, err := r.ResolveModel(context.Background(), "")
	if err == nil {
		t.Fatal("ResolveModel error = nil, want the injected loader's sentinel error")
	}
	if called != 1 {
		t.Fatalf("injected loader called %d times, want 1", called)
	}
	if gotPath != "/boot/path" {
		t.Fatalf("loader path = %q, want the boot path", gotPath)
	}
	// The init error is sticky: a second ResolveModel returns it without
	// re-invoking the loader (initial.Do fired once).
	if _, err := r.ResolveModel(context.Background(), ""); err == nil {
		t.Fatal("second ResolveModel error = nil, want the sticky init error")
	}
	if called != 1 {
		t.Fatalf("loader re-invoked (%d), want exactly 1 (initial.Do guards)", called)
	}
}

// setSpeculativeLoader swaps the target+draft loader before the first
// ResolveModel; this is how serve --native keeps the draft-detection path but
// loads both models through the no-cgo native contract.
func TestHotSwapResolver_SetSpeculativeLoader_UsesInjectedPairLoader_Good(t *testing.T) {
	r := newHotSwapResolver("/target/path", "/draft/path", 6, nil)
	called := 0
	var gotTarget, gotDraft string
	var gotBlock int
	r.setSpeculativeLoader(func(target, draft string, block int, _ ...mlx.LoadOption) (inference.TextModel, error) {
		called++
		gotTarget = target
		gotDraft = draft
		gotBlock = block
		return nil, core.NewError("pair loader sentinel")
	})

	_, err := r.ResolveModel(context.Background(), "")
	if err == nil {
		t.Fatal("ResolveModel error = nil, want the injected pair loader's sentinel error")
	}
	if called != 1 {
		t.Fatalf("pair loader called %d times, want 1", called)
	}
	if gotTarget != "/target/path" || gotDraft != "/draft/path" || gotBlock != 6 {
		t.Fatalf("pair loader args target=%q draft=%q block=%d, want staged target/draft/block", gotTarget, gotDraft, gotBlock)
	}
}
