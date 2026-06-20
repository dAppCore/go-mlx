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
