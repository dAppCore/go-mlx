// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/mlx/pkg/safetensors"
)

// TestRegisterLoader covers the loader registry: register → lookup → invoke, plus the no-op guards
// (empty id, nil fn must not clear an existing loader). Uses a probe id so it never collides with a
// real arch registration.
func TestRegisterLoader(t *testing.T) {
	const id = "zz_registry_probe"
	if LookupLoader(id) != nil {
		t.Fatalf("probe id %q unexpectedly already registered", id)
	}
	called := 0
	RegisterLoader(id, func(string) (*LoadedModel, *safetensors.DirMapping, error) { called++; return nil, nil, nil })
	fn := LookupLoader(id)
	if fn == nil {
		t.Fatal("expected a loader after RegisterLoader")
	}
	_, _, _ = fn("x")
	if called != 1 {
		t.Fatalf("registered loader not invoked (called=%d)", called)
	}
	RegisterLoader("", fn)  // empty id → no-op
	RegisterLoader(id, nil) // nil fn → no-op, keeps the existing loader
	if LookupLoader(id) == nil {
		t.Fatal("nil-fn RegisterLoader must not clear an existing loader")
	}
}
