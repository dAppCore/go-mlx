// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// unregisterMixerLoader removes a test-only kind so a registry test leaves no
// trace in the shared package-global map. Tests register under a unique key and
// defer this so concurrent sibling tests never see the synthetic loader.
func unregisterMixerLoader(kind string) { delete(mixerLoaders, kind) }

func TestMixerRegistry_RegisterMixerLoader_Good(t *testing.T) {
	const kind = "test-mixer-good"
	defer unregisterMixerLoader(kind)

	called := false
	RegisterMixerLoader(kind, func(MixerBuildCtx) (MixerCompute, error) {
		called = true
		return nil, nil
	})

	loader, ok := MixerLoaderFor(kind)
	if !ok || loader == nil {
		t.Fatalf("MixerLoaderFor(%q) = (%v,%v), want a registered loader", kind, loader, ok)
	}
	if _, _ = loader(MixerBuildCtx{}); !called {
		t.Fatal("resolved loader did not invoke the registered function")
	}
	if internal := lookupMixerLoader(kind); internal == nil {
		t.Fatal("lookupMixerLoader returned nil for a registered kind")
	}
}

func TestMixerRegistry_RegisterMixerLoader_Bad(t *testing.T) {
	// Empty kind and nil fn are both no-ops: nothing lands in the map and the
	// resolver still reports a clean miss.
	RegisterMixerLoader("", func(MixerBuildCtx) (MixerCompute, error) { return nil, nil })
	if _, ok := MixerLoaderFor(""); ok {
		t.Fatal("empty-kind registration was accepted, want no-op")
	}

	const kind = "test-mixer-nilfn"
	defer unregisterMixerLoader(kind)
	RegisterMixerLoader(kind, nil)
	if loader, ok := MixerLoaderFor(kind); ok || loader != nil {
		t.Fatalf("nil-fn registration was accepted: (%v,%v), want no-op", loader, ok)
	}
	if lookupMixerLoader(kind) != nil {
		t.Fatal("lookupMixerLoader found a loader after a nil-fn no-op register")
	}
}

func TestMixerRegistry_MixerLoaderFor_Ugly(t *testing.T) {
	// A later register for the same kind overrides the earlier one (documented
	// behaviour) — the resolver returns the most recent loader.
	const kind = "test-mixer-override"
	defer unregisterMixerLoader(kind)

	RegisterMixerLoader(kind, func(MixerBuildCtx) (MixerCompute, error) {
		return nil, errFirstLoader
	})
	RegisterMixerLoader(kind, func(MixerBuildCtx) (MixerCompute, error) {
		return nil, errSecondLoader
	})

	loader, ok := MixerLoaderFor(kind)
	if !ok || loader == nil {
		t.Fatalf("MixerLoaderFor after override = (%v,%v), want the second loader", loader, ok)
	}
	if _, err := loader(MixerBuildCtx{}); err != errSecondLoader {
		t.Fatalf("resolved loader err = %v, want the overriding loader's sentinel", err)
	}

	// An unregistered kind is a clean miss, never a panic or a stale loader.
	if loader, ok := MixerLoaderFor("test-mixer-never-registered"); ok || loader != nil {
		t.Fatalf("MixerLoaderFor(unregistered) = (%v,%v), want (nil,false)", loader, ok)
	}
}

var (
	errFirstLoader  = errSentinel("first")
	errSecondLoader = errSentinel("second")
)

type errSentinel string

func (e errSentinel) Error() string { return string(e) }
