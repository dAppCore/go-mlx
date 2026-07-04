// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import (
	"testing"

	shared "dappco.re/go/inference/scheme"
)

// The registry is ONE store behind two import paths: a scheme registered
// through this package's wrapper resolves through the shared package's own
// functions, and a scheme registered through the shared package resolves
// through this package's wrapper. Either direction failing would mean the
// alias had grown a local shadow registry — the one thing this compatibility
// layer must never do.
func TestRegistryCrossVisibility_Good(t *testing.T) {
	// Register via the LOCAL wrapper, resolve via the SHARED import.
	if res := RegisterMixer(mixerInfo{"cross-vis-local-to-shared", StateRecurrent}); !res.OK {
		t.Fatalf("RegisterMixer (local wrapper) failed: %+v", res)
	}
	m, ok := shared.MixerFor("cross-vis-local-to-shared")
	if !ok {
		t.Fatal("mixer registered via the local wrapper did not resolve via shared.MixerFor")
	}
	if m.State() != shared.StateRecurrent {
		t.Errorf("state = %v, want StateRecurrent", m.State())
	}

	// Register via the SHARED import, resolve via the LOCAL wrapper.
	if res := shared.RegisterCache(cacheInfo{"cross-vis-shared-to-local", StateKVCache}); !res.OK {
		t.Fatalf("shared.RegisterCache failed: %+v", res)
	}
	c, ok := CacheFor("cross-vis-shared-to-local")
	if !ok {
		t.Fatal("cache registered via shared.RegisterCache did not resolve via the local wrapper CacheFor")
	}
	if c.Serves() != StateKVCache {
		t.Errorf("serves = %v, want StateKVCache", c.Serves())
	}
}

// The builtin catalogue is seeded exactly once even though scheme.go imports
// the shared package (whose own init() does the seeding): there is no second,
// local init() re-registering the same names. dappco.re/go's Registry.Set()
// overwrites an existing key rather than erroring or appending a duplicate,
// so a second registration of the identical seed would in fact have been
// silently benign — this test pins the actually-shipped behaviour (a clean
// single entry, because the local seed was removed) rather than leaning on
// that overwrite tolerance.
func TestBuiltinSeedNotDuplicated_Good(t *testing.T) {
	count := func(names []string, want string) int {
		n := 0
		for _, name := range names {
			if name == want {
				n++
			}
		}
		return n
	}
	if n := count(MixerKinds(), "softmax-hybrid"); n != 1 {
		t.Errorf("softmax-hybrid appears %d times in MixerKinds(), want 1", n)
	}
	if n := count(CacheModes(), "q8"); n != 1 {
		t.Errorf("q8 appears %d times in CacheModes(), want 1", n)
	}
	if n := count(CacheModes(), "recurrent"); n != 1 {
		t.Errorf("recurrent appears %d times in CacheModes(), want 1", n)
	}
	if n := count(QuantKinds(), "affine"); n != 1 {
		t.Errorf("affine appears %d times in QuantKinds(), want 1", n)
	}
	if n := count(DTypeNames(), "bfloat16"); n != 1 {
		t.Errorf("bfloat16 appears %d times in DTypeNames(), want 1", n)
	}
}

// BFloat16 and Float32 alias the shared package's exported instances by
// identity, not merely by equal field values — proving the local names are
// the same DType values the shared registry seeded, not parallel copies.
func TestBuiltinDTypesAliasSharedInstances_Good(t *testing.T) {
	if BFloat16 != shared.BFloat16 {
		t.Error("scheme.BFloat16 is not identical to shared.BFloat16")
	}
	if Float32 != shared.Float32 {
		t.Error("scheme.Float32 is not identical to shared.Float32")
	}
}
