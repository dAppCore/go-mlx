// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// kvFactoryFakeMixer is a minimal scheme.Mixer for the factory's mode/build
// decisions — construction + registry resolution only, no Metal runtime.
type kvFactoryFakeMixer struct{ state scheme.StateKind }

func (kvFactoryFakeMixer) Kind() string              { return "kvfactory-fake" }
func (f kvFactoryFakeMixer) State() scheme.StateKind { return f.state }

// kvFactoryFakeModer additionally names a specific cache scheme — the MLA shape.
type kvFactoryFakeModer struct {
	kvFactoryFakeMixer
	mode string
}

func (f kvFactoryFakeModer) CacheMode() string { return f.mode }

// TestCacheModeForMixer_Good: a mixer with no CacheModer gets the default scheme
// for its StateKind; a mixer that names one gets that name.
func TestCacheModeForMixer_Good(t *testing.T) {
	cases := []struct {
		name string
		m    scheme.Mixer
		want string
	}{
		{"kv-default", kvFactoryFakeMixer{scheme.StateKVCache}, cacheModeDefault},
		{"recurrent", kvFactoryFakeMixer{scheme.StateRecurrent}, cacheModeRecurrent},
		{"named-mla", kvFactoryFakeModer{kvFactoryFakeMixer{scheme.StateKVCache}, "mla-latent"}, "mla-latent"},
	}
	for _, c := range cases {
		if got := CacheModeForMixer(c.m); got != c.want {
			t.Errorf("%s: CacheModeForMixer = %q, want %q", c.name, got, c.want)
		}
	}
}

// TestCacheModeForMixer_Bad: an empty CacheMode() falls through to the StateKind
// default rather than naming the empty mode (which would never resolve).
func TestCacheModeForMixer_Bad(t *testing.T) {
	m := kvFactoryFakeModer{kvFactoryFakeMixer{scheme.StateKVCache}, ""}
	if got := CacheModeForMixer(m); got != cacheModeDefault {
		t.Errorf("empty CacheMode() = %q, want fallthrough to %q", got, cacheModeDefault)
	}
}

// TestNewCacheForMode_Good: registered compute-bearing schemes build a cache.
func TestNewCacheForMode_Good(t *testing.T) {
	for _, mode := range []string{"mla-latent", "recurrent"} {
		c, ok := NewCacheForMode(mode, CacheParams{})
		if !ok || c == nil {
			t.Fatalf("NewCacheForMode(%q) = (%v, %v), want a cache", mode, c, ok)
		}
	}
}

// TestNewCacheForMode_Bad: an unregistered mode resolves to (nil, false).
func TestNewCacheForMode_Bad(t *testing.T) {
	if c, ok := NewCacheForMode("no-such-cache-mode", CacheParams{}); ok || c != nil {
		t.Fatalf("NewCacheForMode(unknown) = (%v, %v), want (nil, false)", c, ok)
	}
}

// TestNewCacheForMixer_Good: the factory builds a non-nil cache for each shape —
// the MLA-moder its latent store, a plain KV mixer the default, a recurrent
// mixer its holder.
func TestNewCacheForMixer_Good(t *testing.T) {
	mixers := []scheme.Mixer{
		kvFactoryFakeModer{kvFactoryFakeMixer{scheme.StateKVCache}, "mla-latent"},
		kvFactoryFakeMixer{scheme.StateKVCache},
		kvFactoryFakeMixer{scheme.StateRecurrent},
	}
	for _, m := range mixers {
		if c := NewCacheForMixer(m, CacheParams{}); c == nil {
			t.Fatalf("NewCacheForMixer(%q) = nil, want a cache", m.Kind())
		}
	}
}
