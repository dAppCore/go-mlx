// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

func TestSoftmaxLoader_KindAndState_Good(t *testing.T) {
	var m softmaxMixer
	if got := m.Kind(); got != "full_attention" {
		t.Fatalf("softmaxMixer.Kind() = %q, want %q", got, "full_attention")
	}
	if got := m.State(); got != scheme.StateKVCache {
		t.Fatalf("softmaxMixer.State() = %v, want StateKVCache", got)
	}
}

func TestSoftmaxLoader_buildSoftmax_Bad(t *testing.T) {
	// Wrong / missing Extra: the loader needs the layer *DenseConfig and refuses
	// a build with a descriptive error before touching any tensor.
	if _, err := buildSoftmax(MixerBuildCtx{Extra: nil}); err == nil {
		t.Fatal("buildSoftmax(Extra=nil) err = nil, want a missing-config error")
	}
	if _, err := buildSoftmax(MixerBuildCtx{Extra: "not-a-config"}); err == nil {
		t.Fatal("buildSoftmax(Extra=string) err = nil, want a config-type error")
	}
}

func TestSoftmaxLoader_buildSoftmax_Ugly(t *testing.T) {
	// Extra is the right type but the projection resolver yields no q_proj: the
	// loader must reject the layer rather than build a half-wired attention. The
	// Linear resolver returns nil for every name, so the missing-projection guard
	// fires before any Metal op.
	ctx := MixerBuildCtx{
		Extra:  &DenseConfig{},
		Linear: func(string) *Linear { return nil },
		Weight: func(string) *Array { return nil },
	}
	if _, err := buildSoftmax(ctx); err == nil {
		t.Fatal("buildSoftmax(no projections) err = nil, want a missing-projection error")
	}
}

func TestSoftmaxLoader_CloseMixer_Bad(t *testing.T) {
	// CloseMixer must tolerate a nil receiver and a nil attention block — a
	// half-constructed mixer is freed without panicking.
	var nilMixer *softmaxMixer
	nilMixer.CloseMixer()
	(&softmaxMixer{}).CloseMixer()
}

func TestSoftmaxLoader_Registered_Good(t *testing.T) {
	// The package init() must have registered the full-attention loader so a
	// config-composed model can resolve it; the resolved loader exercises the
	// same missing-config guard as buildSoftmax.
	loader, ok := MixerLoaderFor("full_attention")
	if !ok || loader == nil {
		t.Fatal("full_attention mixer not registered, want the softmax loader")
	}
	if _, err := loader(MixerBuildCtx{Extra: nil}); err == nil {
		t.Fatal("resolved loader accepted a nil config, want an error")
	}
}
