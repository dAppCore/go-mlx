// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// TestSoftmaxMixer_Kind_Good pins the config token a softmax-hybrid Gemma-4 build
// declares — the engine resolves the mixer via scheme.MixerFor("softmax-hybrid"),
// so the literal must stay exactly that.
func TestSoftmaxMixer_Kind_Good(t *testing.T) {
	a := &Gemma4Attention{}
	if got := a.Kind(); got != "softmax-hybrid" {
		t.Fatalf("Kind() = %q, want %q", got, "softmax-hybrid")
	}
}

// TestSoftmaxMixer_State_Good pins that the mixer declares a growing K/V cache
// (scheme.StateKVCache), so the cache layer operates on it and a recurrent cache
// scheme is rejected at load — the mixer-owns-state contract.
func TestSoftmaxMixer_State_Good(t *testing.T) {
	a := &Gemma4Attention{}
	if got := a.State(); got != scheme.StateKVCache {
		t.Fatalf("State() = %v, want scheme.StateKVCache", got)
	}
}
