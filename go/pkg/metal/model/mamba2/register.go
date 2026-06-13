// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import scheme "dappco.re/go/mlx/pkg/scheme"

// register.go adds the Mamba-2 family identity + state contract to the engine's
// scheme catalogue. Blank-importing this package (the model load path does so
// when it sees a "mamba2" mixer kind) registers the weightless family seed; a
// loaded model overwrites it with a compute-bearing *Mixer carrying its
// per-layer Weights (see gemma4/mixer_ssm.go). This is the whole point of the
// registry: a new family member is one Set(), with no edit to the engine or the
// decoder loop.
//
//	import _ "dappco.re/go/mlx/pkg/metal/model/mamba2" // register the mamba2 family
func init() {
	scheme.RegisterMixer(familyInfo{})
}

// familyInfo is the weightless catalogue entry for the Mamba-2 family — identity
// + state kind only, so scheme.MixerFor("mamba2") resolves before any model
// loads. A built layer's *Mixer (with weights) overwrites it via the same Kind.
type familyInfo struct{}

func (familyInfo) Kind() string            { return "mamba2" }
func (familyInfo) State() scheme.StateKind { return scheme.StateRecurrent }
