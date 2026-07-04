// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import scheme "dappco.re/go/mlx/pkg/scheme"

// init registers the GSA mixer's compute with the scheme registry, overwriting
// the metadata-only catalogue entry (if any) with the value that also carries
// metal.MixerCompute. Importing this package for its side effect makes
// scheme.MixerFor(gsa.MixerKind) resolve to a real, compute-bearing mixer.
//
// A zero-value &Mixer{} carries the identity (Kind/State) the registry needs —
// State() reports scheme.StateRecurrent, so the engine pairs it with a
// recurrent-state cache (not a growing K/V cache). The engine fills its weights
// + slot geometry from the model config at load.
func init() { scheme.RegisterMixer(&Mixer{}) }
