// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import scheme "dappco.re/go/mlx/pkg/scheme"

// init registers the MoBA mixer's compute with the scheme registry, overwriting
// the metadata-only catalogue entry (if any) with the value that also carries
// metal.MixerCompute. Importing this package for its side effect makes
// scheme.MixerFor(moba.MixerKind) resolve to a real, compute-bearing mixer.
//
// A zero-value &Mixer{} carries the identity (Kind/State) the registry needs;
// the engine fills its weights + block geometry from the model config at load.
func init() { scheme.RegisterMixer(&Mixer{}) }
