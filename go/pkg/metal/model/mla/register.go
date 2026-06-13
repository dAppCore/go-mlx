// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import scheme "dappco.re/go/mlx/pkg/scheme"

// init registers the MLA mixer's compute with the scheme registry, overwriting
// the metadata-only catalogue entry (if any) with the value that also carries
// metal.MixerCompute. Importing this package for its side effect makes
// scheme.MixerFor(mla.MixerKind) resolve to a real, compute-bearing mixer — the
// "register a scheme, never an engine branch" contract from scheme.go.
//
// A zero-value &Mixer{} carries the identity (Kind/State) the registry needs;
// the engine fills its weights from the model config + safetensors at load.
func init() { scheme.RegisterMixer(&Mixer{}) }
