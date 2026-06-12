// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/profile"
)

// errDiffusionServeDisabled quarantines diffusion_gemma from the serve while
// the within-request memory accumulation is open (#77): the denoise loop
// accumulates live arrays per step — ~36GB peak at 21k prefix x 3 canvases
// in the instrumented probe, 176GB footprint on a real book chapter at 13
// canvases. The per-request allocator-cache clear (8edc7964) cures the
// turn-over-turn growth but cannot touch within-request growth. Short
// `diffuse` CLI runs remain fine.
var errDiffusionServeDisabled = core.NewError(
	"diffusion_gemma serving is temporarily disabled: long generations accumulate memory within a single request (#77) — use the `diffuse` CLI verb for short runs, or serve the gemma-4-26B-A4B AR twin")

// probeServedArchitecture resolves a model directory's architecture from its
// config.json WITHOUT loading weights — the serve gate must answer before the
// multi-GB load, and the hot-swap path must answer before dropping the
// currently-serving model.
func probeServedArchitecture(modelPath string) string {
	res := core.ReadFile(core.PathJoin(modelPath, "config.json"))
	if !res.OK {
		return ""
	}
	data, _ := res.Value.([]byte)
	var probe struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
		TextConfig    struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
		return ""
	}
	return profile.ResolveArchitecture(probe.ModelType, probe.TextConfig.ModelType, probe.Architectures)
}

// serveArchitectureGate returns the refusal for architectures the serve will
// not load right now. Empty error = serve it.
func serveArchitectureGate(modelPath string) error {
	if probeServedArchitecture(modelPath) == "diffusion_gemma" {
		return errDiffusionServeDisabled
	}
	return nil
}
