// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/profile"
)

// errDiffusionServeDisabled quarantines diffusion_gemma from the serve while
// the zero-flag-regime memory regression is open (multi-turn serve books
// spike to OOM since b6f1d81 — bisected; the 26B AR twin is flat on the same
// stack). The engine itself is healthy: the `diffuse` CLI verb serves the
// model fine and stays available.
var errDiffusionServeDisabled = core.NewError(
	"diffusion_gemma serving is temporarily disabled: multi-turn serving of this architecture has an open memory regression — use the `diffuse` CLI verb, or serve the gemma-4-26B-A4B AR twin")

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
