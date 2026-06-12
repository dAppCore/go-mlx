// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// annotateMetallib adds metallib provenance + a kernel-launch proof to the
// discovery report (under -probe-device). "do I have everything I need to
// run inference here?" includes "do the GPU kernels actually load" — a
// bundle or install with a missing/misplaced metallib fails only at first
// Metal op, so discover forces that op and reports the result.
func annotateMetallib(report *inference.MachineDiscoveryReport) {
	path, fromEnv := metal.MetallibResolution()
	if report.Labels == nil {
		report.Labels = map[string]string{}
	}
	report.Labels["metallib_path"] = path
	report.Labels["metallib_source"] = classifyMetallibSource(path, fromEnv, core.TempDir())
	report.Labels["metallib_kernel"] = probeMetalKernel()
}

// classifyMetallibSource names where the resolved metallib came from. Path
// shape is the discriminator:
//   - the embed_metallib extract lands under <tmp>/lthn-mlx/<hash>/
//   - NSBundle resolution lands under <bundle>/Contents/Resources/
//   - the dev-tree walk lands under .../dist/lib/
//   - anything else pre-set in the env is the operator's own choice
func classifyMetallibSource(path string, fromEnv bool, tmpDir string) string {
	switch {
	case path == "":
		return "unresolved"
	case tmpDir != "" && core.HasPrefix(path, core.PathJoin(tmpDir, "lthn-mlx")+"/"):
		return "embedded"
	case core.Contains(path, "/Contents/Resources/"):
		return "bundle"
	case core.HasSuffix(core.PathDir(path), "dist/lib"):
		return "dev-tree"
	case fromEnv:
		return "env"
	default:
		return "external"
	}
}

// probeMetalKernel proves the GPU pipeline end-to-end: one tiny op forces
// MLX's Metal device construction, which loads the metallib (lib/mlx
// device.cpp load_default_library). "ok" means kernels launch with the
// resolved metallib — no model, microseconds.
func probeMetalKernel() (result string) {
	if !metal.MetalAvailable() {
		return "skipped: no usable Metal device"
	}
	// Array creation panics on MLX errors by contract (creation failing is
	// normally a programmer error) — but a missing/misplaced metallib fails
	// exactly there, and reporting that failure is this probe's job.
	defer func() {
		if r := recover(); r != nil {
			result = core.Sprintf("failed: %v", r)
		}
	}()
	a := metal.FromValues([]float32{1, 2, 3, 4}, 4)
	defer metal.Free(a)
	b := metal.AddScalar(a, 1)
	defer metal.Free(b)
	if err := metal.Eval(b); err != nil {
		return "failed: " + err.Error()
	}
	return "ok"
}
