// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"

	core "dappco.re/go"
)

// requireMetalRuntime gates Metal-runtime tests in package metal. It is the
// shared guard the gemma4 architecture-test extraction moved out with the
// gemma4 suite (it previously lived in this package's gemma4_test.go); the
// ~20 callers that stayed in package metal still need it, so the helper is
// recovered here. Tests skip unless built with -tags metal_runtime and a usable
// Metal device is present.
func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// seqArray builds a float32 array of the given shape filled with start + 0.01*i.
// Like requireMetalRuntime, this shared metal test helper was carried out with
// the gemma4 suite (it survives in gemma4's model_test.go too); the metal-side
// callers (model/prompt_cache/moe_model tests) still need it, so it is recovered
// here in package-metal form.
func seqArray(start float32, shape ...int) *Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	for i := range size {
		data[i] = start + 0.01*float32(i)
	}
	return FromValues(data, shape...)
}

// TestMetal_MetallibResolution_Good: after Init the process reports a non-empty
// resolved metallib path. The package test binary runs with MLX_METALLIB_PATH
// exported, so when the from-env flag is set the reported path must equal that
// env value (Init captures it verbatim before any device op).
func TestMetal_MetallibResolution_Good(t *testing.T) {
	Init()
	path, fromEnv := MetallibResolution()
	if path == "" {
		t.Fatal("MetallibResolution() path = \"\", want a resolved metallib path")
	}
	if fromEnv {
		if env := core.Env("MLX_METALLIB_PATH"); env != "" && path != env {
			t.Errorf("from-env path = %q, want the env value %q", path, env)
		}
	}
}

// TestMetal_defaultMetallibPath_Good: the resolver always yields a non-empty
// candidate — a bundled/CWD-walked absolute path when one exists, otherwise the
// bare "mlx.metallib" filename fallback. It must never return an empty string.
func TestMetal_defaultMetallibPath_Good(t *testing.T) {
	if got := defaultMetallibPath(); got == "" {
		t.Error("defaultMetallibPath() = \"\", want a candidate path or the bare filename")
	}
}

// TestMetal_MaterializeAsync_Good: queuing a tiny synthetic array for async GPU
// evaluation must not error; a subsequent Synchronize drains it cleanly. Errors
// are logged, not returned, so the post-condition is a clean LastError.
func TestMetal_MaterializeAsync_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValue(float32(2.5))
	defer Free(a)
	MaterializeAsync(a)
	Synchronize(DefaultStream())
	if err := LastError(); err != nil {
		t.Fatalf("LastError after MaterializeAsync = %v, want nil", err)
	}
	// An empty async queue request is also a defined no-op.
	MaterializeAsync()
}
