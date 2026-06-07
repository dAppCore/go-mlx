// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
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
