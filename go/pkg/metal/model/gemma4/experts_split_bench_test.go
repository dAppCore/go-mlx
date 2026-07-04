// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// BenchmarkExpertIDSplitLastDimArray_Gemma4Decode benchmarks splitLastDimArray —
// the gate/up last-dim split on the fused Gemma 4 MoE gate_up projection. It
// moved here from package metal's expert_id_matvec_bench_test.go with the
// gemma4-internal splitLastDimArray function; the metal-resident expert-id matvec
// benches stay in package metal.
func BenchmarkExpertIDSplitLastDimArray_Gemma4Decode(b *testing.B) {
	gateUp := metal.RandomUniform(-1, 1, []int32{2, 4096}, metal.DTypeFloat32)
	defer metal.Free(gateUp)
	metal.Materialize(gateUp)

	b.ReportAllocs()
	for b.Loop() {
		gate, up, ok := splitLastDimArray(gateUp)
		if !ok {
			b.Fatal("splitLastDimArray returned !ok")
		}
		metal.Materialize(gate, up)
		metal.Free(gate, up)
	}
}
