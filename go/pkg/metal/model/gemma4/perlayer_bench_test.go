// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Per-Layer Embedding (PLE) graph benchmarks for the Gemma 4 slice/split
// methods. They moved here from package metal's ple_bench_test.go with the
// Gemma4Model type; the streamed-view and full-split paths each fetch the
// per-layer slice via mlx_take rather than materialising the whole PLE table.

func BenchmarkPLE_PerLayerInputViewsStreamed_Graph(b *testing.B) {
	combined := metal.RandomUniform(-1, 1, []int32{1, 1, 26, 256}, metal.DTypeFloat32)
	defer metal.Free(combined)
	metal.Materialize(combined)
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			HiddenSizePerLayerInput: 256,
			NumHiddenLayers:         26,
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		for i := int32(0); i < model.Cfg.NumHiddenLayers; i++ {
			slice := model.perLayerInputForLayer(combined, 1, 1, i)
			metal.Free(slice)
		}
	}
}

func BenchmarkPLE_SplitPerLayerInputTensor_Graph(b *testing.B) {
	combined := metal.RandomUniform(-1, 1, []int32{1, 1, 26, 256}, metal.DTypeFloat32)
	defer metal.Free(combined)
	metal.Materialize(combined)
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			HiddenSizePerLayerInput: 256,
			NumHiddenLayers:         26,
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		slices := model.splitPerLayerInputTensor(combined.Clone())
		metal.Free(slices...)
	}
}
