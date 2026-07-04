// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Tests + benchmarks for gemma4ProportionalFreqs — Gemma 4's proportional RoPE
// frequency table. They moved here from package metal's fast_test.go /
// rope_bench_test.go with the gemma4-internal table builder. The metal RoPE
// kernels (RoPE / RoPEWithFreqs) are public and exercised via metal.* here; the
// metal-resident RoPE benches/tests stay in package metal.

func TestFast_RoPE_DefaultFreqsMatchesBasePath_Good(t *testing.T) {
	requireMetalRuntime(t)
	x := metal.RandomUniform(-1, 1, []int32{1, 4, 3, 16}, metal.DTypeFloat32)
	freqs := gemma4ProportionalFreqs(16, 16, 10000, 1)
	defer metal.Free(x, freqs)

	basePath := metal.RoPE(x, 16, false, 10000, 1, 7)
	freqPath := metal.RoPEWithFreqs(x, 16, false, 0, 1, 7, freqs)
	defer metal.Free(basePath, freqPath)
	if err := metal.Eval(basePath, freqPath); err != nil {
		t.Fatalf("Eval RoPE paths: %v", err)
	}
	floatSliceApprox(t, freqPath.Floats(), basePath.Floats())
}

func BenchmarkRoPE_Decode_BaseLocal10k_WithFreqs(b *testing.B) {
	x := metal.RandomUniform(0, 1, []int32{1, 8, 1, 128}, metal.DTypeFloat32)
	freqs := gemma4ProportionalFreqs(128, 128, 10000.0, 1.0)
	defer metal.Free(x, freqs)
	metal.Materialize(x, freqs)
	b.ReportAllocs()
	for b.Loop() {
		y := metal.RoPEWithFreqs(x, 128, false, 0, 1.0, 0, freqs)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkRoPE_WithFreqs_Decode_D256(b *testing.B) {
	x := metal.RandomUniform(0, 1, []int32{1, 4, 1, 256}, metal.DTypeFloat32)
	freqs := gemma4ProportionalFreqs(256, 256, 1000000.0, 8.0)
	defer metal.Free(x, freqs)
	metal.Materialize(x, freqs)
	b.ReportAllocs()
	for b.Loop() {
		y := metal.RoPEWithFreqs(x, 256, false, 0, 1.0, 0, freqs)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkRoPE_WithFreqs_Prefill_4k_D256(b *testing.B) {
	x := metal.RandomUniform(0, 1, []int32{1, 4, 4096, 256}, metal.DTypeFloat32)
	freqs := gemma4ProportionalFreqs(256, 256, 1000000.0, 8.0)
	defer metal.Free(x, freqs)
	metal.Materialize(x, freqs)
	b.ReportAllocs()
	for b.Loop() {
		y := metal.RoPEWithFreqs(x, 256, false, 0, 1.0, 0, freqs)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// --- gemma4ProportionalFreqs — table construction cost ---

func BenchmarkRoPE_BuildProportionalFreqs_D256(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		freqs := gemma4ProportionalFreqs(256, 256, 1000000.0, 8.0)
		metal.Materialize(freqs)
		metal.Free(freqs)
	}
}

func BenchmarkRoPE_BuildProportionalFreqs_D128(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		freqs := gemma4ProportionalFreqs(128, 128, 1000000.0, 8.0)
		metal.Materialize(freqs)
		metal.Free(freqs)
	}
}
