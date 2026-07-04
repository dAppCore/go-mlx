// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Benchmarks for logitSoftcap — Gemma's 30.0 tanh-softcap on output logits —
// plus the end-to-end logit-compose tail that applies it. They moved here from
// package metal's decode_loop_bench_test.go with the gemma4-internal
// logitSoftcap function. The metal-resident output-projection/native-logit
// benches (which do not call logitSoftcap) stay in package metal.

// --- logitSoftcap — Gemma's 30.0 tanh-softcap on output logits ---

func BenchmarkDecodeLoop_LogitSoftcap_Vocab32k(b *testing.B) {
	x := metal.RandomUniform(-10, 10, []int32{1, 32000}, metal.DTypeFloat32)
	defer metal.Free(x)
	metal.Materialize(x)
	b.SetBytes(int64(32000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := logitSoftcap(x, 30.0)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkDecodeLoop_LogitSoftcap_Vocab128k(b *testing.B) {
	x := metal.RandomUniform(-10, 10, []int32{1, 128000}, metal.DTypeFloat32)
	defer metal.Free(x)
	metal.Materialize(x)
	b.SetBytes(int64(128000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := logitSoftcap(x, 30.0)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkDecodeLoop_LogitSoftcap_Vocab256k(b *testing.B) {
	x := metal.RandomUniform(-10, 10, []int32{1, 256000}, metal.DTypeFloat32)
	defer metal.Free(x)
	metal.Materialize(x)
	b.SetBytes(int64(256000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := logitSoftcap(x, 30.0)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// --- Output projection + softcap (Go graph path) ---

func BenchmarkDecodeLoop_LastTokenOutputQ4GoGraph_H2048_Vocab262k(b *testing.B) {
	hidden, normWeight, output := benchmarkLogitSoftcapQ4OutputFixture(b, 2048, 262208)
	defer metal.Free(hidden, normWeight)
	defer metal.FreeLinear(output)
	b.ReportAllocs()
	for b.Loop() {
		normed := metal.RMSNorm(hidden, normWeight, 1e-6)
		logits := output.Forward(normed)
		metal.Free(normed)
		capped := logitSoftcap(logits, 30)
		metal.Free(logits)
		if err := metal.Eval(capped); err != nil {
			metal.Free(capped)
			b.Fatalf("Eval(graph logits): %v", err)
		}
		metal.Free(capped)
	}
}

func benchmarkLogitSoftcapQ4OutputFixture(b *testing.B, hiddenDim, vocab int) (*metal.Array, *metal.Array, *metal.Linear) {
	b.Helper()
	if hiddenDim%64 != 0 {
		b.Fatalf("hiddenDim=%d must be divisible by group size 64", hiddenDim)
	}
	hidden := metal.RandomUniform(-1, 1, []int32{1, 1, int32(hiddenDim)}, metal.DTypeFloat32)
	normWeight := metal.RandomUniform(0.5, 1.5, []int32{int32(hiddenDim)}, metal.DTypeFloat32)
	packedWidth := hiddenDim / 8
	groups := hiddenDim / 64
	weightWords := make([]uint32, vocab*packedWidth)
	for i := range weightWords {
		weightWords[i] = uint32(i*1664525 + 1013904223)
	}
	scales := make([]float32, vocab*groups)
	biases := make([]float32, vocab*groups)
	for i := range scales {
		scales[i] = 0.005 * float32((i%17)+1)
		biases[i] = -0.03 + 0.002*float32(i%31)
	}
	output := metal.NewQuantizedLinear(
		metal.FromValues(weightWords, vocab, packedWidth),
		metal.FromValues(scales, vocab, groups),
		metal.FromValues(biases, vocab, groups),
		nil,
		64,
		4,
	)
	metal.Materialize(hidden, normWeight, output.Weight, output.Scales, output.Biases)
	return hidden, normWeight, output
}

// --- End-to-end logit compose (last hidden → token) ---

// Compose the realistic per-token tail: matmul (output proj) + softcap
// + argmax. This is the post-final-block compute, the closest a
// non-model-loading bench can get to per-token decode cost.
func BenchmarkDecodeLoop_LogitCompose_E2E_H2048_Vocab32k(b *testing.B) {
	x := metal.RandomUniform(-1, 1, []int32{1, 2048}, metal.DTypeFloat32)
	w := metal.RandomUniform(-0.05, 0.05, []int32{2048, 32000}, metal.DTypeFloat32)
	defer metal.Free(x, w)
	metal.Materialize(x, w)
	b.ReportAllocs()
	for b.Loop() {
		logits := metal.Matmul(x, w)
		capped := logitSoftcap(logits, 30.0)
		metal.Free(logits)
		tok := metal.Argmax(capped, -1, false)
		metal.Materialize(tok)
		metal.Free(capped, tok)
	}
}

func BenchmarkDecodeLoop_LogitCompose_E2E_H3072_Vocab262k(b *testing.B) {
	x := metal.RandomUniform(-1, 1, []int32{1, 3072}, metal.DTypeFloat32)
	w := metal.RandomUniform(-0.05, 0.05, []int32{3072, 262208}, metal.DTypeFloat32)
	defer metal.Free(x, w)
	metal.Materialize(x, w)
	b.ReportAllocs()
	for b.Loop() {
		logits := metal.Matmul(x, w)
		capped := logitSoftcap(logits, 30.0)
		metal.Free(logits)
		tok := metal.Argmax(capped, -1, false)
		metal.Materialize(tok)
		metal.Free(capped, tok)
	}
}
