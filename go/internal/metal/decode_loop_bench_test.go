// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Per-token decode loop bench coverage map (W7-E, Wave 7).
//
// The per-token hot path during generation is:
//
//   1. Forward pass produces hidden state.
//   2. Last-token slice + RMSNorm + output projection -> logits.
//   3. (Optional) softcap (Gemma 3/4 applies 30.0).
//   4. Sample (greedy / temp / top-k / top-p).
//   5. Eval the resulting token tensor.
//
// IDEAS.md flags this as a critical seam: every per-token cgo
// boundary cost amortises across hundreds of tokens, so the Eval
// boundary cost + the native fused last-token output paths
// (nativeLastTokenOutputLogits, nativeGreedyDecodeToken) are
// load-bearing.
//
// Coverage:
//   - Eval boundary cost at varying op-count (small / medium / large
//     graphs) — what's the per-call cgo + Metal graph flush cost?
//   - nativeGreedyDecodeToken — the fused argmax + tensor-create call.
//   - logitSoftcap — Gemma's 30-tanh softcap applied to output logits.
//   - Full logit-to-token compose: argmax + softcap + softmax on a
//     1×vocab tensor.
//   - End-to-end "next token" simulation at varying vocab sizes (the
//     output projection cost dominates for large vocab).

import (
	"testing"

	core "dappco.re/go"
)

// --- Eval boundary cost (cgo + Metal graph flush) ---

// Tiny graph (1 op) — measures the cgo overhead floor for an Eval call.
func BenchmarkDecodeLoop_Eval_TinyGraph_1op(b *testing.B) {
	a := RandomUniform(0, 1, []int32{64}, DTypeFloat32)
	defer Free(a)
	Materialize(a)
	b.ReportAllocs()
	for b.Loop() {
		y := Add(a, a)
		if err := Eval(y); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(y)
	}
}

// Small graph (8 ops). Real decode steps push 50-100 ops per token,
// so this tier probes the constant-overhead bucket.
func BenchmarkDecodeLoop_Eval_SmallGraph_8ops(b *testing.B) {
	a := RandomUniform(0, 1, []int32{256}, DTypeFloat32)
	defer Free(a)
	Materialize(a)
	b.ReportAllocs()
	for b.Loop() {
		y1 := Add(a, a)
		y2 := Add(y1, a)
		y3 := Add(y2, a)
		y4 := Add(y3, a)
		y5 := Mul(y4, a)
		y6 := Mul(y5, a)
		y7 := Mul(y6, a)
		y8 := Mul(y7, a)
		if err := Eval(y8); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(y1, y2, y3, y4, y5, y6, y7, y8)
	}
}

// Medium graph (32 ops) — closer to a layer's worth of ops.
func BenchmarkDecodeLoop_Eval_MediumGraph_32ops(b *testing.B) {
	a := RandomUniform(0, 1, []int32{256}, DTypeFloat32)
	defer Free(a)
	Materialize(a)
	b.ReportAllocs()
	for b.Loop() {
		intermediates := make([]*Array, 0, 32)
		prev := a
		for i := 0; i < 32; i++ {
			var next *Array
			if i%2 == 0 {
				next = Add(prev, a)
			} else {
				next = Mul(prev, a)
			}
			intermediates = append(intermediates, next)
			prev = next
		}
		if err := Eval(prev); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(intermediates...)
	}
}

// Eval on multiple outputs at once — does flushing N outputs cost
// more than flushing the same N joined into a single output?
func BenchmarkDecodeLoop_Eval_MultiOutput_8(b *testing.B) {
	a := RandomUniform(0, 1, []int32{64}, DTypeFloat32)
	defer Free(a)
	Materialize(a)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 8)
		for i := range outs {
			outs[i] = Add(a, a)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

// --- nativeGreedyDecodeToken — fused argmax for compiled-greedy path ---

// Vocab sweep: 32k (Llama), 128k (Gemma 3), 256k (Gemma 4 E2B).
func BenchmarkDecodeLoop_NativeGreedyDecode_Vocab32k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	b.ReportAllocs()
	for b.Loop() {
		tok, err := nativeGreedyDecodeToken(logits)
		if err != nil {
			b.Fatalf("nativeGreedyDecodeToken: %v", err)
		}
		Materialize(tok)
		Free(tok)
	}
}

func BenchmarkDecodeLoop_NativeGreedyDecode_Vocab128k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 128000}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	b.ReportAllocs()
	for b.Loop() {
		tok, err := nativeGreedyDecodeToken(logits)
		if err != nil {
			b.Fatalf("nativeGreedyDecodeToken: %v", err)
		}
		Materialize(tok)
		Free(tok)
	}
}

func BenchmarkDecodeLoop_NativeGreedyDecode_Vocab256k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 256000}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	b.ReportAllocs()
	for b.Loop() {
		tok, err := nativeGreedyDecodeToken(logits)
		if err != nil {
			b.Fatalf("nativeGreedyDecodeToken: %v", err)
		}
		Materialize(tok)
		Free(tok)
	}
}

func BenchmarkDecodeLoop_LastTokenLogitsSingleStep_FastReshape_Vocab262k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	b.ReportAllocs()
	for b.Loop() {
		last, err := lastTokenLogits(logits)
		if err != nil {
			b.Fatalf("lastTokenLogits: %v", err)
		}
		if err := Eval(last); err != nil {
			Free(last)
			b.Fatalf("Eval(last): %v", err)
		}
		Free(last)
	}
}

func BenchmarkDecodeLoop_LastTokenLogitsSingleStep_LegacySlice_Vocab262k(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	b.ReportAllocs()
	for b.Loop() {
		last, err := benchmarkDecodeLoopLegacyLastTokenLogits(logits)
		if err != nil {
			b.Fatalf("legacy last logits: %v", err)
		}
		if err := Eval(last); err != nil {
			Free(last)
			b.Fatalf("Eval(last): %v", err)
		}
		Free(last)
	}
}

func benchmarkDecodeLoopLegacyLastTokenLogits(logits *Array) (*Array, error) {
	if logits == nil || !logits.Valid() {
		return nil, core.NewError("mlx: logits are empty")
	}
	ndim := logits.NumDims()
	if ndim <= 0 {
		return nil, core.NewError("mlx: logits rank is invalid")
	}
	if ndim == 1 {
		return Reshape(logits, 1, int32(logits.Dim(0))), nil
	}
	if ndim == 2 {
		rows := logits.Dim(0)
		if rows <= 0 {
			return nil, core.NewError("mlx: logits sequence is empty")
		}
		last := SliceAxis(logits, 0, int32(rows-1), int32(rows))
		out := Reshape(last, 1, int32(last.Dim(last.NumDims()-1)))
		Free(last)
		return out, nil
	}
	seqAxis := ndim - 2
	seqLen := logits.Dim(seqAxis)
	if seqLen <= 0 {
		return nil, core.NewError("mlx: logits sequence is empty")
	}
	last := SliceAxis(logits, seqAxis, int32(seqLen-1), int32(seqLen))
	out := Reshape(last, 1, int32(last.Dim(last.NumDims()-1)))
	Free(last)
	return out, nil
}

// --- logitSoftcap — Gemma's 30.0 tanh-softcap on output logits ---

func BenchmarkDecodeLoop_LogitSoftcap_Vocab32k(b *testing.B) {
	x := RandomUniform(-10, 10, []int32{1, 32000}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(32000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := logitSoftcap(x, 30.0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkDecodeLoop_LogitSoftcap_Vocab128k(b *testing.B) {
	x := RandomUniform(-10, 10, []int32{1, 128000}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(128000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := logitSoftcap(x, 30.0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkDecodeLoop_LogitSoftcap_Vocab256k(b *testing.B) {
	x := RandomUniform(-10, 10, []int32{1, 256000}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(256000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := logitSoftcap(x, 30.0)
		Materialize(y)
		Free(y)
	}
}

// --- Output projection (hidden → vocab) ---

// The output projection is the biggest matmul in the decode loop.
// Last-hidden × W^T = logits, with W shape [vocab, hidden].
func BenchmarkDecodeLoop_OutputProjection_H2048_Vocab32k(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{2048, 32000}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Matmul(x, w)
		Materialize(y)
		Free(y)
	}
}

// Larger vocab — Gemma 4 E4B's 262208-token vocab.
func BenchmarkDecodeLoop_OutputProjection_H2048_Vocab262k(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{2048, 262208}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Matmul(x, w)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkDecodeLoop_OutputProjection_H3072_Vocab262k(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 3072}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{3072, 262208}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(3072 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Matmul(x, w)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkDecodeLoop_LastTokenOutputQ4Native_H2048_Vocab262k(b *testing.B) {
	hidden, normWeight, output := benchmarkDecodeLoopQ4OutputFixture(b, 2048, 262208)
	defer Free(hidden, normWeight)
	defer freeLinear(output)
	b.ReportAllocs()
	for b.Loop() {
		logits, ok, err := nativeLastTokenOutputLogits(hidden, normWeight, output, 1e-6, 30)
		if err != nil {
			b.Fatalf("nativeLastTokenOutputLogits: %v", err)
		}
		if !ok {
			b.Fatal("nativeLastTokenOutputLogits unavailable")
		}
		if err := Eval(logits); err != nil {
			Free(logits)
			b.Fatalf("Eval(native logits): %v", err)
		}
		Free(logits)
	}
}

func BenchmarkDecodeLoop_LastTokenOutputQ4GoGraph_H2048_Vocab262k(b *testing.B) {
	hidden, normWeight, output := benchmarkDecodeLoopQ4OutputFixture(b, 2048, 262208)
	defer Free(hidden, normWeight)
	defer freeLinear(output)
	b.ReportAllocs()
	for b.Loop() {
		normed := RMSNorm(hidden, normWeight, 1e-6)
		logits := output.Forward(normed)
		Free(normed)
		capped := logitSoftcap(logits, 30)
		Free(logits)
		if err := Eval(capped); err != nil {
			Free(capped)
			b.Fatalf("Eval(graph logits): %v", err)
		}
		Free(capped)
	}
}

func benchmarkDecodeLoopQ4OutputFixture(b *testing.B, hiddenDim, vocab int) (*Array, *Array, *Linear) {
	b.Helper()
	if hiddenDim%64 != 0 {
		b.Fatalf("hiddenDim=%d must be divisible by group size 64", hiddenDim)
	}
	hidden := RandomUniform(-1, 1, []int32{1, 1, int32(hiddenDim)}, DTypeFloat32)
	normWeight := RandomUniform(0.5, 1.5, []int32{int32(hiddenDim)}, DTypeFloat32)
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
	output := NewQuantizedLinear(
		FromValues(weightWords, vocab, packedWidth),
		FromValues(scales, vocab, groups),
		FromValues(biases, vocab, groups),
		nil,
		64,
		4,
	)
	Materialize(hidden, normWeight, output.Weight, output.Scales, output.Biases)
	return hidden, normWeight, output
}

// --- End-to-end logit compose (last hidden → token) ---

// Compose the realistic per-token tail: matmul (output proj) + softcap
// + argmax. This is the post-final-block compute, the closest a
// non-model-loading bench can get to per-token decode cost.
func BenchmarkDecodeLoop_LogitCompose_E2E_H2048_Vocab32k(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{2048, 32000}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.ReportAllocs()
	for b.Loop() {
		logits := Matmul(x, w)
		capped := logitSoftcap(logits, 30.0)
		Free(logits)
		tok := Argmax(capped, -1, false)
		Materialize(tok)
		Free(capped, tok)
	}
}

func BenchmarkDecodeLoop_LogitCompose_E2E_H3072_Vocab262k(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 3072}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{3072, 262208}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.ReportAllocs()
	for b.Loop() {
		logits := Matmul(x, w)
		capped := logitSoftcap(logits, 30.0)
		Free(logits)
		tok := Argmax(capped, -1, false)
		Materialize(tok)
		Free(capped, tok)
	}
}

// --- Softmax over logit shape (sampling prep) ---

func BenchmarkDecodeLoop_Softmax_Vocab262k(b *testing.B) {
	x := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(262208 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
		Free(y)
	}
}

// --- Argmax sweep on vocab sizes ---

func BenchmarkDecodeLoop_Argmax_Vocab32k(b *testing.B) {
	x := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(32000 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Argmax(x, -1, false)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkDecodeLoop_Argmax_Vocab262k(b *testing.B) {
	x := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(262208 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Argmax(x, -1, false)
		Materialize(y)
		Free(y)
	}
}

// --- suppressTokenArray — per-step suppression mask build ---

// Per-decode-step cost when the generation cfg supplies a suppress
// list (banned tokens, EOS suppression, etc.). Allocates a fresh
// int32 array each call.
func BenchmarkDecodeLoop_SuppressTokenArray_16(b *testing.B) {
	ids := make([]int32, 16)
	for i := range ids {
		ids[i] = int32(i + 100)
	}
	b.ReportAllocs()
	for b.Loop() {
		array := suppressTokenArray(ids)
		Free(array)
	}
}

func BenchmarkDecodeLoop_SuppressTokenArray_256(b *testing.B) {
	ids := make([]int32, 256)
	for i := range ids {
		ids[i] = int32(i + 100)
	}
	b.ReportAllocs()
	for b.Loop() {
		array := suppressTokenArray(ids)
		Free(array)
	}
}

func BenchmarkDecodeLoop_LastTokenGreedySuppressed_FreshArray(b *testing.B) {
	hidden := RandomUniform(-1, 1, []int32{1, 1, 64}, DTypeFloat32)
	normWeight := RandomUniform(0.9, 1.1, []int32{64}, DTypeFloat32)
	outputWeight := RandomUniform(-0.05, 0.05, []int32{1024, 64}, DTypeFloat32)
	output := NewLinear(outputWeight, nil)
	suppressTokens := make([]int32, 16)
	for i := range suppressTokens {
		suppressTokens[i] = int32(i)
	}
	defer Free(hidden, normWeight, outputWeight)
	Materialize(hidden, normWeight, outputWeight)

	b.ReportAllocs()
	for b.Loop() {
		tok, ok, err := nativeLastTokenGreedyToken(hidden, normWeight, output, 1e-6, suppressTokens...)
		if err != nil {
			b.Fatalf("nativeLastTokenGreedyToken: %v", err)
		}
		if !ok {
			b.Fatal("nativeLastTokenGreedyToken unavailable")
		}
		Materialize(tok)
		Free(tok)
	}
}

func BenchmarkDecodeLoop_LastTokenGreedySuppressed_BorrowedArray(b *testing.B) {
	hidden := RandomUniform(-1, 1, []int32{1, 1, 64}, DTypeFloat32)
	normWeight := RandomUniform(0.9, 1.1, []int32{64}, DTypeFloat32)
	outputWeight := RandomUniform(-0.05, 0.05, []int32{1024, 64}, DTypeFloat32)
	output := NewLinear(outputWeight, nil)
	suppressTokens := make([]int32, 16)
	for i := range suppressTokens {
		suppressTokens[i] = int32(i)
	}
	suppress := suppressTokenArray(suppressTokens)
	defer Free(hidden, normWeight, outputWeight, suppress)
	Materialize(hidden, normWeight, outputWeight, suppress)

	b.ReportAllocs()
	for b.Loop() {
		tok, ok, err := nativeLastTokenGreedyTokenWithArray(hidden, normWeight, output, 1e-6, suppress, suppressTokens...)
		if err != nil {
			b.Fatalf("nativeLastTokenGreedyTokenWithArray: %v", err)
		}
		if !ok {
			b.Fatal("nativeLastTokenGreedyTokenWithArray unavailable")
		}
		Materialize(tok)
		Free(tok)
	}
}
