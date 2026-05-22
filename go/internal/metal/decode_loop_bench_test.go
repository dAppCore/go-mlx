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

import "testing"

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
