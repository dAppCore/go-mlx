// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"runtime"
	"testing"
)

// --- Helpers ---

// randomMatrix creates a random float32 matrix of the given shape.
func randomMatrix(rows, cols int32) *Array {
	return RandomUniform(0, 1, []int32{rows, cols}, DTypeFloat32)
}

// randomVector creates a random float32 vector.
func randomVector(n int32) *Array {
	return RandomUniform(0, 1, []int32{n}, DTypeFloat32)
}

// random4D creates a random float32 4D tensor [B, H, L, D].
func random4D(b, h, l, d int32) *Array {
	return RandomUniform(0, 1, []int32{b, h, l, d}, DTypeFloat32)
}

// --- MatMul benchmarks (various sizes) ---

func BenchmarkMatMul_128x128(b *testing.B) {
	a := randomMatrix(128, 128)
	w := randomMatrix(128, 128)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_512x512(b *testing.B) {
	a := randomMatrix(512, 512)
	w := randomMatrix(512, 512)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_1024x1024(b *testing.B) {
	a := randomMatrix(1024, 1024)
	w := randomMatrix(1024, 1024)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_2048x2048(b *testing.B) {
	a := randomMatrix(2048, 2048)
	w := randomMatrix(2048, 2048)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_4096x4096(b *testing.B) {
	a := randomMatrix(4096, 4096)
	w := randomMatrix(4096, 4096)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

// Token-shaped matmul: [1, D] x [D, V] — single-token forward through output projection.
func BenchmarkMatMul_1x2048_x_2048x32000(b *testing.B) {
	x := randomMatrix(1, 2048)
	w := randomMatrix(2048, 32000)
	Materialize(x, w)
	for b.Loop() {
		c := Matmul(x, w)
		Materialize(c)
	}
}

// --- Softmax benchmarks ---

func BenchmarkSoftmax_1x1024(b *testing.B) {
	x := randomMatrix(1, 1024)
	Materialize(x)
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
	}
}

func BenchmarkSoftmax_32x32000(b *testing.B) {
	x := randomMatrix(32, 32000)
	Materialize(x)
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
	}
}

func BenchmarkSoftmax_1x128000(b *testing.B) {
	x := randomMatrix(1, 128000)
	Materialize(x)
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
	}
}

// --- Element-wise arithmetic ---

func BenchmarkAdd_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	c := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	Materialize(a, c)
	for b.Loop() {
		y := Add(a, c)
		Materialize(y)
	}
}

func BenchmarkMul_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	c := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	Materialize(a, c)
	for b.Loop() {
		y := Mul(a, c)
		Materialize(y)
	}
}

func BenchmarkSiLU_1M(b *testing.B) {
	a := RandomUniform(-3, 3, []int32{1000000}, DTypeFloat32)
	Materialize(a)
	for b.Loop() {
		y := SiLU(a)
		Materialize(y)
	}
}

// --- Fused Metal kernels ---

func BenchmarkRMSNorm_1x2048(b *testing.B) {
	x := randomMatrix(1, 2048)
	w := randomVector(2048)
	Materialize(x, w)
	for b.Loop() {
		y := RMSNorm(x, w, 1e-5)
		Materialize(y)
	}
}

func BenchmarkRMSNorm_32x2048(b *testing.B) {
	x := randomMatrix(32, 2048)
	w := randomVector(2048)
	Materialize(x, w)
	for b.Loop() {
		y := RMSNorm(x, w, 1e-5)
		Materialize(y)
	}
}

func BenchmarkLayerNorm_32x2048(b *testing.B) {
	x := randomMatrix(32, 2048)
	w := randomVector(2048)
	bias := randomVector(2048)
	Materialize(x, w, bias)
	for b.Loop() {
		y := LayerNorm(x, w, bias, 1e-5)
		Materialize(y)
	}
}

func BenchmarkRoPE_1x1x32x128(b *testing.B) {
	// Single head, 32 positions, 128 dims — typical decode step shape.
	x := random4D(1, 1, 32, 128)
	Materialize(x)
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
	}
}

func BenchmarkRoPE_1x32x512x128(b *testing.B) {
	// 32 heads, 512 positions — typical prefill shape.
	x := random4D(1, 32, 512, 128)
	Materialize(x)
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
	}
}

// --- Scaled Dot-Product Attention ---

func BenchmarkSDPA_1head_seq32(b *testing.B) {
	scale := float32(1.0 / math.Sqrt(128.0))
	q := random4D(1, 1, 32, 128)
	k := random4D(1, 1, 32, 128)
	v := random4D(1, 1, 32, 128)
	Materialize(q, k, v)
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
	}
}

func BenchmarkSDPA_32head_seq128(b *testing.B) {
	scale := float32(1.0 / math.Sqrt(128.0))
	q := random4D(1, 32, 128, 128)
	k := random4D(1, 32, 128, 128)
	v := random4D(1, 32, 128, 128)
	Materialize(q, k, v)
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
	}
}

func BenchmarkSDPA_32head_seq512(b *testing.B) {
	scale := float32(1.0 / math.Sqrt(128.0))
	q := random4D(1, 32, 512, 128)
	k := random4D(1, 32, 512, 128)
	v := random4D(1, 32, 512, 128)
	Materialize(q, k, v)
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
	}
}

// --- Neural network layers ---

func BenchmarkLinear_1x2048_to_2048(b *testing.B) {
	w := randomMatrix(2048, 2048)
	Materialize(w)
	layer := NewLinear(w, nil)
	x := randomMatrix(1, 2048)
	Materialize(x)
	for b.Loop() {
		y := layer.Forward(x)
		Materialize(y)
	}
}

func BenchmarkLinear_32x2048_to_8192(b *testing.B) {
	w := randomMatrix(8192, 2048)
	Materialize(w)
	layer := NewLinear(w, nil)
	x := randomMatrix(32, 2048)
	Materialize(x)
	for b.Loop() {
		y := layer.Forward(x)
		Materialize(y)
	}
}

func BenchmarkEmbedding_32tokens_vocab32000_dim2048(b *testing.B) {
	w := randomMatrix(32000, 2048)
	Materialize(w)
	emb := &Embedding{Weight: w}
	indices := FromValues(make([]int32, 32), 32)
	// Fill with random valid indices
	for i := range 32 {
		indices = FromValues([]int32{int32(i % 32000)}, 1)
	}
	indices = RandomUniform(0, 31999, []int32{32}, DTypeFloat32)
	indices = AsType(indices, DTypeInt32)
	Materialize(indices)
	for b.Loop() {
		y := emb.Forward(indices)
		Materialize(y)
	}
}

// --- Reductions ---

func BenchmarkSum_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	Materialize(a)
	for b.Loop() {
		y := Sum(a, 0, false)
		Materialize(y)
	}
}

func BenchmarkArgmax_1x32000(b *testing.B) {
	a := randomMatrix(1, 32000)
	Materialize(a)
	for b.Loop() {
		y := Argmax(a, -1, false)
		Materialize(y)
	}
}

// --- Sampling ---

func BenchmarkSampler_Greedy(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(0, 0, 0, 0) // greedy
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_TopK50_Temp1(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(1.0, 0, 0, 50)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_TopP09_Temp1(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(1.0, 0.9, 0, 0)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_Full_TopP09_MinP01_TopK50(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(0.8, 0.9, 0.1, 50) // temp=0.8, topP=0.9, minP=0.1, topK=50
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_LegacyTopPThenTopK_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := chain{Temperature(1.0), TopP(0.95), TopKSampler(64)}
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopP_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := newSampler(1.0, 0.95, 0, 64)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopPTokenReadNoEval_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := newSampler(1.0, 0.95, 0, 64)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		_ = tok.Int()
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopPTokenReadNoEvalChecked_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := newSampler(1.0, 0.95, 0, 64)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		_ = tok.Int()
		if err := lastError(); err != nil {
			Free(tok)
			b.Fatalf("token read: %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopPWithSuppression_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	s := newSamplerWithSuppression(1.0, 0.95, 0, 64, suppress)
	defer closeSampler(s)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_CompiledTopKThenTopP_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{sampleTopKTopPToken(inputs[0], 64, 0.95)}
	}, false)
	defer compiled.Free()
	b.ResetTimer()
	for b.Loop() {
		tok := compiled.Call(logits)[0]
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(compiled sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_CompiledTopKThenTopPCallOne_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{sampleTopKTopPToken(inputs[0], 64, 0.95)}
	}, false)
	defer compiled.Free()
	b.ResetTimer()
	for b.Loop() {
		tok := compiled.CallOne(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(compiled sample): %v", err)
		}
		Free(tok)
	}
}

// BenchmarkSampler_MinP01_Temp1 isolates min-p path which uses Softmax + MaxAxis
// + MulScalar + Greater(scalar) + Where.  Targets W11-R inline-Greater opportunity.
func BenchmarkSampler_MinP01_Temp1(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(1.0, 0, 0.1, 0)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

// BenchmarkSampler_Temperature_PerToken isolates pure Temperature.Sample —
// already routes through MulScalar (W11-F).  Useful as floor reference.
func BenchmarkSampler_Temperature_PerToken(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := Temperature(0.7)
	for b.Loop() {
		y := s.Sample(logits)
		Materialize(y)
	}
}

// BenchmarkSampler_SuppressedGreedy_Gemma exercises the suppressedGreedy
// fast-path used by the Gemma assistant when only suppression is configured.
// Triggers suppressTokenLogits scalar FromValue (-inf) on each call.
func BenchmarkSampler_SuppressedGreedy_Gemma(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105}
	s := newSamplerWithSuppression(0, 0, 0, 0, suppress)
	defer closeSampler(s)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
		Free(tok)
	}
}

// BenchmarkApplyRepeatPenalty_Hist64 exercises applyRepeatPenalty with a
// realistic 64-token history.  Targets W10-V scratch pool + W11-R FromValue
// crossings (zero / invPenalty / penaltyVal).
func BenchmarkApplyRepeatPenalty_Hist64(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	hist := make([]int32, 64)
	for i := range hist {
		hist[i] = int32(i * 17 % 32000)
	}
	for b.Loop() {
		y := applyRepeatPenalty(logits, hist, 1.1)
		Materialize(y)
	}
}

// BenchmarkHostUnsuppressedGreedyToken_Gemma exercises the Gemma-sized
// host-side fallback that allocates suppressed map every call.  Stress on
// W10-V map elimination.
func BenchmarkHostUnsuppressedGreedyToken_Gemma(b *testing.B) {
	values := make([]float32, 258885)
	values[0] = 100
	values[123] = 10
	logits := FromValues(values, 1, len(values))
	Materialize(logits)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	for b.Loop() {
		tok, err := hostUnsuppressedGreedyToken(logits, suppress)
		if err != nil {
			b.Fatal(err)
		}
		Materialize(tok)
		Free(tok)
	}
}

// BenchmarkInspectAttentionCache_Realistic exercises the host-side
// inspectAttentionCache fan-out used by attention probes. Cache shape
// [1, 32, 1024, 128] = 4M float32 = 16MB — the per-call copy that the
// W11-R zero-copy view pattern eliminates.
func BenchmarkInspectAttentionCache_Realistic(b *testing.B) {
	cache := NewKVCache()
	// [1, 32 heads, 1024 tokens, 128 head_dim] = 4_194_304 float32 = 16 MB
	const heads, seqLen, headDim = 32, 1024, 128
	size := 1 * heads * seqLen * headDim
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.0001
	}
	k := FromValues(data, 1, heads, seqLen, headDim)
	v := FromValues(data, 1, heads, seqLen, headDim)
	outK, outV := cache.Update(k, v, seqLen)
	Materialize(outK, outV)
	Detach(outK)
	Detach(outV)
	for b.Loop() {
		snapshot, ok := inspectAttentionCache(cache, seqLen)
		if !ok {
			b.Fatal("inspectAttentionCache returned not-ok")
		}
		if snapshot.NumHeads != heads {
			b.Fatalf("snapshot.NumHeads = %d, want %d", snapshot.NumHeads, heads)
		}
	}
}

// BenchmarkSummarizeProbeLogitsCompact_Gemma exercises the topK fan-out
// used by ProbeLogits.  TopK = 8 by default, so the topValues.Floats()
// candidate copies only 32 bytes per call, but the per-op alloc count
// matters when probes fire per-decoded-token.
func BenchmarkSummarizeProbeLogitsCompact_Gemma(b *testing.B) {
	const vocab = 258885
	values := make([]float32, vocab)
	for i := range values {
		values[i] = float32(i%1000) * 0.001
	}
	row := FromValues(values, 1, vocab)
	Materialize(row)
	shape := []int32{1, vocab}
	for b.Loop() {
		summary, _, err := summarizeProbeLogitsCompact(row, shape, vocab, defaultProbeTopK)
		if err != nil {
			b.Fatal(err)
		}
		if len(summary.Top) != defaultProbeTopK {
			b.Fatalf("len(Top) = %d, want %d", len(summary.Top), defaultProbeTopK)
		}
	}
}

// BenchmarkInspectKVCacheRange_Realistic exercises the per-block KV
// snapshot fan-out used by KVSnapshot capture. Same 16MB cache slice
// drives the kSliced.Floats() + vSliced.Floats() pair on the !RawKVOnly path.
func BenchmarkInspectKVCacheRange_Realistic(b *testing.B) {
	cache := NewKVCache()
	const heads, seqLen, headDim = 32, 1024, 128
	size := 1 * heads * seqLen * headDim
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.0001
	}
	k := FromValues(data, 1, heads, seqLen, headDim)
	v := FromValues(data, 1, heads, seqLen, headDim)
	outK, outV := cache.Update(k, v, seqLen)
	Materialize(outK, outV)
	Detach(outK)
	Detach(outV)
	opts := KVSnapshotCaptureOptions{}
	for b.Loop() {
		snapshot, ok := inspectKVCacheRangeWithOptions(cache, 0, seqLen, opts)
		if !ok {
			b.Fatal("inspectKVCacheRangeWithOptions returned not-ok")
		}
		if snapshot.NumHeads != heads {
			b.Fatalf("snapshot.NumHeads = %d, want %d", snapshot.NumHeads, heads)
		}
	}
}

// BenchmarkMaterialiseFloat32View_Slow_NB sizes the legacy helper across the
// realistic tensor-size range — characterises the cgo Materialize crossing
// cost as a function of payload bytes.  Compare against the
// BenchmarkMaterialiseFloat32ViewFast_FastPath_NB series to read off the
// crossover threshold.
func benchMaterialiseSlow(b *testing.B, n int) {
	b.Helper()
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i)
	}
	arr := FromValues(values, 1, n)
	Materialize(arr)
	defer Free(arr)
	for b.Loop() {
		src, converted, err := materialiseFloat32View(arr)
		if err != nil {
			b.Fatal(err)
		}
		_ = src.Size()
		runtime.KeepAlive(src)
		Free(converted)
	}
}

func benchMaterialiseFast(b *testing.B, n int) {
	b.Helper()
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i)
	}
	arr := FromValues(values, 1, n)
	Materialize(arr)
	defer Free(arr)
	for b.Loop() {
		view, cleanup, err := materialiseFloat32ViewFast(arr)
		if err != nil {
			b.Fatal(err)
		}
		_ = len(view)
		cleanup()
	}
}

// benchFloats sizes the legacy *Array.Floats() copy at the same size points
// so the fast-path crossover threshold can be read off directly.
func benchFloats(b *testing.B, n int) {
	b.Helper()
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i)
	}
	arr := FromValues(values, 1, n)
	Materialize(arr)
	defer Free(arr)
	for b.Loop() {
		out := arr.Floats()
		_ = len(out)
	}
}

func BenchmarkMaterialiseFloat32View_Floats_128B(b *testing.B)  { benchFloats(b, 32) }
func BenchmarkMaterialiseFloat32View_Floats_1KB(b *testing.B)   { benchFloats(b, 256) }
func BenchmarkMaterialiseFloat32View_Floats_10KB(b *testing.B)  { benchFloats(b, 2560) }
func BenchmarkMaterialiseFloat32View_Floats_100KB(b *testing.B) { benchFloats(b, 25600) }
func BenchmarkMaterialiseFloat32View_Floats_1MB(b *testing.B)   { benchFloats(b, 262144) }

func BenchmarkMaterialiseFloat32View_Slow_128B(b *testing.B)  { benchMaterialiseSlow(b, 32) }
func BenchmarkMaterialiseFloat32View_Slow_1KB(b *testing.B)   { benchMaterialiseSlow(b, 256) }
func BenchmarkMaterialiseFloat32View_Slow_10KB(b *testing.B)  { benchMaterialiseSlow(b, 2560) }
func BenchmarkMaterialiseFloat32View_Slow_100KB(b *testing.B) { benchMaterialiseSlow(b, 25600) }
func BenchmarkMaterialiseFloat32View_Slow_1MB(b *testing.B)   { benchMaterialiseSlow(b, 262144) }
func BenchmarkMaterialiseFloat32ViewFast_128B(b *testing.B)   { benchMaterialiseFast(b, 32) }
func BenchmarkMaterialiseFloat32ViewFast_1KB(b *testing.B)    { benchMaterialiseFast(b, 256) }
func BenchmarkMaterialiseFloat32ViewFast_10KB(b *testing.B)   { benchMaterialiseFast(b, 2560) }
func BenchmarkMaterialiseFloat32ViewFast_100KB(b *testing.B)  { benchMaterialiseFast(b, 25600) }
func BenchmarkMaterialiseFloat32ViewFast_1MB(b *testing.B)    { benchMaterialiseFast(b, 262144) }
