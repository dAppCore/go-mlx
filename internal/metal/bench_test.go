//go:build darwin && arm64 && !nomlx

package metal

import (
	"math"
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
