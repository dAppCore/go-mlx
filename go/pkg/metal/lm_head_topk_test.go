// SPDX-Licence-Identifier: EUPL-1.2

package metal

import (
	"sort"
	"testing"
)

// lmHeadTopKFixture builds a deterministic q4-g64 head: raw packed weights +
// scales/biases exactly as QuantizedMatmul consumes them — the kernel and the
// reference read the same bit layout, so the reference IS the oracle.
func lmHeadTopKFixture(t *testing.T, hidden, vocab int32) (x, w, scales, biases *Array) {
	t.Helper()
	if err := SeedRandom(95); err != nil {
		t.Fatalf("SeedRandom: %v", err)
	}
	const packFactor = 8 // 8 4-bit values per uint32
	xF := RandomUniform(-1, 1, []int32{1, hidden}, DTypeFloat32)
	x = AsType(xF, DTypeBFloat16)
	Free(xF)
	wF := RandomUniform(0, 2.1e9, []int32{vocab, hidden / packFactor}, DTypeFloat32)
	w = AsType(wF, DTypeUint32)
	Free(wF)
	scalesF := RandomUniform(0.01, 0.1, []int32{vocab, hidden / 64}, DTypeFloat32)
	scales = AsType(scalesF, DTypeBFloat16)
	Free(scalesF)
	biasesF := RandomUniform(-0.5, 0.5, []int32{vocab, hidden / 64}, DTypeFloat32)
	biases = AsType(biasesF, DTypeBFloat16)
	Free(biasesF)
	Materialize(x, w, scales, biases)
	return x, w, scales, biases
}

// lmHeadTopKReference computes the oracle: the full logits row via
// QuantizedMatmul, then a CPU descending sort of indices.
func lmHeadTopKReference(t *testing.T, x, w, scales, biases *Array) []float32 {
	t.Helper()
	logits := QuantizedMatmul(x, w, scales, biases, true, 64, 4)
	defer Free(logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval reference logits: %v", err)
	}
	f32 := AsType(logits, DTypeFloat32)
	defer Free(f32)
	if err := Eval(f32); err != nil {
		t.Fatalf("Eval reference f32: %v", err)
	}
	return append([]float32(nil), f32.Floats()...)
}

func lmHeadTopKReferenceOrder(logits []float32) []int {
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return logits[order[a]] > logits[order[b]]
	})
	return order
}

func TestLMHeadTopK_FusedMatchesReference_Good(t *testing.T) {
	requireMetalRuntime(t)
	const hidden, vocab, topK = 1024, 1000, 20
	x, w, scales, biases := lmHeadTopKFixture(t, hidden, vocab)
	defer Free(x, w, scales, biases)

	if !Q4LMHeadTopKEligible(x, w, scales, biases, 64, topK) {
		t.Fatal("fixture is ineligible — gate or fixture drifted")
	}
	values, indices, err := NativeQ4LMHeadTopK(x, w, scales, biases, 64, topK)
	if err != nil {
		t.Fatalf("NativeQ4LMHeadTopK: %v", err)
	}
	defer Free(values, indices)
	if err := Eval(values, indices); err != nil {
		t.Fatalf("Eval fused outputs: %v", err)
	}

	logits := lmHeadTopKReference(t, x, w, scales, biases)
	order := lmHeadTopKReferenceOrder(logits)

	gotIdx := indices.Ints()
	gotVal := values.Floats()
	if len(gotIdx) != topK || len(gotVal) != topK {
		t.Fatalf("fused outputs = %d/%d entries, want %d", len(gotIdx), len(gotVal), topK)
	}
	// The top-k SETS must match the oracle; per-index values must match the
	// reference logits closely (both sides accumulate fp32 over bf16 inputs,
	// but reduction order differs).
	want := map[int]bool{}
	for _, i := range order[:topK] {
		want[i] = true
	}
	for pos, idx := range gotIdx {
		if idx < 0 || idx >= len(logits) {
			t.Fatalf("fused index %d out of range at pos %d", idx, pos)
		}
		if !want[idx] {
			t.Fatalf("fused top-%d contains %d, reference top set %v (got %v)", topK, idx, order[:topK], gotIdx)
		}
		ref := logits[idx]
		diff := gotVal[pos] - ref
		if diff < 0 {
			diff = -diff
		}
		tol := float32(0.02)
		if ref > 1 || ref < -1 {
			scale := ref
			if scale < 0 {
				scale = -scale
			}
			tol *= scale
		}
		if diff > tol {
			t.Fatalf("fused value[%d]=%g, reference logits[%d]=%g (|diff| %g > %g)", pos, gotVal[pos], idx, ref, diff, tol)
		}
	}
	for pos := 1; pos < topK; pos++ {
		if gotVal[pos] > gotVal[pos-1] {
			t.Fatalf("fused values not descending at %d: %v", pos, gotVal)
		}
	}
}

func TestLMHeadTopK_ArgmaxExact_Good(t *testing.T) {
	requireMetalRuntime(t)
	const hidden, vocab = 1024, 1000
	x, w, scales, biases := lmHeadTopKFixture(t, hidden, vocab)
	defer Free(x, w, scales, biases)

	values, indices, err := NativeQ4LMHeadTopK(x, w, scales, biases, 64, 1)
	if err != nil {
		t.Fatalf("NativeQ4LMHeadTopK(k=1): %v", err)
	}
	defer Free(values, indices)
	if err := Eval(values, indices); err != nil {
		t.Fatalf("Eval fused argmax: %v", err)
	}

	logits := lmHeadTopKReference(t, x, w, scales, biases)
	order := lmHeadTopKReferenceOrder(logits)
	if got := indices.Ints(); len(got) != 1 || got[0] != order[0] {
		t.Fatalf("fused argmax = %v, reference argmax = %d (logit %g vs runner-up %g)",
			got, order[0], logits[order[0]], logits[order[1]])
	}
}

func TestLMHeadTopK_Eligibility_Bad(t *testing.T) {
	requireMetalRuntime(t)
	const hidden, vocab = 1024, 1000
	x, w, scales, biases := lmHeadTopKFixture(t, hidden, vocab)
	defer Free(x, w, scales, biases)

	if Q4LMHeadTopKEligible(x, w, scales, biases, 64, 0) {
		t.Fatal("k=0 must be ineligible")
	}
	if Q4LMHeadTopKEligible(x, w, scales, biases, 64, Q4LMHeadTopKMaxK+1) {
		t.Fatal("k>64 must be ineligible")
	}
	if Q4LMHeadTopKEligible(x, w, scales, biases, 48, 8) {
		t.Fatal("group_size 48 must be ineligible")
	}
	if Q4LMHeadTopKEligible(nil, w, scales, biases, 64, 8) {
		t.Fatal("nil x must be ineligible")
	}

	oddX := Zeros([]int32{1, 768}, DTypeBFloat16)
	defer Free(oddX)
	if Q4LMHeadTopKEligible(oddX, w, scales, biases, 64, 8) {
		t.Fatal("K=768 (not a multiple of 512) must be ineligible")
	}

	badRank := Zeros([]int32{1, 1, hidden}, DTypeBFloat16)
	defer Free(badRank)
	if Q4LMHeadTopKEligible(badRank, w, scales, biases, 64, 8) {
		t.Fatal("rank-3 x must be ineligible")
	}
}

// The padding tile case: vocab 130 with the 128-rows-per-tile launch makes
// tile 2 almost entirely -INFINITY padding — none of it may surface.
func TestLMHeadTopK_PaddingTiles_Ugly(t *testing.T) {
	requireMetalRuntime(t)
	const hidden, vocab, topK = 512, 130, 64
	x, w, scales, biases := lmHeadTopKFixture(t, hidden, vocab)
	defer Free(x, w, scales, biases)

	values, indices, err := NativeQ4LMHeadTopK(x, w, scales, biases, 64, topK)
	if err != nil {
		t.Fatalf("NativeQ4LMHeadTopK: %v", err)
	}
	defer Free(values, indices)
	if err := Eval(values, indices); err != nil {
		t.Fatalf("Eval fused outputs: %v", err)
	}

	logits := lmHeadTopKReference(t, x, w, scales, biases)
	order := lmHeadTopKReferenceOrder(logits)
	want := map[int]bool{}
	for _, i := range order[:topK] {
		want[i] = true
	}
	for pos, idx := range indices.Ints() {
		if idx < 0 || idx >= vocab {
			t.Fatalf("padding leaked: index %d at pos %d", idx, pos)
		}
		if !want[idx] {
			t.Fatalf("fused top-%d contains %d outside the reference set", topK, idx)
		}
	}
}

// Bench: fused head+topk vs the materialise-then-pick baseline at output-proj
// dims (H=2048, V=32000). Run bounded: -benchtime=20x.
func BenchmarkLMHeadTopK_Fused_Q4G64_H2048_V32k(b *testing.B) {
	if !MetalAvailable() {
		b.Skip("Metal runtime unavailable")
	}
	const H, V, K = 2048, 32000, 20
	x := AsType(RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32), DTypeBFloat16)
	w := AsType(RandomUniform(0, 2.1e9, []int32{V, H / 8}, DTypeFloat32), DTypeUint32)
	scales := AsType(RandomUniform(0.01, 0.1, []int32{V, H / 64}, DTypeFloat32), DTypeBFloat16)
	biases := AsType(RandomUniform(-0.5, 0.5, []int32{V, H / 64}, DTypeFloat32), DTypeBFloat16)
	defer Free(x, w, scales, biases)
	Materialize(x, w, scales, biases)

	b.ReportAllocs()
	for b.Loop() {
		values, indices, err := NativeQ4LMHeadTopK(x, w, scales, biases, 64, K)
		if err != nil {
			b.Fatal(err)
		}
		Materialize(values, indices)
		Free(values, indices)
	}
}

func BenchmarkLMHeadTopK_Baseline_QMMFullRow_H2048_V32k(b *testing.B) {
	if !MetalAvailable() {
		b.Skip("Metal runtime unavailable")
	}
	const H, V = 2048, 32000
	x := AsType(RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32), DTypeBFloat16)
	w := AsType(RandomUniform(0, 2.1e9, []int32{V, H / 8}, DTypeFloat32), DTypeUint32)
	scales := AsType(RandomUniform(0.01, 0.1, []int32{V, H / 64}, DTypeFloat32), DTypeBFloat16)
	biases := AsType(RandomUniform(-0.5, 0.5, []int32{V, H / 64}, DTypeFloat32), DTypeBFloat16)
	defer Free(x, w, scales, biases)
	Materialize(x, w, scales, biases)

	b.ReportAllocs()
	for b.Loop() {
		logits := QuantizedMatmul(x, w, scales, biases, true, 64, 4)
		Materialize(logits)
		Free(logits)
	}
}
