// SPDX-Licence-Identifier: EUPL-1.2

//q:build darwin && arm64

//go:build darwin && arm64

package metal

// MoE router/expert bench coverage map (W7-E, Wave 7).
//
// Gemma 4 MoE: routers select top-K experts (typically K=2) from a
// pool of N experts per layer. The output of each token is a
// weighted sum of the chosen experts' outputs.
//
// MiniMax M2 MoE: 128 experts, top-2 routing, plus 1 shared expert.
// (IDEAS.md §5: naive implementations dispatch 128 tiny kernels;
// the fused path uses gather + block-sparse matmul.)
//
// Coverage:
//   - Top-K selection (TopK) on a router-scores tensor of [tokens, experts]
//     at common (N, K) pairs.
//   - Gather-based expert lookup: Take(expert_outputs, top_indices)
//     vs masked-accumulate fallback for comparison.
//   - Sum/Argmax router-score primitives.
//   - Softmax over router scores (which determines per-expert weights).
//
// Note: the fully-fused nativeGemma4RouterMatVec and expertIDMatVec
// paths require quantised weight tensors (Q4/Q8) with specific
// group-size + scale/bias layouts. Those require model-state setup
// well beyond synthetic tensors. We bench the component primitives
// only here — full-system MoE benches need a model fixture and
// belong in a separate harness.

import "testing"

var moeRuntimeAvailableBenchSink bool

func BenchmarkMoETextLayersRuntimeAvailable_Dense64(b *testing.B) {
	layers := make([]*DenseDecoderLayer, 64)
	for i := range layers {
		layers[i] = &DenseDecoderLayer{MLP: &SiLUMLP{}}
	}
	b.ReportAllocs()
	for b.Loop() {
		moeRuntimeAvailableBenchSink = MoETextLayersRuntimeAvailable(layers, func(layer *DenseDecoderLayer) MoETextLayerParts {
			return MoETextLayerParts{Dense: layer, OK: layer != nil}
		})
	}
}

// --- Top-K selection (router output ranking) ---

// Gemma 4 small router: N=8 experts, K=2.
func BenchmarkMoE_TopK_Experts8_K2(b *testing.B) {
	scores := RandomUniform(0, 1, []int32{1, 8}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.ReportAllocs()
	for b.Loop() {
		y := TopK(scores, 2)
		Materialize(y)
		Free(y)
	}
}

// Gemma 4 mid router: N=32 experts, K=2.
func BenchmarkMoE_TopK_Experts32_K2(b *testing.B) {
	scores := RandomUniform(0, 1, []int32{1, 32}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.ReportAllocs()
	for b.Loop() {
		y := TopK(scores, 2)
		Materialize(y)
		Free(y)
	}
}

// MiniMax M2 router: N=128 experts, K=2.
func BenchmarkMoE_TopK_Experts128_K2(b *testing.B) {
	scores := RandomUniform(0, 1, []int32{1, 128}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.ReportAllocs()
	for b.Loop() {
		y := TopK(scores, 2)
		Materialize(y)
		Free(y)
	}
}

// MiniMax + extras: K=8 — speculative tuning / Multi-Token Prediction.
func BenchmarkMoE_TopK_Experts128_K8(b *testing.B) {
	scores := RandomUniform(0, 1, []int32{1, 128}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.ReportAllocs()
	for b.Loop() {
		y := TopK(scores, 8)
		Materialize(y)
		Free(y)
	}
}

// --- Softmax over router scores (weight normalisation) ---

func BenchmarkMoE_SoftmaxRouterScores_Experts8(b *testing.B) {
	scores := RandomUniform(-2, 2, []int32{1, 8}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.ReportAllocs()
	for b.Loop() {
		y := Softmax(scores)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkMoE_SoftmaxRouterScores_Experts128(b *testing.B) {
	scores := RandomUniform(-2, 2, []int32{1, 128}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.ReportAllocs()
	for b.Loop() {
		y := Softmax(scores)
		Materialize(y)
		Free(y)
	}
}

// Batch of tokens — router pass for a prefill chunk.
func BenchmarkMoE_SoftmaxRouterScores_Batch512_Experts128(b *testing.B) {
	scores := RandomUniform(-2, 2, []int32{512, 128}, DTypeFloat32)
	defer Free(scores)
	Materialize(scores)
	b.SetBytes(int64(512 * 128 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Softmax(scores)
		Materialize(y)
		Free(y)
	}
}

// --- Gather (Take) for expert output lookup ---

// Per-token gather of K expert outputs: a way of materialising the
// top-K selection without dispatching N tiny kernels.
//
// expert_outputs shape: [N_experts, hidden].
// indices shape: [K] for a single token.
// Take(expert_outputs, indices, 0) = [K, hidden].
//
// Per IDEAS.md §5, this gather + weighted-sum approach replaces
// 128 expert kernels with 1 gather + 1 weighted-sum.
func BenchmarkMoE_GatherTopK_Experts128_Hidden2048(b *testing.B) {
	expertOutputs := RandomUniform(-1, 1, []int32{128, 2048}, DTypeFloat32)
	// Top-2 indices, e.g. picking experts 17 and 42.
	indicesData := []int32{17, 42}
	indices := FromValues(indicesData, 2)
	defer Free(expertOutputs, indices)
	Materialize(expertOutputs, indices)
	b.SetBytes(int64(2 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Take(expertOutputs, indices, 0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkMoE_GatherTopK_Experts32_Hidden2048(b *testing.B) {
	expertOutputs := RandomUniform(-1, 1, []int32{32, 2048}, DTypeFloat32)
	indicesData := []int32{5, 11}
	indices := FromValues(indicesData, 2)
	defer Free(expertOutputs, indices)
	Materialize(expertOutputs, indices)
	b.SetBytes(int64(2 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Take(expertOutputs, indices, 0)
		Materialize(y)
		Free(y)
	}
}

// --- Naive masked-accumulate fallback (IDEAS.md §5 anti-pattern) ---

// Compute weighted sum across all 128 experts using a mask + reduce.
// This is the path you DON'T want — 128 active terms instead of 2.
// Bench it so the gather path can be quantified as the win.
func BenchmarkMoE_MaskedAccumulate_Experts128_Hidden2048(b *testing.B) {
	expertOutputs := RandomUniform(-1, 1, []int32{128, 2048}, DTypeFloat32)
	// Sparse weights: only 2 of 128 are non-zero (top-2 selection).
	weights := make([]float32, 128)
	weights[17] = 0.6
	weights[42] = 0.4
	weightArr := FromValues(weights, 128, 1)
	defer Free(expertOutputs, weightArr)
	Materialize(expertOutputs, weightArr)

	b.SetBytes(int64(128 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		// Weighted sum: weightArr * expertOutputs (broadcast), then
		// reduce along expert axis.
		weighted := Mul(weightArr, expertOutputs)
		summed := Sum(weighted, 0, false)
		Materialize(summed)
		Free(weighted, summed)
	}
}

// --- Weighted-sum after gather (top-K aggregation) ---

// After Take, weighted-sum across K experts to produce the per-token
// MoE output. This is the second half of the fused MoE compute.
func BenchmarkMoE_GatherPlusWeightedSum_K2_Hidden2048(b *testing.B) {
	expertOutputs := RandomUniform(-1, 1, []int32{128, 2048}, DTypeFloat32)
	indicesData := []int32{17, 42}
	indices := FromValues(indicesData, 2)
	// Per-K weight: top-K weights from router softmax.
	kWeights := FromValues([]float32{0.6, 0.4}, 2, 1)
	defer Free(expertOutputs, indices, kWeights)
	Materialize(expertOutputs, indices, kWeights)

	b.SetBytes(int64(2 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		gathered := Take(expertOutputs, indices, 0)
		weighted := Mul(kWeights, gathered)
		Free(gathered)
		summed := Sum(weighted, 0, false)
		Materialize(summed)
		Free(weighted, summed)
	}
}

// --- Router projection — hidden -> router scores ---

// Router projection: matmul[1, hidden] × [hidden, N_experts] -> [1, N_experts].
func BenchmarkMoE_RouterProjection_Hidden2048_Experts128(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{2048, 128}, DTypeFloat32)
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

func BenchmarkMoE_RouterProjection_Hidden2048_Experts32(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{2048, 32}, DTypeFloat32)
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

// --- End-to-end synthetic MoE forward (gather-based) ---

// Full per-token MoE compute: router projection -> softmax -> TopK ->
// gather -> weighted-sum. Synthetic but representative.
func BenchmarkMoE_E2E_GatherBased_Experts32_Hidden2048(b *testing.B) {
	const H, N, K = 2048, 32, 2
	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	routerW := RandomUniform(-0.05, 0.05, []int32{H, N}, DTypeFloat32)
	expertOutputs := RandomUniform(-1, 1, []int32{N, H}, DTypeFloat32)
	defer Free(x, routerW, expertOutputs)
	Materialize(x, routerW, expertOutputs)

	b.ReportAllocs()
	for b.Loop() {
		// Router projection.
		scores := Matmul(x, routerW)
		// Top-K selection (we use indices argmax via TopK; the kernel
		// returns the top-K values, not indices — so for a true gather
		// we'd need Argpartition or similar. For bench purposes we use
		// the scores tensor directly for weighting and dummy indices.)
		topVals := TopK(scores, K)
		// Synthetic indices — in real code these come from the TopK
		// indices path; here we use the first K experts to keep the
		// gather predictable.
		indices := FromValues([]int32{0, 1}, K)
		// Softmax across the top-K values to get per-K weights.
		topProbs := Softmax(topVals)
		// Gather.
		gathered := Take(expertOutputs, indices, 0)
		// Weighted sum.
		reshaped := Reshape(topProbs, K, 1)
		weighted := Mul(reshaped, gathered)
		out := Sum(weighted, 0, false)
		Materialize(out)
		Free(scores, topVals, indices, topProbs, gathered, reshaped, weighted, out)
	}
}
