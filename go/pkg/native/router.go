// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sort"

	core "dappco.re/go"
)

// bf16ToF32 decodes one little-endian bf16 (2 bytes: lo, hi) to float32 — the
// inverse of f32ToBF16, for reading a device result back to the host.
func bf16ToF32(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// topKByScore returns the indices of the topK highest scores, highest first, ties
// broken by lower index — a deterministic, stable selection (stable sort preserves
// the original index order for equal scores).
func topKByScore(scores []float32, topK int) []int32 {
	order := make([]int32, len(scores))
	for i := range order {
		order[i] = int32(i)
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	out := make([]int32, topK)
	copy(out, order[:topK])
	return out
}

// softmaxAt returns softmax over the scores at idx (max-subtracted for stability),
// in idx order, as float32.
func softmaxAt(scores []float32, idx []int32) []float32 {
	maxS := float32(math.Inf(-1))
	for _, e := range idx {
		if scores[e] > maxS {
			maxS = scores[e]
		}
	}
	w := make([]float32, len(idx))
	var sum float32
	for i, e := range idx {
		w[i] = float32(math.Exp(float64(scores[e] - maxS)))
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	return w
}

// MoERouter runs the gemma4 MoE router: it RMS-norms x with the pre-scaled router
// norm weight, projects to per-expert scores, selects the topK highest-scoring
// experts and softmaxes their scores — optionally multiplying each by its per-expert
// scale. Returns (idx, weights) ready to feed MoEExperts.
//
// normWScaled is the router norm weight ALREADY scaled by RootSize (= dModel^-0.5),
// folded once at load exactly like the metal model caches ScaleScaled = Scale·RootSize
// — so this sub-slice needs no on-device scalar-mul. perExpertScale (numExperts bf16)
// is optional; pass nil to skip it. routerW is [numExperts × dModel] row-major bf16
// (each expert is a row), x is dModel bf16; idx is topK int32, weights topK bf16.
//
// Correctness-first HOST top-k: the score vector (numExperts) is read back and the
// top-k + softmax run on the host. This is the simplest correct routing and the
// re-encode path's natural shape, but the per-token GPU→host readback won't fit the
// ICB single-submit path — an on-device top-k (the metallib Argpartition /
// NativeMoERouterTopK kernel) is a later sub-slice for that path.
//
// The routing decision is order-INVARIANT: each selected expert's weight is
// independent of the order idx is returned in (softmax is over the selected scores;
// the downstream combine is a commutative weighted sum). The parity gate therefore
// compares expert→weight maps, not positional sequences.
func MoERouter(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: normWScaled must be dModel bf16 bytes")
	}
	if len(routerW) != numExperts*dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: routerW must be numExperts*dModel bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	if topK <= 0 || topK > numExperts {
		return nil, nil, core.NewError("native.MoERouter: topK must be in 1..numExperts")
	}

	// on-device: RMS-norm then project to per-expert scores (parity-proven ops).
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		return nil, nil, err
	}
	scoresB, err := MatVecBF16(routerW, normed, numExperts, dModel)
	if err != nil {
		return nil, nil, err
	}
	idx, weights := routerSelect(scoresB, perExpertScale, numExperts, topK)
	return idx, weights, nil
}

// routerSelect performs the host top-k + softmax (+ optional per-expert scale) over the raw
// per-expert scores (numExperts bf16) — the routing decision shared by MoERouter and
// MoERouterQuant (they differ only in how the scores are projected: bf16 gemv vs 4-bit qmv).
func routerSelect(scoresB, perExpertScale []byte, numExperts, topK int) ([]int32, []byte) {
	scores := make([]float32, numExperts)
	for e := 0; e < numExperts; e++ {
		scores[e] = bf16ToF32(scoresB[e*bf16Size], scoresB[e*bf16Size+1])
	}
	idx := topKByScore(scores, topK)
	w := softmaxAt(scores, idx)
	if perExpertScale != nil {
		for i, e := range idx {
			w[i] *= bf16ToF32(perExpertScale[int(e)*bf16Size], perExpertScale[int(e)*bf16Size+1])
		}
	}
	weights := make([]byte, topK*bf16Size)
	for i, v := range w {
		h := f32ToBF16(v)
		weights[i*bf16Size] = byte(h)
		weights[i*bf16Size+1] = byte(h >> 8)
	}
	return idx, weights
}

// MoERouterQuant is MoERouter with a 4-bit expert-score projection (gemma4 26B-A4B's
// router.proj is affine-quantised). RMS-norm (normWScaled pre-folded by RootSize, as in
// MoERouter) → QMVBF16 to the per-expert scores → the shared host top-k + softmax. routerProj
// is the [numExperts × dModel] quant weight; groupSize/bits the checkpoint's quant.
func MoERouterQuant(x, normWScaled []byte, routerProj QuantWeight, perExpertScale []byte, numExperts, topK, dModel, groupSize, bits int, eps float32) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: normWScaled must be dModel bf16 bytes")
	}
	if topK <= 0 || topK > numExperts {
		return nil, nil, core.NewError("native.MoERouterQuant: topK must be in 1..numExperts")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, nil, core.NewError("native.MoERouterQuant: groupSize must divide dModel")
	}
	wantPacked, wantSB := numExperts*dModel*bits/8, numExperts*(dModel/groupSize)*bf16Size
	if len(routerProj.Packed) != wantPacked || len(routerProj.Scales) != wantSB || len(routerProj.Biases) != wantSB {
		return nil, nil, core.NewError("native.MoERouterQuant: routerProj size mismatch vs numExperts×dModel")
	}
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		return nil, nil, err
	}
	scoresB, err := QMVBF16(normed, routerProj.Packed, routerProj.Scales, routerProj.Biases, numExperts, dModel, groupSize, bits)
	if err != nil {
		return nil, nil, err
	}
	idx, weights := routerSelect(scoresB, perExpertScale, numExperts, topK)
	return idx, weights, nil
}
