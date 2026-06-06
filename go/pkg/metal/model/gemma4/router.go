// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"dappco.re/go/mlx/pkg/metal"
)

func (r *Gemma4Router) forward(x *metal.Array) (*metal.Array, *metal.Array) {
	scaled := r.ScaleScaled
	if scaled == nil {
		scaled = metal.MulScalar(r.Scale, r.RootSize)
		defer metal.Free(scaled)
	}
	normed := metal.RMSNorm(x, scaled, r.Eps)
	expertScores := r.Proj.Forward(normed)
	metal.Free(normed)

	numExperts := expertScores.Dim(expertScores.NumDims() - 1)
	topK := int(r.TopK)
	if topK <= 0 || topK > numExperts {
		topK = numExperts
	}
	kth := numExperts - topK
	topKIndices := metal.Argpartition(expertScores, kth, -1)
	sliced := metal.SliceAxis(topKIndices, -1, int32(kth), int32(numExperts))
	metal.Free(topKIndices)
	topKIndices = sliced

	topKWeights := metal.TakeAlongAxis(expertScores, topKIndices, -1)
	metal.Free(expertScores)
	topKWeightsSoftmax := metal.Softmax(topKWeights)
	metal.Free(topKWeights)
	if r.PerExpertScale == nil || !r.PerExpertScale.Valid() {
		return topKIndices, topKWeightsSoftmax
	}
	perExpertScale := metal.Take(r.PerExpertScale, topKIndices, 0)
	weighted := metal.Mul(topKWeightsSoftmax, perExpertScale)
	metal.Free(topKWeightsSoftmax, perExpertScale)
	return topKIndices, weighted
}

// NewCache creates per-layer KV caches for Gemma 4.
