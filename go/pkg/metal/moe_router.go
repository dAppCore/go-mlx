// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// MoERouterProjection is the model-family neutral representation of a router
// projection. Qwen, Mixtral, GPT-OSS, Kimi, Gemma 4, and MiniMax wrap this
// shape differently in their loaders, but the per-token projection is the same
// hidden -> expert-score matvec.
type MoERouterProjection struct {
	Weight    *Array
	Scales    *Array
	Biases    *Array
	GroupSize int
	Bits      int
}

func (r MoERouterProjection) Linear() *Linear {
	if r.Weight == nil || !r.Weight.Valid() {
		return nil
	}
	if r.Scales != nil && r.Scales.Valid() {
		return NewQuantizedLinear(r.Weight, r.Scales, r.Biases, nil, r.GroupSize, r.Bits)
	}
	return NewLinear(r.Weight, nil)
}

func moeRouterScores(input *Array, router MoERouterProjection) (*Array, bool, error) {
	proj := router.Linear()
	if proj == nil {
		return nil, false, nil
	}
	if scores, ok, err := nativeMoERouterMatVecScores(input, proj); ok || err != nil {
		return scores, ok, err
	}
	return proj.Forward(input), true, nil
}

func moeRouterSelectTopK(input *Array, router MoERouterProjection, perExpertScale *Array, topK int) (*Array, *Array, bool, error) {
	scores, ok, err := moeRouterScores(input, router)
	if err != nil || !ok {
		return nil, nil, ok, err
	}
	defer Free(scores)

	expertIDs, routeWeights, ok, err := nativeMoERouterTopK(scores, perExpertScale, topK)
	if err != nil || !ok {
		return nil, nil, ok, err
	}
	return expertIDs, routeWeights, true, nil
}

// moeRouterProjection adapts the neutral MoERouter weight set into the
// projection shape consumed by the top-k routing algorithm.
func moeRouterProjection(router *MoERouter) MoERouterProjection {
	if router == nil {
		return MoERouterProjection{}
	}
	return MoERouterProjection{
		Weight:    router.Weight,
		Scales:    router.Scales,
		Biases:    router.Biases,
		GroupSize: router.GroupSize,
		Bits:      router.Bits,
	}
}

// moeRouterTopK runs the router projection then selects the top-k experts for a
// loaded MoERouter. Shared by every sparse MoE model on the metal SDK.
func moeRouterTopK(input *Array, router *MoERouter, topK int) (*Array, *Array, bool, error) {
	if topK <= 0 {
		return nil, nil, false, core.NewError("mlx: moe router top-k must be positive")
	}
	return moeRouterSelectTopK(input, moeRouterProjection(router), nil, topK)
}
