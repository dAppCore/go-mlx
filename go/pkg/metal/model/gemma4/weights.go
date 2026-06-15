// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/profile"
)

func gemma4QuantPredicate(path string, defaultConfig *metal.QuantizationConfig) *metal.QuantizationConfig {
	if core.HasSuffix(path, "router.proj") {
		if defaultConfig != nil {
			q := *defaultConfig
			q.Mode = metal.NormalizeQuantizationMode(q.Mode)
			if metal.IsAffineQuantizationMode(q.Mode) {
				q.GroupSize = 64
				q.Bits = 8
			}
			return &q
		}
		return &metal.QuantizationConfig{GroupSize: 64, Bits: 8}
	}
	if defaultConfig != nil {
		return defaultConfig
	}
	// When weights already carry quantization side tensors but config.json omits
	// the quantization block, let MLX use its affine defaults instead of
	// silently downgrading the layer to an incorrect dense projection.
	return &metal.QuantizationConfig{}
}

func gemma4QuantForWeight(path string, defaultConfig *metal.QuantizationConfig, weight, scales *metal.Array) *metal.QuantizationConfig {
	q := gemma4QuantPredicate(path, defaultConfig)
	if q == nil {
		return nil
	}
	resolved := *q
	resolved.Mode = metal.NormalizeQuantizationMode(resolved.Mode)
	if resolved.Mode == "mxfp4" && resolved.Bits == 0 {
		resolved.Bits = 4
	}
	if resolved.Mode == "mxfp8" && resolved.Bits == 0 {
		resolved.Bits = 8
	}
	if (resolved.Mode == "mxfp4" || resolved.Mode == "mxfp8") && resolved.GroupSize == 0 {
		resolved.GroupSize = 32
	}
	if resolved.Mode == "nvfp4" {
		if resolved.Bits == 0 {
			resolved.Bits = 4
		}
		if resolved.GroupSize == 0 {
			resolved.GroupSize = 16
		}
	}
	if !metal.IsAffineQuantizationMode(resolved.Mode) &&
		resolved.GroupSize > 0 &&
		inferGemma4QuantBits(weight, scales, resolved.GroupSize) == 0 {
		if inferred := inferGemma4QuantBits(weight, scales, 64); inferred > 0 {
			resolved.Mode = "affine"
			resolved.GroupSize = 64
			resolved.Bits = inferred
		}
	}
	if metal.IsAffineQuantizationMode(resolved.Mode) && resolved.GroupSize <= 0 && weight != nil && weight.Valid() && weight.Dtype() == metal.DTypeUint32 {
		if inferred := inferGemma4QuantBits(weight, scales, 64); inferred > 0 {
			resolved.GroupSize = 64
			resolved.Bits = inferred
		}
	}
	if metal.IsAffineQuantizationMode(resolved.Mode) {
		if inferred := inferGemma4QuantBits(weight, scales, resolved.GroupSize); inferred > 0 {
			resolved.Bits = inferred
		}
	}
	return &resolved
}

func inferGemma4QuantBits(weight, scales *metal.Array, groupSize int) int {
	if weight == nil || scales == nil || groupSize <= 0 || !weight.Valid() || !scales.Valid() {
		return 0
	}
	wShape := weight.Shape()
	sShape := scales.Shape()
	if len(wShape) == 0 || len(sShape) == 0 {
		return 0
	}
	weightCols := int(wShape[len(wShape)-1])
	scaleCols := int(sShape[len(sShape)-1])
	if weightCols <= 0 || scaleCols <= 0 {
		return 0
	}
	numerator := weightCols * 32
	denominator := scaleCols * groupSize
	if denominator <= 0 || numerator%denominator != 0 {
		return 0
	}
	bits := numerator / denominator
	switch bits {
	case 2, 3, 4, 5, 6, 8:
		return bits
	default:
		return 0
	}
}

func splitGemma4GateUpArray(a *metal.Array) (*metal.Array, *metal.Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	var shapeBuf [metal.MaxTensorRank]int32
	shape := a.ShapeInto(shapeBuf[:0])
	if len(shape) == 0 {
		return nil, nil, false
	}
	axis := len(shape) - 2
	if len(shape) == 1 {
		axis = 0
	} else if len(shape) == 2 {
		// Expert tensors are typically [num_experts, 2*hidden]. Split the
		// feature axis instead of the expert axis.
		axis = 1
	}
	mid := shape[axis] / 2
	if mid <= 0 || shape[axis]%2 != 0 {
		return nil, nil, false
	}
	var startsBuf, endsBuf [metal.MaxTensorRank]int32
	starts := startsBuf[:len(shape)]
	ends := endsBuf[:len(shape)]
	copy(ends, shape)
	ends[axis] = mid
	left := metal.Slice(a, starts, ends)
	if !left.IsRowContiguous() {
		contiguous := metal.Contiguous(left)
		metal.Free(left)
		metal.Materialize(contiguous)
		left = contiguous
	}
	starts[axis] = mid
	ends[axis] = shape[axis]
	right := metal.Slice(a, starts, ends)
	if !right.IsRowContiguous() {
		contiguous := metal.Contiguous(right)
		metal.Free(right)
		metal.Materialize(contiguous)
		right = contiguous
	}
	return left, right, true
}

func sanitizeGemma4Weights(raw map[string]*metal.Array) map[string]*metal.Array {
	return sanitizeGemma4WeightsAs(gemma4Architecture, raw)
}

// sanitizeGemma4WeightsAs canonicalises checkpoint weight names under the
// given architecture profile — diffusion_gemma re-roots its model.decoder.*
// trunk through here while sharing every other gemma4 rule.
func sanitizeGemma4WeightsAs(architecture string, raw map[string]*metal.Array) map[string]*metal.Array {
	sanitized := make(map[string]*metal.Array, len(raw))
	retained := make(map[*metal.Array]struct{}, len(raw))
	discarded := make([]*metal.Array, 0)
	for name, arr := range raw {
		canonical, skip := canonicalGemma4WeightNameAs(architecture, name)
		if skip {
			discarded = append(discarded, arr)
			continue
		}
		for _, suffix := range []string{".weight", ".scales", ".biases", ".bias"} {
			if core.HasSuffix(canonical, ".experts.gate_up_proj"+suffix) {
				base := core.TrimSuffix(canonical, suffix)
				base = core.TrimSuffix(base, ".gate_up_proj")
				fused := base + ".switch_glu.gate_up_proj" + suffix
				if prev, ok := sanitized[fused]; ok && prev != arr {
					delete(retained, prev)
					discarded = append(discarded, prev)
				}
				sanitized[fused] = arr
				if arr != nil {
					retained[arr] = struct{}{}
				}
				gate, up, ok := splitGemma4GateUpArray(arr)
				if !ok {
					goto nextWeight
				}
				sanitized[base+".switch_glu.gate_proj"+suffix] = gate
				sanitized[base+".switch_glu.up_proj"+suffix] = up
				goto nextWeight
			}
			if core.HasSuffix(canonical, ".experts.down_proj"+suffix) {
				canonical = core.TrimSuffix(canonical, ".down_proj"+suffix) + ".switch_glu.down_proj" + suffix
				break
			}
		}
		if prev, ok := sanitized[canonical]; ok && prev != arr {
			delete(retained, prev)
			discarded = append(discarded, prev)
		}
		sanitized[canonical] = arr
		if arr != nil {
			retained[arr] = struct{}{}
		}
	nextWeight:
	}
	freed := make(map[*metal.Array]struct{}, len(discarded))
	for _, arr := range discarded {
		if arr == nil {
			continue
		}
		if _, ok := retained[arr]; ok {
			continue
		}
		if _, ok := freed[arr]; ok {
			continue
		}
		metal.Free(arr)
		freed[arr] = struct{}{}
	}
	return sanitized
}

func trimGemma4WrapperPrefix(name string) (string, bool) {
	return profile.TrimWeightWrapperPrefix(gemma4Architecture, name)
}

func canonicalGemma4WeightName(name string) (string, bool) {
	return canonicalGemma4WeightNameAs(gemma4Architecture, name)
}

func canonicalGemma4WeightNameAs(architecture, name string) (string, bool) {
	canonical, ok := profile.CanonicalWeightName(architecture, name)
	if !ok {
		return "", true
	}
	return canonical, false
}

func gemma4Ones(shape []int32) *metal.Array {
	base := metal.Zeros(shape, metal.DTypeFloat32)
	ones := metal.AddScalar(base, 1.0)
	metal.Free(base)
	return ones
}

func gemma4WeightAny(weights map[string]*metal.Array, names ...string) *metal.Array {
	for _, name := range names {
		if arr := metal.ResolveWeight(weights, name); arr != nil {
			return arr
		}
	}
	return nil
}

func inferGemma4HeadDim(weights map[string]*metal.Array, layerTypes []string, numAttentionHeads int32, target string) int32 {
	for i, layerType := range layerTypes {
		if layerType != target {
			continue
		}
		if qProj := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)); qProj != nil {
			shape := qProj.Shape()
			if len(shape) > 0 && numAttentionHeads > 0 && shape[0]%numAttentionHeads == 0 {
				return shape[0] / numAttentionHeads
			}
		}
	}
	return 0
}

func inferGemma4PerLayerInputSize(weights map[string]*metal.Array, numHiddenLayers int32) int32 {
	if numHiddenLayers <= 0 {
		return 0
	}
	if w := gemma4WeightAny(weights, "model.per_layer_model_projection.weight"); w != nil {
		shape := w.Shape()
		if len(shape) >= 2 {
			outFeatures := int32(1)
			for _, dim := range shape[:len(shape)-1] {
				outFeatures *= dim
			}
			if outFeatures%numHiddenLayers == 0 {
				return outFeatures / numHiddenLayers
			}
		}
	}
	for i := range numHiddenLayers {
		if w := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.per_layer_input_gate.weight", i)); w != nil {
			shape := w.Shape()
			if len(shape) >= 2 && shape[0] > 0 {
				return shape[0]
			}
		}
		if w := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.per_layer_projection.weight", i)); w != nil {
			shape := w.Shape()
			if len(shape) >= 2 && shape[len(shape)-1] > 0 {
				return shape[len(shape)-1]
			}
		}
	}
	if w := gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight"); w != nil {
		shape := w.Shape()
		switch len(shape) {
		case 2:
			if shape[1]%numHiddenLayers == 0 {
				return shape[1] / numHiddenLayers
			}
		case 3:
			if shape[1] == numHiddenLayers {
				return shape[2]
			}
			if shape[2] == numHiddenLayers {
				return shape[1]
			}
		default:
			if len(shape) > 1 {
				featureSize := int32(1)
				for _, dim := range shape[1:] {
					featureSize *= dim
				}
				if featureSize%numHiddenLayers == 0 {
					return featureSize / numHiddenLayers
				}
			}
		}
	}
	return 0
}

func gemma4Linear(weights map[string]*metal.Array, prefix string, defaultQ *metal.QuantizationConfig) *metal.Linear {
	weight := gemma4WeightAny(weights, prefix+".weight")
	if weight == nil {
		return nil
	}
	scales := gemma4WeightAny(weights, prefix+".scales")
	biases := gemma4WeightAny(weights, prefix+".biases")
	bias := gemma4WeightAny(weights, prefix+".bias")
	if scales != nil {
		if q := gemma4QuantForWeight(prefix, defaultQ, weight, scales); q != nil {
			return metal.NewQuantizedLinearWithMode(weight, scales, biases, bias, q.GroupSize, q.Bits, q.Mode)
		}
	}
	return metal.NewLinear(weight, bias)
}

// gemma4Embedding loads an embedding table at prefix, carrying affine
// quantization (scales/biases/bits/group) when the checkpoint stores a
// quantized table. Load-bearing for QAT drafters whose tied output projection
// runs through Embedding.AsLinear: without the quant metadata the packed weight
// is read as dense and the projection produces garbage logits (no draft token).
// Returns nil when the weight is absent.
func gemma4Embedding(weights map[string]*metal.Array, prefix string, defaultQ *metal.QuantizationConfig) *metal.Embedding {
	weight := gemma4WeightAny(weights, prefix+".weight")
	if weight == nil {
		return nil
	}
	embed := &metal.Embedding{Weight: weight}
	if scales := gemma4WeightAny(weights, prefix+".scales"); scales != nil {
		embed.Scales = scales
		embed.Biases = gemma4WeightAny(weights, prefix+".biases")
		if q := gemma4QuantForWeight(prefix, defaultQ, weight, scales); q != nil {
			embed.GroupSize = q.GroupSize
			embed.Bits = q.Bits
			embed.QuantizationMode = q.Mode
		}
	}
	return embed
}

func gemma4SwitchLinear(weights map[string]*metal.Array, defaultQ *metal.QuantizationConfig, prefixes ...string) *metal.SwitchLinear {
	for _, prefix := range prefixes {
		weight := gemma4WeightAny(weights, prefix+".weight")
		if weight == nil {
			continue
		}
		scales := gemma4WeightAny(weights, prefix+".scales")
		biases := gemma4WeightAny(weights, prefix+".biases")
		bias := gemma4WeightAny(weights, prefix+".bias")
		if scales != nil {
			if q := gemma4QuantForWeight(prefix, defaultQ, weight, scales); q != nil {
				return metal.NewQuantizedSwitchLinearWithMode(weight, scales, biases, bias, q.GroupSize, q.Bits, q.Mode)
			}
		}
		return metal.NewSwitchLinear(weight, bias)
	}
	return nil
}

func gemma4OutputLinear(weights map[string]*metal.Array, cfg *Gemma4TextConfig, embed *metal.Embedding) (*metal.Linear, error) {
	if output := gemma4Linear(weights, "lm_head", cfg.Quantization); output != nil {
		return output, nil
	}
	if cfg.TieWordEmbeddings {
		if embed == nil {
			return nil, core.E("gemma4.outputLinear", "tied output requested without embed_tokens", nil)
		}
		return embed.AsLinear(), nil
	}
	return nil, core.E("gemma4.outputLinear", "missing lm_head.weight with tie_word_embeddings=false", nil)
}

func buildGemma4CacheLayout(layers []*Gemma4DecoderLayer, numShared int32) ([]int32, []int32) {
	previous := make([]int32, len(layers))
	cacheIndexByLayer := make([]int32, len(layers))
	for i := range previous {
		previous[i] = int32(i)
		cacheIndexByLayer[i] = -1
	}
	if len(layers) == 0 {
		return previous, cacheIndexByLayer
	}
	firstShared := max(int32(len(layers))-numShared, 0)
	if firstShared > int32(len(layers)) {
		firstShared = int32(len(layers))
	}
	latestByType := make(map[string]int32)
	nextCacheIndex := int32(0)
	for i := int32(0); i < int32(len(layers)); i++ {
		layerType := layers[i].LayerType
		ownsCache := i < firstShared
		if !ownsCache {
			if prev, ok := latestByType[layerType]; ok {
				previous[i] = prev
			} else {
				// Small toy configs can place the first layer of an attention type
				// in the shared-KV region. Promote it to an owner so decoding keeps
				// a persistent cache instead of silently recomputing from scratch.
				ownsCache = true
			}
		}
		if ownsCache {
			previous[i] = i
			latestByType[layerType] = i
			cacheIndexByLayer[i] = nextCacheIndex
			nextCacheIndex++
		}
	}
	return previous, cacheIndexByLayer
}

func buildGemma4PreviousKVs(layers []*Gemma4DecoderLayer, numShared int32) []int32 {
	previous, _ := buildGemma4CacheLayout(layers, numShared)
	return previous
}

func gemma4RotatedDims(headDim int32, params RopeParams) int32 {
	factor := params.PartialRotaryFactor
	if factor <= 0 {
		factor = 1
	}
	dims := int32(math.Round(float64(float32(headDim) * factor)))
	if dims <= 0 {
		dims = headDim
	}
	if dims > headDim {
		dims = headDim
	}
	if dims%2 != 0 {
		dims--
	}
	if dims <= 0 {
		dims = headDim
	}
	return dims
}

func gemma4ProportionalFreqs(headDim int32, rotatedDims int32, base float32, factor float32) *metal.Array {
	if rotatedDims <= 0 {
		return nil
	}
	exponents := metal.Arange(0, float64(rotatedDims), 2, metal.DTypeFloat32)
	scale := float32(1.0 / float32(headDim))
	exponentsScaled := metal.MulScalar(exponents, scale)
	metal.Free(exponents)
	baseScalar := metal.FromValue(base)
	freqs := metal.Power(baseScalar, exponentsScaled)
	metal.Free(baseScalar, exponentsScaled)
	if factor != 0 && factor != 1 {
		scaled := metal.MulScalar(freqs, factor)
		metal.Free(freqs)
		freqs = scaled
	}
	if rotatedDims < headDim {
		extra := make([]float32, (headDim-rotatedDims)/2)
		for i := range extra {
			extra[i] = float32(math.Inf(1))
		}
		inf := metal.FromValues(extra, len(extra))
		combined := metal.Concatenate2(freqs, inf, 0)
		metal.Free(freqs, inf)
		freqs = combined
	}
	return freqs
}

func gemma4AttentionScale(headDim int32) float32 {
	return 1.0
}

func gemma4TrackArrays(retained map[*metal.Array]struct{}, arrays ...*metal.Array) {
	for _, arr := range arrays {
		if arr == nil || !arr.Valid() {
			continue
		}
		retained[arr] = struct{}{}
	}
}

func gemma4TrackEmbedding(retained map[*metal.Array]struct{}, embedding *metal.Embedding) {
	if embedding == nil {
		return
	}
	gemma4TrackArrays(retained, embedding.Weight, embedding.Scales, embedding.Biases)
}

func gemma4TrackLinear(retained map[*metal.Array]struct{}, linear *metal.Linear) {
	if linear == nil {
		return
	}
	gemma4TrackArrays(retained, linear.Weight, linear.Scales, linear.Biases, linear.Bias)
}

func gemma4TrackSwitchLinear(retained map[*metal.Array]struct{}, linear *metal.SwitchLinear) {
	if linear == nil {
		return
	}
	gemma4TrackArrays(retained, linear.Weight, linear.Scales, linear.Biases, linear.Bias)
}

func gemma4RetainedWeights(m *Gemma4Model) map[*metal.Array]struct{} {
	retained := make(map[*metal.Array]struct{})
	if m == nil {
		return retained
	}

	gemma4TrackEmbedding(retained, m.EmbedTokens)
	gemma4TrackEmbedding(retained, m.EmbedTokensPerLayer)
	if m.MultiModalProjector != nil {
		gemma4TrackLinear(retained, m.MultiModalProjector.Projection)
		gemma4TrackLinear(retained, m.MultiModalProjector.Linear1)
		gemma4TrackLinear(retained, m.MultiModalProjector.Linear2)
	}
	if m.AudioProjector != nil {
		gemma4TrackLinear(retained, m.AudioProjector.Projection)
	}
	gemma4TrackAudioEncoder(retained, m.AudioEncoder)
	gemma4TrackLinear(retained, m.PerLayerModelProj)
	gemma4TrackLinear(retained, m.Output)
	if m.Norm != nil {
		gemma4TrackArrays(retained, m.Norm.Weight)
	}
	if m.PerLayerProjNorm != nil {
		gemma4TrackArrays(retained, m.PerLayerProjNorm.Weight)
	}

	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		if layer.InputNorm != nil {
			gemma4TrackArrays(retained, layer.InputNorm.Weight)
		}
		if layer.PostAttnNorm != nil {
			gemma4TrackArrays(retained, layer.PostAttnNorm.Weight)
		}
		if layer.PreFFNorm != nil {
			gemma4TrackArrays(retained, layer.PreFFNorm.Weight)
		}
		if layer.PostFFNorm != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm.Weight)
		}
		if layer.PreFFNorm2 != nil {
			gemma4TrackArrays(retained, layer.PreFFNorm2.Weight)
		}
		if layer.PostFFNorm1 != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm1.Weight)
		}
		if layer.PostFFNorm2 != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm2.Weight)
		}
		if layer.PostPerLayerInputNorm != nil {
			gemma4TrackArrays(retained, layer.PostPerLayerInputNorm.Weight)
		}
		gemma4TrackArrays(retained, layer.LayerScalar)
		gemma4TrackLinear(retained, layer.PerLayerInputGate)
		gemma4TrackLinear(retained, layer.PerLayerProjection)

		if attn := layer.Attention; attn != nil {
			gemma4TrackLinear(retained, attn.QProj)
			gemma4TrackLinear(retained, attn.KProj)
			gemma4TrackLinear(retained, attn.VProj)
			gemma4TrackLinear(retained, attn.OProj)
			if attn.QNorm != nil {
				gemma4TrackArrays(retained, attn.QNorm.Weight)
			}
			if attn.KNorm != nil {
				gemma4TrackArrays(retained, attn.KNorm.Weight)
			}
		}

		if mlp := layer.MLP; mlp != nil {
			gemma4TrackLinear(retained, mlp.GateProj)
			gemma4TrackLinear(retained, mlp.UpProj)
			gemma4TrackLinear(retained, mlp.DownProj)
		}

		if router := layer.Router; router != nil {
			gemma4TrackLinear(retained, router.Proj)
			gemma4TrackArrays(retained, router.Scale, router.PerExpertScale)
		}

		if experts := layer.Experts; experts != nil {
			gemma4TrackSwitchLinear(retained, experts.GateUpProj)
			gemma4TrackSwitchLinear(retained, experts.GateProj)
			gemma4TrackSwitchLinear(retained, experts.UpProj)
			gemma4TrackSwitchLinear(retained, experts.DownProj)
		}
	}

	return retained
}

func gemma4LazyRetainedWeights(m *Gemma4Model) map[*metal.Array]struct{} {
	lazy := make(map[*metal.Array]struct{})
	if m == nil {
		return lazy
	}
	gemma4TrackEmbedding(lazy, m.EmbedTokensPerLayer)
	return lazy
}

func gemma4FreeUnusedWeights(weights map[string]*metal.Array, retained map[*metal.Array]struct{}) {
	freed := make(map[*metal.Array]struct{})
	for _, arr := range weights {
		if arr == nil || !arr.Valid() {
			continue
		}
		if _, ok := retained[arr]; ok {
			continue
		}
		if _, ok := freed[arr]; ok {
			continue
		}
		metal.Free(arr)
		freed[arr] = struct{}{}
	}
}

func gemma4MaterializableRetainedWeights(retained, lazy map[*metal.Array]struct{}) []*metal.Array {
	all := make([]*metal.Array, 0, len(retained))
	for arr := range retained {
		if arr == nil || !arr.Valid() {
			continue
		}
		if _, ok := lazy[arr]; ok {
			continue
		}
		all = append(all, arr)
	}
	return all
}

func gemma4MaterializeRetainedWeights(retained, lazy map[*metal.Array]struct{}) {
	all := gemma4MaterializableRetainedWeights(retained, lazy)
	metal.Materialize(all...)
}

func precomputeGemma4ScaledWeights(m *Gemma4Model) {
	if m.Norm != nil {
		m.NormScaled = metal.Copy(m.Norm.Weight)
	}
	if m.PerLayerProjNorm != nil && m.PerLayerProjNorm.Weight != nil {
		m.PerLayerProjNormScaled = metal.Copy(m.PerLayerProjNorm.Weight)
	}

	var scaled []*metal.Array
	scaled = append(scaled, m.NormScaled, m.PerLayerProjNormScaled)

	for _, layer := range m.Layers {
		if layer.InputNorm != nil && layer.InputNorm.Weight != nil {
			layer.InputNormScaled = metal.Copy(layer.InputNorm.Weight)
		}
		if layer.PostAttnNorm != nil && layer.PostAttnNorm.Weight != nil {
			layer.PostAttnNormScaled = metal.Copy(layer.PostAttnNorm.Weight)
		}
		if layer.PreFFNorm != nil && layer.PreFFNorm.Weight != nil {
			layer.PreFFNormScaled = metal.Copy(layer.PreFFNorm.Weight)
		}
		if layer.PostFFNorm != nil && layer.PostFFNorm.Weight != nil {
			layer.PostFFNormScaled = metal.Copy(layer.PostFFNorm.Weight)
		}
		if layer.PreFFNorm2 != nil && layer.PreFFNorm2.Weight != nil {
			layer.PreFFNorm2Scaled = metal.Copy(layer.PreFFNorm2.Weight)
		}
		if layer.PostFFNorm1 != nil && layer.PostFFNorm1.Weight != nil {
			layer.PostFFNorm1Scaled = metal.Copy(layer.PostFFNorm1.Weight)
		}
		if layer.PostFFNorm2 != nil && layer.PostFFNorm2.Weight != nil {
			layer.PostFFNorm2Scaled = metal.Copy(layer.PostFFNorm2.Weight)
		}
		if layer.PostPerLayerInputNorm != nil && layer.PostPerLayerInputNorm.Weight != nil {
			layer.PostPerLayerInputNormScaled = metal.Copy(layer.PostPerLayerInputNorm.Weight)
		}
		if layer.Attention != nil {
			if layer.Attention.QNorm != nil && layer.Attention.QNorm.Weight != nil {
				layer.Attention.QNormScaled = metal.Copy(layer.Attention.QNorm.Weight)
			}
			if layer.Attention.KNorm != nil && layer.Attention.KNorm.Weight != nil {
				layer.Attention.KNormScaled = metal.Copy(layer.Attention.KNorm.Weight)
			}
			scaled = append(scaled, layer.Attention.QNormScaled, layer.Attention.KNormScaled, layer.Attention.RopeFreqs)
		}
		if layer.Router != nil && layer.Router.Scale != nil {
			layer.Router.ScaleScaled = metal.MulScalar(layer.Router.Scale, layer.Router.RootSize)
			scaled = append(scaled, layer.Router.ScaleScaled)
		}
		scaled = append(
			scaled,
			layer.InputNormScaled,
			layer.PostAttnNormScaled,
			layer.PreFFNormScaled,
			layer.PostFFNormScaled,
			layer.PreFFNorm2Scaled,
			layer.PostFFNorm1Scaled,
			layer.PostFFNorm2Scaled,
			layer.PostPerLayerInputNormScaled,
		)
	}
	metal.Materialize(scaled...)
}

func (m *Gemma4Model) ensureCacheLayout() {
	if len(m.PreviousKVs) == len(m.Layers) && len(m.CacheIndexByLayer) == len(m.Layers) {
		return
	}
	previous, cacheIndexByLayer := buildGemma4CacheLayout(m.Layers, m.Cfg.NumKVSharedLayers)
	m.PreviousKVs = previous
	m.CacheIndexByLayer = cacheIndexByLayer
}

// LoadGemma4 loads a Gemma 4 text model from a directory.
