// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func gemma4NormalizePerLayerTensor(x *metal.Array, batchSize, seqLen, numLayers, hiddenSize int32) *metal.Array {
	if x == nil || !x.Valid() {
		return x
	}

	// Stack-allocated shape scratch — per-layer tensor reshape is in the
	// per-token decode path. Avoids the per-call []int32 heap alloc from
	// x.Shape() (24 B/op × NumHiddenLayers × tokens).
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	switch len(shape) {
	case 4:
		if shape[2] == numLayers && shape[3] == hiddenSize {
			return x
		}
		if shape[2] == hiddenSize && shape[3] == numLayers {
			return metal.Transpose4(x, 0, 1, 3, 2)
		}
	case 3:
		if shape[2] == numLayers*hiddenSize {
			return metal.Reshape(x, batchSize, seqLen, numLayers, hiddenSize)
		}
	}

	return metal.Reshape(x, batchSize, seqLen, numLayers, hiddenSize)
}

func (m *Gemma4Model) computePerLayerInputs(tokens, hidden *metal.Array) []*metal.Array {
	// Stack-allocated shape scratch — per-token decode hot path. Calling
	// tokens.Shape() twice paid two []int32 heap allocs (24 B/op each).
	var tokShapeBuf [metal.MaxTensorRank]int32
	tokShape := tokens.ShapeInto(tokShapeBuf[:0])
	B, L := tokShape[0], tokShape[1]
	combined := m.computePerLayerInputTensor(tokens, hidden, B, L)
	return m.splitPerLayerInputTensor(combined)
}

func (m *Gemma4Model) computePerLayerInputTensor(tokens, hidden *metal.Array, B, L int32) *metal.Array {
	if disableGemma4PerLayerInputs {
		return nil
	}
	if m.EmbedTokensPerLayer == nil || m.PerLayerModelProj == nil || m.PerLayerProjNorm == nil || m.PerLayerProjNormScaled == nil {
		return nil
	}
	if combined, ok := m.compiledPerLayerInputTensor(tokens, hidden); ok {
		return combined
	}
	return m.perLayerInputTensor(tokens, hidden, B, L)
}

func (m *Gemma4Model) perLayerInputTensor(tokens, hidden *metal.Array, B, L int32) *metal.Array {
	perLayer := m.EmbedTokensPerLayer.Forward(tokens)
	scaled := metal.MulScalar(perLayer, m.Cfg.PerLayerInputEmbeddingScale)
	metal.Free(perLayer)
	perLayer = gemma4NormalizePerLayerTensor(scaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	if perLayer != scaled {
		metal.Free(scaled)
	}

	projected := m.PerLayerModelProj.Forward(hidden)
	projectedScaled := metal.MulScalar(projected, m.Cfg.PerLayerProjectionScale)
	metal.Free(projected)
	projected = gemma4NormalizePerLayerTensor(projectedScaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	if projected != projectedScaled {
		metal.Free(projectedScaled)
	}
	projectedNormed := metal.RMSNorm(projected, m.PerLayerProjNormScaled, m.Cfg.RMSNormEps)
	metal.Free(projected)

	combined := metal.Add(projectedNormed, perLayer)
	metal.Free(projectedNormed, perLayer)
	combinedScaled := metal.MulScalar(combined, gemma4PerLayerCombineScale)
	metal.Free(combined)
	combined = combinedScaled
	return combined
}

func (m *Gemma4Model) splitPerLayerInputTensor(combined *metal.Array) []*metal.Array {
	if combined == nil || !combined.Valid() {
		return nil
	}
	defer metal.Free(combined)

	perLayerInputs := make([]*metal.Array, m.Cfg.NumHiddenLayers)
	var shapeBuf [metal.MaxTensorRank]int32
	shape := combined.ShapeInto(shapeBuf[:0])
	if len(shape) == 4 {
		for i := range m.Cfg.NumHiddenLayers {
			perLayerInputs[i] = m.perLayerInputForLayer(combined, shape[0], shape[1], i)
		}
		return perLayerInputs
	}

	// Generic fallback for malformed or legacy shapes. The normal Gemma 4 path
	// is rank-4 and should use the allocation-free Slice4/Reshape3 helper above.
	squeezeAxis2 := []int{2}
	for i := range m.Cfg.NumHiddenLayers {
		sliced := metal.SliceAxis(combined, 2, i, i+1)
		perLayerInputs[i] = metal.Squeeze(sliced, squeezeAxis2...)
		metal.Free(sliced)
	}
	return perLayerInputs
}

func (m *Gemma4Model) perLayerInputForLayer(combined *metal.Array, B, L, layer int32) *metal.Array {
	if combined == nil || !combined.Valid() || layer < 0 || layer >= m.Cfg.NumHiddenLayers {
		return nil
	}
	if combined.NumDims() != 4 {
		sliced := metal.SliceAxis(combined, 2, layer, layer+1)
		out := metal.Reshape3(sliced, B, L, m.Cfg.HiddenSizePerLayerInput)
		metal.Free(sliced)
		return out
	}
	sliced := metal.Slice4(combined, 0, 0, layer, 0, B, L, layer+1, m.Cfg.HiddenSizePerLayerInput)
	out := metal.Reshape3(sliced, B, L, m.Cfg.HiddenSizePerLayerInput)
	metal.Free(sliced)
	return out
}

func (m *Gemma4Model) compiledPerLayerInputTensor(tokens, hidden *metal.Array) (_ *metal.Array, ok bool) {
	if !enableCompiledGemma4PerLayerInputs || m.compiledPerLayerInputsFailed {
		return nil, false
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			core.Error("mlx: compiled Gemma 4 per-layer inputs failed; falling back to Go graph", "error", recovered)
			m.compiledPerLayerInputsFailed = true
			if m.compiledPerLayerInputs != nil {
				m.compiledPerLayerInputs.Free()
				m.compiledPerLayerInputs = nil
			}
			ok = false
		}
	}()
	if m.compiledPerLayerInputs == nil || !m.compiledPerLayerInputs.Valid() {
		m.compiledPerLayerInputs = metal.CompileShapeless(func(inputs []*metal.Array) []*metal.Array {
			if len(inputs) < 2 {
				return nil
			}
			shape := inputs[0].Shape()
			if len(shape) < 2 {
				return nil
			}
			out := m.perLayerInputTensor(inputs[0], inputs[1], shape[0], shape[1])
			return []*metal.Array{out}
		}, true)
	}
	outs := m.compiledPerLayerInputs.Call(tokens, hidden)
	if len(outs) != 1 || outs[0] == nil || !outs[0].Valid() {
		metal.Free(outs...)
		m.compiledPerLayerInputsFailed = true
		return nil, false
	}
	return outs[0], true
}
