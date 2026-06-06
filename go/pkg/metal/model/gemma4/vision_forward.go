// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Vision/audio projection + multimodal forward passes (encode, inject, project,
// 2-D RoPE).

func (m *Gemma4Model) ForwardMultiModal(tokens *metal.Array, imagePixels []*metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardUnifiedMultiModal(tokens, imagePixels, nil, caches)
}

func (m *Gemma4Model) ForwardUnifiedMultiModal(tokens *metal.Array, imagePixels []*metal.Array, audioFeatures []*metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardUnifiedVideoMultiModal(tokens, imagePixels, audioFeatures, nil, caches)
}

func (m *Gemma4Model) ForwardUnifiedVideoMultiModal(tokens *metal.Array, imagePixels []*metal.Array, audioFeatures []*metal.Array, videoFrames []*metal.Array, caches []metal.Cache) *metal.Array {
	hasImages := len(imagePixels) > 0 && (m.VisionTower != nil || m.MultiModalProjector != nil)
	hasAudio := len(audioFeatures) > 0 && m.AudioProjector != nil
	hasVideo := len(videoFrames) > 0 && (m.VisionTower != nil || m.MultiModalProjector != nil)
	if !hasImages && !hasAudio && !hasVideo {
		return m.Forward(tokens, caches)
	}

	// Stack-allocated shape scratch — multimodal forward-pass entrypoint.
	// Reused as the tokenShape argument to injectGemma4ImageFeatures.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	if len(shape) != 2 {
		return m.Forward(tokens, caches)
	}

	tokenIDs := tokens.DataInt32()
	imageTokenCount, audioTokenCount, videoTokenCount := gemma4UnifiedModalTokenCounts(m.Cfg, tokenIDs)
	if imageTokenCount == 0 && audioTokenCount == 0 && videoTokenCount == 0 {
		return m.Forward(tokens, caches)
	}

	h := m.EmbedTokens.Forward(tokens)
	scaledH := metal.MulScalar(h, m.Cfg.EmbeddingScale)
	metal.Free(h)
	h = scaledH

	if imageTokenCount > 0 && hasImages {
		imageFeatures := m.encodeGemma4Images(imagePixels)
		if imageFeatures == nil || !imageFeatures.Valid() {
			metal.Free(h)
			return m.Forward(tokens, caches)
		}
		h = m.injectGemma4ImageFeatures(h, tokenIDs, shape, imageFeatures)
		metal.Free(imageFeatures)
	}

	if audioTokenCount > 0 && hasAudio {
		projectedAudio := m.encodeGemma4Audio(audioFeatures)
		if projectedAudio == nil || !projectedAudio.Valid() {
			metal.Free(h)
			return m.Forward(tokens, caches)
		}
		h = m.injectGemma4TokenFeatures(h, tokenIDs, shape, projectedAudio, m.Cfg.AudioTokenID, "audio")
		metal.Free(projectedAudio)
	}
	if videoTokenCount > 0 && hasVideo {
		videoFeatures := m.encodeGemma4Images(videoFrames)
		if videoFeatures == nil || !videoFeatures.Valid() {
			metal.Free(h)
			return m.Forward(tokens, caches)
		}
		h = m.injectGemma4TokenFeatures(h, tokenIDs, shape, videoFeatures, m.Cfg.VideoTokenID, "video")
		metal.Free(videoFeatures)
	}
	return m.forwardGemma4EmbeddingsMasked(tokens, h, nil, caches)
}

func gemma4UnifiedModalTokenCounts(cfg *Gemma4TextConfig, tokenIDs []int32) (imageCount, audioCount, videoCount int) {
	if cfg == nil {
		return 0, 0, 0
	}
	for _, id := range tokenIDs {
		switch {
		case cfg.ImageTokenID != 0 && id == cfg.ImageTokenID:
			imageCount++
		case cfg.AudioTokenID != 0 && id == cfg.AudioTokenID:
			audioCount++
		case cfg.VideoTokenID != 0 && id == cfg.VideoTokenID:
			videoCount++
		}
	}
	return imageCount, audioCount, videoCount
}

func (m *Gemma4Model) encodeGemma4Images(imagePixels []*metal.Array) *metal.Array {
	features := make([]*metal.Array, 0, len(imagePixels))
	for _, image := range imagePixels {
		if image == nil || !image.Valid() {
			continue
		}
		encoded := image
		if m.VisionTower != nil {
			encoded = m.VisionTower.Forward(image)
			if encoded == nil || !encoded.Valid() {
				continue
			}
		}
		projected := encoded
		if m.MultiModalProjector != nil {
			projected = m.MultiModalProjector.Forward(encoded)
			if encoded != image {
				metal.Free(encoded)
			}
		}
		if projected == image {
			projected = image.Clone()
		}
		features = append(features, projected)
	}
	if len(features) == 0 {
		return nil
	}
	if len(features) == 1 {
		return features[0]
	}
	combined := metal.Concatenate(features, 0)
	metal.Free(features...)
	return combined
}

func (m *Gemma4Model) injectGemma4ImageFeatures(h *metal.Array, tokenIDs []int32, tokenShape []int32, features *metal.Array) *metal.Array {
	return m.injectGemma4TokenFeatures(h, tokenIDs, tokenShape, features, m.Cfg.ImageTokenID, "image")
}

func (m *Gemma4Model) encodeGemma4Audio(audioFeatures []*metal.Array) *metal.Array {
	features := make([]*metal.Array, 0, len(audioFeatures))
	for _, feature := range audioFeatures {
		if feature == nil || !feature.Valid() {
			continue
		}
		projected := feature
		if m.AudioProjector != nil {
			projected = m.AudioProjector.Forward(feature)
		} else {
			projected = feature.Clone()
		}
		if projected == feature {
			projected = feature.Clone()
		}
		features = append(features, projected)
	}
	if len(features) == 0 {
		return nil
	}
	if len(features) == 1 {
		return features[0]
	}
	combined := metal.Concatenate(features, 0)
	metal.Free(features...)
	return combined
}

func (m *Gemma4Model) injectGemma4TokenFeatures(h *metal.Array, tokenIDs []int32, tokenShape []int32, features *metal.Array, tokenID int32, label string) *metal.Array {
	featureRows := features
	if features.NumDims() == 3 {
		// Stack-allocated shape scratch — image-feature reshape called per
		// multimodal forward pass.
		var shapeBuf [metal.MaxTensorRank]int32
		shape := features.ShapeInto(shapeBuf[:0])
		featureRows = metal.Reshape(features, shape[0]*shape[1], shape[2])
		defer metal.Free(featureRows)
	}
	if featureRows.NumDims() != 2 {
		return h
	}

	// h.Shape()[2] previously allocated; use Dim(2) instead.
	B, L, H := tokenShape[0], tokenShape[1], int32(h.Dim(2))
	if int32(featureRows.Dim(1)) != H {
		core.Error("gemma4: "+label+" features hidden size mismatch", "features", featureRows.Dim(1), "hidden", H)
		return h
	}
	nFeatures := int32(featureRows.Dim(0))
	tokenSlots := int32(0)
	for _, id := range tokenIDs {
		if id == tokenID {
			tokenSlots++
		}
	}
	if nFeatures != tokenSlots {
		core.Error("gemma4: "+label+" feature count mismatch", "features", nFeatures, "tokens", tokenSlots)
	}
	featureIdx := int32(0)
	for flatIdx, id := range tokenIDs {
		if id != tokenID {
			continue
		}
		if featureIdx >= nFeatures {
			break
		}
		b := int32(flatIdx) / L
		pos := int32(flatIdx) % L
		if b >= B {
			break
		}

		row := metal.SliceAxis(featureRows, 0, featureIdx, featureIdx+1)
		update := metal.Reshape(row, 1, 1, H)
		next := metal.SliceUpdateInplace(h, update, []int32{b, pos, 0}, []int32{b + 1, pos + 1, H})
		metal.Free(h, row, update)
		h = next
		featureIdx++
	}
	return h
}

func (m *Gemma4Model) forwardGemma4EmbeddingsMasked(tokens *metal.Array, h *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	m.ensureCacheLayout()

	// Stack-allocated shape scratch — per-forward-pass hot path.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	perLayerInputs := m.computePerLayerInputs(tokens, h)
	defer metal.Free(perLayerInputs...)

	var ownedMasks []*metal.Array
	fullMask := mask
	slidingMask := mask
	if mask == nil {
		if L > 1 && m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
			slidingMask = buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
			ownedMasks = append(ownedMasks, slidingMask)
		}
	} else if m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
		windowMask := buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
		combined := gemma4CombineMasks(mask, windowMask)
		metal.Free(windowMask)
		slidingMask = combined
		ownedMasks = append(ownedMasks, combined)
	}
	defer metal.Free(ownedMasks...)

	intermediates := make([]sharedKV, len(m.Layers))
	sharedSources := make([]bool, len(m.Layers))
	for i, prevIdx := range m.PreviousKVs {
		if i >= len(sharedSources) {
			break
		}
		if prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(sharedSources)) {
			sharedSources[prevIdx] = true
		}
	}
	for i, layer := range m.Layers {
		var prev sharedKV
		if prevIdx := m.PreviousKVs[i]; prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(intermediates)) {
			prev = intermediates[prevIdx]
		}

		var cache metal.Cache
		if m.PreviousKVs[i] == int32(i) && i < len(m.CacheIndexByLayer) {
			if cacheIdx := m.CacheIndexByLayer[i]; cacheIdx >= 0 && int(cacheIdx) < len(caches) {
				cache = caches[cacheIdx]
			}
		}

		layerMask := fullMask
		if layer.IsSliding {
			layerMask = slidingMask
		}

		var pli *metal.Array
		if len(perLayerInputs) > i {
			pli = perLayerInputs[i]
		}

		materializePagedKVForReuse := m.PreviousKVs[i] == int32(i) && sharedSources[i]
		nextH, kv := layer.forward(h, cache, B, L, layerMask, pli, prev, m.Cfg, nil, nil, materializePagedKVForReuse)
		metal.Free(h)
		h = nextH
		intermediates[i] = kv
	}
	defer func() {
		for i, kv := range intermediates {
			if m.PreviousKVs[i] != int32(i) {
				continue
			}
			metal.Free(kv.Keys, kv.Values)
		}
	}()

	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	return out
}

func (v *Gemma4VisionModel) Forward(pixelValues *metal.Array) *metal.Array {
	if v == nil || v.PatchEmbedder == nil {
		return nil
	}
	h, gridH, gridW := v.PatchEmbedder.Forward(pixelValues)
	if h == nil || !h.Valid() {
		return nil
	}

	encoded := v.Encoder.Forward(h, gridH, gridW)
	metal.Free(h)
	if v.PostLayernorm != nil && v.PostLayernorm.Weight != nil && v.PostLayernorm.Weight.Valid() {
		normed := metal.RMSNorm(encoded, v.PostLayernorm.Weight, v.Cfg.RMSNormEps)
		metal.Free(encoded)
		encoded = normed
	}
	pooled := v.Pooler.Forward(encoded, gridH, gridW)
	metal.Free(encoded)

	if v.Cfg.Standardize && v.StdBias != nil && v.StdScale != nil {
		centered := metal.Subtract(pooled, v.StdBias)
		metal.Free(pooled)
		pooled = metal.Mul(centered, v.StdScale)
		metal.Free(centered)
	}
	return pooled
}

func (p *Gemma4VisionPatchEmbedder) Forward(pixelValues *metal.Array) (*metal.Array, int32, int32) {
	patches, projected, gridH, gridW := p.prepare(pixelValues)
	if patches == nil || !patches.Valid() {
		return nil, 0, 0
	}

	hidden := patches
	if !projected {
		shifted := metal.AddScalar(patches, -0.5)
		scaled := metal.MulScalar(shifted, 2.0)
		metal.Free(shifted)
		if scaled != patches {
			metal.Free(patches)
		}
		hidden = p.InputProj.Forward(scaled)
		metal.Free(scaled)
	}

	if p.PositionEmbeddingTable != nil && p.PositionEmbeddingTable.Valid() {
		// hidden.Shape()[0] previously allocated; Dim(0) is one C call zero allocs.
		pos := p.positionEmbeddings(int32(hidden.Dim(0)), gridH, gridW)
		if pos != nil && pos.Valid() {
			next := metal.Add(hidden, pos)
			metal.Free(hidden, pos)
			hidden = next
		}
	}
	return hidden, gridH, gridW
}

func (p *Gemma4VisionPatchEmbedder) prepare(pixelValues *metal.Array) (*metal.Array, bool, int32, int32) {
	// Stack-allocated shape scratch — vision patch embed prepare; per-image
	// hot path. The Transpose(0,2,3,1) on the rank-4 branches is rank-4 by
	// case-construction, so Transpose4 applies.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := pixelValues.ShapeInto(shapeBuf[:0])
	channels := p.NumChannels
	if channels <= 0 {
		channels = 3
	}
	patchDim := channels * p.PatchSize * p.PatchSize
	switch len(shape) {
	case 2:
		gridH, gridW := gemma4VisionGridForPatchCount(shape[0], p.poolKernel())
		return metal.Reshape(pixelValues, 1, shape[0], shape[1]), false, gridH, gridW
	case 3:
		if shape[2] == patchDim {
			gridH, gridW := gemma4VisionGridForPatchCount(shape[1], p.poolKernel())
			return pixelValues.Clone(), false, gridH, gridW
		}
		if shape[2] == channels {
			expanded := metal.ExpandDims(pixelValues, 0)
			return p.prepareRawNHWC(expanded, true)
		}
		if shape[0] == channels {
			expanded := metal.ExpandDims(pixelValues, 0)
			transposed := metal.Transpose4(expanded, 0, 2, 3, 1)
			metal.Free(expanded)
			return p.prepareRawNHWC(transposed, true)
		}
	case 4:
		if shape[3] == channels {
			return p.prepareRawNHWC(pixelValues.Clone(), true)
		}
		if shape[1] == channels {
			transposed := metal.Transpose4(pixelValues, 0, 2, 3, 1)
			return p.prepareRawNHWC(transposed, true)
		}
	}
	return nil, false, 0, 0
}

func (p *Gemma4VisionPatchEmbedder) prepareRawNHWC(nhwc *metal.Array, owned bool) (*metal.Array, bool, int32, int32) {
	// Stack-allocated shape scratch — per-image patch-embed convolution
	// path. Both nhwc and conv are rank-4 NHWC tensors.
	var shapeBuf, convShapeBuf [metal.MaxTensorRank]int32
	shape := nhwc.ShapeInto(shapeBuf[:0])
	if len(shape) != 4 || p.PatchConvWeight == nil || !p.PatchConvWeight.Valid() {
		if owned {
			metal.Free(nhwc)
		}
		return nil, false, 0, 0
	}
	gridH := shape[1] / p.PatchSize
	gridW := shape[2] / p.PatchSize

	shifted := metal.AddScalar(nhwc, -0.5)
	scaled := metal.MulScalar(shifted, 2.0)
	metal.Free(shifted)
	if owned {
		metal.Free(nhwc)
	}

	conv := metal.Conv2d(scaled, p.PatchConvWeight, int(p.PatchSize), int(p.PatchSize), 0, 0, 1, 1, 1)
	metal.Free(scaled)
	convShape := conv.ShapeInto(convShapeBuf[:0])
	patches := metal.Reshape(conv, convShape[0], convShape[1]*convShape[2], convShape[3])
	metal.Free(conv)
	return patches, true, gridH, gridW
}

func (p *Gemma4VisionPatchEmbedder) poolKernel() int32 {
	if p == nil {
		return 1
	}
	if p.PoolingKernelSize <= 0 {
		return 1
	}
	return p.PoolingKernelSize
}

func (p *Gemma4VisionPatchEmbedder) positionEmbeddings(batch, gridH, gridW int32) *metal.Array {
	table := p.PositionEmbeddingTable
	// Stack-allocated shape scratch — per-vision-pass position embedding.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := table.ShapeInto(shapeBuf[:0])
	if len(shape) < 2 {
		return nil
	}

	count := int(batch * gridH * gridW)
	xIDs := make([]int32, count)
	yIDs := make([]int32, count)
	for b := range batch {
		base := int(b * gridH * gridW)
		for y := range gridH {
			for x := range gridW {
				idx := base + int(y*gridW+x)
				xIDs[idx] = x
				yIDs[idx] = y
			}
		}
	}
	xIdx := metal.FromValues(xIDs, int(batch), int(gridH*gridW))
	yIdx := metal.FromValues(yIDs, int(batch), int(gridH*gridW))
	defer metal.Free(xIdx, yIdx)

	if len(shape) == 3 && shape[0] >= 2 {
		xTableSlice := metal.SliceAxis(table, 0, 0, 1)
		xTable := metal.Squeeze(xTableSlice, 0)
		yTableSlice := metal.SliceAxis(table, 0, 1, 2)
		yTable := metal.Squeeze(yTableSlice, 0)
		xEmb := metal.Take(xTable, xIdx, 0)
		yEmb := metal.Take(yTable, yIdx, 0)
		pos := metal.Add(xEmb, yEmb)
		metal.Free(xTableSlice, xTable, yTableSlice, yTable, xEmb, yEmb)
		return pos
	}

	flatIDs := make([]int32, count)
	for i := range flatIDs {
		flatIDs[i] = int32(i) % (gridH * gridW)
	}
	flatIdx := metal.FromValues(flatIDs, int(batch), int(gridH*gridW))
	pos := metal.Take(table, flatIdx, 0)
	metal.Free(flatIdx)
	return pos
}

func (e *Gemma4VisionEncoder) Forward(x *metal.Array, grid ...int32) *metal.Array {
	gridH, gridW := int32(0), int32(0)
	if len(grid) >= 2 {
		gridH, gridW = grid[0], grid[1]
	}
	if (gridH <= 0 || gridW <= 0) && x != nil && x.NumDims() >= 2 {
		gridH, gridW = gemma4VisionGridForPatchCount(int32(x.Dim(1)), 1)
	}
	h := x
	cfg := e.Cfg
	if cfg == nil {
		cfg = normalizeGemma4VisionConfig(&Gemma4VisionConfig{})
	}
	for _, layer := range e.Layers {
		next := layer.Forward(h, gridH, gridW, cfg)
		if h != x {
			metal.Free(h)
		}
		h = next
	}
	return h
}

func (l *Gemma4VisionEncoderLayer) Forward(x *metal.Array, gridH, gridW int32, cfg *Gemma4VisionConfig) *metal.Array {
	residual := x
	normed := metal.RMSNorm(x, l.InputNorm.Weight, cfg.RMSNormEps)
	attnOut := l.Attention.Forward(normed, gridH, gridW, cfg)
	metal.Free(normed)
	attnNormed := metal.RMSNorm(attnOut, l.PostAttnNorm.Weight, cfg.RMSNormEps)
	metal.Free(attnOut)
	h := metal.Add(residual, attnNormed)
	metal.Free(attnNormed)

	residual = h
	ffIn := metal.RMSNorm(h, l.PreFFNorm.Weight, cfg.RMSNormEps)
	ff := l.MLP.Forward(ffIn)
	metal.Free(ffIn)
	ffNormed := metal.RMSNorm(ff, l.PostFFNorm.Weight, cfg.RMSNormEps)
	metal.Free(ff)
	out := metal.Add(residual, ffNormed)
	metal.Free(h, ffNormed)
	return out
}

func (a *Gemma4VisionAttention) Forward(x *metal.Array, gridH, gridW int32, cfg *Gemma4VisionConfig) *metal.Array {
	// Stack-allocated shape scratch — per-vision-attention-layer hot path.
	// All rank-4 Transposes on the V and out paths use the scalar-pass
	// Transpose4 (axes 0,2,1,3 — rank-4 by construction).
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	qProj := a.QProj.Forward(x)
	q := metal.Reshape(qProj, B, L, a.NHeads, a.HeadDim)
	metal.Free(qProj)
	qNorm := metal.RMSNorm(q, a.QNorm.Weight, cfg.RMSNormEps)
	metal.Free(q)
	q = gemma4VisionRoPEAndTranspose(qNorm, gridH, gridW, a.RopeBase, a.HeadDim)
	metal.Free(qNorm)

	kProj := a.KProj.Forward(x)
	k := metal.Reshape(kProj, B, L, a.NKVHeads, a.HeadDim)
	metal.Free(kProj)
	kNorm := metal.RMSNorm(k, a.KNorm.Weight, cfg.RMSNormEps)
	metal.Free(k)
	k = gemma4VisionRoPEAndTranspose(kNorm, gridH, gridW, a.RopeBase, a.HeadDim)
	metal.Free(kNorm)

	vProj := a.VProj.Forward(x)
	v := metal.Reshape(vProj, B, L, a.NKVHeads, a.HeadDim)
	metal.Free(vProj)
	vNorm := metal.RMSNormNoScale(v, cfg.RMSNormEps)
	metal.Free(v)
	v = metal.Transpose4(vNorm, 0, 2, 1, 3)
	metal.Free(vNorm)

	repeatFactor := a.NHeads / a.NKVHeads
	kAttn, vAttn := k, v
	repeated := false
	if repeatFactor > 1 {
		kAttn = metal.RepeatKV(k, repeatFactor)
		vAttn = metal.RepeatKV(v, repeatFactor)
		repeated = true
	}

	out := metal.ScaledDotProductAttention(q, kAttn, vAttn, a.Attention, false)
	metal.Free(q, k, v)
	if repeated {
		metal.Free(kAttn, vAttn)
	}

	transposed := metal.Transpose4(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, a.NHeads*a.HeadDim)
	metal.Free(transposed)
	result := a.OProj.Forward(reshaped)
	metal.Free(reshaped)
	return result
}

func gemma4VisionRoPEAndTranspose(x *metal.Array, gridH, gridW int32, base float32, headDim int32) *metal.Array {
	// Rank-4 transposes (axes 0,2,1,3) — substrate Transpose4 form.
	if rotated := gemma4VisionApply2DRoPE(x, gridH, gridW, base); rotated != nil {
		transposed := metal.Transpose4(rotated, 0, 2, 1, 3)
		metal.Free(rotated)
		return transposed
	}
	transposed := metal.Transpose4(x, 0, 2, 1, 3)
	out := metal.RoPE(transposed, int(headDim), false, base, 1.0, 0)
	metal.Free(transposed)
	return out
}

func gemma4VisionApply2DRoPE(x *metal.Array, gridH, gridW int32, base float32) *metal.Array {
	// Stack-allocated shape scratch — per-vision-layer 2D RoPE; rank-4 by
	// guard. The three rank-4 Slice calls use the scalar-pass Slice4 form.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	if len(shape) != 4 || base == 0 {
		return nil
	}
	B, L, N, D := shape[0], shape[1], shape[2], shape[3]
	if D < 4 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || gridH*gridW != L {
		gridH, gridW = gemma4VisionGridForPatchCount(L, 1)
	}
	if gridH <= 0 || gridW <= 0 || gridH*gridW != L {
		return nil
	}

	rotatedPerDim := 2 * (D / 4)
	if rotatedPerDim <= 0 || rotatedPerDim%2 != 0 {
		return nil
	}
	rotatedTotal := rotatedPerDim * 2

	cosX, sinX, cosY, sinY := gemma4Vision2DRoPETables(B, L, gridH, gridW, rotatedPerDim, base)
	defer metal.Free(cosX, sinX, cosY, sinY)

	xPart := metal.Slice4(x, 0, 0, 0, 0, B, L, N, rotatedPerDim)
	yPart := metal.Slice4(x, 0, 0, 0, rotatedPerDim, B, L, N, rotatedTotal)
	xRot := gemma4VisionRotatePart(xPart, cosX, sinX)
	yRot := gemma4VisionRotatePart(yPart, cosY, sinY)
	metal.Free(xPart, yPart)

	parts := []*metal.Array{xRot, yRot}
	if rotatedTotal < D {
		rest := metal.Slice4(x, 0, 0, 0, rotatedTotal, B, L, N, D)
		parts = append(parts, rest)
	}
	out := metal.Concatenate(parts, 3)
	metal.Free(parts...)
	return out
}

func gemma4Vision2DRoPETables(batch, seqLen, gridH, gridW, dim int32, base float32) (*metal.Array, *metal.Array, *metal.Array, *metal.Array) {
	freqCount := dim / 2
	invFreq := make([]float64, int(freqCount))
	for i := range freqCount {
		invFreq[int(i)] = 1.0 / math.Pow(float64(base), float64(2*i)/float64(dim))
	}

	size := int(batch * seqLen * dim)
	cosX := make([]float32, size)
	sinX := make([]float32, size)
	cosY := make([]float32, size)
	sinY := make([]float32, size)
	for b := range batch {
		for pos := range seqLen {
			x := float64(pos % gridW)
			y := float64(pos / gridW)
			baseIdx := int((b*seqLen + pos) * dim)
			for d := range dim {
				freq := invFreq[int(d%freqCount)]
				cx := x * freq
				cy := y * freq
				idx := baseIdx + int(d)
				cosX[idx] = float32(math.Cos(cx))
				sinX[idx] = float32(math.Sin(cx))
				cosY[idx] = float32(math.Cos(cy))
				sinY[idx] = float32(math.Sin(cy))
			}
		}
	}

	shape := []int{int(batch), int(seqLen), 1, int(dim)}
	return metal.FromValues(cosX, shape...), metal.FromValues(sinX, shape...), metal.FromValues(cosY, shape...), metal.FromValues(sinY, shape...)
}

func gemma4VisionRotatePart(x, cos, sin *metal.Array) *metal.Array {
	// Stack-allocated shape scratch — per-vision-layer rotate half;
	// x is always rank-4 [B,L,N,D] by caller (gemma4VisionApply2DRoPE).
	var shapeBuf [metal.MaxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	D := shape[3]
	half := D / 2
	first := metal.Slice4(x, 0, 0, 0, 0, shape[0], shape[1], shape[2], half)
	second := metal.Slice4(x, 0, 0, 0, half, shape[0], shape[1], shape[2], D)
	negativeSecond := metal.Negative(second)
	rotated := metal.Concatenate2(negativeSecond, first, 3)
	scaled := metal.Mul(x, cos)
	rotatedScaled := metal.Mul(rotated, sin)
	out := metal.Add(scaled, rotatedScaled)
	metal.Free(first, second, negativeSecond, rotated, scaled, rotatedScaled)
	return out
}

func (m *Gemma4VisionMLP) Forward(x *metal.Array) *metal.Array {
	gate := m.GateProj.Forward(x)
	activated := metal.GeluActivation(gate)
	metal.Free(gate)
	var hidden *metal.Array
	if m.UpProj != nil {
		up := m.UpProj.Forward(x)
		hidden = metal.Mul(activated, up)
		metal.Free(activated, up)
	} else {
		hidden = activated
	}
	out := m.DownProj.Forward(hidden)
	metal.Free(hidden)
	return out
}

func (p *Gemma4VisionPooler) Forward(hidden *metal.Array, gridH, gridW int32) *metal.Array {
	// Stack-allocated shape scratch — per-vision-pass pooler entrypoint.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := hidden.ShapeInto(shapeBuf[:0])
	B, L, H := shape[0], shape[1], shape[2]
	k := p.PoolingKernelSize
	var pooled *metal.Array

	if k > 1 && gridH > 0 && gridW > 0 && gridH%k == 0 && gridW%k == 0 && gridH*gridW == L {
		pooled = p.poolByGrid(hidden, B, gridH, gridW, H, k)
	} else if k > 1 && L%(k*k) == 0 {
		outLen := L / (k * k)
		grouped := metal.Reshape(hidden, B, outLen, k*k, H)
		mean := metal.Mean(grouped, 2, false)
		metal.Free(grouped)
		pooled = metal.Reshape(mean, B*outLen, H)
		metal.Free(mean)
	} else {
		pooled = metal.Reshape(hidden, B*L, H)
	}

	scaled := metal.MulScalar(pooled, p.EmbeddingScale)
	metal.Free(pooled)
	return scaled
}

func (p *Gemma4VisionPooler) poolByGrid(hidden *metal.Array, B, gridH, gridW, H, k int32) *metal.Array {
	rows := gridH / k
	cols := gridW / k
	groups := make([]*metal.Array, 0, rows*cols)
	for y := range rows {
		for x := range cols {
			indices := make([]int32, 0, k*k)
			for dy := range k {
				for dx := range k {
					indices = append(indices, (y*k+dy)*gridW+(x*k+dx))
				}
			}
			idx := metal.FromValues(indices, len(indices))
			patches := metal.Take(hidden, idx, 1)
			mean := metal.Mean(patches, 1, false)
			expanded := metal.ExpandDims(mean, 1)
			metal.Free(idx, patches, mean)
			groups = append(groups, expanded)
		}
	}
	combined := metal.Concatenate(groups, 1)
	metal.Free(groups...)
	flat := metal.Reshape(combined, B*rows*cols, H)
	metal.Free(combined)
	return flat
}

func (p *Gemma4MultiModalProjector) Forward(x *metal.Array) *metal.Array {
	if p == nil {
		return x.Clone()
	}
	normed := metal.RMSNormNoScale(x, p.Eps)
	if p.Projection != nil {
		out := p.Projection.Forward(normed)
		metal.Free(normed)
		return out
	}
	if p.Linear1 != nil && p.Linear2 != nil {
		hidden := p.Linear1.Forward(normed)
		activated := metal.GeluActivation(hidden)
		metal.Free(hidden, normed)
		out := p.Linear2.Forward(activated)
		metal.Free(activated)
		return out
	}
	return normed
}
