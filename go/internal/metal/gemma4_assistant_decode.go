// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	core "dappco.re/go"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. Speculative-decoding validation fires per-draft-step
// (MTP draft block + verify) which runs many times per generation.
var (
	errTargetPagedNoVisible          = core.NewError("target paged cache has no visible pages")
	errTargetCacheTooShort           = core.NewError("target cache state shorter than visible length")
	errTargetCacheStateEmpty         = core.NewError("target cache state is empty")
	errTargetCacheLenEmpty           = core.NewError("target cache length is empty")
	errTargetCacheNil                = core.NewError("target cache is nil")
	errTargetCacheEmpty              = core.NewError("target cache is empty")
	errRotatingCacheEmpty            = core.NewError("rotating cache state is empty")
	errKVCacheStateEmpty             = core.NewError("KV cache state is empty")
	errAsstVerifyNeedTargetLogits    = core.NewError("gemma4.assistant verify requires target logits")
	errAsstVerifyNeedTargetCaches    = core.NewError("gemma4.assistant verify requires target caches")
	errAsstVerifyNeedDraftTokens     = core.NewError("gemma4.assistant verify requires draft tokens")
	errAsstVerifyNeedTargetModel     = core.NewError("gemma4.assistant verify requires a target model")
	errAsstVerifyNoTargetToken       = core.NewError("gemma4.assistant verify produced no target token")
	errAsstOrderedEmbedNotImpl       = core.NewError("gemma4.assistant ordered embedding logits are not implemented yet")
	errAsstDraftStepTokenInvalid     = core.NewError("gemma4.assistant draft step token is invalid")
	errAsstDraftStepNeedTargetCaches = core.NewError("gemma4.assistant draft step requires populated target caches")
	errAsstDraftStepNeedPair         = core.NewError("gemma4.assistant draft step requires a validated pair")
	errAsstDraftStepHiddenInvalid    = core.NewError("gemma4.assistant draft step previous hidden is invalid")
	errAsstDraftStepLayerIncomplete  = core.NewError("gemma4.assistant draft step layer is incomplete")
	errAsstDraftBlockNoToken         = core.NewError("gemma4.assistant draft block produced no token")
	errAsstDraftBlockMaxZero         = core.NewError("gemma4.assistant draft block maxDraftTokens must be > 0")
	errAsstCloneInvalid              = core.NewError("gemma4.assistant cannot clone invalid array")
	errAsstAttnMissingKV             = core.NewError("gemma4.assistant attention missing target K/V")
	errAsstAttnIncomplete            = core.NewError("gemma4.assistant attention is incomplete")
	errCacheStateEmpty               = core.NewError("cache state is empty")
)

// Gemma4AssistantDraftStepResult is the caller-owned output of one MTP draft
// step. Hidden is projected back to the target backbone hidden size so it can
// seed the next assistant step.
type Gemma4AssistantDraftStepResult struct {
	Logits *Array
	Token  *Array
	Hidden *Array
}

// Gemma4AssistantDraftBlockResult is the caller-owned output of chained MTP
// assistant proposals. Hidden is the final projected backbone hidden state.
type Gemma4AssistantDraftBlockResult struct {
	Tokens []int32
	Hidden *Array
}

// Gemma4AssistantVerifyResult reports target-side verification of a proposed
// assistant draft block. Caches, Logits, and Hidden are caller-owned.
type Gemma4AssistantVerifyResult struct {
	DraftedTokens    []int32
	TargetTokens     []int32
	AcceptedTokens   []int32
	RejectedTokens   []int32
	AcceptedCount    int
	RejectedCount    int
	ReplacementToken int32
	AllAccepted      bool
	Caches           []Cache
	Logits           *Array
	Hidden           *Array
}

// Close releases arrays returned by DraftStep.
func (result *Gemma4AssistantDraftStepResult) Close() {
	if result == nil {
		return
	}
	Free(result.Logits, result.Token, result.Hidden)
	result.Logits = nil
	result.Token = nil
	result.Hidden = nil
}

// Close releases arrays returned by DraftBlock.
func (result *Gemma4AssistantDraftBlockResult) Close() {
	if result == nil {
		return
	}
	Free(result.Hidden)
	result.Hidden = nil
	result.Tokens = nil
}

// Close releases arrays and caches returned by VerifyDraftBlock.
func (result *Gemma4AssistantVerifyResult) Close() {
	if result == nil {
		return
	}
	freeCaches(result.Caches)
	Free(result.Logits, result.Hidden)
	result.Caches = nil
	result.Logits = nil
	result.Hidden = nil
	result.DraftedTokens = nil
	result.TargetTokens = nil
	result.AcceptedTokens = nil
	result.RejectedTokens = nil
}

type gemma4AssistantTargetKV struct {
	kv    sharedKV
	owned []*Array
}

func (targetKV gemma4AssistantTargetKV) free() {
	Free(targetKV.owned...)
}

// DraftStep proposes one token from the assistant using the target model's
// existing K/V cache streams and the previous target-backbone hidden state.
func (pair *Gemma4AssistantPair) DraftStep(lastToken int32, previousHidden *Array, targetCaches []Cache) (*Gemma4AssistantDraftStepResult, error) {
	if pair == nil || pair.Target == nil || pair.Assistant == nil {
		return nil, errAsstDraftStepNeedPair
	}
	if lastToken < 0 {
		return nil, errAsstDraftStepTokenInvalid
	}
	if previousHidden == nil || !previousHidden.Valid() {
		return nil, errAsstDraftStepHiddenInvalid
	}
	if len(targetCaches) == 0 {
		return nil, errAsstDraftStepNeedTargetCaches
	}
	if pair.Assistant.UseOrderedEmbeddings {
		return nil, errAsstOrderedEmbedNotImpl
	}
	if err := validateGemma4AssistantPair(pair.Target, pair.Assistant); err != nil {
		return nil, err
	}

	targetKVs, err := pair.targetKVByLayerType(targetCaches)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, targetKV := range targetKVs {
			targetKV.free()
		}
	}()

	tokenValue := fromSingleInt32(lastToken)
	tokenInput := Reshape(tokenValue, 1, 1)
	tokenEmbedding := pair.Target.EmbedTokens.Forward(tokenInput)
	scaledTokenEmbedding := MulScalar(tokenEmbedding, pair.Target.Cfg.EmbeddingScale)
	Free(tokenValue, tokenInput, tokenEmbedding)

	backboneHidden, ownBackboneHidden, err := gemma4AssistantBackboneHidden(previousHidden, pair.Assistant.BackboneHiddenSize)
	if err != nil {
		Free(scaledTokenEmbedding)
		return nil, err
	}
	combined := concatenate2(scaledTokenEmbedding, backboneHidden, 2)
	Free(scaledTokenEmbedding)
	if ownBackboneHidden {
		Free(backboneHidden)
	}

	h := pair.Assistant.PreProjection.Forward(combined)
	Free(combined)
	for _, layer := range pair.Assistant.Layers {
		targetKV, ok := targetKVs[layer.LayerType]
		if !ok || !targetKV.kv.hasState() {
			Free(h)
			return nil, core.NewError("gemma4.assistant draft step missing target K/V stream for " + layer.LayerType)
		}
		next, err := layer.forwardDraftStep(h, targetKV.kv, pair.Assistant.Cfg)
		Free(h)
		if err != nil {
			return nil, err
		}
		h = next
	}

	normed := pair.Assistant.Norm.Forward(h, pair.Assistant.Cfg.RMSNormEps)
	Free(h)
	hidden := pair.Assistant.PostProjection.Forward(normed)
	logits := pair.Assistant.EmbedTokens.AsLinear().Forward(normed)
	Free(normed)
	if pair.Assistant.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(logits, pair.Assistant.Cfg.FinalLogitSoftcapping)
		Free(logits)
		logits = softcapped
	}
	token := Argmax(logits, -1, false)
	return &Gemma4AssistantDraftStepResult{Logits: logits, Token: token, Hidden: hidden}, nil
}

// DraftBlock chains assistant MTP steps and returns a CPU-visible draft token
// block. Verification still belongs to the target-side accept/reject path.
func (pair *Gemma4AssistantPair) DraftBlock(lastToken int32, previousHidden *Array, targetCaches []Cache, maxDraftTokens int) (*Gemma4AssistantDraftBlockResult, error) {
	if maxDraftTokens <= 0 {
		return nil, errAsstDraftBlockMaxZero
	}
	tokens := make([]int32, 0, maxDraftTokens)
	currentToken := lastToken
	currentHidden := previousHidden
	ownsCurrentHidden := false
	for len(tokens) < maxDraftTokens {
		step, err := pair.DraftStep(currentToken, currentHidden, targetCaches)
		if ownsCurrentHidden {
			Free(currentHidden)
			currentHidden = nil
			ownsCurrentHidden = false
		}
		if err != nil {
			return nil, err
		}
		if err := Eval(step.Token, step.Hidden); err != nil {
			step.Close()
			return nil, core.E("gemma4.assistant draft block", "eval draft step", err)
		}
		values := step.Token.DataInt32()
		if len(values) == 0 {
			step.Close()
			return nil, errAsstDraftBlockNoToken
		}
		currentToken = values[0]
		tokens = append(tokens, currentToken)
		currentHidden = step.Hidden
		step.Hidden = nil
		ownsCurrentHidden = true
		step.Close()
	}
	return &Gemma4AssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden}, nil
}

// VerifyDraftBlock compares an assistant draft block against greedy target
// predictions. The caller's target caches are cloned before verification, so
// rejected draft tokens never pollute the live generation cache.
func (pair *Gemma4AssistantPair) VerifyDraftBlock(targetLogits *Array, draftTokens []int32, targetCaches []Cache) (*Gemma4AssistantVerifyResult, error) {
	if pair == nil || pair.Target == nil {
		return nil, errAsstVerifyNeedTargetModel
	}
	if targetLogits == nil || !targetLogits.Valid() {
		return nil, errAsstVerifyNeedTargetLogits
	}
	if len(draftTokens) == 0 {
		return nil, errAsstVerifyNeedDraftTokens
	}
	if len(targetCaches) == 0 {
		return nil, errAsstVerifyNeedTargetCaches
	}
	verifyCaches, err := cloneGemma4AssistantVerifyCaches(targetCaches)
	if err != nil {
		return nil, err
	}

	result := &Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), draftTokens...),
		Caches:        verifyCaches,
	}
	currentLogits := targetLogits
	currentLogitsOwned := false
	var currentHidden *Array
	currentHiddenOwned := false

	for idx, draftToken := range draftTokens {
		targetToken, err := gemma4AssistantGreedyToken(currentLogits)
		if err != nil {
			result.Close()
			if currentLogitsOwned {
				Free(currentLogits)
			}
			if currentHiddenOwned {
				Free(currentHidden)
			}
			return nil, err
		}
		result.TargetTokens = append(result.TargetTokens, targetToken)
		if targetToken != draftToken {
			result.AcceptedCount = len(result.AcceptedTokens)
			result.RejectedCount = len(draftTokens) - idx
			result.RejectedTokens = append([]int32(nil), draftTokens[idx:]...)
			result.ReplacementToken = targetToken
			if currentLogitsOwned {
				result.Logits = currentLogits
				currentLogitsOwned = false
			} else {
				result.Logits, err = cloneGemma4AssistantArray(currentLogits)
				if err != nil {
					result.Close()
					if currentHiddenOwned {
						Free(currentHidden)
					}
					return nil, err
				}
			}
			if currentHiddenOwned {
				result.Hidden = currentHidden
				currentHiddenOwned = false
			}
			return result, nil
		}

		result.AcceptedTokens = append(result.AcceptedTokens, draftToken)
		tokenArray := fromSingleInt32(draftToken)
		tokenInput := Reshape(tokenArray, 1, 1)
		nextLogits, nextHidden := pair.Target.ForwardLastTokenLogitsAndHidden(tokenInput, nil, verifyCaches)
		Free(tokenArray, tokenInput)
		if err := Eval(nextLogits, nextHidden); err != nil {
			result.Close()
			Free(nextLogits, nextHidden)
			if currentLogitsOwned {
				Free(currentLogits)
			}
			if currentHiddenOwned {
				Free(currentHidden)
			}
			return nil, core.E("gemma4.assistant verify", "target accepted token", err)
		}
		detachCaches(verifyCaches)
		if currentLogitsOwned {
			Free(currentLogits)
		}
		if currentHiddenOwned {
			Free(currentHidden)
		}
		currentLogits = nextLogits
		currentLogitsOwned = true
		currentHidden = nextHidden
		currentHiddenOwned = true
	}

	result.AcceptedCount = len(result.AcceptedTokens)
	result.AllAccepted = true
	if currentLogitsOwned {
		result.Logits = currentLogits
		currentLogitsOwned = false
	} else {
		result.Logits, err = cloneGemma4AssistantArray(currentLogits)
		if err != nil {
			result.Close()
			if currentHiddenOwned {
				Free(currentHidden)
			}
			return nil, err
		}
	}
	if currentHiddenOwned {
		result.Hidden = currentHidden
		currentHiddenOwned = false
	}
	return result, nil
}

func (pair *Gemma4AssistantPair) targetKVByLayerType(caches []Cache) (map[string]gemma4AssistantTargetKV, error) {
	pair.Target.ensureCacheLayout()
	out := make(map[string]gemma4AssistantTargetKV)
	for layerIdx, layer := range pair.Target.Layers {
		if layer == nil || layer.LayerType == "" {
			continue
		}
		ownerIdx := layerIdx
		if layerIdx < len(pair.Target.PreviousKVs) && pair.Target.PreviousKVs[layerIdx] >= 0 {
			ownerIdx = int(pair.Target.PreviousKVs[layerIdx])
		}
		if ownerIdx >= len(pair.Target.CacheIndexByLayer) {
			continue
		}
		cacheIdx := pair.Target.CacheIndexByLayer[ownerIdx]
		if cacheIdx < 0 || int(cacheIdx) >= len(caches) {
			continue
		}
		targetKV, err := gemma4AssistantKVFromCache(caches[cacheIdx])
		if err != nil {
			for _, existing := range out {
				existing.free()
			}
			return nil, core.E("gemma4.assistant draft step", core.Sprintf("target layer %d", layerIdx), err)
		}
		if previous, ok := out[layer.LayerType]; ok {
			previous.free()
		}
		out[layer.LayerType] = targetKV
	}
	for _, layer := range pair.Assistant.Layers {
		if layer == nil {
			continue
		}
		targetKV, ok := out[layer.LayerType]
		if !ok || !targetKV.kv.hasState() {
			for _, existing := range out {
				existing.free()
			}
			return nil, core.NewError("gemma4.assistant draft step missing populated target K/V stream for " + layer.LayerType)
		}
	}
	return out, nil
}

func gemma4AssistantKVFromCache(cache Cache) (gemma4AssistantTargetKV, error) {
	if cache == nil || cache.Len() <= 0 {
		return gemma4AssistantTargetKV{}, errTargetCacheEmpty
	}
	if paged, ok := cache.(*PagedKVCache); ok {
		pages := paged.PageState()
		if pages.Length <= 0 || len(pages.Keys) == 0 || len(pages.Keys) != len(pages.Values) {
			pages.Free()
			return gemma4AssistantTargetKV{}, errTargetPagedNoVisible
		}
		return gemma4AssistantTargetKV{
			kv:    sharedKV{Pages: pages, Offset: cache.Offset()},
			owned: pages.Owned,
		}, nil
	}

	state, owned := cacheReadState(cache)
	if len(state) < 2 || state[0] == nil || state[1] == nil || !state[0].Valid() || !state[1].Valid() {
		Free(owned...)
		return gemma4AssistantTargetKV{}, errTargetCacheStateEmpty
	}
	keys, values := state[0], state[1]
	visible := int32(cache.Len())
	if visible <= 0 {
		Free(owned...)
		return gemma4AssistantTargetKV{}, errTargetCacheLenEmpty
	}
	// Stack-allocated shape scratch — assistant verify cache trim is called
	// per draft step. Both Slice calls are rank-4 by guard (len ≥ 4).
	var kShapeBuf, vShapeBuf [maxTensorRank]int32
	kShape := keys.ShapeInto(kShapeBuf[:0])
	vShape := values.ShapeInto(vShapeBuf[:0])
	if len(kShape) >= 4 && len(vShape) >= 4 {
		if kShape[2] < visible || vShape[2] < visible {
			Free(owned...)
			return gemma4AssistantTargetKV{}, errTargetCacheTooShort
		}
		if kShape[2] != visible {
			keys = Slice4(keys, 0, 0, 0, 0, kShape[0], kShape[1], visible, kShape[3])
			owned = append(owned, keys)
		}
		if vShape[2] != visible {
			values = Slice4(values, 0, 0, 0, 0, vShape[0], vShape[1], visible, vShape[3])
			owned = append(owned, values)
		}
	}
	return gemma4AssistantTargetKV{
		kv:    sharedKV{Keys: keys, Values: values, Offset: cache.Offset()},
		owned: owned,
	}, nil
}

func cloneGemma4AssistantVerifyCaches(caches []Cache) ([]Cache, error) {
	cloned := make([]Cache, len(caches))
	for i, cache := range caches {
		next, err := cloneGemma4AssistantVerifyCache(cache)
		if err != nil {
			freeCaches(cloned)
			return nil, core.E("gemma4.assistant verify", core.Sprintf("clone cache %d", i), err)
		}
		cloned[i] = next
	}
	return cloned, nil
}

func cloneGemma4AssistantVerifyCache(cache Cache) (Cache, error) {
	if cache == nil {
		return nil, errTargetCacheNil
	}
	if cache.Len() <= 0 {
		switch c := cache.(type) {
		case *RotatingKVCache:
			return NewRotatingKVCache(c.maxSize), nil
		case *FixedKVCache:
			return NewFixedKVCache(c.maxSize), nil
		case *PagedKVCache:
			return NewPagedKVCache(c.maxSize, c.pageSize), nil
		case *QuantizedKVCache:
			return NewQuantizedKVCache(c.maxSize, c.keyBits, c.valueBits), nil
		default:
			return NewKVCache(), nil
		}
	}
	switch c := cache.(type) {
	case *KVCache:
		state, owned := cacheReadState(c)
		defer Free(owned...)
		if len(state) < 2 {
			return nil, errKVCacheStateEmpty
		}
		keys, values, err := cloneGemma4AssistantCacheState(state[0], state[1], c.Len())
		if err != nil {
			return nil, err
		}
		return &KVCache{keys: keys, values: values, offset: c.offset, step: c.step}, nil
	case *RotatingKVCache:
		state, owned := cacheReadState(c)
		defer Free(owned...)
		if len(state) < 2 {
			return nil, errRotatingCacheEmpty
		}
		keys, values, err := cloneGemma4AssistantCacheState(state[0], state[1], c.Len())
		if err != nil {
			return nil, err
		}
		return &RotatingKVCache{keys: keys, values: values, offset: c.offset, maxSize: c.maxSize, step: c.step, idx: c.Len()}, nil
	case *FixedKVCache:
		state := c.FixedState()
		if state.Keys == nil || state.Values == nil {
			state.Free()
			return NewFixedKVCache(c.maxSize), nil
		}
		return &FixedKVCache{keys: state.Keys, values: state.Values, offset: c.offset, length: c.length, maxSize: c.maxSize}, nil
	case *PagedKVCache:
		pages := c.PageState()
		defer pages.Free()
		kPages, vPages, err := copyPagedCachePrefix(pages.Keys, pages.Values, c.Len())
		if err != nil {
			return nil, err
		}
		return &PagedKVCache{kPages: kPages, vPages: vPages, pageLens: pagedPageLensForPages(kPages, c.length), offset: c.offset, length: c.length, maxSize: c.maxSize, pageSize: c.pageSize}, nil
	case *QuantizedKVCache:
		return &QuantizedKVCache{
			keys:       Copy(c.keys),
			values:     Copy(c.values),
			keyScale:   Copy(c.keyScale),
			valueScale: Copy(c.valueScale),
			keyDtype:   c.keyDtype,
			valueDtype: c.valueDtype,
			keyShape:   append([]int32(nil), c.keyShape...),
			valueShape: append([]int32(nil), c.valueShape...),
			offset:     c.offset,
			maxSize:    c.maxSize,
			step:       c.step,
			keyBits:    c.keyBits,
			valueBits:  c.valueBits,
		}, nil
	default:
		state, owned := cacheReadState(cache)
		defer Free(owned...)
		if len(state) < 2 {
			return nil, errCacheStateEmpty
		}
		keys, values, err := cloneGemma4AssistantCacheState(state[0], state[1], cache.Len())
		if err != nil {
			return nil, err
		}
		return &KVCache{keys: keys, values: values, offset: cache.Offset(), step: 256}, nil
	}
}

func cloneGemma4AssistantCacheState(keys, values *Array, tokenLen int) (*Array, *Array, error) {
	keyCopy, err := copyCachePrefix(keys, tokenLen)
	if err != nil {
		return nil, nil, err
	}
	valueCopy, err := copyCachePrefix(values, tokenLen)
	if err != nil {
		Free(keyCopy)
		return nil, nil, err
	}
	return keyCopy, valueCopy, nil
}

func gemma4AssistantGreedyToken(logits *Array) (int32, error) {
	token := Argmax(logits, -1, false)
	defer Free(token)
	if err := Eval(token); err != nil {
		return 0, err
	}
	values := token.DataInt32()
	if len(values) == 0 {
		return 0, errAsstVerifyNoTargetToken
	}
	return values[0], nil
}

func cloneGemma4AssistantArray(array *Array) (*Array, error) {
	if array == nil || !array.Valid() {
		return nil, errAsstCloneInvalid
	}
	cloned := Copy(array)
	if err := Eval(cloned); err != nil {
		Free(cloned)
		return nil, err
	}
	Detach(cloned)
	return cloned, nil
}

func gemma4AssistantBackboneHidden(hidden *Array, backboneHidden int32) (*Array, bool, error) {
	// Stack-allocated shape scratch — per-assistant-draft-step path.
	var shapeBuf [maxTensorRank]int32
	shape := hidden.ShapeInto(shapeBuf[:0])
	switch {
	case len(shape) == 3 && shape[0] == 1 && shape[1] == 1 && shape[2] == backboneHidden:
		return hidden, false, nil
	case len(shape) == 2 && shape[0] == 1 && shape[1] == backboneHidden:
		return Reshape(hidden, 1, 1, backboneHidden), true, nil
	case len(shape) == 1 && shape[0] == backboneHidden:
		return Reshape(hidden, 1, 1, backboneHidden), true, nil
	default:
		return nil, false, core.NewError(core.Sprintf("gemma4.assistant previous hidden shape = %v, want [1 1 %d]", shape, backboneHidden))
	}
}

func (layer *Gemma4AssistantLayer) forwardDraftStep(x *Array, targetKV sharedKV, cfg *Gemma4TextConfig) (*Array, error) {
	if layer == nil || layer.Attention == nil || layer.MLP == nil {
		return nil, errAsstDraftStepLayerIncomplete
	}
	// Stack-allocated shape scratch — per-assistant-draft-step per-layer
	// hot path. Avoids the per-call []int32 heap alloc.
	var shapeBuf [maxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	if len(shape) != 3 {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft step layer input shape = %v, want [batch sequence hidden]", shape))
	}
	B, L := shape[0], shape[1]
	if B != 1 || L != 1 {
		return nil, core.NewError(core.Sprintf("gemma4.assistant draft step only supports [1 1 hidden], got %v", shape))
	}

	normed := layer.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut, err := layer.Attention.forwardWithTargetKV(normed, targetKV, B, L, cfg)
	Free(normed)
	if err != nil {
		return nil, err
	}
	attnNormed := layer.PostAttnNorm.Forward(attnOut, cfg.RMSNormEps)
	Free(attnOut)
	h := Add(x, attnNormed)
	Free(attnNormed)

	ffIn := layer.PreFFNorm.Forward(h, cfg.RMSNormEps)
	ff := layer.MLP.forward(ffIn)
	Free(ffIn)
	ffResidual := layer.PostFFNorm.Forward(ff, cfg.RMSNormEps)
	Free(ff)

	hNext := Add(h, ffResidual)
	Free(h, ffResidual)
	if layer.LayerScalar != nil && layer.LayerScalar.Valid() {
		scaled := Mul(hNext, layer.LayerScalar)
		Free(hNext)
		hNext = scaled
	}
	return hNext, nil
}

func (attn *Gemma4AssistantAttention) forwardWithTargetKV(x *Array, targetKV sharedKV, B, L int32, cfg *Gemma4TextConfig) (*Array, error) {
	if attn == nil || attn.QProj == nil || attn.OProj == nil || attn.QNorm == nil {
		return nil, errAsstAttnIncomplete
	}
	if !targetKV.hasState() {
		return nil, errAsstAttnMissingKV
	}

	qProj := attn.QProj.Forward(x)
	q := AsStrided(qProj, []int32{B, attn.NHeads, L, attn.HeadDim},
		[]int64{int64(L * attn.NHeads * attn.HeadDim), int64(attn.HeadDim), int64(attn.NHeads * attn.HeadDim), 1}, 0)
	Free(qProj)
	oldQ := q
	q = attn.QNorm.Forward(q, cfg.RMSNormEps)
	Free(oldQ)
	qRoPE := attn.applyRoPE(q, targetKV.Offset)
	Free(q)
	q = qRoPE

	var out *Array
	if targetKV.hasPages() {
		keyHeads := int32(0)
		if len(targetKV.Pages.Keys) > 0 && targetKV.Pages.Keys[0] != nil && targetKV.Pages.Keys[0].Valid() {
			keyHeads = int32(targetKV.Pages.Keys[0].Dim(1))
		}
		kPages, vPages := targetKV.Pages.Keys, targetKV.Pages.Values
		var repeated []*Array
		if keyHeads > 0 && attn.NHeads > keyHeads && attn.NHeads%keyHeads == 0 && len(kPages) > 1 && pagedStateNeedsMaterializedRepeat(targetKV.Pages, attn.NHeads/keyHeads) {
			kPages, vPages, repeated = repeatPagedState(targetKV.Pages, attn.NHeads/keyHeads)
		}
		out = ScaledDotProductAttentionPaged(q, kPages, vPages, attn.Scale)
		Free(repeated...)
	} else {
		out = ScaledDotProductAttention(q, targetKV.Keys, targetKV.Values, attn.Scale, false)
	}
	Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — scalar-pass
	// Transpose4 form (eliminates the []int axes heap alloc).
	transposed := Transpose4(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, attn.NHeads*attn.HeadDim)
	Free(transposed)
	result := attn.OProj.Forward(reshaped)
	Free(reshaped)
	return result, nil
}

func (attn *Gemma4AssistantAttention) applyRoPE(x *Array, offset int) *Array {
	if attn.RopeFreqs != nil {
		return RoPEWithFreqs(x, int(attn.HeadDim), false, 0, 1.0, offset, attn.RopeFreqs)
	}
	return RoPE(x, int(attn.RopeRotatedDim), false, attn.RopeBase, 1.0, offset)
}
