// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
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
	errAsstOrderedNeedCentroids      = core.NewError("gemma4.assistant ordered embeddings require masked_embedding.centroids")
	errAsstOrderedNeedTokenOrdering  = core.NewError("gemma4.assistant ordered embeddings require masked_embedding.token_ordering")
	errAsstOrderedTopKInvalid        = core.NewError("gemma4.assistant ordered embeddings centroid_intermediate_top_k is invalid")
	errAsstOrderedVocabInvalid       = core.NewError("gemma4.assistant ordered embeddings vocab_size is invalid")
	errAsstOrderedAllCandidatesMuted = core.NewError("gemma4.assistant ordered embeddings produced no unsuppressed candidate")
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

const gemma4AssistantLogitsFloor = -3.4028234663852886e38

// Gemma4AssistantDraftStepResult is the caller-owned output of one MTP draft
// step. Hidden is projected back to the target backbone hidden size so it can
// seed the next assistant step.
type Gemma4AssistantDraftStepResult struct {
	Logits *metal.Array
	Token  *metal.Array
	Hidden *metal.Array
}

// Gemma4AssistantDraftBlockResult is the caller-owned output of chained MTP
// assistant proposals. Hidden is the final projected backbone hidden state.
type Gemma4AssistantDraftBlockResult struct {
	Tokens []int32
	Hidden *metal.Array
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
	Caches           []metal.Cache
	Logits           *metal.Array
	Hidden           *metal.Array
}

// Close releases arrays returned by DraftStep.
func (result *Gemma4AssistantDraftStepResult) Close() {
	if result == nil {
		return
	}
	metal.Free(result.Logits, result.Token, result.Hidden)
	result.Logits = nil
	result.Token = nil
	result.Hidden = nil
}

// Close releases arrays returned by DraftBlock.
func (result *Gemma4AssistantDraftBlockResult) Close() {
	if result == nil {
		return
	}
	metal.Free(result.Hidden)
	result.Hidden = nil
	result.Tokens = nil
}

// Close releases arrays and caches returned by VerifyDraftBlock.
func (result *Gemma4AssistantVerifyResult) Close() {
	if result == nil {
		return
	}
	metal.FreeCaches(result.Caches)
	metal.Free(result.Logits, result.Hidden)
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
	owned []*metal.Array
}

func (targetKV gemma4AssistantTargetKV) free() {
	metal.Free(targetKV.owned...)
}

// DraftStep proposes one token from the assistant using the target model's
// existing K/V cache streams and the previous target-backbone hidden state.
func (pair *Gemma4AssistantPair) DraftStep(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache) (*Gemma4AssistantDraftStepResult, error) {
	normed, hidden, err := pair.draftStepActivations(lastToken, previousHidden, targetCaches)
	if err != nil {
		return nil, err
	}
	logits, err := pair.Assistant.outputLogits(normed)
	metal.Free(normed)
	if err != nil {
		metal.Free(hidden)
		return nil, err
	}
	if pair.Assistant.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(logits, pair.Assistant.Cfg.FinalLogitSoftcapping)
		metal.Free(logits)
		logits = softcapped
	}
	token := metal.Argmax(logits, -1, false)
	return &Gemma4AssistantDraftStepResult{Logits: logits, Token: token, Hidden: hidden}, nil
}

func (pair *Gemma4AssistantPair) draftStepGreedy(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache, suppressTokens []int32) (*Gemma4AssistantDraftStepResult, error) {
	normed, hidden, err := pair.draftStepActivations(lastToken, previousHidden, targetCaches)
	if err != nil {
		return nil, err
	}
	if pair.Assistant.UseOrderedEmbeddings {
		token, err := pair.Assistant.orderedEmbeddingGreedyToken(normed, suppressTokens)
		metal.Free(normed)
		if err != nil {
			metal.Free(hidden)
			return nil, err
		}
		return &Gemma4AssistantDraftStepResult{Token: token, Hidden: hidden}, nil
	}

	logits, err := pair.Assistant.outputLogits(normed)
	metal.Free(normed)
	if err != nil {
		metal.Free(hidden)
		return nil, err
	}
	if pair.Assistant.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(logits, pair.Assistant.Cfg.FinalLogitSoftcapping)
		metal.Free(logits)
		logits = softcapped
	}
	token := metal.Argmax(logits, -1, false)
	return &Gemma4AssistantDraftStepResult{Logits: logits, Token: token, Hidden: hidden}, nil
}

func (pair *Gemma4AssistantPair) draftStepActivations(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache) (*metal.Array, *metal.Array, error) {
	if pair == nil || pair.Target == nil || pair.Assistant == nil {
		return nil, nil, errAsstDraftStepNeedPair
	}
	if lastToken < 0 {
		return nil, nil, errAsstDraftStepTokenInvalid
	}
	if previousHidden == nil || !previousHidden.Valid() {
		return nil, nil, errAsstDraftStepHiddenInvalid
	}
	if len(targetCaches) == 0 {
		return nil, nil, errAsstDraftStepNeedTargetCaches
	}
	if err := validateGemma4AssistantPair(pair.Target, pair.Assistant); err != nil {
		return nil, nil, err
	}

	targetKVs, err := pair.targetKVByLayerType(targetCaches)
	if err != nil {
		return nil, nil, err
	}
	defer func() {
		for _, targetKV := range targetKVs {
			targetKV.free()
		}
	}()

	tokenInput := metal.FromSingleInt32Matrix(lastToken)
	tokenEmbedding := pair.Target.EmbedTokens.Forward(tokenInput)
	scaledTokenEmbedding := metal.MulScalar(tokenEmbedding, pair.Target.Cfg.EmbeddingScale)
	metal.Free(tokenInput, tokenEmbedding)

	backboneHidden, ownBackboneHidden, err := gemma4AssistantBackboneHidden(previousHidden, pair.Assistant.BackboneHiddenSize)
	if err != nil {
		metal.Free(scaledTokenEmbedding)
		return nil, nil, err
	}
	combined := metal.Concatenate2(scaledTokenEmbedding, backboneHidden, 2)
	metal.Free(scaledTokenEmbedding)
	if ownBackboneHidden {
		metal.Free(backboneHidden)
	}

	h := pair.Assistant.PreProjection.Forward(combined)
	metal.Free(combined)
	for _, layer := range pair.Assistant.Layers {
		targetKV, ok := targetKVs[layer.LayerType]
		if !ok || !targetKV.kv.hasState() {
			metal.Free(h)
			return nil, nil, core.NewError("gemma4.assistant draft step missing target K/V stream for " + layer.LayerType)
		}
		next, err := layer.forwardDraftStep(h, targetKV.kv, pair.Assistant.Cfg)
		metal.Free(h)
		if err != nil {
			return nil, nil, err
		}
		h = next
	}

	normed := pair.Assistant.Norm.Forward(h, pair.Assistant.Cfg.RMSNormEps)
	metal.Free(h)
	hidden := pair.Assistant.PostProjection.Forward(normed)
	return normed, hidden, nil
}

func (m *Gemma4AssistantModel) outputLogits(hiddenStates *metal.Array) (*metal.Array, error) {
	if m == nil || !m.UseOrderedEmbeddings {
		return m.EmbedTokens.AsLinear().Forward(hiddenStates), nil
	}
	return m.orderedEmbeddingLogits(hiddenStates)
}

type gemma4AssistantOrderedEmbeddingCandidates struct {
	batch         int32
	seqLen        int32
	vocabSize     int32
	tokenCount    int32
	selectedCount int32
	selectedFlat  *metal.Array
	sparseLogits  *metal.Array
}

func (c *gemma4AssistantOrderedEmbeddingCandidates) free() {
	if c == nil {
		return
	}
	metal.Free(c.selectedFlat, c.sparseLogits)
	c.selectedFlat = nil
	c.sparseLogits = nil
}

func (m *Gemma4AssistantModel) orderedEmbeddingLogits(hiddenStates *metal.Array) (*metal.Array, error) {
	candidates, err := m.orderedEmbeddingCandidates(hiddenStates)
	if err != nil {
		return nil, err
	}
	defer candidates.free()

	fillScalar := metal.FromValue(float32(gemma4AssistantLogitsFloor))
	if dtype := candidates.sparseLogits.Dtype(); dtype != metal.DTypeFloat32 {
		typedFill := metal.AsType(fillScalar, dtype)
		metal.Free(fillScalar)
		fillScalar = typedFill
	}
	fullFlat := metal.BroadcastTo(fillScalar, []int32{candidates.tokenCount, candidates.vocabSize})
	metal.Free(fillScalar)
	scattered := metal.PutAlongAxis(fullFlat, candidates.selectedFlat, candidates.sparseLogits, -1)
	metal.Free(fullFlat)
	logits := metal.Reshape3(scattered, candidates.batch, candidates.seqLen, candidates.vocabSize)
	metal.Free(scattered)
	return logits, nil
}

func (m *Gemma4AssistantModel) orderedEmbeddingGreedyToken(hiddenStates *metal.Array, suppressTokens []int32) (*metal.Array, error) {
	candidates, err := m.orderedEmbeddingCandidates(hiddenStates)
	if err != nil {
		return nil, err
	}
	defer candidates.free()

	sparseLogits := candidates.sparseLogits
	filteredLogits, filteredOwned, err := suppressOrderedEmbeddingSparseLogits(candidates.selectedFlat, sparseLogits, suppressTokens)
	if err != nil {
		return nil, err
	}
	if filteredOwned {
		sparseLogits = filteredLogits
		defer metal.Free(filteredLogits)
	}

	indices := metal.Argmax(sparseLogits, -1, true)
	tokenFlat := metal.TakeAlongAxis(candidates.selectedFlat, indices, -1)
	metal.Free(indices)
	token := metal.Reshape2(tokenFlat, candidates.batch, candidates.seqLen)
	metal.Free(tokenFlat)
	return token, nil
}

func suppressOrderedEmbeddingSparseLogits(selectedFlat, sparseLogits *metal.Array, suppressTokens []int32) (*metal.Array, bool, error) {
	if len(suppressTokens) == 0 {
		return sparseLogits, false, nil
	}

	scratchPtr := metal.SuppressIDsScratch.Get().(*[]int32)
	scratch := (*scratchPtr)[:0]
	if cap(scratch) < len(suppressTokens) {
		scratch = make([]int32, 0, len(suppressTokens))
	}
	for _, id := range suppressTokens {
		if id >= 0 {
			scratch = append(scratch, id)
		}
	}
	if len(scratch) == 0 {
		*scratchPtr = scratch
		metal.SuppressIDsScratch.Put(scratchPtr)
		return sparseLogits, false, nil
	}

	suppressIDs := metal.FromValues(scratch, 1, 1, len(scratch))
	expandedSelected := metal.ExpandDims(selectedFlat, -1)
	matches := metal.Equal(expandedSelected, suppressIDs)
	metal.Free(expandedSelected, suppressIDs)
	suppressed := metal.AnyAxis(matches, -1, false)
	metal.Free(matches)
	filtered := metal.WhereScalarArray(suppressed, float32(gemma4AssistantLogitsFloor), sparseLogits)
	metal.Free(suppressed)

	*scratchPtr = scratch
	metal.SuppressIDsScratch.Put(scratchPtr)
	return filtered, true, nil
}

func (m *Gemma4AssistantModel) orderedEmbeddingCandidates(hiddenStates *metal.Array) (*gemma4AssistantOrderedEmbeddingCandidates, error) {
	if m.MaskedCentroids == nil || m.MaskedCentroids.Weight == nil || !m.MaskedCentroids.Weight.Valid() {
		return nil, errAsstOrderedNeedCentroids
	}
	if m.TokenOrdering == nil || !m.TokenOrdering.Valid() {
		return nil, errAsstOrderedNeedTokenOrdering
	}
	if m.Cfg == nil || m.Cfg.VocabSize <= 0 {
		return nil, errAsstOrderedVocabInvalid
	}
	vocabSize := m.Cfg.VocabSize
	numCentroids := m.NumCentroids
	topK := m.CentroidIntermediateTopK
	if numCentroids <= 0 || topK <= 0 || topK > numCentroids {
		return nil, errAsstOrderedTopKInvalid
	}
	if vocabSize%numCentroids != 0 {
		return nil, core.NewError("gemma4.assistant token_ordering requires vocab_size divisible by num_centroids")
	}
	var orderingShapeBuf [metal.MaxTensorRank]int32
	orderingShape := m.TokenOrdering.ShapeInto(orderingShapeBuf[:0])
	var clusters *metal.Array
	clustersOwned := false
	if len(orderingShape) == 1 && orderingShape[0] == vocabSize {
		clusters = metal.Reshape2(m.TokenOrdering, numCentroids, vocabSize/numCentroids)
		clustersOwned = true
	} else if len(orderingShape) == 2 && orderingShape[0] == numCentroids && orderingShape[1] == vocabSize/numCentroids {
		clusters = m.TokenOrdering
	} else {
		return nil, core.NewError(core.Sprintf("gemma4.assistant token_ordering shape = %v, want [%d] or [%d %d]", orderingShape, vocabSize, numCentroids, vocabSize/numCentroids))
	}
	var hiddenShapeBuf [metal.MaxTensorRank]int32
	hiddenShape := hiddenStates.ShapeInto(hiddenShapeBuf[:0])
	if len(hiddenShape) != 3 || hiddenShape[2] != m.Cfg.HiddenSize {
		return nil, core.NewError(core.Sprintf("gemma4.assistant ordered hidden shape = %v, want [batch sequence %d]", hiddenShape, m.Cfg.HiddenSize))
	}

	batch, seqLen, hiddenSize := hiddenShape[0], hiddenShape[1], hiddenShape[2]
	tokenCount := batch * seqLen
	vocabPerCentroid := vocabSize / numCentroids
	selectedCount := topK * vocabPerCentroid

	flatHidden := metal.Reshape2(hiddenStates, tokenCount, hiddenSize)
	centroidScores := m.MaskedCentroids.Forward(flatHidden)
	kth := int(numCentroids - topK)
	partitioned := metal.Argpartition(centroidScores, kth, -1)
	metal.Free(centroidScores)
	topCentroids := metal.Slice2(partitioned, 0, int32(kth), tokenCount, numCentroids)
	metal.Free(partitioned)

	selected := metal.Take(clusters, topCentroids, 0)
	if clustersOwned {
		metal.Free(clusters)
	}
	metal.Free(topCentroids)
	selectedFlat := metal.Reshape2(selected, tokenCount, selectedCount)
	metal.Free(selected)

	candidateEmbeddings := m.EmbedTokens.Forward(selectedFlat)
	expandedHidden := metal.ExpandDims(flatHidden, 1)
	products := metal.Mul(expandedHidden, candidateEmbeddings)
	sparseLogits := metal.Sum(products, -1, false)
	metal.Free(flatHidden, candidateEmbeddings, expandedHidden, products)
	return &gemma4AssistantOrderedEmbeddingCandidates{
		batch:         batch,
		seqLen:        seqLen,
		vocabSize:     vocabSize,
		tokenCount:    tokenCount,
		selectedCount: selectedCount,
		selectedFlat:  selectedFlat,
		sparseLogits:  sparseLogits,
	}, nil
}

// DraftBlock chains assistant MTP steps and returns a CPU-visible draft token
// block. Verification still belongs to the target-side accept/reject path.
func (pair *Gemma4AssistantPair) DraftBlock(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache, maxDraftTokens int) (*Gemma4AssistantDraftBlockResult, error) {
	return pair.DraftBlockWithSuppression(lastToken, previousHidden, targetCaches, maxDraftTokens, nil)
}

// DraftBlockWithSuppression chains assistant MTP steps while preserving the
// generation token-suppression policy used by the target decoder.
func (pair *Gemma4AssistantPair) DraftBlockWithSuppression(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache, maxDraftTokens int, suppressTokens []int32) (*Gemma4AssistantDraftBlockResult, error) {
	if maxDraftTokens <= 0 {
		return nil, errAsstDraftBlockMaxZero
	}
	tokens := make([]int32, 0, maxDraftTokens)
	currentToken := lastToken
	currentHidden := previousHidden
	ownsCurrentHidden := false
	for len(tokens) < maxDraftTokens {
		step, err := pair.draftStepGreedy(currentToken, currentHidden, targetCaches, suppressTokens)
		if ownsCurrentHidden {
			metal.Free(currentHidden)
			currentHidden = nil
			ownsCurrentHidden = false
		}
		if err != nil {
			return nil, err
		}
		if err := metal.Eval(step.Token, step.Hidden); err != nil {
			step.Close()
			return nil, core.E("gemma4.assistant draft block", "eval draft step", err)
		}
		currentToken, err = gemma4AssistantDraftStepToken(step, suppressTokens)
		if err != nil {
			step.Close()
			return nil, err
		}
		tokens = append(tokens, currentToken)
		currentHidden = step.Hidden
		step.Hidden = nil
		ownsCurrentHidden = true
		step.Close()
	}
	return &Gemma4AssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden}, nil
}

func gemma4AssistantDraftStepToken(step *Gemma4AssistantDraftStepResult, suppressTokens []int32) (int32, error) {
	if step == nil || step.Token == nil {
		return 0, errAsstDraftBlockNoToken
	}
	values := step.Token.DataInt32()
	if len(values) == 0 {
		return 0, errAsstDraftBlockNoToken
	}
	id := values[0]
	if !metal.TokenIDSuppressed(id, suppressTokens) {
		return id, nil
	}
	if step.Logits == nil || !step.Logits.Valid() {
		return 0, errAsstOrderedAllCandidatesMuted
	}
	replacement, replacementID, _, err := metal.SampleTokenIDWithSuppressionGuard(step.Logits, metal.Greedy{}, suppressTokens, false)
	if err != nil {
		return 0, err
	}
	metal.Free(step.Token)
	step.Token = replacement
	return replacementID, nil
}

// VerifyDraftBlock compares an assistant draft block against metal.Greedy target
// predictions. The caller's target caches are cloned before verification, so
// rejected draft tokens never pollute the live generation cache.
func (pair *Gemma4AssistantPair) VerifyDraftBlock(targetLogits *metal.Array, draftTokens []int32, targetCaches []metal.Cache) (*Gemma4AssistantVerifyResult, error) {
	return pair.VerifyDraftBlockWithSuppression(targetLogits, draftTokens, targetCaches, nil)
}

// VerifyDraftBlockWithSuppression compares assistant proposals against target
// metal.Greedy predictions after applying the same token-suppression policy used by
// normal generation.
func (pair *Gemma4AssistantPair) VerifyDraftBlockWithSuppression(targetLogits *metal.Array, draftTokens []int32, targetCaches []metal.Cache, suppressTokens []int32) (*Gemma4AssistantVerifyResult, error) {
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
	var currentHidden *metal.Array
	currentHiddenOwned := false

	for idx, draftToken := range draftTokens {
		targetToken, err := gemma4AssistantGreedyToken(currentLogits, suppressTokens)
		if err != nil {
			result.Close()
			if currentLogitsOwned {
				metal.Free(currentLogits)
			}
			if currentHiddenOwned {
				metal.Free(currentHidden)
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
						metal.Free(currentHidden)
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
		tokenInput := metal.FromSingleInt32Matrix(draftToken)
		nextLogits, nextHidden := pair.Target.ForwardLastTokenLogitsAndHidden(tokenInput, nil, verifyCaches)
		metal.Free(tokenInput)
		if err := metal.Eval(nextLogits, nextHidden); err != nil {
			result.Close()
			metal.Free(nextLogits, nextHidden)
			if currentLogitsOwned {
				metal.Free(currentLogits)
			}
			if currentHiddenOwned {
				metal.Free(currentHidden)
			}
			return nil, core.E("gemma4.assistant verify", "target accepted token", err)
		}
		metal.DetachCaches(verifyCaches)
		if currentLogitsOwned {
			metal.Free(currentLogits)
		}
		if currentHiddenOwned {
			metal.Free(currentHidden)
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
				metal.Free(currentHidden)
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

func (pair *Gemma4AssistantPair) targetKVByLayerType(caches []metal.Cache) (map[string]gemma4AssistantTargetKV, error) {
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

func gemma4AssistantKVFromCache(cache metal.Cache) (gemma4AssistantTargetKV, error) {
	if cache == nil || cache.Len() <= 0 {
		return gemma4AssistantTargetKV{}, errTargetCacheEmpty
	}
	if paged, ok := cache.(*metal.PagedKVCache); ok {
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

	state, owned := metal.CacheReadState(cache)
	if len(state) < 2 || state[0] == nil || state[1] == nil || !state[0].Valid() || !state[1].Valid() {
		metal.Free(owned...)
		return gemma4AssistantTargetKV{}, errTargetCacheStateEmpty
	}
	keys, values := state[0], state[1]
	visible := int32(cache.Len())
	if visible <= 0 {
		metal.Free(owned...)
		return gemma4AssistantTargetKV{}, errTargetCacheLenEmpty
	}
	// Stack-allocated shape scratch — assistant verify cache trim is called
	// per draft step. Both Slice calls are rank-4 by guard (len ≥ 4).
	var kShapeBuf, vShapeBuf [metal.MaxTensorRank]int32
	kShape := keys.ShapeInto(kShapeBuf[:0])
	vShape := values.ShapeInto(vShapeBuf[:0])
	if len(kShape) >= 4 && len(vShape) >= 4 {
		if kShape[2] < visible || vShape[2] < visible {
			metal.Free(owned...)
			return gemma4AssistantTargetKV{}, errTargetCacheTooShort
		}
		if kShape[2] != visible {
			keys = metal.Slice4(keys, 0, 0, 0, 0, kShape[0], kShape[1], visible, kShape[3])
			owned = append(owned, keys)
		}
		if vShape[2] != visible {
			values = metal.Slice4(values, 0, 0, 0, 0, vShape[0], vShape[1], visible, vShape[3])
			owned = append(owned, values)
		}
	}
	return gemma4AssistantTargetKV{
		kv:    sharedKV{Keys: keys, Values: values, Offset: cache.Offset()},
		owned: owned,
	}, nil
}

func cloneGemma4AssistantVerifyCaches(caches []metal.Cache) ([]metal.Cache, error) {
	cloned := make([]metal.Cache, len(caches))
	for i, cache := range caches {
		next, err := cloneGemma4AssistantVerifyCache(cache)
		if err != nil {
			metal.FreeCaches(cloned)
			return nil, core.E("gemma4.assistant verify", core.Sprintf("clone cache %d", i), err)
		}
		cloned[i] = next
	}
	return cloned, nil
}

func cloneGemma4AssistantVerifyCache(cache metal.Cache) (metal.Cache, error) {
	if cache == nil {
		return nil, errTargetCacheNil
	}
	if cache.Len() <= 0 {
		switch c := cache.(type) {
		case *metal.RotatingKVCache:
			return metal.NewRotatingKVCache(c.maxSize), nil
		case *metal.FixedKVCache:
			return metal.NewFixedKVCache(c.maxSize), nil
		case *metal.PagedKVCache:
			return metal.NewPagedKVCache(c.maxSize, c.pageSize), nil
		case *metal.QuantizedKVCache:
			return metal.NewQuantizedKVCache(c.maxSize, c.keyBits, c.valueBits), nil
		default:
			return metal.NewKVCache(), nil
		}
	}
	switch c := cache.(type) {
	case *metal.KVCache:
		state, owned := metal.CacheReadState(c)
		defer metal.Free(owned...)
		if len(state) < 2 {
			return nil, errKVCacheStateEmpty
		}
		keys, values, err := cloneGemma4AssistantCacheState(state[0], state[1], c.Len())
		if err != nil {
			return nil, err
		}
		return &metal.KVCache{keys: keys, values: values, offset: c.offset, step: c.step}, nil
	case *metal.RotatingKVCache:
		state, owned := metal.CacheReadState(c)
		defer metal.Free(owned...)
		if len(state) < 2 {
			return nil, errRotatingCacheEmpty
		}
		keys, values, err := cloneGemma4AssistantCacheState(state[0], state[1], c.Len())
		if err != nil {
			return nil, err
		}
		return &metal.RotatingKVCache{keys: keys, values: values, offset: c.offset, maxSize: c.maxSize, step: c.step, idx: c.Len()}, nil
	case *metal.FixedKVCache:
		state := c.FixedState()
		if state.Keys == nil || state.Values == nil {
			state.Free()
			return metal.NewFixedKVCache(c.maxSize), nil
		}
		return &metal.FixedKVCache{keys: state.Keys, values: state.Values, offset: c.offset, length: c.length, maxSize: c.maxSize}, nil
	case *metal.PagedKVCache:
		pages := c.PageState()
		defer pages.Free()
		kPages, vPages, err := metal.CopyPagedCachePrefix(pages.Keys, pages.Values, c.Len())
		if err != nil {
			return nil, err
		}
		return &metal.PagedKVCache{kPages: kPages, vPages: vPages, pageLens: metal.PagedPageLensForPages(kPages, c.length), offset: c.offset, length: c.length, maxSize: c.maxSize, pageSize: c.pageSize}, nil
	case *metal.QuantizedKVCache:
		return &metal.QuantizedKVCache{
			keys:       metal.Copy(c.keys),
			values:     metal.Copy(c.values),
			keyScale:   metal.Copy(c.keyScale),
			valueScale: metal.Copy(c.valueScale),
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
		state, owned := metal.CacheReadState(cache)
		defer metal.Free(owned...)
		if len(state) < 2 {
			return nil, errCacheStateEmpty
		}
		keys, values, err := cloneGemma4AssistantCacheState(state[0], state[1], cache.Len())
		if err != nil {
			return nil, err
		}
		return &metal.KVCache{keys: keys, values: values, offset: cache.Offset(), step: 256}, nil
	}
}

func cloneGemma4AssistantCacheState(keys, values *metal.Array, tokenLen int) (*metal.Array, *metal.Array, error) {
	keyCopy, err := metal.CopyCachePrefix(keys, tokenLen)
	if err != nil {
		return nil, nil, err
	}
	valueCopy, err := metal.CopyCachePrefix(values, tokenLen)
	if err != nil {
		metal.Free(keyCopy)
		return nil, nil, err
	}
	return keyCopy, valueCopy, nil
}

func gemma4AssistantGreedyToken(logits *metal.Array, suppressTokens ...[]int32) (int32, error) {
	if len(suppressTokens) > 0 && len(suppressTokens[0]) > 0 {
		token, id, _, err := metal.SampleTokenIDWithSuppressionGuard(logits, metal.Greedy{}, suppressTokens[0], false)
		metal.Free(token)
		return id, err
	}
	token := metal.Argmax(logits, -1, false)
	defer metal.Free(token)
	if err := metal.Eval(token); err != nil {
		return 0, err
	}
	values := token.DataInt32()
	if len(values) == 0 {
		return 0, errAsstVerifyNoTargetToken
	}
	return values[0], nil
}

func cloneGemma4AssistantArray(array *metal.Array) (*metal.Array, error) {
	if array == nil || !array.Valid() {
		return nil, errAsstCloneInvalid
	}
	cloned := metal.Copy(array)
	if err := metal.Eval(cloned); err != nil {
		metal.Free(cloned)
		return nil, err
	}
	metal.Detach(cloned)
	return cloned, nil
}

func gemma4AssistantBackboneHidden(hidden *metal.Array, backboneHidden int32) (*metal.Array, bool, error) {
	// Stack-allocated shape scratch — per-assistant-draft-step path.
	var shapeBuf [metal.MaxTensorRank]int32
	shape := hidden.ShapeInto(shapeBuf[:0])
	switch {
	case len(shape) == 3 && shape[0] == 1 && shape[1] == 1 && shape[2] == backboneHidden:
		return hidden, false, nil
	case len(shape) == 2 && shape[0] == 1 && shape[1] == backboneHidden:
		return metal.Reshape(hidden, 1, 1, backboneHidden), true, nil
	case len(shape) == 1 && shape[0] == backboneHidden:
		return metal.Reshape(hidden, 1, 1, backboneHidden), true, nil
	default:
		return nil, false, core.NewError(core.Sprintf("gemma4.assistant previous hidden shape = %v, want [1 1 %d]", shape, backboneHidden))
	}
}

func (layer *Gemma4AssistantLayer) forwardDraftStep(x *metal.Array, targetKV sharedKV, cfg *Gemma4TextConfig) (*metal.Array, error) {
	if layer == nil || layer.Attention == nil || layer.MLP == nil {
		return nil, errAsstDraftStepLayerIncomplete
	}
	// Stack-allocated shape scratch — per-assistant-draft-step per-layer
	// hot path. Avoids the per-call []int32 heap alloc.
	var shapeBuf [metal.MaxTensorRank]int32
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
	metal.Free(normed)
	if err != nil {
		return nil, err
	}
	attnNormed := layer.PostAttnNorm.Forward(attnOut, cfg.RMSNormEps)
	metal.Free(attnOut)
	h := metal.Add(x, attnNormed)
	metal.Free(attnNormed)

	ffIn := layer.PreFFNorm.Forward(h, cfg.RMSNormEps)
	ff := layer.MLP.forward(ffIn)
	metal.Free(ffIn)
	ffResidual := layer.PostFFNorm.Forward(ff, cfg.RMSNormEps)
	metal.Free(ff)

	hNext := metal.Add(h, ffResidual)
	metal.Free(h, ffResidual)
	if layer.LayerScalar != nil && layer.LayerScalar.Valid() {
		scaled := metal.Mul(hNext, layer.LayerScalar)
		metal.Free(hNext)
		hNext = scaled
	}
	return hNext, nil
}

func (attn *Gemma4AssistantAttention) forwardWithTargetKV(x *metal.Array, targetKV sharedKV, B, L int32, cfg *Gemma4TextConfig) (*metal.Array, error) {
	if attn == nil || attn.QProj == nil || attn.OProj == nil || attn.QNorm == nil {
		return nil, errAsstAttnIncomplete
	}
	if !targetKV.hasState() {
		return nil, errAsstAttnMissingKV
	}

	qProj := attn.QProj.Forward(x)
	q := metal.AsStrided(qProj, []int32{B, attn.NHeads, L, attn.HeadDim},
		[]int64{int64(L * attn.NHeads * attn.HeadDim), int64(attn.HeadDim), int64(attn.NHeads * attn.HeadDim), 1}, 0)
	metal.Free(qProj)
	oldQ := q
	q = attn.QNorm.Forward(q, cfg.RMSNormEps)
	metal.Free(oldQ)
	qRoPE := attn.applyRoPE(q, targetKV.Offset)
	metal.Free(q)
	q = qRoPE

	var out *metal.Array
	if targetKV.hasPages() {
		keyHeads := int32(0)
		if len(targetKV.Pages.Keys) > 0 && targetKV.Pages.Keys[0] != nil && targetKV.Pages.Keys[0].Valid() {
			keyHeads = int32(targetKV.Pages.Keys[0].Dim(1))
		}
		kPages, vPages := targetKV.Pages.Keys, targetKV.Pages.Values
		var repeated []*metal.Array
		if keyHeads > 0 && attn.NHeads > keyHeads && attn.NHeads%keyHeads == 0 && len(kPages) > 1 && metal.PagedStateNeedsMaterializedRepeat(targetKV.Pages, attn.NHeads/keyHeads) {
			kPages, vPages, repeated = metal.RepeatPagedState(targetKV.Pages, attn.NHeads/keyHeads)
		}
		out = metal.ScaledDotProductAttentionPaged(q, kPages, vPages, attn.Scale)
		metal.Free(repeated...)
	} else {
		out = metal.ScaledDotProductAttention(q, targetKV.Keys, targetKV.Values, attn.Scale, false)
	}
	metal.Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — scalar-pass
	// Transpose4 form (eliminates the []int axes heap alloc).
	transposed := metal.Transpose4(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, attn.NHeads*attn.HeadDim)
	metal.Free(transposed)
	result := attn.OProj.Forward(reshaped)
	metal.Free(reshaped)
	return result, nil
}

func (attn *Gemma4AssistantAttention) applyRoPE(x *metal.Array, offset int) *metal.Array {
	if attn.RopeFreqs != nil {
		return metal.RoPEWithFreqs(x, int(attn.HeadDim), false, 0, 1.0, offset, attn.RopeFreqs)
	}
	return metal.RoPE(x, int(attn.RopeRotatedDim), false, attn.RopeBase, 1.0, offset)
}
