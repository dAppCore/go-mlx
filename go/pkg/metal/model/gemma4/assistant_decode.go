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
	// Logits holds the drafter's per-position output logits ([1, vocab] each),
	// retained ONLY for the speculative-SAMPLING path (temperature > 0) so the
	// verifier can form the drafter distribution q(x) for the min(1, p/q) accept
	// and the (p-q)+ residual. nil on the greedy path and for ordered-embedding
	// drafters (which expose no logits — those requests fall back to plain
	// decode). The caller Frees these.
	Logits []*metal.Array
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

// orderedEmbeddingDenseLogits projects an ordered-embedding (EAGLE) drafter step
// into a DENSE [batch, seqLen, vocab] logit vector: it scatters the sparse
// candidate logits into a -1e9-filled vocab vector at their token ids, so
// non-candidate tokens vanish under softmax. This lets a sparse drafter feed the
// speculative-SAMPLING path (which needs a full distribution q) through the same
// pipeline as a dense drafter — the maths is unchanged (q = 0 off-support).
func (m *Gemma4AssistantModel) orderedEmbeddingDenseLogits(hiddenStates *metal.Array) (*metal.Array, error) {
	candidates, err := m.orderedEmbeddingCandidates(hiddenStates)
	if err != nil {
		return nil, err
	}
	defer candidates.free()
	vocab := int32(m.Cfg.VocabSize)
	// selectedFlat / sparseLogits are 2D [tokenCount, K]; the dense base must
	// match rank, so [tokenCount, vocab]. PutAlongAxis scatters each row's K
	// candidate logits into its vocab row at the candidate token ids.
	tokenCount := int32(candidates.selectedFlat.Dim(0))
	negOne := metal.FromValues([]float32{-1e9}, 1, 1)
	base := metal.BroadcastTo(negOne, []int32{tokenCount, vocab})
	metal.Free(negOne)
	dense := metal.PutAlongAxis(base, candidates.selectedFlat, candidates.sparseLogits, -1)
	metal.Free(base)
	if dense == nil || !dense.Valid() {
		metal.Free(dense)
		return nil, errAsstOrderedAllCandidatesMuted
	}
	return dense, nil
}

// draftStepSampled is the speculative-SAMPLING counterpart of draftStepGreedy:
// it returns DENSE per-step logits for both drafter kinds (a dense drafter's
// native output, or an ordered-embedding drafter scattered into dense vocab
// space) and no token — draftBlockSampled samples the token from these logits.
func (pair *Gemma4AssistantPair) draftStepSampled(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache) (*Gemma4AssistantDraftStepResult, error) {
	normed, hidden, err := pair.draftStepActivations(lastToken, previousHidden, targetCaches)
	if err != nil {
		return nil, err
	}
	var logits *metal.Array
	if pair.Assistant.UseOrderedEmbeddings {
		logits, err = pair.Assistant.orderedEmbeddingDenseLogits(normed)
	} else {
		logits, err = pair.Assistant.outputLogits(normed)
	}
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
	return &Gemma4AssistantDraftStepResult{Logits: logits, Hidden: hidden}, nil
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
		if !ok || !targetKV.kv.HasState() {
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
	// The EAGLE head consumes the post-final-norm target feature (the vector the
	// target's LM head reads), not the pre-norm hidden the target carries. The
	// block seed is target-produced, so normalise it once here; the chained steps
	// below already run in the assistant's own (post-projection) feature space.
	currentHidden := metal.RMSNorm(previousHidden, pair.Target.NormScaled, pair.Target.Cfg.RMSNormEps)
	ownsCurrentHidden := true
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

// draftBlockSampled is the temperature>0 counterpart of DraftBlockWithSuppression:
// it SAMPLES each draft token from the drafter distribution (not argmax) and
// RETAINS the per-position logits so the verifier can form q(x) for the
// min(1, p/q) accept. The cache/hidden bookkeeping mirrors the greedy loop
// exactly; only token selection and logit retention differ. It requires a
// logits-exposing drafter — the serve gate keeps ordered-embedding drafters on
// the plain path at temperature>0, so step.Logits is expected non-nil here.
func (pair *Gemma4AssistantPair) draftBlockSampled(lastToken int32, previousHidden *metal.Array, targetCaches []metal.Cache, maxDraftTokens int, suppressTokens []int32, cfg metal.GenerateConfig, uniform func() float32) (*Gemma4AssistantDraftBlockResult, error) {
	if maxDraftTokens <= 0 {
		return nil, errAsstDraftBlockMaxZero
	}
	tokens := make([]int32, 0, maxDraftTokens)
	draftLogits := make([]*metal.Array, 0, maxDraftTokens)
	freeRetained := func() { metal.Free(draftLogits...) }
	currentToken := lastToken
	currentHidden := metal.RMSNorm(previousHidden, pair.Target.NormScaled, pair.Target.Cfg.RMSNormEps)
	ownsCurrentHidden := true
	for len(tokens) < maxDraftTokens {
		step, err := pair.draftStepSampled(currentToken, currentHidden, targetCaches)
		if ownsCurrentHidden {
			metal.Free(currentHidden)
			currentHidden = nil
			ownsCurrentHidden = false
		}
		if err != nil {
			freeRetained()
			return nil, err
		}
		if step.Logits == nil {
			step.Close()
			freeRetained()
			return nil, core.NewError("gemma4.assistant: speculative sampling requires a logits-exposing drafter")
		}
		if err := metal.Eval(step.Hidden); err != nil {
			step.Close()
			freeRetained()
			return nil, core.E("gemma4.assistant draft block (sampled)", "eval draft step", err)
		}
		currentToken = metal.SampleTokenFromLogits(step.Logits, cfg.Temperature, cfg.TopP, cfg.MinP, int(cfg.TopK), suppressTokens, uniform)
		draftLogits = append(draftLogits, step.Logits)
		step.Logits = nil // retained for verify — keep it out of step.Close
		tokens = append(tokens, currentToken)
		currentHidden = step.Hidden
		step.Hidden = nil
		ownsCurrentHidden = true
		step.Close()
	}
	return &Gemma4AssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden, Logits: draftLogits}, nil
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
	verifyCaches, err := metal.CloneCachePrefixes(targetCaches)
	if err != nil {
		return nil, err
	}

	result := &Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), draftTokens...),
		Caches:        verifyCaches,
	}
	// Verify the WHOLE draft block in ONE target forward (the speculative win —
	// previously this looped a separate target forward per accepted token, which
	// is no faster than plain decode). allLogits[:,i,:] is the target's greedy
	// prediction after consuming draftTokens[i].
	vDraft := metal.FromValues(draftTokens, len(draftTokens))
	draftInput := metal.Reshape2(vDraft, 1, int32(len(draftTokens)))
	metal.Free(vDraft)
	allLogits, allHidden := pair.Target.ForwardAllTokenLogitsAndHidden(draftInput, verifyCaches)
	metal.Free(draftInput)
	if allLogits == nil || allHidden == nil || !allLogits.Valid() || !allHidden.Valid() {
		metal.Free(allLogits, allHidden)
		result.Close()
		return nil, errAsstVerifyNoTargetToken
	}
	if err := metal.Eval(allLogits, allHidden); err != nil {
		metal.Free(allLogits, allHidden)
		result.Close()
		return nil, core.E("gemma4.assistant verify", "batched target forward", err)
	}
	metal.DetachCaches(verifyCaches)
	defer metal.Free(allLogits, allHidden)

	// Accept the longest prefix where the draft matches the target's greedy pick.
	// Position 0 is judged by the incoming targetLogits; position i (>0) by the
	// target prediction after draftTokens[i-1] (allLogits row i-1).
	k := 0
	for i := range draftTokens {
		predLogits := targetLogits
		if i > 0 {
			predLogits = metal.SliceAxis(allLogits, 1, int32(i-1), int32(i))
		}
		targetToken, terr := gemma4AssistantGreedyToken(predLogits, suppressTokens)
		if i > 0 {
			metal.Free(predLogits)
		}
		if terr != nil {
			result.Close()
			return nil, terr
		}
		if i == 0 {
			result.TargetTokens = append(result.TargetTokens, targetToken)
		}
		if targetToken != draftTokens[i] {
			break
		}
		result.AcceptedTokens = append(result.AcceptedTokens, draftTokens[i])
		k++
	}
	result.AcceptedCount = k

	// The bonus/replacement is the target's greedy pick after the accepted prefix.
	bonusLogits := targetLogits
	bonusOwned := false
	if k > 0 {
		bonusLogits = metal.SliceAxis(allLogits, 1, int32(k-1), int32(k))
		bonusOwned = true
	}

	if k == len(draftTokens) {
		// Whole block accepted: the clone already holds exactly the accepted
		// tokens (no truncate). Carry the last-position logits + hidden forward.
		result.AllAccepted = true
		result.Logits, err = cloneGemma4AssistantArray(bonusLogits)
		if bonusOwned {
			metal.Free(bonusLogits)
		}
		if err != nil {
			result.Close()
			return nil, err
		}
		hiddenSlice := metal.SliceAxis(allHidden, 1, int32(k-1), int32(k))
		result.Hidden, err = cloneGemma4AssistantArray(hiddenSlice)
		metal.Free(hiddenSlice)
		if err != nil {
			result.Close()
			return nil, err
		}
		return result, nil
	}

	// Partial accept: record the replacement and drop the rejected draft tokens
	// from the clone in place (or rebuild if the cache cannot truncate).
	result.RejectedCount = len(draftTokens) - k
	result.RejectedTokens = append([]int32(nil), draftTokens[k:]...)
	replacement, rerr := gemma4AssistantGreedyToken(bonusLogits, suppressTokens)
	if rerr != nil {
		if bonusOwned {
			metal.Free(bonusLogits)
		}
		result.Close()
		return nil, rerr
	}
	result.ReplacementToken = replacement
	// Contract: reject returns the rejection-position logits (argmax == replacement)
	// and, when some drafts were accepted, the hidden after the last accepted token.
	result.Logits, err = cloneGemma4AssistantArray(bonusLogits)
	if bonusOwned {
		metal.Free(bonusLogits)
	}
	if err != nil {
		result.Close()
		return nil, err
	}
	if k > 0 {
		hiddenSlice := metal.SliceAxis(allHidden, 1, int32(k-1), int32(k))
		result.Hidden, err = cloneGemma4AssistantArray(hiddenSlice)
		metal.Free(hiddenSlice)
		if err != nil {
			result.Close()
			return nil, err
		}
	}

	if !gemma4TruncateVerifyCaches(verifyCaches, len(draftTokens)-k) {
		rebuilt, berr := pair.rebuildAcceptedPrefixCaches(targetCaches, draftTokens[:k])
		if berr != nil {
			result.Close()
			return nil, berr
		}
		metal.FreeCaches(verifyCaches)
		result.Caches = rebuilt
	}
	// On the reject path the generate loop forwards the replacement onto
	// result.Caches, producing the next logits/hidden — none are returned here.
	return result, nil
}

// gemma4TruncateVerifyCaches drops the last `dropped` tokens from every verify
// cache in place. Returns false if any cache cannot truncate cheaply (rotating
// cache past its window) so the caller rebuilds instead.
func gemma4TruncateVerifyCaches(caches []metal.Cache, dropped int) bool {
	for _, c := range caches {
		if !metal.CacheTruncateTo(c, c.Len()-dropped) {
			return false
		}
	}
	return true
}

// rebuildAcceptedPrefixCaches clones the live caches and replays only the
// accepted prefix in one batched forward — the correctness fallback for caches
// that cannot truncate in place.
func (pair *Gemma4AssistantPair) rebuildAcceptedPrefixCaches(targetCaches []metal.Cache, accepted []int32) ([]metal.Cache, error) {
	caches, err := metal.CloneCachePrefixes(targetCaches)
	if err != nil {
		return nil, err
	}
	if len(accepted) == 0 {
		return caches, nil
	}
	vInput := metal.FromValues(accepted, len(accepted))
	input := metal.Reshape2(vInput, 1, int32(len(accepted)))
	metal.Free(vInput)
	logits, hidden := pair.Target.ForwardAllTokenLogitsAndHidden(input, caches)
	metal.Free(input)
	if err := metal.Eval(logits, hidden); err != nil {
		metal.Free(logits, hidden)
		metal.FreeCaches(caches)
		return nil, core.E("gemma4.assistant verify", "rebuild accepted prefix", err)
	}
	metal.DetachCaches(caches)
	metal.Free(logits, hidden)
	return caches, nil
}

// verifyDraftBlockSampled is the temperature>0 counterpart of
// VerifyDraftBlockWithSuppression. It forwards the block once, then instead of
// the greedy longest-argmax-prefix it runs the speculative-SAMPLING decision
// (accept each drafted token with prob min(1, p/q); committed lead tokens with a
// nil draft logit accept unconditionally). The next token is the residual draw
// on a partial accept, or a fresh sample from the target distribution p on a full
// accept — always produced so the loop can carry it as the next carryLead. The
// forward, cache truncation, and carried logits/hidden mirror the greedy path.
//
// draftLogits[i] is the drafter's [1, vocab] logits for block[i], or nil for a
// committed lead. The caller owns draftLogits (this does not free them).
func (pair *Gemma4AssistantPair) verifyDraftBlockSampled(targetLogits *metal.Array, draftTokens []int32, draftLogits []*metal.Array, targetCaches []metal.Cache, cfg metal.GenerateConfig, uniform func() float32, suppressTokens []int32) (*Gemma4AssistantVerifyResult, error) {
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
	verifyCaches, err := metal.CloneCachePrefixes(targetCaches)
	if err != nil {
		return nil, err
	}
	result := &Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), draftTokens...),
		Caches:        verifyCaches,
	}
	vDraft := metal.FromValues(draftTokens, len(draftTokens))
	draftInput := metal.Reshape2(vDraft, 1, int32(len(draftTokens)))
	metal.Free(vDraft)
	allLogits, allHidden := pair.Target.ForwardAllTokenLogitsAndHidden(draftInput, verifyCaches)
	metal.Free(draftInput)
	if allLogits == nil || allHidden == nil || !allLogits.Valid() || !allHidden.Valid() {
		metal.Free(allLogits, allHidden)
		result.Close()
		return nil, errAsstVerifyNoTargetToken
	}
	if err := metal.Eval(allLogits, allHidden); err != nil {
		metal.Free(allLogits, allHidden)
		result.Close()
		return nil, core.E("gemma4.assistant verify (sampled)", "batched target forward", err)
	}
	metal.DetachCaches(verifyCaches)
	defer metal.Free(allLogits, allHidden)

	// Per-position target logits: position 0 is judged by the incoming logits;
	// position i (>0) by the prediction after block[i-1] (allLogits row i-1).
	perPosTarget := make([]*metal.Array, len(draftTokens))
	ownedTarget := make([]bool, len(draftTokens))
	for i := range draftTokens {
		if i == 0 {
			perPosTarget[i] = targetLogits
		} else {
			perPosTarget[i] = metal.SliceAxis(allLogits, 1, int32(i-1), int32(i))
			ownedTarget[i] = true
		}
	}
	accepted, residualReplacement, allAccepted, derr := metal.SpeculativeVerifyDecision(perPosTarget, draftLogits, draftTokens, cfg, uniform, suppressTokens)
	for i := range perPosTarget {
		if ownedTarget[i] {
			metal.Free(perPosTarget[i])
		}
	}
	if derr != nil {
		result.Close()
		return nil, derr
	}
	k := len(accepted)
	result.AcceptedTokens = accepted
	result.AcceptedCount = k
	result.AllAccepted = allAccepted

	// bonus/replacement is judged by the prediction after the accepted prefix.
	bonusLogits := targetLogits
	bonusOwned := false
	if k > 0 {
		bonusLogits = metal.SliceAxis(allLogits, 1, int32(k-1), int32(k))
		bonusOwned = true
	}
	if allAccepted {
		// Whole block accepted → sample the bonus from the target distribution p.
		result.ReplacementToken = metal.SampleTokenFromLogits(bonusLogits, cfg.Temperature, cfg.TopP, cfg.MinP, int(cfg.TopK), suppressTokens, uniform)
	} else {
		result.ReplacementToken = residualReplacement
		result.RejectedCount = len(draftTokens) - k
		result.RejectedTokens = append([]int32(nil), draftTokens[k:]...)
	}
	result.Logits, err = cloneGemma4AssistantArray(bonusLogits)
	if bonusOwned {
		metal.Free(bonusLogits)
	}
	if err != nil {
		result.Close()
		return nil, err
	}
	if k > 0 {
		hiddenSlice := metal.SliceAxis(allHidden, 1, int32(k-1), int32(k))
		result.Hidden, err = cloneGemma4AssistantArray(hiddenSlice)
		metal.Free(hiddenSlice)
		if err != nil {
			result.Close()
			return nil, err
		}
	}
	if !gemma4TruncateVerifyCaches(verifyCaches, len(draftTokens)-k) {
		rebuilt, berr := pair.rebuildAcceptedPrefixCaches(targetCaches, draftTokens[:k])
		if berr != nil {
			result.Close()
			return nil, berr
		}
		metal.FreeCaches(verifyCaches)
		result.Caches = rebuilt
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
		if !ok || !targetKV.kv.HasState() {
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
	ff := layer.MLP.Forward(ffIn)
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
	if !targetKV.HasState() {
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
	if targetKV.HasPages() {
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
