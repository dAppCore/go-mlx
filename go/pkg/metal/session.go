// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"iter"
	"slices"
	"sync"
	"time"

	core "dappco.re/go"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. Sharing instances also makes errors.Is comparable for
// callers without parsing message text.
var (
	errByteLenShape                 = core.NewError("byte length does not match shape")
	errInvalidShape                 = core.NewError("invalid shape")
	errMissingNativeSlab            = core.NewError("missing native slab")
	errKVStreamInvalidTokenState    = core.NewError("mlx: KV block stream has invalid token state")
	errKVStreamNoBoundaries         = core.NewError("mlx: KV block stream has no block boundaries")
	errKVBlockYieldNil              = core.NewError("mlx: KV block yield is nil")
	errSnapshotArchMismatch         = core.NewError("mlx: KV snapshot architecture does not match model")
	errSnapshotBlockSize            = core.NewError("mlx: KV snapshot block size must be > 0")
	errSnapshotCacheIndex           = core.NewError("mlx: KV snapshot cache index exceeds model cache count")
	errSnapshotExceedsFixedCap      = core.NewError("mlx: KV snapshot exceeds fixed cache capacity")
	errSnapshotInvalidHeadDims      = core.NewError("mlx: KV snapshot has invalid head dimensions")
	errSnapshotInvalidTensorDims    = core.NewError("mlx: KV snapshot has invalid tensor dimensions")
	errSnapshotNoLayers             = core.NewError("mlx: KV snapshot has no layers")
	errSnapshotNoRestorableLogits   = core.NewError("mlx: KV snapshot has no restorable logits")
	errSnapshotNoSeqLen             = core.NewError("mlx: KV snapshot has no sequence length")
	errSnapshotNil                  = core.NewError("mlx: KV snapshot is nil")
	errSnapshotKeyTensorSize        = core.NewError("mlx: KV snapshot key tensor has unexpected size")
	errSnapshotKVLenDiffer          = core.NewError("mlx: KV snapshot key/value cache lengths differ")
	errSnapshotLayerNoHeads         = core.NewError("mlx: KV snapshot layer has no heads")
	errSnapshotLogitShape           = core.NewError("mlx: KV snapshot logit shape is invalid")
	errSnapshotLogitsShapeMismatch  = core.NewError("mlx: KV snapshot logits do not match shape")
	errSnapshotMixedTensorHeads     = core.NewError("mlx: KV snapshot mixes native and float32 tensor heads")
	errSnapshotNativeKeySize        = core.NewError("mlx: KV snapshot native key tensor has unexpected size")
	errSnapshotNativeKVShapesDiffer = core.NewError("mlx: KV snapshot native layer key/value shapes differ")
	errSnapshotNativeByteLen        = core.NewError("mlx: KV snapshot native tensor byte length mismatch")
	errSnapshotNativeDtypeMismatch  = core.NewError("mlx: KV snapshot native tensor dtype mismatch")
	errSnapshotNativeValueSize      = core.NewError("mlx: KV snapshot native value tensor has unexpected size")
	errSnapshotValueTensorSize      = core.NewError("mlx: KV snapshot value tensor has unexpected size")
	errModelNoKVCaches              = core.NewError("mlx: model has no KV caches")
	errSessionNoPrefill             = core.NewError("mlx: model session has no prefilled state")
	errSessionNoRestorableLogits    = core.NewError("mlx: model session has no restorable logits")
	errSessionClosed                = core.NewError("mlx: model session is closed")
	errSessionNil                   = core.NewError("mlx: model session is nil")
	errTurboQuantSnapshotLayout     = core.NewError("mlx: TurboQuant KV cache snapshots require a versioned TurboQuant physical layout")
	errUnsupportedKVCacheType       = core.NewError("mlx: unsupported KV cache type")
	errUnsupportedNativeDtype       = core.NewError("mlx: unsupported KV snapshot native tensor dtype")
	errUnsupportedSnapshotVersion   = core.NewError("mlx: unsupported KV snapshot version")
	errForwardNilLogits             = core.NewError("model forward returned nil logits")
	errAppendPromptEmpty            = core.NewError("ModelSession.AppendPrompt: empty prompt after tokenisation")
	errAppendTokensEmpty            = core.NewError("ModelSession.AppendTokens: empty prompt tokens")
	errForkCacheNotSnapshotable     = core.NewError("ModelSession.Fork: cache is not snapshotable")
	errPrefillPromptEmpty           = core.NewError("ModelSession.Prefill: empty prompt after tokenisation")
	errPrefillTokensEmpty           = core.NewError("ModelSession.PrefillTokens: empty prompt tokens")
	errUnsupportedDtype             = core.NewError("unsupported dtype")
)

// SessionHandle is the native model-state session interface.
type SessionHandle interface {
	Prefill(context.Context, string) error
	AppendPrompt(context.Context, string) error
	Generate(context.Context, GenerateConfig) iter.Seq[Token]
	CaptureKV(context.Context) (*KVSnapshot, error)
	RangeKVBlocks(context.Context, int, KVSnapshotCaptureOptions, func(KVSnapshotBlock) (bool, error)) error
	Fork(context.Context) (SessionHandle, error)
	Reset()
	Close() error
	Err() error
}

// ModelSession owns one persistent KV/logit state for a loaded model.
type ModelSession struct {
	mu              sync.Mutex
	model           *Model
	caches          []Cache
	logits          *Array
	tokens          []int32
	generated       []int32
	tokenOffset     int
	err             error
	prefillDuration time.Duration
	closed          bool
}

// NewSession creates a persistent model-state session.
func (m *Model) NewSession() SessionHandle {
	return &ModelSession{model: m}
}

// Prefill tokenises prompt and stores its KV/logit state in the session.
func (s *ModelSession) Prefill(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForMutation(); err != nil {
		s.err = err
		return err
	}
	s.resetState()
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	start := time.Now()
	var prefillErr error
	if deviceErr := s.model.withDevice(func() {
		tokens := s.model.tokenizer.Encode(prompt)
		if len(tokens) == 0 {
			prefillErr = errPrefillPromptEmpty
			return
		}
		caches := s.model.newCaches()
		logits, err := s.model.prefillTokenBlock(ctx, tokens, caches)
		if err != nil {
			FreeCaches(caches)
			prefillErr = core.E("ModelSession.Prefill", "prefill", err)
			return
		}
		s.caches = caches
		s.logits = logits
		s.tokens = append([]int32(nil), tokens...)
		s.generated = nil
		s.tokenOffset = len(tokens)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if prefillErr != nil {
		s.err = prefillErr
		return prefillErr
	}
	s.prefillDuration = time.Since(start)
	return nil
}

// PrefillChunks tokenises bounded prompt chunks and stores their KV/logit state
// in the session.
func (s *ModelSession) PrefillChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForMutation(); err != nil {
		s.err = err
		return err
	}
	s.resetState()
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	start := time.Now()
	var prefillErr error
	if deviceErr := s.model.withDevice(func() {
		caches := s.model.newCaches()
		tokens, logits, err := s.model.prefillPromptChunksWithPrefix(ctx, chunks, caches, false, "ModelSession.PrefillChunks")
		if err != nil {
			FreeCaches(caches)
			prefillErr = err
			return
		}
		s.caches = caches
		s.logits = logits
		s.tokens = append([]int32(nil), tokens...)
		s.generated = nil
		s.tokenOffset = len(tokens)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if prefillErr != nil {
		s.err = prefillErr
		return prefillErr
	}
	s.prefillDuration = time.Since(start)
	return nil
}

// PrefillTokens stores already-tokenised prompt state in the session.
func (s *ModelSession) PrefillTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForMutation(); err != nil {
		s.err = err
		return err
	}
	s.resetState()
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	start := time.Now()
	var prefillErr error
	if deviceErr := s.model.withDevice(func() {
		promptTokens := append([]int32(nil), tokens...)
		if len(promptTokens) == 0 {
			prefillErr = errPrefillTokensEmpty
			return
		}
		caches := s.model.newCaches()
		logits, err := s.model.prefillTokenBlock(ctx, promptTokens, caches)
		if err != nil {
			FreeCaches(caches)
			prefillErr = core.E("ModelSession.PrefillTokens", "prefill", err)
			return
		}
		s.caches = caches
		s.logits = logits
		s.tokens = promptTokens
		s.generated = nil
		s.tokenOffset = len(promptTokens)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if prefillErr != nil {
		s.err = prefillErr
		return prefillErr
	}
	s.prefillDuration = time.Since(start)
	return nil
}

// AppendPrompt tokenises prompt and appends its KV/logit state to the current
// session without resetting the retained prefix.
func (s *ModelSession) AppendPrompt(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForAppend(); err != nil {
		s.err = err
		return err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	start := time.Now()
	var appendErr error
	if deviceErr := s.model.withDevice(func() {
		tokens := s.model.tokenizer.Encode(prompt)
		if len(s.tokens) > 0 {
			tokens = stripImplicitChunkBOS(s.model.tokenizer, tokens)
		}
		if len(tokens) == 0 {
			appendErr = errAppendPromptEmpty
			return
		}
		logits, err := s.model.prefillTokenBlock(ctx, tokens, s.caches)
		if err != nil {
			appendErr = core.E("ModelSession.AppendPrompt", "prefill", err)
			return
		}
		oldLogits := s.logits
		s.logits = logits
		Free(oldLogits)
		s.tokens = append(s.tokens, tokens...)
		s.tokenOffset += len(tokens)
		s.prefillDuration += time.Since(start)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if appendErr != nil {
		s.err = appendErr
		return appendErr
	}
	return nil
}

// AppendTokens appends already-tokenised prompt state without replaying the
// retained prefix.
func (s *ModelSession) AppendTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForAppend(); err != nil {
		s.err = err
		return err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	start := time.Now()
	var appendErr error
	if deviceErr := s.model.withDevice(func() {
		promptTokens := append([]int32(nil), tokens...)
		if len(s.tokens) > 0 {
			promptTokens = stripImplicitChunkBOS(s.model.tokenizer, promptTokens)
		}
		if len(promptTokens) == 0 {
			appendErr = errAppendTokensEmpty
			return
		}
		logits, err := s.model.prefillTokenBlock(ctx, promptTokens, s.caches)
		if err != nil {
			appendErr = core.E("ModelSession.AppendTokens", "prefill", err)
			return
		}
		oldLogits := s.logits
		s.logits = logits
		Free(oldLogits)
		s.tokens = append(s.tokens, promptTokens...)
		s.tokenOffset += len(promptTokens)
		s.prefillDuration += time.Since(start)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if appendErr != nil {
		s.err = appendErr
		return appendErr
	}
	return nil
}

// AppendPromptChunks tokenises bounded prompt chunks and appends their KV/logit
// state without replaying the retained prefix.
func (s *ModelSession) AppendPromptChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForAppend(); err != nil {
		s.err = err
		return err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	start := time.Now()
	var appendErr error
	if deviceErr := s.model.withDevice(func() {
		tokens, logits, err := s.model.prefillPromptChunksWithPrefix(ctx, chunks, s.caches, len(s.tokens) > 0, "ModelSession.AppendPromptChunks")
		if err != nil {
			appendErr = err
			return
		}
		oldLogits := s.logits
		s.logits = logits
		Free(oldLogits)
		s.tokens = append(s.tokens, tokens...)
		s.tokenOffset += len(tokens)
		s.prefillDuration += time.Since(start)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if appendErr != nil {
		s.err = appendErr
		return appendErr
	}
	return nil
}

// Generate streams tokens from the retained session state.
func (s *ModelSession) Generate(ctx context.Context, cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.err = nil
		if err := s.readyForGeneration(); err != nil {
			s.err = err
			return
		}
		release, err := s.model.acquireSlot(ctx)
		if err != nil {
			s.err = err
			return
		}
		defer release()

		if deviceErr := s.model.withDevice(func() {
			if seedErr := applyGenerationSeed(cfg); seedErr != nil {
				s.err = seedErr
				return
			}
			s.generateLocked(ctx, cfg, yield)
		}); deviceErr != nil {
			s.err = deviceErr
		}
	}
}

func (s *ModelSession) generateLocked(ctx context.Context, cfg GenerateConfig, yield func(Token) bool) {
	totalStart := time.Now()
	ResetPeakMemory()
	sampler := NewSamplerWithSuppression(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK, cfg.SuppressTokens)
	defer CloseSampler(sampler)
	earlySuppressTokens := cfg.SuppressTokens
	earlySampler := sampler
	earlySamplerDistinct := false
	if cfg.MinTokensBeforeStop > 0 {
		earlySuppressTokens = generationStopSuppressionTokens(cfg.SuppressTokens, cfg.StopTokens, s.model.tokenizer)
		if len(earlySuppressTokens) != len(cfg.SuppressTokens) {
			earlySampler = NewSamplerWithSuppression(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK, earlySuppressTokens)
			earlySamplerDistinct = true
		}
	}
	if earlySamplerDistinct {
		defer CloseSampler(earlySampler)
	}
	promptLen := max(s.tokenOffset, len(s.tokens))
	genCount := 0
	var firstTokenDuration time.Duration
	var history []int32
	if cfg.RepeatPenalty > 1.0 {
		history = append([]int32(nil), s.generated...)
	}
	tokenPhases := newTokenPhaseTraceBuffer(cfg)
	emitProbeCachePressure(cfg.ProbeSink, ProbePhasePrefill, promptLen, len(s.generated), -1, s.caches)
	emitProbeMemoryPressure(cfg.ProbeSink, ProbePhasePrefill, -1)

	defer func() {
		decodeDur := time.Since(totalStart)
		processMemory := GetProcessMemory()
		metrics := Metrics{
			PromptTokens:               promptLen,
			GeneratedTokens:            genCount,
			FirstTokenDuration:         firstTokenDuration,
			PrefillDuration:            s.prefillDuration,
			DecodeDuration:             decodeDur,
			TotalDuration:              s.prefillDuration + decodeDur,
			PeakMemoryBytes:            GetPeakMemory(),
			ActiveMemoryBytes:          GetActiveMemory(),
			CacheMemoryBytes:           GetCacheMemory(),
			ProcessVirtualMemoryBytes:  processMemory.VirtualMemoryBytes,
			ProcessResidentMemoryBytes: processMemory.ResidentMemoryBytes,
			ProcessPeakResidentBytes:   processMemory.PeakResidentMemoryBytes,
			CacheProfile:               modelCacheProfile(s.model.model, s.caches),
			TurboQuantKVPayload:        turboQuantKVCachesPayloadEstimate(s.caches),
			TokenPhases:                tokenPhases,
		}
		if s.prefillDuration > 0 {
			metrics.PrefillTokensPerSec = float64(promptLen) / s.prefillDuration.Seconds()
		}
		if decodeDur > 0 {
			metrics.DecodeTokensPerSec = float64(genCount) / decodeDur.Seconds()
		}
		s.model.lastMetrics = metrics
	}()

	for i := range cfg.MaxTokens {
		tracePhases := cfg.TraceTokenPhases
		var phaseStart, phaseLast time.Time
		var phase TokenPhaseTrace
		if tracePhases {
			phaseStart = time.Now()
			phaseLast = phaseStart
			phase = TokenPhaseTrace{Step: i}
		}
		select {
		case <-ctx.Done():
			s.err = ctx.Err()
			return
		default:
		}

		var next *Array
		var sampledID int32
		sampledIDSet := false
		nextEvaluated := false
		stepCfg := cfg
		stepSampler := sampler
		stepSuppressTokens := cfg.SuppressTokens
		if generationStopSuppressionActive(genCount, cfg) {
			stepCfg.SuppressTokens = earlySuppressTokens
			stepSampler = earlySampler
			stepSuppressTokens = earlySuppressTokens
		}
		if nativeGreedyDecodeAvailable(stepCfg, history, s.logits) {
			var err error
			next, err = nativeGreedyDecodeToken(s.logits)
			if err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("native Greedy decode step %d", i), err)
				return
			}
			if tracePhases {
				phase.LogitsDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
		} else {
			lastPos, err := lastTokenLogits(s.logits)
			if err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("last logits step %d", i), err)
				return
			}

			if cfg.RepeatPenalty > 1.0 && len(history) > 0 {
				oldLastPos := lastPos
				lastPos = applyRepeatPenalty(lastPos, history, cfg.RepeatPenalty)
				Free(oldLastPos)
			}
			if tracePhases {
				phase.LogitsDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
			if err := emitProbeLogits(cfg.ProbeSink, ProbePhaseDecode, i, lastPos); err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("probe logits step %d", i), err)
				Free(lastPos)
				return
			}
			if tracePhases && cfg.ProbeSink != nil {
				phase.CacheProbeDuration += time.Since(phaseLast)
			}
			if tracePhases {
				phaseLast = time.Now()
			}

			var sampleErr error
			var sampleTimings sampleTokenTimings
			next, sampledID, sampleTimings, sampleErr = SampleTokenIDWithSuppressionGuard(lastPos, stepSampler, stepSuppressTokens, tracePhases)
			Free(lastPos)
			if sampleErr != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("sample step %d", i), sampleErr)
				return
			}
			sampledIDSet = true
			nextEvaluated = true
			if tracePhases {
				phase.SampleDuration = sampleTimings.Build
				phase.SampleEvalDuration = sampleTimings.Eval
				phase.TokenReadDuration += sampleTimings.TokenRead
				phaseLast = time.Now()
			}
		}
		if !nextEvaluated {
			if err := Eval(next); err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("sample step %d", i), err)
				Free(next)
				return
			}
			if tracePhases {
				phase.SampleEvalDuration += time.Since(phaseLast)
				phaseLast = time.Now()
			}
		}
		detachEvalState(s.logits, s.caches)
		if tracePhases {
			phase.DetachDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}
		id := sampledID
		if !sampledIDSet {
			id = int32(next.Int())
			if tracePhases {
				phase.TokenReadDuration += time.Since(phaseLast)
				phaseLast = time.Now()
			}
		}
		Free(next)
		text := s.model.tokenizer.DecodeToken(id)
		if tracePhases {
			phase.TokenID = id
			if cfg.TraceTokenText {
				phase.TokenText = text
			}
			phase.DecodeTextDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}
		emitProbeToken(cfg.ProbeSink, ProbePhaseDecode, i, id, text, promptLen, len(s.generated)+1)
		if tracePhases {
			phase.ProbeTokenDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}

		stop := s.model.tokenizer.HasEOSToken() && id == s.model.tokenizer.EOSToken()
		stop = stop || slices.Contains(cfg.StopTokens, id)
		if stop {
			if tracePhases {
				phase.FinalToken = true
				tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
			}
			return
		}
		if tracePhases {
			resetNativePhaseTraceEvents()
		}
		if err := s.advanceTokenLocked(ctx, id, i); err != nil {
			s.err = err
			return
		}
		if tracePhases {
			phase.ForwardDuration = time.Since(phaseLast)
			phase.NativeEvents = takeNativePhaseTraceEvents()
			phaseLast = time.Now()
		}
		// Retained sessions use the same lazy next-logits boundary as
		// Model.Generate; prefetching logits plus the dirty K/V handles keeps
		// the next sample step from inheriting the whole decode graph without
		// re-evaluating every historical page.
		var prefetchTimings asyncDecodePrefetchTimings
		var prefetchErr error
		if tracePhases {
			prefetchTimings, prefetchErr = asyncDecodePrefetchWithCachesTrace("ModelSession.Generate", i, "session next logits and dirty KV", s.logits, s.caches)
		} else {
			prefetchErr = asyncDecodePrefetchWithCaches("ModelSession.Generate", i, "session next logits and dirty KV", s.logits, s.caches)
		}
		if prefetchErr != nil {
			s.err = prefetchErr
			return
		}
		if tracePhases {
			phase.PrefetchDuration = time.Since(phaseLast)
			phase.PrefetchLogitsDuration = prefetchTimings.Logits
			phase.PrefetchCacheDuration = prefetchTimings.Cache
			phaseLast = time.Now()
		}
		if cfg.RepeatPenalty > 1.0 {
			history = append(history, id)
		}
		emitProbeCachePressure(cfg.ProbeSink, ProbePhaseDecode, promptLen, len(s.generated), i, s.caches)
		emitProbeMemoryPressure(cfg.ProbeSink, ProbePhaseDecode, i)
		if tracePhases && cfg.ProbeSink != nil {
			phase.CacheProbeDuration += time.Since(phaseLast)
		}
		if tracePhases {
			phaseLast = time.Now()
		}
		genCount++
		if firstTokenDuration == 0 {
			firstTokenDuration = time.Since(totalStart)
		}
		if !yield(Token{ID: id, Text: text}) {
			if tracePhases {
				phase.FinalToken = true
				tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
			}
			return
		}
		if tracePhases {
			phase.YieldDuration = time.Since(phaseLast)
			tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
		}
	}
}

func (s *ModelSession) advanceTokenLocked(ctx context.Context, id int32, step int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	input := FromSingleInt32Matrix(id)

	nextLogits, _ := s.model.forwardLastTokenLogits(input, nil, s.caches)
	Free(input)
	if nextLogits == nil || !nextLogits.Valid() {
		if err := LastError(); err != nil {
			return core.E("ModelSession.Generate", core.Sprintf("decode step %d", step), err)
		}
		return core.E("ModelSession.Generate", core.Sprintf("decode step %d", step), errForwardNilLogits)
	}
	oldLogits := s.logits
	s.logits = nextLogits
	Free(oldLogits)
	s.tokens = append(s.tokens, id)
	s.generated = append(s.generated, id)
	s.tokenOffset++
	return nil
}

// CaptureKV copies the session's current KV cache tensors to CPU memory.
func (s *ModelSession) CaptureKV(ctx context.Context) (*KVSnapshot, error) {
	return s.CaptureKVWithOptions(ctx, KVSnapshotCaptureOptions{})
}

// CaptureKVWithOptions copies the session's current KV cache tensors to CPU
// memory with explicit capture options.
func (s *ModelSession) CaptureKVWithOptions(ctx context.Context, opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForGeneration(); err != nil {
		s.err = err
		return nil, err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return nil, err
	}
	defer release()

	var (
		snapshot *KVSnapshot
		capture  error
	)
	if deviceErr := s.model.withDevice(func() {
		snapshot, capture = s.model.snapshotKVCachesWithOptions(s.tokens, s.caches, opts, s.logits)
		if snapshot != nil {
			snapshot.Generated = append([]int32(nil), s.generated...)
			if s.tokenOffset > 0 {
				snapshot.TokenOffset = s.tokenOffset
			}
		}
	}); deviceErr != nil {
		s.err = deviceErr
		return nil, deviceErr
	}
	if capture != nil {
		s.err = capture
	}
	return snapshot, capture
}

// RangeKVBlocks streams contiguous KV blocks from the retained session state
// without first assembling a full CPU-side KV snapshot.
func (s *ModelSession) RangeKVBlocks(ctx context.Context, blockSize int, opts KVSnapshotCaptureOptions, yield func(KVSnapshotBlock) (bool, error)) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if yield == nil {
		return errKVBlockYieldNil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForGeneration(); err != nil {
		s.err = err
		return err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	var streamErr error
	if deviceErr := s.model.withDevice(func() {
		streamErr = s.rangeKVBlocksLocked(ctx, blockSize, opts, yield)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if streamErr != nil {
		s.err = streamErr
	}
	return streamErr
}

func (s *ModelSession) rangeKVBlocksLocked(ctx context.Context, blockSize int, opts KVSnapshotCaptureOptions, yield func(KVSnapshotBlock) (bool, error)) error {
	if blockSize <= 0 {
		return errSnapshotBlockSize
	}
	seqLen := len(s.tokens)
	if seqLen <= 0 {
		return errKVStreamInvalidTokenState
	}
	snapshotTokens := s.tokens
	baseOffset := max(s.tokenOffset-seqLen, 0)
	boundaries := s.model.kvBlockBoundaries(blockSize, seqLen, s.caches)
	if len(boundaries) < 2 {
		return errKVStreamNoBoundaries
	}
	for i := 0; i < len(boundaries)-1; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		start := boundaries[i]
		end := boundaries[i+1]
		block, err := s.model.snapshotKVCacheBlockWithOptions(snapshotTokens, s.caches, baseOffset, start, end, end == seqLen, opts, s.logits)
		if err != nil {
			return err
		}
		ok, err := yield(KVSnapshotBlock{
			Index:      i,
			TokenStart: start,
			TokenCount: end - start,
			Snapshot:   block,
		})
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
	}
	return nil
}

// RestoreKV replaces the session's retained state with a restorable KV snapshot.
func (s *ModelSession) RestoreKV(ctx context.Context, snapshot *KVSnapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForMutation(); err != nil {
		s.err = err
		return err
	}
	if snapshot == nil {
		err := errSnapshotNil
		s.err = err
		return err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	var restoreErr error
	if deviceErr := s.model.withDevice(func() {
		restoreErr = s.restoreKVLocked(snapshot)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if restoreErr != nil {
		s.err = restoreErr
	}
	return restoreErr
}

// RestoreKVBlocks replaces the session state from streamed KV blocks without
// first assembling a CPU-side full-prefix snapshot.
func (s *ModelSession) RestoreKVBlocks(ctx context.Context, source KVSnapshotBlockSource) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForMutation(); err != nil {
		s.err = err
		return err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return err
	}
	defer release()

	var restoreErr error
	if deviceErr := s.model.withDevice(func() {
		restoreErr = s.restoreKVBlocksLocked(ctx, source)
	}); deviceErr != nil {
		s.err = deviceErr
		return deviceErr
	}
	if restoreErr != nil {
		s.err = restoreErr
		return restoreErr
	}
	return nil
}

func (s *ModelSession) restoreKVBlocksLocked(ctx context.Context, source KVSnapshotBlockSource) error {
	entry, err := s.model.newPromptCacheEntryFromKVBlocks(ctx, source)
	if err != nil {
		return err
	}
	defer entry.free()
	caches, err := restoreSessionCachesTransferringPaged(entry.caches)
	if err != nil {
		return err
	}
	var logits *Array
	if entry.logits != nil {
		logits = Copy(entry.logits)
		if err := Eval(logits); err != nil {
			Free(logits)
			FreeCaches(caches)
			return core.E("ModelSession.RestoreKVBlocks", "restore logits", err)
		}
		Detach(logits)
	}
	s.resetState()
	s.caches = caches
	s.logits = logits
	s.tokens = append([]int32(nil), entry.tokens...)
	s.generated = nil
	s.tokenOffset = len(entry.tokens)
	s.prefillDuration = 0
	return nil
}

func (s *ModelSession) restoreKVLocked(snapshot *KVSnapshot) error {
	if err := s.model.validateKVSnapshot(snapshot); err != nil {
		return err
	}
	caches, err := s.model.restoreKVCachesFromSnapshot(snapshot)
	if err != nil {
		return core.E("ModelSession.RestoreKV", "restore cache", err)
	}
	var logits *Array
	if len(snapshot.Logits) > 0 || len(snapshot.LogitShape) > 0 {
		logits, err = restoreSnapshotLogits(snapshot)
		if err != nil {
			FreeCaches(caches)
			return core.E("ModelSession.RestoreKV", "restore logits", err)
		}
	}
	s.resetState()
	s.caches = caches
	s.logits = logits
	s.tokens = append([]int32(nil), snapshot.Tokens...)
	s.generated = append([]int32(nil), snapshot.Generated...)
	s.tokenOffset = snapshot.TokenOffset
	if s.tokenOffset == 0 {
		s.tokenOffset = len(s.tokens)
	}
	return nil
}

// Fork creates an independent session with a deep-copied model state.
func (s *ModelSession) Fork(ctx context.Context) (SessionHandle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	if err := s.readyForGeneration(); err != nil {
		s.err = err
		return nil, err
	}
	release, err := s.model.acquireSlot(ctx)
	if err != nil {
		s.err = err
		return nil, err
	}
	defer release()

	var forked *ModelSession
	if deviceErr := s.model.withDevice(func() {
		forked, err = s.forkLocked()
	}); deviceErr != nil {
		s.err = deviceErr
		return nil, deviceErr
	}
	if err != nil {
		s.err = err
		return nil, err
	}
	return forked, nil
}

func (s *ModelSession) forkLocked() (*ModelSession, error) {
	snapshots := make([]cacheSnapshot, len(s.caches))
	for i, cache := range s.caches {
		snapshot, ok, err := snapshotSessionCache(cache)
		if err != nil {
			return nil, core.E("ModelSession.Fork", "snapshot cache", err)
		}
		if !ok {
			return nil, errForkCacheNotSnapshotable
		}
		snapshots[i] = snapshot
	}
	caches, err := restoreSessionCachesTransferringPaged(snapshots)
	if err != nil {
		freeCacheSnapshots(snapshots)
		return nil, core.E("ModelSession.Fork", "restore cache", err)
	}
	logits := Copy(s.logits)
	if err := Eval(logits); err != nil {
		Free(logits)
		FreeCaches(caches)
		freeCacheSnapshots(snapshots)
		return nil, core.E("ModelSession.Fork", "copy logits", err)
	}
	Detach(logits)
	freeCacheSnapshots(snapshots)
	return &ModelSession{
		model:           s.model,
		caches:          caches,
		logits:          logits,
		tokens:          append([]int32(nil), s.tokens...),
		generated:       append([]int32(nil), s.generated...),
		tokenOffset:     s.tokenOffset,
		prefillDuration: s.prefillDuration,
	}, nil
}

// Reset releases retained state and leaves the session ready for another prefill.
func (s *ModelSession) Reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = nil
	s.resetState()
}

// Close releases retained state. A closed session cannot be reused.
func (s *ModelSession) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resetState()
	s.closed = true
	s.err = nil
	return nil
}

// Err returns the last session error.
func (s *ModelSession) Err() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *ModelSession) readyForMutation() error {
	if s == nil || s.model == nil || s.model.model == nil || s.model.tokenizer == nil {
		return errSessionNil
	}
	if s.closed {
		return errSessionClosed
	}
	return nil
}

func (s *ModelSession) readyForGeneration() error {
	if err := s.readyForMutation(); err != nil {
		return err
	}
	if len(s.caches) == 0 && len(s.tokens) == 0 && s.tokenOffset <= 0 {
		return errSessionNoPrefill
	}
	if s.logits == nil || !s.logits.Valid() {
		return errSessionNoRestorableLogits
	}
	return nil
}

func (s *ModelSession) readyForAppend() error {
	if err := s.readyForMutation(); err != nil {
		return err
	}
	if len(s.caches) == 0 {
		return errSessionNoPrefill
	}
	return nil
}

func (s *ModelSession) resetState() {
	Free(s.logits)
	s.logits = nil
	FreeCaches(s.caches)
	s.caches = nil
	s.tokens = nil
	s.generated = nil
	s.tokenOffset = 0
	s.prefillDuration = 0
}

func snapshotSessionCache(cache Cache) (cacheSnapshot, bool, error) {
	if cache == nil || cache.State() == nil || cache.Len() <= 0 {
		return cacheSnapshot{}, false, nil
	}
	var (
		state      []*Array
		ownedState []*Array
		snapshot   cacheSnapshot
	)
	switch c := cache.(type) {
	case *RotatingKVCache:
		state = c.orderedState()
		ownedState = state
		snapshot.rotating = true
		snapshot.maxSize = c.maxSize
		snapshot.step = c.step
	case *KVCache:
		state = c.State()
		snapshot.step = c.step
	case *QuantizedKVCache:
		return snapshotQuantizedCache(c, c.Len(), c.Offset())
	case *PagedKVCache:
		return snapshotPagedCache(c, c.Len(), c.Offset())
	case *FixedKVCache:
		state, ownedState = c.ReadState()
		snapshot.mode = KVCacheModeFixed
		snapshot.maxSize = c.maxSize
	default:
		return cacheSnapshot{}, false, nil
	}
	defer Free(ownedState...)
	if len(state) < 2 || !state[0].Valid() || !state[1].Valid() {
		return cacheSnapshot{}, false, nil
	}

	length := cache.Len()
	keys, err := CopyCachePrefix(state[0], length)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	values, err := CopyCachePrefix(state[1], length)
	if err != nil {
		Free(keys)
		return cacheSnapshot{}, false, err
	}
	snapshot.keys = keys
	snapshot.values = values
	snapshot.offset = cache.Offset()
	snapshot.length = length
	return snapshot, true, nil
}

func restoreSessionCaches(snapshots []cacheSnapshot) ([]Cache, error) {
	return restoreSessionCachesWithPagedTransfer(snapshots, false)
}

func restoreSessionCachesTransferringPaged(snapshots []cacheSnapshot) ([]Cache, error) {
	return restoreSessionCachesWithPagedTransfer(snapshots, true)
}

func restoreSessionCachesWithPagedTransfer(snapshots []cacheSnapshot, transferPaged bool) ([]Cache, error) {
	caches := make([]Cache, len(snapshots))
	totalArrays := 0
	for i := range snapshots {
		totalArrays += snapshots[i].arrayCount()
	}
	evalArrays := make([]*Array, 0, totalArrays)
	for i := range snapshots {
		snapshot := &snapshots[i]
		length := snapshotCacheLength(*snapshot)
		if snapshot.keys == nil || snapshot.values == nil || length <= 0 {
			if snapshot.mode != KVCacheModePaged && snapshot.mode != KVCacheModeTurboQuant {
				continue
			}
		}
		if err := validateRestorableCacheSnapshotMode(snapshot.mode); err != nil {
			FreeCaches(caches)
			return nil, err
		}
		if snapshot.mode == KVCacheModeQ8 || snapshot.mode == KVCacheModeKQ8VQ4 {
			cache, next, err := appendRestoreQuantizedCacheSnapshot(evalArrays, *snapshot, length, snapshot.offset)
			if err != nil {
				FreeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = next
			continue
		}
		if snapshot.mode == KVCacheModePaged {
			var (
				cache Cache
				next  []*Array
				err   error
			)
			if transferPaged && canTransferPagedCacheSnapshot(*snapshot, length) {
				cache, next, err = appendRestorePagedCacheSnapshotTransfer(evalArrays, snapshot, length, snapshot.offset)
			} else {
				cache, next, err = appendRestorePagedCacheSnapshot(evalArrays, *snapshot, length, snapshot.offset, "")
			}
			if err != nil {
				FreeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = next
			continue
		}
		if snapshot.mode == KVCacheModeTurboQuant {
			cache, next, err := appendRestoreTurboQuantCacheSnapshot(evalArrays, *snapshot, length, snapshot.offset)
			if err != nil {
				FreeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = next
			continue
		}
		if snapshot.mode == KVCacheModeFixed {
			cache, next, err := appendRestoreFixedCacheSnapshot(evalArrays, *snapshot, length, snapshot.offset, 0, "")
			if err != nil {
				FreeCaches(caches)
				return nil, err
			}
			caches[i] = cache
			evalArrays = next
			continue
		}
		keys, err := CopyCachePrefix(snapshot.keys, length)
		if err != nil {
			FreeCaches(caches)
			return nil, err
		}
		values, err := CopyCachePrefix(snapshot.values, length)
		if err != nil {
			Free(keys)
			FreeCaches(caches)
			return nil, err
		}
		evalArrays = append(evalArrays, keys, values)
		if snapshot.rotating {
			maxSize := snapshot.maxSize
			if maxSize <= 0 {
				maxSize = length
			}
			// idx is the temporal length of valid content (0..maxSize). The
			// rotating cache now keeps storage in temporal order, so the
			// restored content lives at slots [0, length) without further
			// rehydration.
			caches[i] = &RotatingKVCache{
				keys:    keys,
				values:  values,
				offset:  snapshot.offset,
				maxSize: maxSize,
				step:    snapshot.step,
				idx:     length,
			}
			continue
		}
		caches[i] = &KVCache{
			keys:   keys,
			values: values,
			offset: snapshot.offset,
			step:   snapshot.step,
		}
	}
	if err := Eval(evalArrays...); err != nil {
		FreeCaches(caches)
		return nil, core.E("session cache", "restore", err)
	}
	Detach(evalArrays...)
	return caches, nil
}

func snapshotCacheLength(snapshot cacheSnapshot) int {
	if snapshot.length > 0 {
		return snapshot.length
	}
	if snapshot.keys != nil && snapshot.keys.Valid() {
		shape := snapshot.keys.Shape()
		if len(shape) >= 3 {
			return int(shape[2])
		}
	}
	return snapshot.offset
}

func freeCacheSnapshots(snapshots []cacheSnapshot) {
	for _, snapshot := range snapshots {
		freeCacheSnapshot(snapshot)
	}
}

func (m *Model) validateKVSnapshot(snapshot *KVSnapshot) error {
	if snapshot == nil {
		return errSnapshotNil
	}
	if snapshot.Version <= 0 || snapshot.Version > KVSnapshotVersion {
		return errUnsupportedSnapshotVersion
	}
	info := m.Info()
	if snapshot.Architecture != "" && info.Architecture != "" && snapshot.Architecture != info.Architecture {
		return errSnapshotArchMismatch
	}
	if snapshot.SeqLen <= 0 || snapshot.HeadDim <= 0 {
		return errSnapshotInvalidTensorDims
	}
	if len(snapshot.Layers) == 0 {
		return errSnapshotNoLayers
	}
	return nil
}

func (m *Model) restoreKVCachesFromSnapshot(snapshot *KVSnapshot) ([]Cache, error) {
	templates := m.newCaches()
	defer FreeCaches(templates)
	if len(templates) == 0 {
		return nil, errModelNoKVCaches
	}
	snapshots := make([]cacheSnapshot, len(templates))
	populated := make([]bool, len(templates))
	for _, layer := range snapshot.Layers {
		if !kvLayerSnapshotHasState(layer) || layer.CacheIndex < 0 {
			continue
		}
		if layer.CacheIndex >= len(templates) {
			freeCacheSnapshots(snapshots)
			return nil, errSnapshotCacheIndex
		}
		if populated[layer.CacheIndex] {
			continue
		}
		cacheSnapshot, err := cacheSnapshotFromKVLayer(snapshot, layer, templates[layer.CacheIndex])
		if err != nil {
			freeCacheSnapshots(snapshots)
			return nil, err
		}
		snapshots[layer.CacheIndex] = cacheSnapshot
		populated[layer.CacheIndex] = true
	}
	for i, ok := range populated {
		if !ok {
			freeCacheSnapshots(snapshots)
			return nil, core.E("ModelSession.RestoreKV", core.Sprintf("missing cache %d", i), nil)
		}
	}
	caches, err := restoreSessionCachesTransferringPaged(snapshots)
	freeCacheSnapshots(snapshots)
	return caches, err
}

func cacheSnapshotFromKVLayer(snapshot *KVSnapshot, layer KVLayerSnapshot, template Cache) (cacheSnapshot, error) {
	if snapshot == nil {
		return cacheSnapshot{}, errSnapshotNil
	}
	globalSeqLen := snapshot.SeqLen
	if globalSeqLen <= 0 {
		globalSeqLen = len(snapshot.Tokens)
	}
	if globalSeqLen <= 0 {
		return cacheSnapshot{}, errSnapshotNoSeqLen
	}
	if len(layer.TurboQuantPayloads) > 0 || layer.CacheMode == KVCacheModeTurboQuant {
		return cacheSnapshotFromTurboQuantKVLayer(snapshot, layer, template, globalSeqLen)
	}
	keyArray, valueArray, seqLen, err := kvLayerArrays(snapshot, layer, globalSeqLen)
	if err != nil {
		return cacheSnapshot{}, err
	}
	offset := snapshot.TokenOffset
	if offset <= 0 {
		offset = globalSeqLen
	}
	result := cacheSnapshot{
		keys:   keyArray,
		values: valueArray,
		offset: offset,
		length: seqLen,
		step:   defaultPagedKVPageSize,
	}
	switch c := template.(type) {
	case *RotatingKVCache:
		result.rotating = true
		result.maxSize = c.maxSize
		result.step = c.step
	case *KVCache:
		result.step = c.step
	case *QuantizedKVCache:
		if c.keyBits == 8 && c.valueBits == 8 {
			result.mode = KVCacheModeQ8
			result.keyDtype = keyArray.Dtype()
			result.valueDtype = valueArray.Dtype()
			result.keyBits = c.keyBits
			result.valueBits = c.valueBits
			result.keys, result.keyScale, result.keyShape = quantizeCacheArray(keyArray, c.keyBits)
			result.values, result.valueScale, result.valueShape = quantizeCacheArray(valueArray, c.valueBits)
			Free(keyArray, valueArray)
		}
		result.step = c.step
		if c.maxSize > 0 {
			result.rotating = true
			result.maxSize = c.maxSize
		}
	case *FixedKVCache:
		if c.maxSize > 0 && seqLen > c.maxSize {
			Free(keyArray, valueArray)
			return cacheSnapshot{}, errSnapshotExceedsFixedCap
		}
		result.mode = KVCacheModeFixed
		result.maxSize = c.maxSize
		result.storageDType = c.storageDType
		result.hasStorageDType = c.hasStorageDType
	case *PagedKVCache:
		pagesK, pagesV, adopted, err := pageCacheArrays(keyArray, valueArray, c.pageSize)
		if err != nil {
			Free(keyArray, valueArray)
			return cacheSnapshot{}, err
		}
		result.mode = KVCacheModePaged
		result.kPages = pagesK
		result.vPages = pagesV
		if !adopted {
			Free(keyArray, valueArray)
		}
		result.keys = nil
		result.values = nil
		result.step = c.pageSize
		result.storageDType = c.storageDType
		result.hasStorageDType = c.hasStorageDType
		if c.maxSize > 0 {
			result.rotating = true
			result.maxSize = c.maxSize
		}
	case nil:
	default:
		Free(keyArray, valueArray)
		return cacheSnapshot{}, errUnsupportedKVCacheType
	}
	return result, nil
}

func kvLayerSnapshotHasState(layer KVLayerSnapshot) bool {
	return len(layer.TurboQuantPayloads) > 0 || len(layer.Heads) > 0 || (len(layer.KeyBytes) > 0 && len(layer.ValueBytes) > 0)
}

func cacheSnapshotFromTurboQuantKVLayer(snapshot *KVSnapshot, layer KVLayerSnapshot, template Cache, globalSeqLen int) (cacheSnapshot, error) {
	if len(layer.TurboQuantPayloads) == 0 {
		return cacheSnapshot{}, errTurboQuantSnapshotLayout
	}
	tokenLen := turboQuantKVPayloadTokenLen(layer.TurboQuantPayloads)
	if tokenLen <= 0 {
		return cacheSnapshot{}, errTurboQuantSnapshotLayout
	}
	offset := snapshot.TokenOffset
	if offset <= 0 {
		offset = globalSeqLen
	}
	result := cacheSnapshot{
		mode:          KVCacheModeTurboQuant,
		turboPayloads: turboQuantKVClonePayloads(layer.TurboQuantPayloads),
		offset:        offset,
		length:        tokenLen,
		step:          defaultTurboQuantKVCachePageSize,
	}
	if len(layer.TurboQuantPayloads) > 0 && layer.TurboQuantPayloads[0].Layout.PageSize > 0 {
		result.step = layer.TurboQuantPayloads[0].Layout.PageSize
	}
	if c, ok := template.(*TurboQuantKVCache); ok && c != nil {
		result.maxSize = c.maxSize
		if c.maxSize > 0 {
			result.rotating = true
		}
	}
	return result, nil
}

func kvLayerArrays(snapshot *KVSnapshot, layer KVLayerSnapshot, globalSeqLen int) (*Array, *Array, int, error) {
	if len(layer.TurboQuantPayloads) > 0 {
		keyArray, valueArray, err := decodeTurboQuantKVSnapshotFloatArrays(layer.TurboQuantPayloads)
		if err != nil {
			return nil, nil, 0, err
		}
		return keyArray, valueArray, turboQuantKVPayloadTokenLen(layer.TurboQuantPayloads), nil
	}
	if len(layer.KeyBytes) > 0 || len(layer.ValueBytes) > 0 {
		keyArray, valueArray, seqLen, err := kvLayerNativeSlabArrays(layer)
		if err != nil {
			return nil, nil, 0, err
		}
		return keyArray, valueArray, seqLen, nil
	}

	numHeads := len(layer.Heads)
	if numHeads <= 0 {
		return nil, nil, 0, errSnapshotLayerNoHeads
	}
	seqLen, keyDim, valueDim, err := inferSnapshotLayerCacheShape(layer.Heads, globalSeqLen, snapshot.HeadDim)
	if err != nil {
		return nil, nil, 0, err
	}

	for _, head := range layer.Heads {
		if err := validateSnapshotHeadTensorCacheShape(head, seqLen, keyDim, true); err != nil {
			return nil, nil, 0, err
		}
		if err := validateSnapshotHeadTensorCacheShape(head, seqLen, valueDim, false); err != nil {
			return nil, nil, 0, err
		}
	}

	keyArray, keyNative, err := kvLayerNativeArray(layer.Heads, seqLen, keyDim, true)
	if err != nil {
		return nil, nil, 0, err
	}
	if !keyNative {
		keys := make([]float32, 0, numHeads*seqLen*keyDim)
		for _, head := range layer.Heads {
			keys = append(keys, head.Key...)
		}
		keyArray = FromValues(keys, 1, numHeads, seqLen, keyDim)
	}
	valueArray, valueNative, err := kvLayerNativeArray(layer.Heads, seqLen, valueDim, false)
	if err != nil {
		Free(keyArray)
		return nil, nil, 0, err
	}
	if !valueNative {
		values := make([]float32, 0, numHeads*seqLen*valueDim)
		for _, head := range layer.Heads {
			values = append(values, head.Value...)
		}
		valueArray = FromValues(values, 1, numHeads, seqLen, valueDim)
	}
	return keyArray, valueArray, seqLen, nil
}

// fromOwnedRawBytes pins a private clone of raw. Snapshot byte slices may be
// backed by transient decode buffers, and a restored cache outlives them, so
// cache storage always owns its memory.
func fromOwnedRawBytes(raw []byte, shape []int, dtype DType) (*Array, error) {
	return fromPinnedRawBytes(slices.Clone(raw), shape, dtype)
}

func kvLayerNativeSlabArrays(layer KVLayerSnapshot) (*Array, *Array, int, error) {
	keyShape, keySeqLen, err := validateKVLayerNativeSlab(layer.KeyBytes, layer.KeyDType, layer.KeyShape)
	if err != nil {
		return nil, nil, 0, core.E("mlx: KV snapshot native layer key", "validate", err)
	}
	valueShape, valueSeqLen, err := validateKVLayerNativeSlab(layer.ValueBytes, layer.ValueDType, layer.ValueShape)
	if err != nil {
		return nil, nil, 0, core.E("mlx: KV snapshot native layer value", "validate", err)
	}
	if keySeqLen != valueSeqLen || keyShape[0] != valueShape[0] || keyShape[1] != valueShape[1] {
		return nil, nil, 0, errSnapshotNativeKVShapesDiffer
	}
	var keyShapeBuf [MaxTensorRank]int
	keyArray, err := fromOwnedRawBytes(layer.KeyBytes, int32ShapeToIntsInto(keyShapeBuf[:0], keyShape), layer.KeyDType)
	if err != nil {
		return nil, nil, 0, err
	}
	var valueShapeBuf [MaxTensorRank]int
	valueArray, err := fromOwnedRawBytes(layer.ValueBytes, int32ShapeToIntsInto(valueShapeBuf[:0], valueShape), layer.ValueDType)
	if err != nil {
		Free(keyArray)
		return nil, nil, 0, err
	}
	return keyArray, valueArray, keySeqLen, nil
}

func validateKVLayerNativeSlab(raw []byte, dtype DType, shape []int32) ([]int32, int, error) {
	if len(raw) == 0 || len(shape) != 4 {
		return nil, 0, errMissingNativeSlab
	}
	byteSize := DTypeByteSize(dtype)
	if byteSize <= 0 {
		return nil, 0, errUnsupportedDtype
	}
	count := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, 0, errInvalidShape
		}
		count *= int(dim)
	}
	if count*byteSize != len(raw) {
		return nil, 0, errByteLenShape
	}
	return shape, int(shape[2]), nil
}

func int32ShapeToInts(shape []int32) []int {
	out := make([]int, len(shape))
	for i, dim := range shape {
		out[i] = int(dim)
	}
	return out
}

func int32ShapeToIntsInto(dst []int, shape []int32) []int {
	for _, dim := range shape {
		dst = append(dst, int(dim))
	}
	return dst
}

func inferSnapshotLayerCacheShape(heads []KVHeadSnapshot, globalSeqLen, fallbackHeadDim int) (int, int, int, error) {
	if len(heads) == 0 {
		return 0, 0, 0, errSnapshotLayerNoHeads
	}
	keyLen, keyDim := inferSnapshotHeadTensorCacheShape(heads[0], globalSeqLen, fallbackHeadDim, true)
	valueLen, valueDim := inferSnapshotHeadTensorCacheShape(heads[0], globalSeqLen, fallbackHeadDim, false)
	if keyLen <= 0 || keyDim <= 0 || valueLen <= 0 || valueDim <= 0 {
		return 0, 0, 0, errSnapshotInvalidHeadDims
	}
	if keyLen != valueLen {
		return 0, 0, 0, errSnapshotKVLenDiffer
	}
	return keyLen, keyDim, valueDim, nil
}

func inferSnapshotHeadTensorCacheShape(head KVHeadSnapshot, globalSeqLen, fallbackHeadDim int, key bool) (int, int) {
	values := head.Value
	if key {
		values = head.Key
	}
	if len(values) > 0 {
		return inferSnapshotTensorElementCacheShape(len(values), globalSeqLen, fallbackHeadDim)
	}
	raw, dtype := kvHeadRawTensor(head, key)
	bytesPerValue := DTypeByteSize(dtype)
	if len(raw) > 0 && bytesPerValue > 0 && len(raw)%bytesPerValue == 0 {
		return inferSnapshotTensorElementCacheShape(len(raw)/bytesPerValue, globalSeqLen, fallbackHeadDim)
	}
	return 0, 0
}

func inferSnapshotTensorCacheShape(values []float32, globalSeqLen, fallbackHeadDim int) (int, int) {
	if len(values) == 0 {
		return 0, 0
	}
	return inferSnapshotTensorElementCacheShape(len(values), globalSeqLen, fallbackHeadDim)
}

func inferSnapshotTensorElementCacheShape(elements, globalSeqLen, fallbackHeadDim int) (int, int) {
	if elements <= 0 {
		return 0, 0
	}
	if globalSeqLen > 0 && elements%globalSeqLen == 0 {
		return globalSeqLen, elements / globalSeqLen
	}
	if fallbackHeadDim > 0 && elements%fallbackHeadDim == 0 {
		return elements / fallbackHeadDim, fallbackHeadDim
	}
	return 0, 0
}

func validateSnapshotHeadTensorCacheShape(head KVHeadSnapshot, seqLen, dim int, key bool) error {
	if seqLen <= 0 || dim <= 0 {
		return errSnapshotInvalidHeadDims
	}
	values := head.Value
	if key {
		values = head.Key
	}
	if len(values) > 0 && len(values) != seqLen*dim {
		if key {
			return errSnapshotKeyTensorSize
		}
		return errSnapshotValueTensorSize
	}
	raw, dtype := kvHeadRawTensor(head, key)
	if len(raw) == 0 {
		if len(values) == 0 {
			if key {
				return errSnapshotKeyTensorSize
			}
			return errSnapshotValueTensorSize
		}
		return nil
	}
	bytesPerValue := DTypeByteSize(dtype)
	if bytesPerValue <= 0 || len(raw) != seqLen*dim*bytesPerValue {
		if key {
			return errSnapshotNativeKeySize
		}
		return errSnapshotNativeValueSize
	}
	return nil
}

func kvLayerNativeArray(heads []KVHeadSnapshot, seqLen, headDim int, key bool) (*Array, bool, error) {
	raw, dtype, ok, err := kvLayerRawTensor(heads, seqLen, headDim, key)
	if err != nil || !ok {
		return nil, ok, err
	}
	array, err := fromOwnedRawBytes(raw, []int{1, len(heads), seqLen, headDim}, dtype)
	if err != nil {
		return nil, false, err
	}
	return array, true, nil
}

func kvLayerRawTensor(heads []KVHeadSnapshot, seqLen, headDim int, key bool) ([]byte, DType, bool, error) {
	if len(heads) == 0 {
		return nil, 0, false, nil
	}
	firstRaw, firstDType := kvHeadRawTensor(heads[0], key)
	if len(firstRaw) == 0 {
		for _, head := range heads[1:] {
			raw, _ := kvHeadRawTensor(head, key)
			if len(raw) > 0 {
				return nil, 0, false, errSnapshotMixedTensorHeads
			}
		}
		return nil, 0, false, nil
	}
	bytesPerValue := DTypeByteSize(firstDType)
	if bytesPerValue <= 0 {
		return nil, 0, false, errUnsupportedNativeDtype
	}
	expectedBytes := seqLen * headDim * bytesPerValue
	if len(heads) == 1 {
		if len(firstRaw) != expectedBytes {
			return nil, 0, false, errSnapshotNativeByteLen
		}
		return firstRaw, firstDType, true, nil
	}
	raw := make([]byte, 0, len(heads)*expectedBytes)
	for _, head := range heads {
		headRaw, headDType := kvHeadRawTensor(head, key)
		if len(headRaw) == 0 {
			return nil, 0, false, errSnapshotMixedTensorHeads
		}
		if headDType != firstDType {
			return nil, 0, false, errSnapshotNativeDtypeMismatch
		}
		if len(headRaw) != expectedBytes {
			return nil, 0, false, errSnapshotNativeByteLen
		}
		raw = append(raw, headRaw...)
	}
	return raw, firstDType, true, nil
}

func kvHeadRawTensor(head KVHeadSnapshot, key bool) ([]byte, DType) {
	if key {
		return head.KeyBytes, head.KeyDType
	}
	return head.ValueBytes, head.ValueDType
}

func inferSnapshotHeadDim(values []float32, seqLen int) int {
	if seqLen <= 0 || len(values)%seqLen != 0 {
		return 0
	}
	return len(values) / seqLen
}

func restoreSnapshotLogits(snapshot *KVSnapshot) (*Array, error) {
	if snapshot == nil {
		return nil, errSnapshotNil
	}
	if len(snapshot.Logits) == 0 || len(snapshot.LogitShape) == 0 {
		return nil, errSnapshotNoRestorableLogits
	}
	shape := make([]int, len(snapshot.LogitShape))
	count := 1
	for i, dim := range snapshot.LogitShape {
		if dim <= 0 {
			return nil, errSnapshotLogitShape
		}
		shape[i] = int(dim)
		count *= int(dim)
	}
	if count != len(snapshot.Logits) {
		return nil, errSnapshotLogitsShapeMismatch
	}
	logits := FromValues(snapshot.Logits, shape...)
	if err := Eval(logits); err != nil {
		Free(logits)
		return nil, err
	}
	Detach(logits)
	return logits, nil
}
