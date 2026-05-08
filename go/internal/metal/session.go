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

// SessionHandle is the native model-state session interface.
type SessionHandle interface {
	Prefill(context.Context, string) error
	Generate(context.Context, GenerateConfig) iter.Seq[Token]
	CaptureKV(context.Context) (*KVSnapshot, error)
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
			prefillErr = core.NewError("ModelSession.Prefill: empty prompt after tokenisation")
			return
		}
		caches := s.model.newCaches()
		logits, err := s.model.prefillTokenBlock(ctx, tokens, caches)
		if err != nil {
			freeCaches(caches)
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
			s.generateLocked(ctx, cfg, yield)
		}); deviceErr != nil {
			s.err = deviceErr
		}
	}
}

func (s *ModelSession) generateLocked(ctx context.Context, cfg GenerateConfig, yield func(Token) bool) {
	totalStart := time.Now()
	ResetPeakMemory()
	sampler := newSampler(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK)
	promptLen := len(s.tokens)
	if s.tokenOffset > promptLen {
		promptLen = s.tokenOffset
	}
	genCount := 0
	history := append([]int32(nil), s.generated...)
	emitProbeCachePressure(cfg.ProbeSink, ProbePhasePrefill, promptLen, len(s.generated), -1, s.caches)
	emitProbeMemoryPressure(cfg.ProbeSink, ProbePhasePrefill, -1)

	defer func() {
		decodeDur := time.Since(totalStart)
		metrics := Metrics{
			PromptTokens:      promptLen,
			GeneratedTokens:   genCount,
			PrefillDuration:   s.prefillDuration,
			DecodeDuration:    decodeDur,
			TotalDuration:     s.prefillDuration + decodeDur,
			PeakMemoryBytes:   GetPeakMemory(),
			ActiveMemoryBytes: GetActiveMemory(),
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
		select {
		case <-ctx.Done():
			s.err = ctx.Err()
			return
		default:
		}

		l1 := SliceAxis(s.logits, 1, int32(s.logits.Dim(1)-1), int32(s.logits.Dim(1)))
		lastPos := Reshape(l1, 1, int32(l1.Dim(2)))
		Free(l1)

		if cfg.RepeatPenalty > 1.0 && len(history) > 0 {
			oldLastPos := lastPos
			lastPos = applyRepeatPenalty(lastPos, history, cfg.RepeatPenalty)
			Free(oldLastPos)
		}

		if err := emitProbeLogits(cfg.ProbeSink, ProbePhaseDecode, i, lastPos); err != nil {
			s.err = core.E("ModelSession.Generate", core.Sprintf("probe logits step %d", i), err)
			Free(lastPos)
			return
		}

		next := sampler.Sample(lastPos)
		if err := Eval(next); err != nil {
			s.err = core.E("ModelSession.Generate", core.Sprintf("sample step %d", i), err)
			Free(lastPos, next)
			return
		}
		id := int32(next.Int())
		Free(lastPos, next)
		text := s.model.tokenizer.DecodeToken(id)
		emitProbeToken(cfg.ProbeSink, ProbePhaseDecode, i, id, text, promptLen, len(s.generated)+1)

		stop := s.model.tokenizer.HasEOSToken() && id == s.model.tokenizer.EOSToken()
		stop = stop || slices.Contains(cfg.StopTokens, id)
		if err := s.advanceTokenLocked(ctx, id, i); err != nil {
			s.err = err
			return
		}
		history = append(history, id)
		emitProbeCachePressure(cfg.ProbeSink, ProbePhaseDecode, promptLen, len(s.generated), i, s.caches)
		emitProbeMemoryPressure(cfg.ProbeSink, ProbePhaseDecode, i)
		if stop {
			return
		}

		genCount++
		if !yield(Token{ID: id, Text: text}) {
			return
		}
	}
}

func (s *ModelSession) advanceTokenLocked(ctx context.Context, id int32, step int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	vInput := FromValues([]int32{id}, 1)
	input := Reshape(vInput, 1, 1)
	Free(vInput)

	nextLogits := s.model.model.Forward(input, s.caches)
	Free(input)
	if err := Eval(nextLogits); err != nil {
		Free(nextLogits)
		return core.E("ModelSession.Generate", core.Sprintf("decode step %d", step), err)
	}
	oldLogits := s.logits
	s.logits = nextLogits
	Free(oldLogits)
	detachEvalState(s.logits, s.caches)
	s.tokens = append(s.tokens, id)
	s.generated = append(s.generated, id)
	s.tokenOffset++
	return nil
}

// CaptureKV copies the session's current KV cache tensors to CPU memory.
func (s *ModelSession) CaptureKV(ctx context.Context) (*KVSnapshot, error) {
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
		snapshot, capture = s.model.snapshotKVCaches(s.tokens, s.caches, s.logits)
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
		err := core.NewError("mlx: KV snapshot is nil")
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

func (s *ModelSession) restoreKVLocked(snapshot *KVSnapshot) error {
	if err := s.model.validateKVSnapshot(snapshot); err != nil {
		return err
	}
	caches, err := s.model.restoreKVCachesFromSnapshot(snapshot)
	if err != nil {
		return core.E("ModelSession.RestoreKV", "restore cache", err)
	}
	logits, err := restoreSnapshotLogits(snapshot)
	if err != nil {
		freeCaches(caches)
		return core.E("ModelSession.RestoreKV", "restore logits", err)
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
			return nil, core.NewError("ModelSession.Fork: cache is not snapshotable")
		}
		snapshots[i] = snapshot
	}
	caches, err := restoreSessionCaches(snapshots)
	if err != nil {
		freeCacheSnapshots(snapshots)
		return nil, core.E("ModelSession.Fork", "restore cache", err)
	}
	logits := Copy(s.logits)
	if err := Eval(logits); err != nil {
		Free(logits)
		freeCaches(caches)
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
		return core.NewError("mlx: model session is nil")
	}
	if s.closed {
		return core.NewError("mlx: model session is closed")
	}
	return nil
}

func (s *ModelSession) readyForGeneration() error {
	if err := s.readyForMutation(); err != nil {
		return err
	}
	if len(s.caches) == 0 || s.logits == nil || !s.logits.Valid() {
		return core.NewError("mlx: model session has no prefilled state")
	}
	return nil
}

func (s *ModelSession) resetState() {
	Free(s.logits)
	s.logits = nil
	freeCaches(s.caches)
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
		state, ownedState = c.ReadState()
		snapshot.step = c.step
		if c.maxSize > 0 {
			snapshot.rotating = true
			snapshot.maxSize = c.maxSize
		}
	case *PagedKVCache:
		state, ownedState = c.ReadState()
		snapshot.step = c.pageSize
		if c.maxSize > 0 {
			snapshot.rotating = true
			snapshot.maxSize = c.maxSize
		}
	default:
		return cacheSnapshot{}, false, nil
	}
	defer Free(ownedState...)
	if len(state) < 2 || !state[0].Valid() || !state[1].Valid() {
		return cacheSnapshot{}, false, nil
	}

	length := cache.Len()
	keys, err := copyCachePrefix(state[0], length)
	if err != nil {
		return cacheSnapshot{}, false, err
	}
	values, err := copyCachePrefix(state[1], length)
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
	caches := make([]Cache, len(snapshots))
	var evalArrays []*Array
	for i, snapshot := range snapshots {
		length := snapshotCacheLength(snapshot)
		if snapshot.keys == nil || snapshot.values == nil || length <= 0 {
			continue
		}
		keys, err := copyCachePrefix(snapshot.keys, length)
		if err != nil {
			freeCaches(caches)
			return nil, err
		}
		values, err := copyCachePrefix(snapshot.values, length)
		if err != nil {
			Free(keys)
			freeCaches(caches)
			return nil, err
		}
		evalArrays = append(evalArrays, keys, values)
		if snapshot.rotating {
			maxSize := snapshot.maxSize
			if maxSize <= 0 {
				maxSize = length
			}
			idx := length
			if idx >= maxSize {
				idx = idx % maxSize
			}
			caches[i] = &RotatingKVCache{
				keys:    keys,
				values:  values,
				offset:  snapshot.offset,
				maxSize: maxSize,
				step:    snapshot.step,
				idx:     idx,
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
		freeCaches(caches)
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
		Free(snapshot.keys, snapshot.values)
	}
}

func (m *Model) validateKVSnapshot(snapshot *KVSnapshot) error {
	if snapshot == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	if snapshot.Version <= 0 || snapshot.Version > KVSnapshotVersion {
		return core.NewError("mlx: unsupported KV snapshot version")
	}
	info := m.Info()
	if snapshot.Architecture != "" && info.Architecture != "" && snapshot.Architecture != info.Architecture {
		return core.NewError("mlx: KV snapshot architecture does not match model")
	}
	if snapshot.SeqLen <= 0 || snapshot.HeadDim <= 0 {
		return core.NewError("mlx: KV snapshot has invalid tensor dimensions")
	}
	if len(snapshot.Layers) == 0 {
		return core.NewError("mlx: KV snapshot has no layers")
	}
	if len(snapshot.Logits) == 0 || len(snapshot.LogitShape) == 0 {
		return core.NewError("mlx: KV snapshot has no restorable logits")
	}
	return nil
}

func (m *Model) restoreKVCachesFromSnapshot(snapshot *KVSnapshot) ([]Cache, error) {
	templates := m.newCaches()
	defer freeCaches(templates)
	if len(templates) == 0 {
		return nil, core.NewError("mlx: model has no KV caches")
	}
	snapshots := make([]cacheSnapshot, len(templates))
	populated := make([]bool, len(templates))
	for _, layer := range snapshot.Layers {
		if len(layer.Heads) == 0 || layer.CacheIndex < 0 {
			continue
		}
		if layer.CacheIndex >= len(templates) {
			freeCacheSnapshots(snapshots)
			return nil, core.NewError("mlx: KV snapshot cache index exceeds model cache count")
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
	caches, err := restoreSessionCaches(snapshots)
	freeCacheSnapshots(snapshots)
	return caches, err
}

func cacheSnapshotFromKVLayer(snapshot *KVSnapshot, layer KVLayerSnapshot, template Cache) (cacheSnapshot, error) {
	if snapshot == nil {
		return cacheSnapshot{}, core.NewError("mlx: KV snapshot is nil")
	}
	seqLen := snapshot.SeqLen
	if seqLen <= 0 {
		seqLen = len(snapshot.Tokens)
	}
	if seqLen <= 0 {
		return cacheSnapshot{}, core.NewError("mlx: KV snapshot has no sequence length")
	}
	numHeads := len(layer.Heads)
	if numHeads <= 0 {
		return cacheSnapshot{}, core.NewError("mlx: KV snapshot layer has no heads")
	}
	keyDim := snapshot.HeadDim
	if keyDim <= 0 {
		keyDim = inferSnapshotHeadDim(layer.Heads[0].Key, seqLen)
	}
	valueDim := inferSnapshotHeadDim(layer.Heads[0].Value, seqLen)
	if keyDim <= 0 || valueDim <= 0 {
		return cacheSnapshot{}, core.NewError("mlx: KV snapshot has invalid head dimensions")
	}

	keys := make([]float32, 0, numHeads*seqLen*keyDim)
	values := make([]float32, 0, numHeads*seqLen*valueDim)
	for _, head := range layer.Heads {
		if len(head.Key) != seqLen*keyDim {
			return cacheSnapshot{}, core.NewError("mlx: KV snapshot key tensor has unexpected size")
		}
		if len(head.Value) != seqLen*valueDim {
			return cacheSnapshot{}, core.NewError("mlx: KV snapshot value tensor has unexpected size")
		}
		keys = append(keys, head.Key...)
		values = append(values, head.Value...)
	}

	keyArray := FromValues(keys, 1, numHeads, seqLen, keyDim)
	valueArray := FromValues(values, 1, numHeads, seqLen, valueDim)
	offset := snapshot.TokenOffset
	if offset <= 0 {
		offset = seqLen
	}
	result := cacheSnapshot{
		keys:   keyArray,
		values: valueArray,
		offset: offset,
		length: seqLen,
		step:   256,
	}
	switch c := template.(type) {
	case *RotatingKVCache:
		result.rotating = true
		result.maxSize = c.maxSize
		result.step = c.step
	case *KVCache:
		result.step = c.step
	case nil:
	default:
		Free(keyArray, valueArray)
		return cacheSnapshot{}, core.NewError("mlx: unsupported KV cache type")
	}
	return result, nil
}

func inferSnapshotHeadDim(values []float32, seqLen int) int {
	if seqLen <= 0 || len(values)%seqLen != 0 {
		return 0
	}
	return len(values) / seqLen
}

func restoreSnapshotLogits(snapshot *KVSnapshot) (*Array, error) {
	if snapshot == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
	}
	if len(snapshot.Logits) == 0 || len(snapshot.LogitShape) == 0 {
		return nil, core.NewError("mlx: KV snapshot has no restorable logits")
	}
	shape := make([]int, len(snapshot.LogitShape))
	count := 1
	for i, dim := range snapshot.LogitShape {
		if dim <= 0 {
			return nil, core.NewError("mlx: KV snapshot logit shape is invalid")
		}
		shape[i] = int(dim)
		count *= int(dim)
	}
	if count != len(snapshot.Logits) {
		return nil, core.NewError("mlx: KV snapshot logits do not match shape")
	}
	logits := FromValues(snapshot.Logits, shape...)
	if err := Eval(logits); err != nil {
		Free(logits)
		return nil, err
	}
	Detach(logits)
	return logits, nil
}
