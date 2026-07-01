// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"
	"math"
	"reflect"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/native"
	pkgtokenizer "dappco.re/go/mlx/pkg/tokenizer"
)

type promptCacheTextTokenModel struct {
	session *promptCacheTextSession
}

func (m *promptCacheTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *promptCacheTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *promptCacheTextTokenModel) Head([]byte) ([]byte, error) { return make([]byte, 64), nil }

func (m *promptCacheTextTokenModel) Vocab() int { return 32 }

func (m *promptCacheTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type promptCacheTextSession struct {
	warmed              []int32
	generated           []int32
	generatedMax        int
	generatedEOS        int
	sampledStopTokens   []int32
	sampledParams       model.SampleParams
	generateEachCalls   int
	cachedEachCalls     int
	cachedSampledCalls  int
	cachedSuppressCalls int
	streamedYieldCount  int
	clearCallCount      int
}

func (s *promptCacheTextSession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *promptCacheTextSession) WarmPromptCache(ids []int32) error {
	s.warmed = append(s.warmed[:0], ids...)
	return nil
}

func (s *promptCacheTextSession) GenerateCached(ids []int32, maxNew, eos int) ([]int32, error) {
	s.generated = append(s.generated[:0], ids...)
	s.generatedMax = maxNew
	s.generatedEOS = eos
	return []int32{10, 11}, nil
}

func (s *promptCacheTextSession) GenerateEach(ids []int32, maxNew, eos int, yield func(int32) bool) ([]int32, error) {
	s.generated = append(s.generated[:0], ids...)
	s.generatedMax = maxNew
	s.generatedEOS = eos
	s.generateEachCalls++
	return s.streamIDs([]int32{12, 13}, yield), nil
}

func (s *promptCacheTextSession) GenerateCachedEach(ids []int32, maxNew, eos int, yield func(int32) bool) ([]int32, error) {
	s.generated = append(s.generated[:0], ids...)
	s.generatedMax = maxNew
	s.generatedEOS = eos
	s.cachedEachCalls++
	return s.streamIDs([]int32{7, 8}, yield), nil
}

func (s *promptCacheTextSession) GenerateCachedSampledEach(ids []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generated = append(s.generated[:0], ids...)
	s.generatedMax = maxNew
	s.sampledStopTokens = append(s.sampledStopTokens[:0], stopTokens...)
	s.sampledParams = params
	s.cachedSampledCalls++
	out := []int32{9, 10}
	gen := make([]int32, 0, len(out))
	for _, id := range out {
		if transform != nil {
			id = transform(id)
		}
		gen = append(gen, id)
		s.streamedYieldCount++
		if yield != nil && !yield(id) {
			break
		}
	}
	return gen, nil
}

func (s *promptCacheTextSession) GenerateCachedEachWithSuppressionAndTransform(ids []int32, maxNew, eos int, suppress []int32, transform native.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generated = append(s.generated[:0], ids...)
	s.generatedMax = maxNew
	s.generatedEOS = eos
	s.cachedSuppressCalls++
	out := make([]int32, 0, maxNew)
	for _, id := range []int32{7, 8} {
		if tokenInSet(id, suppress) {
			continue
		}
		if transform != nil {
			id = transform(id)
		}
		out = append(out, id)
		s.streamedYieldCount++
		if yield != nil && !yield(id) {
			break
		}
		if eos >= 0 && int(id) == eos {
			break
		}
		if len(out) >= maxNew {
			break
		}
	}
	return out, nil
}

func (s *promptCacheTextSession) streamIDs(ids []int32, yield func(int32) bool) []int32 {
	out := make([]int32, 0, len(ids))
	for _, id := range ids {
		out = append(out, id)
		s.streamedYieldCount++
		if yield != nil && !yield(id) {
			break
		}
	}
	return out
}

func (s *promptCacheTextSession) ClearPromptCache() {
	s.clearCallCount++
	s.warmed = nil
}

func (s *promptCacheTextSession) CachedPrefixLen() int { return len(s.warmed) }

type sampledStopTextTokenModel struct {
	session   *sampledStopTextSession
	headCalls int
}

func (m *sampledStopTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *sampledStopTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *sampledStopTextTokenModel) Head([]byte) ([]byte, error) {
	m.headCalls++
	return []byte{0, 0}, nil
}

func (m *sampledStopTextTokenModel) Vocab() int { return 1 }

func (m *sampledStopTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type sampledStopTextSession struct {
	stepCalls int
}

func (s *sampledStopTextSession) Step([]byte) ([]byte, error) {
	s.stepCalls++
	return []byte{0}, nil
}

type sampledFastTextTokenModel struct {
	session   *sampledFastTextSession
	headCalls int
}

func (m *sampledFastTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *sampledFastTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *sampledFastTextTokenModel) Head([]byte) ([]byte, error) {
	m.headCalls++
	return make([]byte, 12*2), nil
}

func (m *sampledFastTextTokenModel) Vocab() int { return 12 }

func (m *sampledFastTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type sampledFastTextSession struct {
	sampledOneShotCalls  int
	sampledRetainedCalls int
	seenIDs              []int32
	seenMax              int
	seenStops            []int32
	seenParams           model.SampleParams
}

func (s *sampledFastTextSession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *sampledFastTextSession) GenerateSampledOneShotEach(ids []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.sampledOneShotCalls++
	return s.generateSampled(ids, maxNew, stopTokens, params, transform, yield)
}

func (s *sampledFastTextSession) GenerateSampledEach(ids []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.sampledRetainedCalls++
	return s.generateSampled(ids, maxNew, stopTokens, params, transform, yield)
}

func (s *sampledFastTextSession) generateSampled(ids []int32, maxNew int, stopTokens []int32, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.seenIDs = append(s.seenIDs[:0], ids...)
	s.seenMax = maxNew
	s.seenStops = append(s.seenStops[:0], stopTokens...)
	s.seenParams = params
	out := []int32{2, 3}
	gen := make([]int32, 0, len(out))
	for _, id := range out {
		if transform != nil {
			id = transform(id)
		}
		gen = append(gen, id)
		if yield != nil && !yield(id) {
			break
		}
	}
	return gen, nil
}

type repeatPenaltyTextTokenModel struct {
	session  *repeatPenaltyTextSession
	headSeen int
}

func (m *repeatPenaltyTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *repeatPenaltyTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *repeatPenaltyTextTokenModel) Head([]byte) ([]byte, error) {
	m.headSeen++
	logits := make([]byte, 3*2)
	logits[1*2], logits[1*2+1] = f32ToBF16BytesForNativeTextTest(1.0)
	logits[2*2], logits[2*2+1] = f32ToBF16BytesForNativeTextTest(0.75)
	return logits, nil
}

func (m *repeatPenaltyTextTokenModel) Vocab() int { return 3 }

func (m *repeatPenaltyTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type repeatPenaltyTextSession struct {
	generateEachCalls int
}

func (s *repeatPenaltyTextSession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *repeatPenaltyTextSession) GenerateEach(ids []int32, maxNew, eos int, yield func(int32) bool) ([]int32, error) {
	s.generateEachCalls++
	out := []int32{1, 1}
	for _, id := range out {
		if yield != nil && !yield(id) {
			break
		}
	}
	return out, nil
}

type classifyLogitsTextTokenModel struct {
	headCalls int
}

func (m *classifyLogitsTextTokenModel) Embed(id int32) ([]byte, error) {
	return []byte{byte(id)}, nil
}

func (m *classifyLogitsTextTokenModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return inputs, nil
}

func (m *classifyLogitsTextTokenModel) Head([]byte) ([]byte, error) {
	m.headCalls++
	logits := make([]byte, 4*2)
	logits[1*2], logits[1*2+1] = f32ToBF16BytesForNativeTextTest(0.25)
	logits[2*2], logits[2*2+1] = f32ToBF16BytesForNativeTextTest(1.0)
	logits[3*2], logits[3*2+1] = f32ToBF16BytesForNativeTextTest(0.5)
	return logits, nil
}

func (m *classifyLogitsTextTokenModel) Vocab() int { return 4 }

func f32ToBF16BytesForNativeTextTest(v float32) (byte, byte) {
	bits := math.Float32bits(v)
	h := uint16(bits >> 16)
	return byte(h), byte(h >> 8)
}

func TestNativeTextModelWarmPromptCacheUsesCachedSession(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	promptIDs := tok.Encode("hello")

	if err := native.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if !reflect.DeepEqual(session.warmed, promptIDs) {
		t.Fatalf("warmed ids = %v, want %v", session.warmed, promptIDs)
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(session.generated, promptIDs) {
		t.Fatalf("GenerateCached prompt ids = %v, want %v", session.generated, promptIDs)
	}
	if session.generatedMax != 2 {
		t.Fatalf("GenerateCached maxNew = %d, want 2", session.generatedMax)
	}
	if !reflect.DeepEqual(got, []int32{7, 8}) {
		t.Fatalf("generated ids = %v, want [7 8]", got)
	}
	if session.cachedEachCalls != 1 {
		t.Fatalf("GenerateCachedEach calls = %d, want 1", session.cachedEachCalls)
	}

	native.ClearPromptCache()
	if session.clearCallCount != 1 {
		t.Fatalf("ClearPromptCache calls = %d, want 1", session.clearCallCount)
	}
}

func TestNativeTextModelWarmPromptCacheUsesCachedSampledSession(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	promptIDs := tok.Encode("hello")
	if err := native.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(3), inference.WithTemperature(0.8), inference.WithTopK(5), inference.WithStopTokens(11)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{9, 10}) {
		t.Fatalf("cached sampled ids = %v, want [9 10]", got)
	}
	if session.cachedSampledCalls != 1 {
		t.Fatalf("GenerateCachedSampledEach calls = %d, want 1", session.cachedSampledCalls)
	}
	if session.cachedEachCalls != 0 || session.generateEachCalls != 0 {
		t.Fatalf("greedy cache calls cached/generate = %d/%d, want 0/0", session.cachedEachCalls, session.generateEachCalls)
	}
	if !reflect.DeepEqual(session.generated, promptIDs) {
		t.Fatalf("cached sampled prompt ids = %v, want %v", session.generated, promptIDs)
	}
	if session.generatedMax != 3 || !reflect.DeepEqual(session.sampledStopTokens, []int32{11}) {
		t.Fatalf("cached sampled max/stops = %d/%v, want 3/[11]", session.generatedMax, session.sampledStopTokens)
	}
	if session.sampledParams.Temperature != 0.8 || session.sampledParams.TopK != 5 {
		t.Fatalf("cached sampled params = temp %.1f topK %d, want 0.8/5", session.sampledParams.Temperature, session.sampledParams.TopK)
	}
}

func TestNativeTextModelWarmPromptCacheMinTokensBeforeStopUsesCachedGreedyStages(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	promptIDs := tok.Encode("hello")
	if err := native.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2), inference.WithStopTokens(7), inference.WithMinTokensBeforeStop(1)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{8, 7}) {
		t.Fatalf("cached min-stop ids = %v, want [8 7]", got)
	}
	if session.cachedSuppressCalls != 1 {
		t.Fatalf("GenerateCachedEachWithSuppressionAndTransform calls = %d, want 1", session.cachedSuppressCalls)
	}
	if session.cachedEachCalls != 1 {
		t.Fatalf("GenerateCachedEach calls = %d, want 1", session.cachedEachCalls)
	}
	if session.cachedSampledCalls != 0 {
		t.Fatalf("GenerateCachedSampledEach calls = %d, want 0", session.cachedSampledCalls)
	}
	if !reflect.DeepEqual(session.generated, append(append([]int32(nil), promptIDs...), 8)) {
		t.Fatalf("second-stage prompt ids = %v, want original prompt plus first generated token", session.generated)
	}
}

func TestNativeTextModelCacheServiceWarmsAndClearsPromptCache(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	var service inference.CacheService = native
	var lister interface {
		CacheEntries(context.Context, map[string]string) ([]inference.CacheBlockRef, error)
	} = native
	labels := map[string]string{"scope": "native"}
	promptIDs := tok.Encode("hello")

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Prompt: "hello",
		Labels: labels,
	})
	if err != nil {
		t.Fatalf("WarmCache: %v", err)
	}
	if !reflect.DeepEqual(session.warmed, promptIDs) {
		t.Fatalf("WarmCache ids = %v, want %v", session.warmed, promptIDs)
	}
	if warmed.Stats.Blocks != 1 {
		t.Fatalf("WarmCache stats blocks = %d, want 1", warmed.Stats.Blocks)
	}
	if len(warmed.Blocks) != 1 || warmed.Blocks[0].TokenCount != len(promptIDs) {
		t.Fatalf("WarmCache blocks = %+v, want one prompt block with %d tokens", warmed.Blocks, len(promptIDs))
	}
	if !reflect.DeepEqual(warmed.Labels, labels) || !reflect.DeepEqual(warmed.Stats.Labels, labels) {
		t.Fatalf("WarmCache labels = %+v stats=%+v, want %v", warmed.Labels, warmed.Stats.Labels, labels)
	}
	entries, err := lister.CacheEntries(context.Background(), labels)
	if err != nil {
		t.Fatalf("CacheEntries: %v", err)
	}
	if len(entries) != 1 || entries[0].ID != "native-prompt" || entries[0].TokenCount != len(promptIDs) {
		t.Fatalf("CacheEntries = %+v, want native prompt block with %d tokens", entries, len(promptIDs))
	}
	if !reflect.DeepEqual(entries[0].Labels, labels) {
		t.Fatalf("CacheEntries labels = %+v, want %v", entries[0].Labels, labels)
	}
	entries[0].Labels["scope"] = "mutated"
	again, err := lister.CacheEntries(context.Background(), labels)
	if err != nil {
		t.Fatalf("CacheEntries again: %v", err)
	}
	if again[0].Labels["scope"] != "native" {
		t.Fatalf("CacheEntries returned mutable internal labels: %+v", again[0].Labels)
	}
	miss, err := lister.CacheEntries(context.Background(), map[string]string{"scope": "other"})
	if err != nil {
		t.Fatalf("CacheEntries miss: %v", err)
	}
	if len(miss) != 0 {
		t.Fatalf("CacheEntries non-matching labels = %+v, want none", miss)
	}

	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats: %v", err)
	}
	if stats.Blocks != 1 {
		t.Fatalf("CacheStats blocks = %d, want 1", stats.Blocks)
	}
	unchanged, err := service.ClearCache(context.Background(), map[string]string{"scope": "other"})
	if err != nil {
		t.Fatalf("ClearCache non-match: %v", err)
	}
	if unchanged.Blocks != 1 {
		t.Fatalf("ClearCache non-match blocks = %d, want 1", unchanged.Blocks)
	}
	if session.clearCallCount != 0 {
		t.Fatalf("ClearCache non-match calls = %d, want 0", session.clearCallCount)
	}

	cleared, err := service.ClearCache(context.Background(), labels)
	if err != nil {
		t.Fatalf("ClearCache: %v", err)
	}
	if session.clearCallCount != 1 {
		t.Fatalf("ClearCache calls = %d, want 1", session.clearCallCount)
	}
	if cleared.Blocks != 0 {
		t.Fatalf("ClearCache stats blocks = %d, want 0", cleared.Blocks)
	}
	if !reflect.DeepEqual(cleared.Labels, labels) {
		t.Fatalf("ClearCache labels = %v, want %v", cleared.Labels, labels)
	}
	entries, err = lister.CacheEntries(context.Background(), nil)
	if err != nil {
		t.Fatalf("CacheEntries after clear: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("CacheEntries after clear = %+v, want none", entries)
	}
}

func TestNativeTextModelSchedulerModelSchedulesPromptGeneration(t *testing.T) {
	var _ inference.SchedulerModel = (*nativeTextModel)(nil)
	var _ inference.CancellableModel = (*nativeTextModel)(nil)
	var _ inference.ProbeableModel = (*nativeTextModel)(nil)

	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	probeEvents := make(chan string, 8)
	native.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		if event.Scheduler == nil {
			return
		}
		select {
		case probeEvents <- event.Scheduler.Event:
		default:
		}
	}))

	handle, tokens, err := native.Schedule(context.Background(), inference.ScheduledRequest{
		ID:      "native-sched",
		Prompt:  "hello",
		Sampler: inference.SamplerConfig{MaxTokens: 2},
		Labels:  map[string]string{"scope": "native"},
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if handle.ID != "native-sched" || handle.Labels["scope"] != "native" {
		t.Fatalf("Schedule handle = %+v, want native-sched with labels", handle)
	}

	var got []int32
	for token := range tokens {
		if token.RequestID != "native-sched" {
			t.Fatalf("scheduled token request ID = %q, want native-sched", token.RequestID)
		}
		if token.Labels["scope"] != "native" || token.Labels["queue_latency_ms"] == "" {
			t.Fatalf("scheduled token labels = %+v, want scope and queue latency", token.Labels)
		}
		got = append(got, token.Token.ID)
	}
	if !reflect.DeepEqual(got, []int32{12, 13}) {
		t.Fatalf("scheduled token ids = %v, want [12 13]", got)
	}
	seenProbe := map[string]bool{}
	for {
		select {
		case event := <-probeEvents:
			seenProbe[event] = true
		default:
			goto probesDrained
		}
	}
probesDrained:
	for _, event := range []string{"queued", "start", "first_token", "complete"} {
		if !seenProbe[event] {
			t.Fatalf("native scheduler probes = %v, want event %q", seenProbe, event)
		}
	}

	cancelled, err := native.CancelRequest(context.Background(), "missing")
	if err != nil {
		t.Fatalf("CancelRequest missing: %v", err)
	}
	if cancelled.ID != "missing" || cancelled.Cancelled || cancelled.Reason != "not_found" {
		t.Fatalf("CancelRequest missing = %+v, want not_found", cancelled)
	}
}

func TestNativeTextModelWarmPromptCacheChunksUsesCachedSession(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	chunks := func(yield func(string) bool) {
		if !yield("hello") {
			return
		}
		yield("hello")
	}
	want := append([]int32(nil), tok.Encode("hello")...)
	second := tok.Encode("hello")
	if tok.HasBOSToken() && len(second) > 0 && second[0] == tok.BOSToken() {
		second = second[1:]
	}
	want = append(want, second...)

	if err := native.WarmPromptCacheChunks(context.Background(), iter.Seq[string](chunks)); err != nil {
		t.Fatalf("WarmPromptCacheChunks: %v", err)
	}
	if !reflect.DeepEqual(session.warmed, want) {
		t.Fatalf("chunk-warmed ids = %v, want %v", session.warmed, want)
	}
}

func TestNativeTextModelGenerateChunksUsesChunkTokenStream(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	chunks := func(yield func(string) bool) {
		if !yield("hello") {
			return
		}
		yield("hello")
	}
	want := append([]int32(nil), tok.Encode("hello")...)
	second := tok.Encode("hello")
	second = stripNativeImplicitChunkBOS(tok, second)
	want = append(want, second...)

	var got []int32
	for tok := range native.GenerateChunks(context.Background(), iter.Seq[string](chunks), inference.WithMaxTokens(2)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("GenerateChunks Err: %v", err)
	}
	if !reflect.DeepEqual(session.generated, want) {
		t.Fatalf("chunk-generated prompt ids = %v, want %v", session.generated, want)
	}
	if !reflect.DeepEqual(got, []int32{12, 13}) {
		t.Fatalf("GenerateChunks ids = %v, want [12 13]", got)
	}
	if session.generateEachCalls != 1 {
		t.Fatalf("GenerateEach calls = %d, want 1", session.generateEachCalls)
	}
}

func TestNativeTextModelChatChunksUsesFormattedChunkStream(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:        &promptCacheTextTokenModel{session: session},
		tok:       tok,
		modelType: "gemma4",
		maxLen:    256,
	}
	messages := []inference.Message{{Role: "user", Content: "hello"}}
	prompt := native.formatChat(messages, inference.DefaultGenerateConfig())
	chunkBytes := 5
	want := []int32{}
	for i := 0; i < len(prompt); i += chunkBytes {
		end := i + chunkBytes
		if end > len(prompt) {
			end = len(prompt)
		}
		ids := tok.Encode(prompt[i:end])
		if i > 0 {
			ids = stripNativeImplicitChunkBOS(tok, ids)
		}
		want = append(want, ids...)
	}

	var got []int32
	for tok := range native.ChatChunks(context.Background(), messages, chunkBytes, inference.WithMaxTokens(2)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("ChatChunks Err: %v", err)
	}
	if !reflect.DeepEqual(session.generated, want) {
		t.Fatalf("chat chunk prompt ids = %v, want %v", session.generated, want)
	}
	if !reflect.DeepEqual(got, []int32{12, 13}) {
		t.Fatalf("ChatChunks ids = %v, want [12 13]", got)
	}
	if session.generateEachCalls != 1 {
		t.Fatalf("GenerateEach calls = %d, want 1", session.generateEachCalls)
	}
}

func TestNativeTextModelStreamsCachedGreedyTokensAsDecoded(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	if err := native.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2)) {
		got = append(got, tok.ID)
		break
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{7}) {
		t.Fatalf("streamed ids before consumer stop = %v, want [7]", got)
	}
	if session.cachedEachCalls != 1 {
		t.Fatalf("GenerateCachedEach calls = %d, want 1", session.cachedEachCalls)
	}
	if session.streamedYieldCount != 1 {
		t.Fatalf("streamed yield count = %d, want 1", session.streamedYieldCount)
	}
}

func TestNativeTextModelCachedGreedyHonoursStopTokens(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	if err := native.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2), inference.WithStopTokens(7)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{7}) {
		t.Fatalf("streamed ids with stop token = %v, want [7]", got)
	}
	if session.streamedYieldCount != 1 {
		t.Fatalf("streamed yield count = %d, want 1", session.streamedYieldCount)
	}
}

func TestNativeTextModelStreamsUncachedGreedyTokensAsDecoded(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2)) {
		got = append(got, tok.ID)
		break
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{12}) {
		t.Fatalf("streamed ids before consumer stop = %v, want [12]", got)
	}
	if session.generateEachCalls != 1 {
		t.Fatalf("GenerateEach calls = %d, want 1", session.generateEachCalls)
	}
	if session.streamedYieldCount != 1 {
		t.Fatalf("streamed yield count = %d, want 1", session.streamedYieldCount)
	}
}

func TestNativeTextModelUncachedGreedyHonoursStopTokens(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2), inference.WithStopTokens(12)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{12}) {
		t.Fatalf("streamed ids with stop token = %v, want [12]", got)
	}
	if session.streamedYieldCount != 1 {
		t.Fatalf("streamed yield count = %d, want 1", session.streamedYieldCount)
	}
}

func TestNativeTextModelSampledHonoursMultipleStopTokensBeforeFullDecode(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &sampledStopTextSession{}
	tm := &sampledStopTextTokenModel{session: session}
	native := &nativeTextModel{
		tm:     tm,
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(4), inference.WithTemperature(1), inference.WithStopTokens(0, 2)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{0}) {
		t.Fatalf("sampled ids with stop token = %v, want [0]", got)
	}
	if tm.headCalls != 1 {
		t.Fatalf("sampled head calls = %d, want 1 stop-token decode step", tm.headCalls)
	}
}

func TestNativeTextModelSampledUsesNativeSessionFastPath(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &sampledFastTextSession{}
	tm := &sampledFastTextTokenModel{session: session}
	native := &nativeTextModel{
		tm:     tm,
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(4), inference.WithTemperature(0.8), inference.WithTopK(5), inference.WithStopTokens(11)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{2, 3}) {
		t.Fatalf("sampled fast-path ids = %v, want [2 3]", got)
	}
	if session.sampledOneShotCalls != 1 || session.sampledRetainedCalls != 0 {
		t.Fatalf("sampled calls oneshot/retained = %d/%d, want 1/0", session.sampledOneShotCalls, session.sampledRetainedCalls)
	}
	if tm.headCalls != 0 {
		t.Fatalf("generic Head calls = %d, want 0 when sampled native session path is available", tm.headCalls)
	}
	if !reflect.DeepEqual(session.seenIDs, tok.Encode("hello")) {
		t.Fatalf("sampled fast-path prompt ids = %v, want encoded hello", session.seenIDs)
	}
	if session.seenMax != 4 || !reflect.DeepEqual(session.seenStops, []int32{11}) {
		t.Fatalf("sampled fast-path max/stops = %d/%v, want 4/[11]", session.seenMax, session.seenStops)
	}
	if session.seenParams.Temperature != 0.8 || session.seenParams.TopK != 5 {
		t.Fatalf("sampled params = temp %.1f topK %d, want 0.8/5", session.seenParams.Temperature, session.seenParams.TopK)
	}
}

func TestNativeTextModelGreedyHonoursRepeatPenalty(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &repeatPenaltyTextSession{}
	tm := &repeatPenaltyTextTokenModel{session: session}
	native := &nativeTextModel{
		tm:     tm,
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range native.Generate(context.Background(), "hello", inference.WithMaxTokens(2), inference.WithRepeatPenalty(2)) {
		got = append(got, tok.ID)
	}
	if err := native.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{1, 2}) {
		t.Fatalf("repeat-penalised ids = %v, want [1 2]", got)
	}
	if session.generateEachCalls != 0 {
		t.Fatalf("GenerateEach calls = %d, want 0 when repeat penalty requires logits", session.generateEachCalls)
	}
	if tm.headSeen != 2 {
		t.Fatalf("Head calls = %d, want 2 logits steps", tm.headSeen)
	}
}

func TestNativeTextModelClassifyReturnsLogits(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	tm := &classifyLogitsTextTokenModel{}
	native := &nativeTextModel{
		tm:     tm,
		tok:    tok,
		maxLen: 32,
	}

	results, err := native.Classify(context.Background(), []string{"hello"}, inference.WithLogits())
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Classify results = %d, want 1", len(results))
	}
	if results[0].Token.ID != 2 {
		t.Fatalf("Classify token = %d, want 2", results[0].Token.ID)
	}
	if len(results[0].Logits) != 4 {
		t.Fatalf("Classify logits len = %d, want 4", len(results[0].Logits))
	}
	if results[0].Logits[2] != 1.0 {
		t.Fatalf("Classify logits[2] = %f, want 1", results[0].Logits[2])
	}
	if tm.headCalls != 1 {
		t.Fatalf("Head calls = %d, want 1", tm.headCalls)
	}
}

func TestNativeTextModelClassifyHonoursSuppressTokens(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	native := &nativeTextModel{
		tm:     &classifyLogitsTextTokenModel{},
		tok:    tok,
		maxLen: 32,
	}

	results, err := native.Classify(context.Background(), []string{"hello"}, inference.WithSuppressTokens(2))
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Classify results = %d, want 1", len(results))
	}
	if results[0].Token.ID != 3 {
		t.Fatalf("Classify token = %d, want 3 when token 2 is suppressed", results[0].Token.ID)
	}
}

func TestNativeTextModelClassifyUpdatesMetrics(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	tm := &classifyLogitsTextTokenModel{}
	native := &nativeTextModel{
		tm:     tm,
		tok:    tok,
		maxLen: 32,
	}
	prompts := []string{"hello", "world"}
	wantPromptTokens := len(tok.Encode(prompts[0])) + len(tok.Encode(prompts[1]))

	if _, err := native.Classify(context.Background(), prompts, inference.WithLogits()); err != nil {
		t.Fatalf("Classify: %v", err)
	}
	metrics := native.Metrics()
	if metrics.PromptTokens != wantPromptTokens {
		t.Fatalf("PromptTokens = %d, want %d", metrics.PromptTokens, wantPromptTokens)
	}
	if metrics.GeneratedTokens != len(prompts) {
		t.Fatalf("GeneratedTokens = %d, want %d", metrics.GeneratedTokens, len(prompts))
	}
	if metrics.PrefillDuration <= 0 || metrics.TotalDuration <= 0 {
		t.Fatalf("metrics durations = prefill %s total %s, want positive", metrics.PrefillDuration, metrics.TotalDuration)
	}
	if metrics.DecodeDuration != 0 {
		t.Fatalf("DecodeDuration = %s, want 0 for classify prefill", metrics.DecodeDuration)
	}
	if metrics.PrefillTokensPerSec <= 0 {
		t.Fatalf("PrefillTokensPerSec = %f, want positive", metrics.PrefillTokensPerSec)
	}
}

func TestNativeTextModelBatchGenerateUpdatesBatchMetrics(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &promptCacheTextSession{}
	native := &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	prompts := []string{"hello", "world"}
	wantPromptTokens := len(tok.Encode(prompts[0])) + len(tok.Encode(prompts[1]))

	results, err := native.BatchGenerate(context.Background(), prompts, inference.WithMaxTokens(2))
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != len(prompts) {
		t.Fatalf("BatchGenerate results = %d, want %d", len(results), len(prompts))
	}
	wantGenerated := 0
	for i := range results {
		if results[i].Err != nil {
			t.Fatalf("BatchGenerate result %d err = %v", i, results[i].Err)
		}
		wantGenerated += len(results[i].Tokens)
	}
	metrics := native.Metrics()
	if metrics.PromptTokens != wantPromptTokens {
		t.Fatalf("PromptTokens = %d, want %d", metrics.PromptTokens, wantPromptTokens)
	}
	if metrics.GeneratedTokens != wantGenerated {
		t.Fatalf("GeneratedTokens = %d, want %d", metrics.GeneratedTokens, wantGenerated)
	}
	if metrics.TotalDuration <= 0 || metrics.DecodeDuration <= 0 {
		t.Fatalf("metrics durations = total %s decode %s, want positive", metrics.TotalDuration, metrics.DecodeDuration)
	}
	if metrics.DecodeTokensPerSec <= 0 {
		t.Fatalf("DecodeTokensPerSec = %f, want positive", metrics.DecodeTokensPerSec)
	}
}
