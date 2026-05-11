// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type blockingScheduleModel struct {
	started chan string
	release chan struct{}
	metrics inference.GenerateMetrics
}

func newBlockingScheduleModel() *blockingScheduleModel {
	return &blockingScheduleModel{
		started: make(chan string, 8),
		release: make(chan struct{}),
	}
}

func (model *blockingScheduleModel) Generate(ctx context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		model.started <- prompt
		select {
		case <-ctx.Done():
			return
		case <-model.release:
		}
		yield(inference.Token{Text: prompt})
	}
}

func (model *blockingScheduleModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt := ""
	if len(messages) > 0 {
		prompt = messages[len(messages)-1].Content
	}
	return model.Generate(ctx, prompt, opts...)
}

func (model *blockingScheduleModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (model *blockingScheduleModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (model *blockingScheduleModel) ModelType() string { return "blocking" }
func (model *blockingScheduleModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3"}
}
func (model *blockingScheduleModel) Metrics() inference.GenerateMetrics { return model.metrics }
func (model *blockingScheduleModel) Err() error                         { return nil }
func (model *blockingScheduleModel) Close() error                       { return nil }

func TestScheduledModel_Good_QueuesRequestsAndEmitsLatencyProbe(t *testing.T) {
	base := newBlockingScheduleModel()
	var events []inference.ProbeEvent
	scheduled := NewScheduledModel(base, SchedulerConfig{
		MaxConcurrent:   1,
		MaxQueue:        1,
		StreamBuffer:    1,
		RequestIDPrefix: "test",
		ProbeSink: inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
			events = append(events, event)
		}),
	})

	first, firstTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "first"})
	if err != nil {
		t.Fatalf("Schedule(first) error = %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "first" {
		t.Fatalf("started = %q, want first", got)
	}
	second, secondTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "second"})
	if err != nil {
		t.Fatalf("Schedule(second) error = %v", err)
	}
	if first.ID == "" || second.ID == "" || first.ID == second.ID {
		t.Fatalf("request IDs = %q/%q, want unique non-empty IDs", first.ID, second.ID)
	}

	assertNoStartedPrompt(t, base.started)
	base.release <- struct{}{}
	firstToken := waitScheduledToken(t, firstTokens)
	if firstToken.RequestID != first.ID || firstToken.Token.Text != "first" {
		t.Fatalf("first token = %+v, want request %q text first", firstToken, first.ID)
	}
	if firstToken.Labels["queue_latency_ms"] == "" || firstToken.Labels["first_token_latency_ms"] == "" {
		t.Fatalf("first token labels = %+v, want latency labels", firstToken.Labels)
	}

	if got := waitStartedPrompt(t, base.started); got != "second" {
		t.Fatalf("started = %q, want second", got)
	}
	base.release <- struct{}{}
	secondToken := waitScheduledToken(t, secondTokens)
	if secondToken.RequestID != second.ID || secondToken.Token.Text != "second" {
		t.Fatalf("second token = %+v, want request %q text second", secondToken, second.ID)
	}
	if !hasSchedulerProbeEvent(events, "first_token") || !hasSchedulerProbeEvent(events, "complete") {
		t.Fatalf("events = %+v, want first_token and complete scheduler probes", events)
	}
}

func TestScheduledModel_Bad_RejectsFullQueue(t *testing.T) {
	base := newBlockingScheduleModel()
	scheduled := NewScheduledModel(base, SchedulerConfig{MaxConcurrent: 1, MaxQueue: 1})

	_, _, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "active", Prompt: "active"})
	if err != nil {
		t.Fatalf("Schedule(active) error = %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "active" {
		t.Fatalf("started = %q, want active", got)
	}
	_, _, err = scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "queued", Prompt: "queued"})
	if err != nil {
		t.Fatalf("Schedule(queued) error = %v", err)
	}
	_, _, err = scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "overflow", Prompt: "overflow"})
	if err == nil {
		t.Fatal("Schedule(overflow) error = nil, want queue full")
	}
}

func TestScheduledModel_CancelRequest_Good_CancelsQueuedRequest(t *testing.T) {
	base := newBlockingScheduleModel()
	scheduled := NewScheduledModel(base, SchedulerConfig{MaxConcurrent: 1, MaxQueue: 1})

	_, activeTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "active", Prompt: "active"})
	if err != nil {
		t.Fatalf("Schedule(active) error = %v", err)
	}
	if got := waitStartedPrompt(t, base.started); got != "active" {
		t.Fatalf("started = %q, want active", got)
	}
	_, queuedTokens, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{ID: "queued", Prompt: "queued"})
	if err != nil {
		t.Fatalf("Schedule(queued) error = %v", err)
	}

	result, err := scheduled.CancelRequest(context.Background(), "queued")
	if err != nil {
		t.Fatalf("CancelRequest() error = %v", err)
	}
	if !result.Cancelled || result.ID != "queued" {
		t.Fatalf("CancelRequest() = %+v, want queued cancellation", result)
	}
	base.release <- struct{}{}
	_ = waitScheduledToken(t, activeTokens)
	if token, ok := <-queuedTokens; ok {
		t.Fatalf("queued token = %+v, want closed channel after cancellation", token)
	}
	assertNoStartedPrompt(t, base.started)
}

type immediateScheduleModel struct {
	tokens       []inference.Token
	err          error
	cancelledID  string
	closed       bool
	classified   []string
	batchPrompts []string
	lastPrompt   string
	lastMessages []inference.Message
	metrics      inference.GenerateMetrics
}

func (model *immediateScheduleModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	model.lastPrompt = prompt
	return model.seq()
}

func (model *immediateScheduleModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	model.lastMessages = append([]inference.Message(nil), messages...)
	return model.seq()
}

func (model *immediateScheduleModel) Classify(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	model.classified = append([]string(nil), prompts...)
	return []inference.ClassifyResult{{Token: inference.Token{Text: "ok"}}}, nil
}

func (model *immediateScheduleModel) BatchGenerate(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	model.batchPrompts = append([]string(nil), prompts...)
	return []inference.BatchResult{{Tokens: []inference.Token{{Text: "batch"}}}}, nil
}

func (model *immediateScheduleModel) ModelType() string { return "immediate" }
func (model *immediateScheduleModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3", NumLayers: 2}
}
func (model *immediateScheduleModel) Metrics() inference.GenerateMetrics {
	if model.metrics.GeneratedTokens == 0 {
		model.metrics.GeneratedTokens = len(model.tokens)
	}
	return model.metrics
}
func (model *immediateScheduleModel) Err() error   { return model.err }
func (model *immediateScheduleModel) Close() error { model.closed = true; return nil }

func (model *immediateScheduleModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	model.cancelledID = id
	return inference.RequestCancelResult{ID: id, Cancelled: id != "", Reason: "base_cancelled"}, nil
}

func (model *immediateScheduleModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range model.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func TestScheduledModel_Good_GenerateChatAndDelegates(t *testing.T) {
	base := &immediateScheduleModel{tokens: []inference.Token{{Text: "A"}, {Text: "B"}}}
	scheduled := NewScheduledModel(base, SchedulerConfig{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})

	var generated []string
	for token := range scheduled.Generate(context.Background(), "prompt", inference.WithMaxTokens(2)) {
		generated = append(generated, token.Text)
	}
	if len(generated) != 2 || generated[0] != "A" || generated[1] != "B" || base.lastPrompt != "prompt" {
		t.Fatalf("generated = %v prompt=%q, want A/B from prompt", generated, base.lastPrompt)
	}

	var chat []string
	for token := range scheduled.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		chat = append(chat, token.Text)
	}
	if len(chat) != 2 || len(base.lastMessages) != 1 || base.lastMessages[0].Content != "hi" {
		t.Fatalf("chat = %v messages=%+v, want delegated chat", chat, base.lastMessages)
	}
	if results, err := scheduled.Classify(context.Background(), []string{"x"}); err != nil || len(results) != 1 || base.classified[0] != "x" {
		t.Fatalf("Classify() = %+v/%v classified=%v", results, err, base.classified)
	}
	if batches, err := scheduled.BatchGenerate(context.Background(), []string{"b"}); err != nil || len(batches) != 1 || base.batchPrompts[0] != "b" {
		t.Fatalf("BatchGenerate() = %+v/%v prompts=%v", batches, err, base.batchPrompts)
	}
	if scheduled.ModelType() != "immediate" || scheduled.Info().Architecture != "qwen3" || scheduled.Metrics().GeneratedTokens != 2 {
		t.Fatalf("model delegates = type %q info %+v metrics %+v", scheduled.ModelType(), scheduled.Info(), scheduled.Metrics())
	}
	if err := scheduled.Close(); err != nil || !base.closed {
		t.Fatalf("Close() = %v closed=%v", err, base.closed)
	}
}

func TestScheduledModel_Bad_NilAndErrorPaths(t *testing.T) {
	var nilScheduler *ScheduledModel
	if _, _, err := nilScheduler.Schedule(context.Background(), inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(nil scheduler) error = nil")
	}
	if result, err := nilScheduler.CancelRequest(context.Background(), "x"); err != nil || result.Reason != "scheduler_nil" {
		t.Fatalf("CancelRequest(nil scheduler) = %+v/%v", result, err)
	}
	if nilScheduler.Err() != nil || nilScheduler.Close() != nil {
		t.Fatal("nil scheduler Err/Close should be nil")
	}
	nilScheduler.SetProbeSink(nil)
	if nilScheduler.ModelType() != "" || nilScheduler.Info().Architecture != "" || nilScheduler.Metrics().GeneratedTokens != 0 {
		t.Fatalf("nil scheduler delegates returned non-zero values")
	}
	if _, err := nilScheduler.Classify(context.Background(), []string{"x"}); err == nil {
		t.Fatal("Classify(nil scheduler) error = nil")
	}
	if _, err := nilScheduler.BatchGenerate(context.Background(), []string{"x"}); err == nil {
		t.Fatal("BatchGenerate(nil scheduler) error = nil")
	}
	var generated []inference.Token
	for token := range nilScheduler.Generate(context.Background(), "prompt") {
		generated = append(generated, token)
	}
	if len(generated) != 0 || nilScheduler.Err() != nil {
		t.Fatalf("nil Generate tokens=%v err=%v, want no tokens and no stored nil-scheduler err", generated, nilScheduler.Err())
	}

	scheduled := NewScheduledModel(nil, SchedulerConfig{})
	if _, _, err := scheduled.Schedule(context.Background(), inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(nil base) error = nil")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	base := &immediateScheduleModel{tokens: []inference.Token{{Text: "x"}}}
	withBase := NewScheduledModel(base, SchedulerConfig{MaxQueue: 1})
	if _, _, err := withBase.Schedule(cancelled, inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule(cancelled context) error = nil")
	}
	if result, err := withBase.CancelRequest(context.Background(), ""); err != nil || result.Reason != "missing_id" {
		t.Fatalf("CancelRequest(empty) = %+v/%v", result, err)
	}
	if result, err := withBase.CancelRequest(context.Background(), "unknown"); err != nil || !result.Cancelled || base.cancelledID != "unknown" {
		t.Fatalf("CancelRequest(fallback) = %+v/%v cancelledID=%q", result, err, base.cancelledID)
	}
}

func TestScheduledModel_Good_ErrAndHelpers(t *testing.T) {
	base := &immediateScheduleModel{tokens: []inference.Token{{Text: "x"}}, err: core.NewError("base failed")}
	scheduled := NewScheduledModel(base, SchedulerConfig{RequestIDPrefix: "req", MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})
	for range scheduled.Generate(context.Background(), "prompt") {
	}
	if err := scheduled.Err(); err == nil || err.Error() != "base failed" {
		t.Fatalf("Err() = %v, want base failed", err)
	}
	scheduled.setErr(core.NewError("stored failed"))
	if err := scheduled.Err(); err == nil || err.Error() != "stored failed" {
		t.Fatalf("stored Err() = %v, want stored failed", err)
	}
	opts := scheduledGenerateOptions(inference.SamplerConfig{
		MaxTokens:     4,
		Temperature:   0.25,
		TopK:          8,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{1, 2},
		ReturnLogits:  true,
	})
	if len(opts) != 7 {
		t.Fatalf("scheduledGenerateOptions len = %d, want 7", len(opts))
	}
	labels := map[string]string{"a": "b"}
	cloned := cloneSchedulerLabels(labels)
	cloned["a"] = "changed"
	if labels["a"] != "b" {
		t.Fatalf("cloneSchedulerLabels mutated source = %+v", labels)
	}
	if millis(-time.Millisecond) != 0 || millisString(time.Millisecond) == "" {
		t.Fatal("millis helpers returned unexpected values")
	}
}

func waitStartedPrompt(t *testing.T, started <-chan string) string {
	t.Helper()
	select {
	case prompt := <-started:
		return prompt
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for prompt start")
		return ""
	}
}

func assertNoStartedPrompt(t *testing.T, started <-chan string) {
	t.Helper()
	select {
	case prompt := <-started:
		t.Fatalf("unexpected started prompt %q", prompt)
	case <-time.After(25 * time.Millisecond):
	}
}

func waitScheduledToken(t *testing.T, tokens <-chan inference.ScheduledToken) inference.ScheduledToken {
	t.Helper()
	select {
	case token, ok := <-tokens:
		if !ok {
			t.Fatal("token channel closed before token")
		}
		return token
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for token")
		return inference.ScheduledToken{}
	}
}

func hasSchedulerProbeEvent(events []inference.ProbeEvent, eventName string) bool {
	for _, event := range events {
		if event.Kind == inference.ProbeEventScheduler && event.Scheduler != nil && event.Scheduler.Event == eventName {
			return true
		}
	}
	return false
}
