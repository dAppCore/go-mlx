// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// SchedulerConfig configures the package-first request scheduler.
type SchedulerConfig struct {
	MaxConcurrent   int
	MaxQueue        int
	StreamBuffer    int
	RequestIDPrefix string
	ProbeSink       inference.ProbeSink
}

// ScheduledModel wraps an inference.TextModel with bounded queueing,
// cancellation, streaming backpressure, and scheduler probe events.
type ScheduledModel struct {
	base            inference.TextModel
	queue           chan *scheduledJob
	maxConcurrent   int
	streamBuffer    int
	requestIDPrefix string
	probeSink       inference.ProbeSink
	nextID          atomic.Uint64

	mu      sync.Mutex
	active  map[string]*scheduledJob
	lastErr error
}

type scheduledJob struct {
	req      inference.ScheduledRequest
	ctx      context.Context
	cancel   context.CancelFunc
	out      chan inference.ScheduledToken
	queuedAt time.Time
}

// NewScheduledModel returns a scheduler wrapper for model. Nil models are
// accepted so callers can construct package surfaces before a backend loads.
func NewScheduledModel(model inference.TextModel, cfg SchedulerConfig) *ScheduledModel {
	maxConcurrent := cfg.MaxConcurrent
	if maxConcurrent <= 0 {
		maxConcurrent = 1
	}
	maxQueue := cfg.MaxQueue
	if maxQueue < 0 {
		maxQueue = 0
	}
	streamBuffer := cfg.StreamBuffer
	if streamBuffer < 0 {
		streamBuffer = 0
	}
	prefix := core.Trim(cfg.RequestIDPrefix)
	if prefix == "" {
		prefix = "mlx-sched"
	}
	scheduler := &ScheduledModel{
		base:            model,
		queue:           make(chan *scheduledJob, maxQueue),
		maxConcurrent:   maxConcurrent,
		streamBuffer:    streamBuffer,
		requestIDPrefix: prefix,
		probeSink:       cfg.ProbeSink,
		active:          map[string]*scheduledJob{},
	}
	for worker := range maxConcurrent {
		go scheduler.worker(worker)
	}
	return scheduler
}

// Schedule enqueues a generation request and returns its streamed tokens.
func (scheduler *ScheduledModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if scheduler == nil || scheduler.base == nil {
		return inference.RequestHandle{}, nil, core.NewError("mlx: scheduler model is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.RequestHandle{}, nil, err
	}
	if core.Trim(req.ID) == "" {
		req.ID = scheduler.nextRequestID()
	}
	reqCtx, cancel := context.WithCancel(ctx)
	job := &scheduledJob{
		req:      req,
		ctx:      reqCtx,
		cancel:   cancel,
		out:      make(chan inference.ScheduledToken, scheduler.streamBuffer),
		queuedAt: time.Now(),
	}
	scheduler.register(job)
	select {
	case scheduler.queue <- job:
		scheduler.emitSchedulerProbe(job, "queued", 0, 0, false)
		return inference.RequestHandle{ID: req.ID, Model: inference.ModelIdentity{ID: req.Model}, Labels: cloneSchedulerLabels(req.Labels)}, job.out, nil
	case <-ctx.Done():
		scheduler.unregister(req.ID)
		cancel()
		close(job.out)
		return inference.RequestHandle{}, nil, ctx.Err()
	default:
		scheduler.unregister(req.ID)
		cancel()
		close(job.out)
		return inference.RequestHandle{}, nil, core.NewError("mlx: scheduler queue is full")
	}
}

// CancelRequest cancels a queued or running request by ID.
func (scheduler *ScheduledModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	if scheduler == nil {
		return inference.RequestCancelResult{ID: id, Reason: "scheduler_nil"}, nil
	}
	if core.Trim(id) == "" {
		return inference.RequestCancelResult{Reason: "missing_id"}, nil
	}
	scheduler.mu.Lock()
	job := scheduler.active[id]
	scheduler.mu.Unlock()
	if job == nil {
		if cancellable, ok := scheduler.base.(inference.CancellableModel); ok {
			return cancellable.CancelRequest(context.Background(), id)
		}
		return inference.RequestCancelResult{ID: id, Reason: "not_found"}, nil
	}
	job.cancel()
	scheduler.emitSchedulerProbe(job, "cancel", time.Since(job.queuedAt), 0, true)
	return inference.RequestCancelResult{ID: id, Cancelled: true, Reason: "cancelled"}, nil
}

// Generate schedules a prompt request and yields tokens with scheduler
// backpressure semantics.
func (scheduler *ScheduledModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		req := inference.ScheduledRequest{Prompt: prompt, Sampler: inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))}
		_, tokens, err := scheduler.Schedule(ctx, req)
		if err != nil {
			scheduler.setErr(err)
			return
		}
		for scheduled := range tokens {
			if !yield(scheduled.Token) {
				_, _ = scheduler.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

// Chat schedules a chat request and yields tokens with scheduler backpressure
// semantics.
func (scheduler *ScheduledModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		req := inference.ScheduledRequest{Messages: append([]inference.Message(nil), messages...), Sampler: inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))}
		_, tokens, err := scheduler.Schedule(ctx, req)
		if err != nil {
			scheduler.setErr(err)
			return
		}
		for scheduled := range tokens {
			if !yield(scheduled.Token) {
				_, _ = scheduler.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

func (scheduler *ScheduledModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if scheduler == nil || scheduler.base == nil {
		return nil, core.NewError("mlx: scheduler model is nil")
	}
	return scheduler.base.Classify(ctx, prompts, opts...)
}

func (scheduler *ScheduledModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if scheduler == nil || scheduler.base == nil {
		return nil, core.NewError("mlx: scheduler model is nil")
	}
	return scheduler.base.BatchGenerate(ctx, prompts, opts...)
}

func (scheduler *ScheduledModel) ModelType() string {
	if scheduler == nil || scheduler.base == nil {
		return ""
	}
	return scheduler.base.ModelType()
}

func (scheduler *ScheduledModel) Info() inference.ModelInfo {
	if scheduler == nil || scheduler.base == nil {
		return inference.ModelInfo{}
	}
	return scheduler.base.Info()
}

func (scheduler *ScheduledModel) Metrics() inference.GenerateMetrics {
	if scheduler == nil || scheduler.base == nil {
		return inference.GenerateMetrics{}
	}
	return scheduler.base.Metrics()
}

func (scheduler *ScheduledModel) Err() error {
	if scheduler == nil {
		return nil
	}
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	if scheduler.lastErr != nil {
		return scheduler.lastErr
	}
	if scheduler.base == nil {
		return nil
	}
	return scheduler.base.Err()
}

func (scheduler *ScheduledModel) Close() error {
	if scheduler == nil || scheduler.base == nil {
		return nil
	}
	return scheduler.base.Close()
}

// SetProbeSink updates the scheduler probe sink.
func (scheduler *ScheduledModel) SetProbeSink(sink inference.ProbeSink) {
	if scheduler == nil {
		return
	}
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	scheduler.probeSink = sink
}

func (scheduler *ScheduledModel) worker(_ int) {
	for job := range scheduler.queue {
		scheduler.run(job)
	}
}

func (scheduler *ScheduledModel) run(job *scheduledJob) {
	defer close(job.out)
	defer scheduler.unregister(job.req.ID)
	queueLatency := time.Since(job.queuedAt)
	if err := job.ctx.Err(); err != nil {
		scheduler.emitSchedulerProbe(job, "cancelled", queueLatency, 0, true)
		return
	}
	startedAt := time.Now()
	scheduler.emitSchedulerProbe(job, "start", queueLatency, 0, false)
	firstToken := true
	for token := range scheduler.baseTokens(job) {
		firstLatency := time.Duration(0)
		if firstToken {
			firstLatency = time.Since(startedAt)
			firstToken = false
			scheduler.emitSchedulerProbe(job, "first_token", queueLatency, firstLatency, false)
		}
		labels := cloneSchedulerLabels(job.req.Labels)
		labels["queue_latency_ms"] = millisString(queueLatency)
		if firstLatency > 0 {
			labels["first_token_latency_ms"] = millisString(firstLatency)
		}
		select {
		case <-job.ctx.Done():
			scheduler.emitSchedulerProbe(job, "cancelled", queueLatency, firstLatency, true)
			return
		case job.out <- inference.ScheduledToken{
			RequestID: job.req.ID,
			Token:     token,
			Metrics:   scheduler.base.Metrics(),
			Labels:    labels,
		}:
		}
	}
	if err := scheduler.base.Err(); err != nil {
		scheduler.setErr(err)
	}
	scheduler.emitSchedulerProbe(job, "complete", queueLatency, 0, false)
}

func (scheduler *ScheduledModel) baseTokens(job *scheduledJob) iter.Seq[inference.Token] {
	opts := scheduledGenerateOptions(job.req.Sampler)
	if len(job.req.Messages) > 0 {
		messages := append([]inference.Message(nil), job.req.Messages...)
		return scheduler.base.Chat(job.ctx, messages, opts...)
	}
	return scheduler.base.Generate(job.ctx, job.req.Prompt, opts...)
}

func (scheduler *ScheduledModel) register(job *scheduledJob) {
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	scheduler.active[job.req.ID] = job
}

func (scheduler *ScheduledModel) unregister(id string) {
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	delete(scheduler.active, id)
}

func (scheduler *ScheduledModel) emitSchedulerProbe(job *scheduledJob, event string, queueLatency, firstTokenLatency time.Duration, cancelled bool) {
	scheduler.mu.Lock()
	sink := scheduler.probeSink
	queueDepth := len(scheduler.queue)
	scheduler.mu.Unlock()
	if sink == nil || job == nil {
		return
	}
	sink.EmitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventScheduler,
		Phase: inference.ProbePhaseQueue,
		Labels: map[string]string{
			"request_id": job.req.ID,
			"event":      event,
			"model":      job.req.Model,
		},
		Scheduler: &inference.ProbeScheduler{
			RequestID:               job.req.ID,
			Event:                   event,
			QueueDepth:              queueDepth,
			QueueLatencyMillis:      millis(queueLatency),
			FirstTokenLatencyMillis: millis(firstTokenLatency),
			TotalLatencyMillis:      millis(time.Since(job.queuedAt)),
			Cancelled:               cancelled,
		},
	})
}

func (scheduler *ScheduledModel) setErr(err error) {
	if scheduler == nil || err == nil {
		return
	}
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	scheduler.lastErr = err
}

func (scheduler *ScheduledModel) nextRequestID() string {
	return core.Sprintf("%s-%d", scheduler.requestIDPrefix, scheduler.nextID.Add(1))
}

func scheduledGenerateOptions(cfg inference.SamplerConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{}
	if cfg.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	opts = append(opts, inference.WithTemperature(cfg.Temperature))
	if cfg.TopK > 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP > 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if cfg.RepeatPenalty > 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(cfg.StopTokens...))
	}
	if cfg.ReturnLogits {
		opts = append(opts, inference.WithLogits())
	}
	return opts
}

func cloneSchedulerLabels(labels map[string]string) map[string]string {
	out := map[string]string{}
	for key, value := range labels {
		out[key] = value
	}
	return out
}

func millisString(duration time.Duration) string {
	return core.Sprintf("%.3f", millis(duration))
}

func millis(duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return float64(duration) / float64(time.Millisecond)
}
