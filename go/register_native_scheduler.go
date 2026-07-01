// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"

	"dappco.re/go/inference"
	"dappco.re/go/inference/scheduler"
)

func (m *nativeTextModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	return m.schedulerModel().Schedule(ctx, req)
}

func (m *nativeTextModel) CancelRequest(ctx context.Context, id string) (inference.RequestCancelResult, error) {
	return m.schedulerModel().CancelRequest(ctx, id)
}

func (m *nativeTextModel) SetProbeSink(sink inference.ProbeSink) {
	if m == nil {
		return
	}
	m.schedulerMu.Lock()
	m.probeSink = sink
	scheduler := m.scheduler
	m.schedulerMu.Unlock()
	if scheduler != nil {
		scheduler.SetProbeSink(sink)
	}
}

func (m *nativeTextModel) schedulerModel() *scheduler.Model {
	if m == nil {
		return scheduler.New(nil, scheduler.Config{})
	}
	m.schedulerMu.Lock()
	defer m.schedulerMu.Unlock()
	if m.scheduler == nil {
		maxConcurrent := DefaultLocalParallelSlots
		if maxConcurrent <= 0 {
			maxConcurrent = 1
		}
		m.scheduler = scheduler.New(nativeSchedulerBase{model: m}, scheduler.Config{
			MaxConcurrent:   maxConcurrent,
			MaxQueue:        maxConcurrent * 4,
			StreamBuffer:    0,
			RequestIDPrefix: "mlx-native",
			ProbeSink:       m.probeSink,
		})
	}
	return m.scheduler
}

type nativeSchedulerBase struct {
	model *nativeTextModel
}

func (base nativeSchedulerBase) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return base.model.Generate(ctx, prompt, opts...)
}

func (base nativeSchedulerBase) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return base.model.Chat(ctx, messages, opts...)
}

func (base nativeSchedulerBase) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return base.model.Classify(ctx, prompts, opts...)
}

func (base nativeSchedulerBase) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return base.model.BatchGenerate(ctx, prompts, opts...)
}

func (base nativeSchedulerBase) ModelType() string {
	return base.model.ModelType()
}

func (base nativeSchedulerBase) Info() inference.ModelInfo {
	return base.model.Info()
}

func (base nativeSchedulerBase) Metrics() inference.GenerateMetrics {
	return base.model.Metrics()
}

func (base nativeSchedulerBase) Err() error {
	return base.model.Err()
}

func (base nativeSchedulerBase) Close() error {
	return base.model.Close()
}
