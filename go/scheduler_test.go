// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/scheduler"
)

// These tests cover the mlx-root scheduler.go shim. Algorithmic
// coverage lives in go-inference/go/scheduler/scheduler_test.go; here
// we verify the alias surface + NewScheduledModel forwarder.

type schedulerShimModel struct {
	prompt string
}

func (m *schedulerShimModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.prompt = prompt
	return func(yield func(inference.Token) bool) { yield(inference.Token{Text: prompt}) }
}

func (m *schedulerShimModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (*schedulerShimModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (*schedulerShimModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (*schedulerShimModel) ModelType() string                  { return "shim" }
func (*schedulerShimModel) Info() inference.ModelInfo          { return inference.ModelInfo{Architecture: "test"} }
func (*schedulerShimModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (*schedulerShimModel) Err() error                         { return nil }
func (*schedulerShimModel) Close() error                       { return nil }

func TestScheduledModel_AliasMatchesSchedulerPackage_Good(t *testing.T) {
	// Type aliases are identical types in Go's type system, so this
	// assignment compiles only if the alias is wired through.
	var _ *ScheduledModel = (*scheduler.Model)(nil)
	var cfg SchedulerConfig = scheduler.Config{MaxConcurrent: 2, MaxQueue: 4}
	if cfg.MaxConcurrent != 2 || cfg.MaxQueue != 4 {
		t.Fatalf("alias round-trip = %+v", cfg)
	}
}

func TestNewScheduledModel_BuildsSchedulerModel_Good(t *testing.T) {
	base := &schedulerShimModel{}
	s := NewScheduledModel(base, SchedulerConfig{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1, RequestIDPrefix: "shim"})
	if s == nil {
		t.Fatal("NewScheduledModel returned nil")
	}
	handle, tokens, err := s.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "p"})
	if err != nil {
		t.Fatalf("Schedule() error = %v", err)
	}
	if handle.ID == "" {
		t.Fatal("handle ID empty")
	}
	got, ok := <-tokens
	if !ok || got.Token.Text != "p" {
		t.Fatalf("tokens drained early or wrong text: %+v ok=%v", got, ok)
	}
}

func TestNewScheduledModel_NilBaseAccepted_Ugly(t *testing.T) {
	s := NewScheduledModel(nil, SchedulerConfig{})
	if s == nil {
		t.Fatal("NewScheduledModel(nil) returned nil; want defensive wrapper")
	}
	if _, _, err := s.Schedule(context.Background(), inference.ScheduledRequest{}); err == nil {
		t.Fatal("Schedule on nil-base wrapper should error")
	}
}
