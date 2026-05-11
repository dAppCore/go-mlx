// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"

	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
)

// WakeAgentMemory returns an availability error on unsupported builds.
func (m *Model) WakeAgentMemory(_ context.Context, _ memvid.Store, _ agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return nil, nil, unsupportedBuildError()
}

// Wake returns an availability error on unsupported builds.
func (m *Model) Wake(_ context.Context, _ memvid.Store, _ agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return nil, nil, unsupportedBuildError()
}

// ForkFromBundle returns an availability error on unsupported builds.
func (m *Model) ForkFromBundle(_ context.Context, _ memvid.Store, _ agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return nil, nil, unsupportedBuildError()
}

// ForkState returns an availability error on unsupported builds.
func (m *Model) ForkState(_ context.Context, _ inference.AgentMemoryWakeRequest) (inference.AgentMemorySession, *inference.AgentMemoryWakeResult, error) {
	return nil, nil, unsupportedBuildError()
}

// WakeAgentMemory returns an availability error on unsupported builds.
func (s *ModelSession) WakeAgentMemory(_ context.Context, _ memvid.Store, _ agent.WakeOptions) (*agent.WakeReport, error) {
	return nil, unsupportedBuildError()
}

// Wake returns an availability error on unsupported builds.
func (s *ModelSession) Wake(_ context.Context, _ memvid.Store, _ agent.WakeOptions) (*agent.WakeReport, error) {
	return nil, unsupportedBuildError()
}

// WakeState returns an availability error on unsupported builds.
func (s *ModelSession) WakeState(_ context.Context, _ inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	return nil, unsupportedBuildError()
}

// SleepAgentMemory returns an availability error on unsupported builds.
func (s *ModelSession) SleepAgentMemory(_ context.Context, _ memvid.Writer, _ agent.SleepOptions) (*agent.SleepReport, error) {
	return nil, unsupportedBuildError()
}

// Sleep returns an availability error on unsupported builds.
func (s *ModelSession) Sleep(_ context.Context, _ memvid.Writer, _ agent.SleepOptions) (*agent.SleepReport, error) {
	return nil, unsupportedBuildError()
}

// SleepState returns an availability error on unsupported builds.
func (s *ModelSession) SleepState(_ context.Context, _ inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	return nil, unsupportedBuildError()
}

// AppendAndSleepAgentMemory returns an availability error on unsupported builds.
func (s *ModelSession) AppendAndSleepAgentMemory(_ context.Context, _ string, _ memvid.Writer, _ agent.SleepOptions) (*agent.SleepReport, error) {
	return nil, unsupportedBuildError()
}

// AppendAndSleep returns an availability error on unsupported builds.
func (s *ModelSession) AppendAndSleep(_ context.Context, _ string, _ memvid.Writer, _ agent.SleepOptions) (*agent.SleepReport, error) {
	return nil, unsupportedBuildError()
}

// GenerateAndSleepAgentMemory returns an availability error on unsupported builds.
func (s *ModelSession) GenerateAndSleepAgentMemory(_ context.Context, _ memvid.Writer, _ agent.SleepOptions, _ ...GenerateOption) (string, *agent.SleepReport, error) {
	return "", nil, unsupportedBuildError()
}

// GenerateAndSleep returns an availability error on unsupported builds.
func (s *ModelSession) GenerateAndSleep(_ context.Context, _ memvid.Writer, _ agent.SleepOptions, _ ...GenerateOption) (string, *agent.SleepReport, error) {
	return "", nil, unsupportedBuildError()
}
