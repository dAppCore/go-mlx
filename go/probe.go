// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "dappco.re/go/mlx/probe"

// Legacy aliases — the canonical probe vocabulary lives at
// dappco.re/go/mlx/probe/. mlx-root callers keep their existing Probe*
// surface via these aliases.
type (
	ProbeEvent           = probe.Event
	ProbeEventKind       = probe.Kind
	ProbePhase           = probe.Phase
	ProbeToken           = probe.Token
	ProbeLogit           = probe.Logit
	ProbeLogits          = probe.Logits
	ProbeEntropy         = probe.Entropy
	ProbeHeadSelection   = probe.HeadSelection
	ProbeLayerCoherence  = probe.LayerCoherence
	ProbeRouterDecision  = probe.RouterDecision
	ProbeExpertResidency = probe.ExpertResidency
	ProbeResidualSummary = probe.ResidualSummary
	ProbeCachePressure   = probe.CachePressure
	ProbeMemoryPressure  = probe.MemoryPressure
	ProbeTraining        = probe.Training
	ProbeSink            = probe.Sink
	ProbeSinkFunc        = probe.SinkFunc
	ProbeBus             = probe.Bus
	ProbeRecorder        = probe.Recorder
)

// Event kind + phase constants forwarded from the probe package.
const (
	ProbeEventToken           = probe.KindToken
	ProbeEventLogits          = probe.KindLogits
	ProbeEventEntropy         = probe.KindEntropy
	ProbeEventSelectedHeads   = probe.KindSelectedHeads
	ProbeEventLayerCoherence  = probe.KindLayerCoherence
	ProbeEventRouterDecision  = probe.KindRouterDecision
	ProbeEventExpertResidency = probe.KindExpertResidency
	ProbeEventResidual        = probe.KindResidual
	ProbeEventCachePressure   = probe.KindCachePressure
	ProbeEventMemoryPressure  = probe.KindMemoryPressure
	ProbeEventTraining        = probe.KindTraining

	ProbePhasePrefill  = probe.PhasePrefill
	ProbePhaseDecode   = probe.PhaseDecode
	ProbePhaseTraining = probe.PhaseTraining
)

// NewProbeBus creates a fanout sink.
//
//	bus := mlx.NewProbeBus(sink)
func NewProbeBus(sinks ...ProbeSink) *ProbeBus {
	return probe.NewBus(sinks...)
}

// NewProbeRecorder returns a recorder sink.
//
//	rec := mlx.NewProbeRecorder()
func NewProbeRecorder() *ProbeRecorder {
	return probe.NewRecorder()
}

// WithProbeSink streams typed probe events during generation.
//
//	model.Generate(prompt, mlx.WithProbeSink(sink))
func WithProbeSink(sink ProbeSink) GenerateOption {
	return func(c *GenerateConfig) {
		c.ProbeSink = sink
	}
}

// WithProbeCallback streams typed probe events to a callback during generation.
//
//	model.Generate(prompt, mlx.WithProbeCallback(func(e mlx.ProbeEvent) { … }))
func WithProbeCallback(callback func(ProbeEvent)) GenerateOption {
	if callback == nil {
		return func(*GenerateConfig) {}
	}
	return WithProbeSink(ProbeSinkFunc(callback))
}
