// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

// metal_convert.go: conversions from the spine surface types into the
// metal.* engine types — the per-call dispatch shufflers every Generate /
// Chat / Classify / BatchGenerate entry runs through.

// Compile-time layout guard for the metal.ProbeLogit / probe.Logit
// reinterpret cast in toProbeLogits. Both types carry int32 + float32 +
// float64 with the same Go field ordering; the assertions below break
// the build if either struct grows / shrinks / changes field order,
// forcing a manual review of the unsafe cast.
var _ [unsafe.Sizeof(metal.ProbeLogit{}) - unsafe.Sizeof(probe.Logit{})]byte
var _ [unsafe.Sizeof(probe.Logit{}) - unsafe.Sizeof(metal.ProbeLogit{})]byte
var _ [unsafe.Offsetof(metal.ProbeLogit{}.TokenID) - unsafe.Offsetof(probe.Logit{}.TokenID)]byte
var _ [unsafe.Offsetof(metal.ProbeLogit{}.Logit) - unsafe.Offsetof(probe.Logit{}.Logit)]byte
var _ [unsafe.Offsetof(metal.ProbeLogit{}.Probability) - unsafe.Offsetof(probe.Logit{}.Probability)]byte

// ToMetalGenerateConfig shuffles a spine GenerateConfig into the metal
// package equivalent. Inlined into every Generate / Chat / Classify
// entry — the per-call allocation pattern here drives the dispatch-side
// budget.
//
//	mcfg := spine.ToMetalGenerateConfig(cfg)
func ToMetalGenerateConfig(cfg GenerateConfig) metal.GenerateConfig {
	return metal.GenerateConfig{
		MaxTokens:           cfg.MaxTokens,
		Temperature:         cfg.Temperature,
		TopK:                cfg.TopK,
		TopP:                cfg.TopP,
		MinP:                cfg.MinP,
		Seed:                cfg.Seed,
		SeedSet:             cfg.SeedSet,
		StopTokens:          cfg.StopTokens,
		SuppressTokens:      cfg.SuppressTokens,
		MinTokensBeforeStop: cfg.MinTokensBeforeStop,
		RepeatPenalty:       cfg.RepeatPenalty,
		ProbeSink:           ToMetalProbeSink(cfg.ProbeSink),
		TraceTokenPhases:    cfg.TraceTokenPhases,
		TraceTokenText:      cfg.TraceTokenText,
		ClearCache:          cfg.GenerationClearCache,
		ClearCacheInterval:  cfg.GenerationClearCacheInterval,
	}
}

// metalProbeSinkAdapter forwards metal.ProbeEvent into a probe.Sink
// after the metal→probe event conversion. Replaces the per-call closure
// allocation in ToMetalProbeSink — the closure form captured `sink`
// into a fresh func per Generate/Chat/Classify call (24 B + GC pressure
// on the per-call hot path even when ProbeSink was non-nil but emitted
// few events). The struct form is heap-allocated once per call but is
// two pointer-sized words and qualifies for stack allocation when the
// metal config doesn't escape.
type metalProbeSinkAdapter struct {
	sink probe.Sink
}

// EmitProbe converts metal.ProbeEvent to probe.Event and forwards to the
// wrapped sink. Called per token during generation when the caller
// supplies a ProbeSink — the conversion still allocates per event but
// the dispatch site no longer allocates a closure per Generate call.
func (a metalProbeSinkAdapter) EmitProbe(event metal.ProbeEvent) {
	a.sink.EmitProbe(toProbeEvent(event))
}

// ToMetalProbeSink wraps a probe.Sink for the metal engine. Nil in,
// nil out — the steady-state Generate call carries no sink.
func ToMetalProbeSink(sink probe.Sink) metal.ProbeSink {
	if sink == nil {
		return nil
	}
	return metalProbeSinkAdapter{sink: sink}
}

func toProbeEvent(event metal.ProbeEvent) probe.Event {
	// Read sub-fields direct through the source pointer — the previous
	// `x := *event.X` dereference-copy form materialised the entire
	// substruct (ProbeLogits alone is ~130 B with three slice headers
	// + a map header) into a local before reading individual fields.
	// toProbeEvent fires per probe event, which under ProbeSink is
	// emitted PER TOKEN during generation — skipping the redundant
	// substruct copy compounds across long generations.
	out := probe.Event{
		Kind:  probe.Kind(event.Kind),
		Phase: probe.Phase(event.Phase),
		Step:  event.Step,
		Meta:  cloneProbeMeta(event.Meta),
	}
	if event.Token != nil {
		token := event.Token
		out.Token = &probe.Token{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if event.Logits != nil {
		logits := event.Logits
		out.Logits = &probe.Logits{
			Shape:      core.SliceClone(logits.Shape),
			VocabSize:  logits.VocabSize,
			MaxTokenID: logits.MaxTokenID,
			MaxLogit:   logits.MaxLogit,
			MinTokenID: logits.MinTokenID,
			MinLogit:   logits.MinLogit,
			MeanLogit:  logits.MeanLogit,
			Top:        toProbeLogits(logits.Top),
			Values:     core.SliceClone(logits.Values),
			Meta:       cloneProbeMeta(logits.Meta),
		}
	}
	if event.Entropy != nil {
		entropy := event.Entropy
		out.Entropy = &probe.Entropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if event.SelectedHeads != nil {
		heads := event.SelectedHeads
		out.SelectedHeads = &probe.HeadSelection{
			Layer:  heads.Layer,
			Heads:  core.SliceClone(heads.Heads),
			Scores: core.SliceClone(heads.Scores),
		}
	}
	if event.LayerCoherence != nil {
		coherence := event.LayerCoherence
		out.LayerCoherence = &probe.LayerCoherence{
			Layer:          coherence.Layer,
			KeyCoherence:   coherence.KeyCoherence,
			ValueCoherence: coherence.ValueCoherence,
			CrossAlignment: coherence.CrossAlignment,
			KVCoupling:     coherence.KVCoupling,
			HeadEntropy:    coherence.HeadEntropy,
			PhaseLock:      coherence.PhaseLock,
		}
	}
	if event.RouterDecision != nil {
		router := event.RouterDecision
		out.RouterDecision = &probe.RouterDecision{
			Layer:       router.Layer,
			TokenID:     router.TokenID,
			ExpertIDs:   core.SliceClone(router.ExpertIDs),
			Weights:     core.SliceClone(router.Weights),
			Temperature: router.Temperature,
		}
	}
	if event.Residual != nil {
		residual := event.Residual
		out.Residual = &probe.ResidualSummary{
			Layer:    residual.Layer,
			Mean:     residual.Mean,
			Variance: residual.Variance,
			RMS:      residual.RMS,
			L2Norm:   residual.L2Norm,
			MaxAbs:   residual.MaxAbs,
		}
	}
	if event.Cache != nil {
		cache := event.Cache
		out.Cache = &probe.CachePressure{
			PromptTokens:    cache.PromptTokens,
			GeneratedTokens: cache.GeneratedTokens,
			LayerCount:      cache.LayerCount,
			CacheTokens:     cache.CacheTokens,
			ProcessedTokens: cache.ProcessedTokens,
			MaxCacheTokens:  cache.MaxCacheTokens,
			Utilization:     cache.Utilization,
			Rotating:        cache.Rotating,
		}
	}
	if event.Memory != nil {
		memory := event.Memory
		out.Memory = &probe.MemoryPressure{
			ActiveBytes: memory.ActiveBytes,
			PeakBytes:   memory.PeakBytes,
			CacheBytes:  memory.CacheBytes,
		}
	}
	if event.Training != nil {
		training := event.Training
		out.Training = &probe.Training{
			Step:         training.Step,
			Epoch:        training.Epoch,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
			GradNorm:     training.GradNorm,
		}
	}
	return out
}

func toProbeLogits(logits []metal.ProbeLogit) []probe.Logit {
	if len(logits) == 0 {
		return nil
	}
	// W8-A2 unsafe reinterpret — metal.ProbeLogit and probe.Logit have
	// bit-identical layout (int32 TokenID + float32 Logit + float64
	// Probability, with the same field order). The compile-time guard
	// at the top of the file fires if either struct ever drifts. Cast
	// the source slice header in-place, then `copy` does one memcpy
	// instead of len(logits) per-field unpacks. Top-K is commonly
	// 50-100 entries per probe event, emitted per-token when ProbeSink
	// is enabled — every saved unpack compounds across the generation.
	src := unsafe.Slice((*probe.Logit)(unsafe.Pointer(&logits[0])), len(logits))
	out := make([]probe.Logit, len(logits))
	copy(out, src)
	return out
}

func cloneProbeMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	return core.MapClone(meta)
}
