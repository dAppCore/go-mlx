// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"iter"

	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
)

// backend_convert.go: conversions from the metal.* engine types to the root
// mlx.* surface types (metrics, tokens, phase traces, classify/batch). The
// root→metal direction (GenerateConfig, probe sinks) lives in spine.

func toRootMetrics(metrics metal.Metrics) Metrics {
	return Metrics{
		PromptTokens:               metrics.PromptTokens,
		GeneratedTokens:            metrics.GeneratedTokens,
		FirstTokenDuration:         metrics.FirstTokenDuration,
		PrefillDuration:            metrics.PrefillDuration,
		DecodeDuration:             metrics.DecodeDuration,
		TotalDuration:              metrics.TotalDuration,
		PrefillTokensPerSec:        metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:         metrics.DecodeTokensPerSec,
		PeakMemoryBytes:            metrics.PeakMemoryBytes,
		ActiveMemoryBytes:          metrics.ActiveMemoryBytes,
		CacheMemoryBytes:           metrics.CacheMemoryBytes,
		ProcessVirtualMemoryBytes:  metrics.ProcessVirtualMemoryBytes,
		ProcessResidentMemoryBytes: metrics.ProcessResidentMemoryBytes,
		ProcessPeakResidentBytes:   metrics.ProcessPeakResidentBytes,
		PromptCacheHits:            metrics.PromptCacheHits,
		PromptCacheMisses:          metrics.PromptCacheMisses,
		PromptCacheHitTokens:       metrics.PromptCacheHitTokens,
		PromptCacheMissTokens:      metrics.PromptCacheMissTokens,
		PromptCacheRestoreDuration: metrics.PromptCacheRestoreDuration,
		CacheProfile:               toRootCacheProfile(metrics.CacheProfile),
		TurboQuantKVPayload:        toRootTurboQuantKVPayloadEstimate(metrics.TurboQuantKVPayload),
		TokenPhases:                toRootTokenPhaseTraces(metrics.TokenPhases),
		DecodeLane:                 metrics.DecodeLane,
		DecodeLaneReason:           metrics.DecodeLaneReason,
		CompiledLayerHits:          metrics.CompiledLayerHits,
		MTP:                        toRootMTPMetrics(metrics.MTP),
		Adapter:                    toRootAdapterInfo(metrics.Adapter),
	}
}

func toRootTurboQuantKVPayloadEstimate(estimate *metal.TurboQuantKVCachePayloadEstimate) *TurboQuantKVPayloadEstimate {
	if estimate == nil {
		return nil
	}
	return &TurboQuantKVPayloadEstimate{
		Pages:                     estimate.Pages,
		PageVectors:               estimate.PageVectors,
		PageElements:              estimate.PageElements,
		KeyCentroidBytes:          estimate.KeyCentroidBytes,
		KeyQJLSignBytes:           estimate.KeyQJLSignBytes,
		KeyNormBytes:              estimate.KeyNormBytes,
		KeyResidualNormBytes:      estimate.KeyResidualNormBytes,
		ValueCentroidBytes:        estimate.ValueCentroidBytes,
		ValueNormBytes:            estimate.ValueNormBytes,
		OutlierMaskBytes:          estimate.OutlierMaskBytes,
		PayloadBytes:              estimate.PayloadBytes,
		PaddedPayloadBytes:        estimate.PaddedPayloadBytes,
		AlignmentPaddingBytes:     estimate.AlignmentPaddingBytes,
		FP16BaselineBytes:         estimate.FP16BaselineBytes,
		PayloadToFP16Ratio:        estimate.PayloadToFP16Ratio,
		PaddedPayloadToFP16Ratio:  estimate.PaddedPayloadToFP16Ratio,
		PayloadSavingsRatio:       estimate.PayloadSavingsRatio,
		PaddedPayloadSavingsRatio: estimate.PaddedPayloadSavingsRatio,
	}
}

func toRootMTPMetrics(metrics *metal.MTPMetrics) *MTPMetrics {
	if metrics == nil {
		return nil
	}
	return &MTPMetrics{
		DraftTokenSchedule:     append([]int(nil), metrics.DraftTokenSchedule...),
		ProposedTokens:         metrics.ProposedTokens,
		AcceptedTokens:         metrics.AcceptedTokens,
		RejectedTokens:         metrics.RejectedTokens,
		TargetVerifyCalls:      metrics.TargetVerifyCalls,
		TargetCalls:            metrics.TargetCalls,
		DraftCalls:             metrics.DraftCalls,
		AcceptanceRate:         metrics.AcceptanceRate,
		VisibleTokensPerSec:    metrics.VisibleTokensPerSec,
		TargetTokensPerSec:     metrics.TargetTokensPerSec,
		WarmDecodeTokensPerSec: metrics.WarmDecodeTokensPerSec,
		WallDuration:           metrics.WallDuration,
		RestoreDuration:        metrics.RestoreDuration,
		TargetVerifyDuration:   metrics.TargetVerifyDuration,
		TargetDuration:         metrics.TargetDuration,
		DraftDuration:          metrics.DraftDuration,
		PeakMemoryBytes:        metrics.PeakMemoryBytes,
	}
}

func toRootCacheProfile(profile *metal.CacheProfile) *CacheProfile {
	if profile == nil {
		return nil
	}
	return &CacheProfile{
		Architecture:       profile.Architecture,
		TotalCaches:        profile.TotalCaches,
		LocalCaches:        profile.LocalCaches,
		GlobalCaches:       profile.GlobalCaches,
		SharedLayers:       profile.SharedLayers,
		CachelessLayers:    profile.CachelessLayers,
		LocalWindowTokens:  profile.LocalWindowTokens,
		MaxLocalTokens:     profile.MaxLocalTokens,
		MaxLocalCapacity:   profile.MaxLocalCapacity,
		MaxGlobalTokens:    profile.MaxGlobalTokens,
		MaxGlobalCapacity:  profile.MaxGlobalCapacity,
		MaxCacheTokens:     profile.MaxCacheTokens,
		MaxCacheCapacity:   profile.MaxCacheCapacity,
		MaxProcessedTokens: profile.MaxProcessedTokens,
		FullCaches:         profile.FullCaches,
		RotatingCaches:     profile.RotatingCaches,
		FixedCaches:        profile.FixedCaches,
		PagedCaches:        profile.PagedCaches,
		QuantizedCaches:    profile.QuantizedCaches,
		UnknownCaches:      profile.UnknownCaches,
		UnboundedCaches:    profile.UnboundedCaches,
		LocalWindowLeaked:  profile.LocalWindowLeaked,
	}
}

func toRootTokenPhaseTraces(phases []metal.TokenPhaseTrace) []TokenPhaseTrace {
	if len(phases) == 0 {
		return nil
	}
	out := make([]TokenPhaseTrace, len(phases))
	// Single arena allocation for the per-phase NativeEvents slices.
	// TraceTokenPhases-enabled metrics emit one TokenPhaseTrace per
	// decoded token, each with a NativeEvents fanout — collapsing the
	// per-phase make into one slab avoids len(phases) small allocs on
	// every Metrics() read with phase tracing enabled.
	totalNative := 0
	for i := range phases {
		totalNative += len(phases[i].NativeEvents)
	}
	var nativeSlab []NativePhaseTrace
	nativeOffset := 0
	if totalNative > 0 {
		nativeSlab = make([]NativePhaseTrace, totalNative)
	}
	// Index iteration — metal.TokenPhaseTrace is ~192 B (19 duration
	// + Step int + TokenID int32 + TokenText string + FinalToken bool
	// + NativeEvents slice header).
	// metal.NativePhaseTrace is small but contains strings and counters; avoid
	// copying it through a range variable on long traced generations.
	// TraceTokenPhases emits ONE phase trace per decoded token, so for
	// long generations the range form was copying many KB of struct
	// data into loop variables before re-emitting it via field rebuild.
	for i := range phases {
		phase := &phases[i]
		nativeSrc := phase.NativeEvents
		var phaseNative []NativePhaseTrace
		if n := len(nativeSrc); n > 0 {
			end := nativeOffset + n
			phaseNative = nativeSlab[nativeOffset:end:end]
			for j := range nativeSrc {
				event := &nativeSrc[j]
				phaseNative[j] = NativePhaseTrace{
					Name:     event.Name,
					Duration: event.Duration,
					Error:    event.Error,
					Pages:    event.Pages,
					Tokens:   event.Tokens,
				}
			}
			nativeOffset = end
		}
		out[i] = TokenPhaseTrace{
			Step:                   phase.Step,
			TokenID:                phase.TokenID,
			TokenText:              phase.TokenText,
			FinalToken:             phase.FinalToken,
			TotalDuration:          phase.TotalDuration,
			LogitsDuration:         phase.LogitsDuration,
			SampleDuration:         phase.SampleDuration,
			SampleEvalDuration:     phase.SampleEvalDuration,
			TokenReadDuration:      phase.TokenReadDuration,
			DecodeTextDuration:     phase.DecodeTextDuration,
			ProbeTokenDuration:     phase.ProbeTokenDuration,
			YieldDuration:          phase.YieldDuration,
			NextInputDuration:      phase.NextInputDuration,
			ForwardDuration:        phase.ForwardDuration,
			PrefetchDuration:       phase.PrefetchDuration,
			PrefetchLogitsDuration: phase.PrefetchLogitsDuration,
			PrefetchCacheDuration:  phase.PrefetchCacheDuration,
			MaterializeDuration:    phase.MaterializeDuration,
			DetachDuration:         phase.DetachDuration,
			CacheProbeDuration:     phase.CacheProbeDuration,
			OtherDuration:          phase.OtherDuration,
			NativeEvents:           phaseNative,
		}
	}
	return out
}

func toRootNativePhaseTraces(events []metal.NativePhaseTrace) []NativePhaseTrace {
	if len(events) == 0 {
		return nil
	}
	out := make([]NativePhaseTrace, len(events))
	// Index iteration — see toRootTokenPhaseTraces; NativePhaseTrace is
	// ~48 B and the range form copied each event into the loop variable
	// before re-emitting via field rebuild.
	for i := range events {
		event := &events[i]
		out[i] = NativePhaseTrace{
			Name:     event.Name,
			Duration: event.Duration,
			Error:    event.Error,
			Pages:    event.Pages,
			Tokens:   event.Tokens,
		}
	}
	return out
}

// toRootAdapterInfo shuffles an already-cloned metal AdapterInfo into the
// root-facing lora.AdapterInfo. All four callers pass slices that the
// metal side already cloned for caller isolation:
//
//   - toRootMetrics — metrics.Adapter comes from m.lastMetrics.Adapter
//     which is assigned via metal.(*Model).Adapter() (cloneMetalAdapterInfo).
//   - adapterFromNativeInfo + (*Model).Adapter — info.Adapter likewise
//     comes from m.Info() → m.Adapter() which clones.
//   - inference_contract.go — passes adapter.model.Adapter() directly.
//
// The previous core.SliceClone(info.TargetKeys) at this layer was a
// redundant second clone — drops a 64 B / 1 alloc per call by sharing
// the already-isolated slice with the root-side handle. Every Info() /
// Metrics() / Adapter() read on a LoRA-loaded model fires this site.
func toRootAdapterInfo(info metal.AdapterInfo) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: info.TargetKeys,
	}
}

func toRootToken(token metal.Token) Token {
	return Token{ID: token.ID, Value: token.Text, Text: token.Text}
}

func emptyTokenSeq() iter.Seq[Token] {
	return func(func(Token) bool) {}
}

func filteredRootTokenSeq(source iter.Seq[metal.Token], filter *parser.Processor) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		for tok := range source {
			text := filter.Process(tok.Text)
			if text == "" {
				continue
			}
			if !yield(Token{ID: tok.ID, Value: text, Text: text}) {
				return
			}
		}
		if text := filter.Flush(); text != "" {
			yield(Token{Value: text, Text: text})
		}
	}
}

func toRootClassifyResults(results []metal.ClassifyResult) []ClassifyResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]ClassifyResult, len(results))
	// Single arena allocation for all per-result Logits slices. Classify
	// is called over multiple prompts at once and each result has a
	// vocab-sized logits vector — collapsing the per-result clone into
	// one slab cuts N allocs to 1 on the return path. Per-result nil vs
	// non-nil empty is preserved (matches the prior core.SliceClone
	// nil-in / empty-in semantics).
	totalLogits := 0
	for i := range results {
		totalLogits += len(results[i].Logits)
	}
	var logitsSlab []float32
	logitsOffset := 0
	if totalLogits > 0 {
		logitsSlab = make([]float32, totalLogits)
	}
	// Index iteration — metal.ClassifyResult carries a Token (3 fields)
	// + Logits slice header. Skip the per-iter struct copy.
	for i := range results {
		result := &results[i]
		var resultLogits []float32
		switch {
		case result.Logits == nil:
			// nil in -> nil out (matches slices.Clone(nil)).
		case len(result.Logits) == 0:
			resultLogits = []float32{}
		default:
			end := logitsOffset + len(result.Logits)
			resultLogits = logitsSlab[logitsOffset:end:end]
			copy(resultLogits, result.Logits)
			logitsOffset = end
		}
		out[i] = ClassifyResult{
			Token:  toRootToken(result.Token),
			Logits: resultLogits,
		}
	}
	return out
}

func toRootBatchResults(results []metal.BatchResult) []BatchResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]BatchResult, len(results))
	// Single arena allocation for all per-result Tokens slices. Avoids
	// len(results) small allocations on BatchGenerate's return path.
	totalTokens := 0
	for i := range results {
		totalTokens += len(results[i].Tokens)
	}
	tokensSlab := make([]Token, totalTokens)
	tokensOffset := 0
	// Index iteration — metal.BatchResult is a Tokens slice header +
	// error interface. metal.Token is a small (ID int32 + Text string)
	// 24 B struct, but for long-generation batches the outer slice can
	// be hundreds long and the inner Tokens slices can be thousands.
	for i := range results {
		result := &results[i]
		tokensSrc := result.Tokens
		tokensEnd := tokensOffset + len(tokensSrc)
		resultTokens := tokensSlab[tokensOffset:tokensEnd:tokensEnd]
		for j := range tokensSrc {
			resultTokens[j] = toRootToken(tokensSrc[j])
		}
		out[i] = BatchResult{
			Tokens: resultTokens,
			Err:    result.Err,
		}
		tokensOffset = tokensEnd
	}
	return out
}

func toRootAttentionSnapshot(result *metal.AttentionResult) *AttentionSnapshot {
	if result == nil {
		return nil
	}
	return &AttentionSnapshot{
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		Keys:          result.Keys,
		Queries:       result.Queries,
		Architecture:  result.Architecture,
	}
}
