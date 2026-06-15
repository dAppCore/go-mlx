// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"time"

	core "dappco.re/go"
)

func asyncDecodePrefetch(step int, label string, out *Array) error {
	return asyncDecodePrefetchFor("Model.Generate", step, label, out)
}

func asyncDecodePrefetchFor(scope string, step int, label string, out *Array) error {
	if !asyncDecodePrefetchEnabled() || out == nil || !out.Valid() {
		return nil
	}
	return asyncDecodePrefetchArraysFor(scope, step, label, out)
}

type asyncDecodePrefetchTimings struct {
	Logits time.Duration
	Cache  time.Duration
}

func asyncDecodePrefetchWithCaches(scope string, step int, label string, out *Array, caches []Cache) error {
	if !asyncDecodePrefetchEnabled() {
		return nil
	}
	var stack [64]*Array
	outputs := stack[:0]
	if out != nil && out.Valid() {
		outputs = append(outputs, out)
	}
	for _, cache := range caches {
		outputs = appendCacheDirtyState(outputs, cache)
	}
	if len(outputs) == 0 {
		return nil
	}
	return asyncDecodePrefetchArraysFor(scope, step, label, outputs...)
}

func asyncDecodePrefetchWithCachesTrace(scope string, step int, label string, out *Array, caches []Cache) (asyncDecodePrefetchTimings, error) {
	var timings asyncDecodePrefetchTimings
	if !asyncDecodePrefetchEnabled() {
		return timings, nil
	}
	var stack [64]*Array
	outputs := stack[:0]
	hasLogits := false
	if out != nil && out.Valid() {
		outputs = append(outputs, out)
		hasLogits = true
	}
	for _, cache := range caches {
		outputs = appendCacheDirtyState(outputs, cache)
	}
	if len(outputs) == 0 {
		return timings, nil
	}
	start := time.Now()
	if err := asyncDecodePrefetchArraysFor(scope, step, label, outputs...); err != nil {
		return timings, err
	}
	elapsed := nonZeroTraceDuration(time.Since(start))
	if hasLogits {
		// Keep trace mode on the same combined eval boundary as production.
		// Splitting logits and dirty K/V into separate EvalAsync calls gives
		// cleaner attribution but changes the graph shape being measured.
		timings.Logits = elapsed
	} else {
		timings.Cache = elapsed
	}
	return timings, nil
}

func asyncDecodePrefetchWithCachesTraceSplit(scope string, step int, label string, out *Array, caches []Cache) (asyncDecodePrefetchTimings, error) {
	var timings asyncDecodePrefetchTimings
	if !asyncDecodePrefetchEnabled() {
		return timings, nil
	}
	if out != nil && out.Valid() {
		start := time.Now()
		if err := asyncDecodePrefetchArraysFor(scope, step, label+" logits", out); err != nil {
			return timings, err
		}
		timings.Logits = nonZeroTraceDuration(time.Since(start))
	}
	var stack [64]*Array
	dirty := stack[:0]
	for _, cache := range caches {
		dirty = appendCacheDirtyState(dirty, cache)
	}
	if len(dirty) > 0 {
		start := time.Now()
		if err := asyncDecodePrefetchArraysFor(scope, step, label+" dirty KV", dirty...); err != nil {
			return timings, err
		}
		timings.Cache = nonZeroTraceDuration(time.Since(start))
	}
	return timings, nil
}

func asyncDecodePrefetchArraysFor(scope string, step int, label string, outputs ...*Array) error {
	if !asyncDecodePrefetchEnabled() || len(outputs) == 0 {
		return nil
	}
	if err := EvalAsync(outputs...); err != nil {
		if core.Trim(scope) == "" {
			scope = "Model.Generate"
		}
		return core.E(scope, core.Sprintf("async prefetch %s step %d", label, step), err)
	}
	return nil
}

func nonZeroTraceDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}

func appendTokenPhaseTrace(phases []TokenPhaseTrace, phase TokenPhaseTrace, start time.Time) []TokenPhaseTrace {
	phase.TotalDuration = time.Since(start)
	if accounted := tokenPhaseAccountedDuration(phase); phase.TotalDuration > accounted {
		phase.OtherDuration = phase.TotalDuration - accounted
	}
	return append(phases, phase)
}

func newTokenPhaseTraceBuffer(cfg GenerateConfig) []TokenPhaseTrace {
	if !cfg.TraceTokenPhases || cfg.MaxTokens <= 0 {
		return nil
	}
	return make([]TokenPhaseTrace, 0, cfg.MaxTokens)
}

func tokenPhaseAccountedDuration(phase TokenPhaseTrace) time.Duration {
	return phase.LogitsDuration +
		phase.SampleDuration +
		phase.SampleEvalDuration +
		phase.TokenReadDuration +
		phase.DecodeTextDuration +
		phase.ProbeTokenDuration +
		phase.YieldDuration +
		phase.NextInputDuration +
		phase.ForwardDuration +
		phase.PrefetchDuration +
		phase.MaterializeDuration +
		phase.DetachDuration +
		phase.CacheProbeDuration
}
