// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"math"
	"strconv"
	"sync/atomic"

	core "dappco.re/go"
	distillinf "dappco.re/go/inference/distill"
	"dappco.re/go/inference/probe"
)

// distillTempStringCache holds the most recently formatted
// temperature → string mapping. The temperature is per-config
// invariant — every gradient step in a run sees the same value — so
// caching by float64 bits skips strconv.FormatFloat's per-call
// allocation on every step after the first. Uses atomic for the
// cache cell so concurrent emits don't race (also matches the
// lock-free read pattern eval.go uses for its per-call invariants).
type distillTempCacheCell struct {
	bits      uint64
	formatted string
}

var distillTempStringCache atomic.Pointer[distillTempCacheCell]

func formatDistillTemperature(temp float64) string {
	bits := math.Float64bits(temp)
	if cached := distillTempStringCache.Load(); cached != nil && cached.bits == bits {
		return cached.formatted
	}
	formatted := strconv.FormatFloat(temp, 'f', 6, 64)
	distillTempStringCache.Store(&distillTempCacheCell{bits: bits, formatted: formatted})
	return formatted
}

func emitDistillProbe(cfg DistillConfig, result *DistillResult, loss *DistillLoss, cacheStatus string, epoch int) {
	if cfg.ProbeSink == nil {
		return
	}
	metaPtr := distillProbeMetaPool.Get().(*map[string]string)
	meta := *metaPtr
	// Don't bother clear()-ing — every key is reassigned each call,
	// so any stale value is overwritten before the map is read by the
	// sink. Pool entries land here with their bucket array already
	// warm (cap 8) from a previous iteration.
	meta["distillation"] = "true"
	meta["loss_kind"] = string(loss.Kind)
	meta["temperature"] = formatDistillTemperature(loss.Temperature)
	meta["tokens"] = core.Itoa(loss.Tokens)
	meta["teacher_cache"] = cacheStatus
	meta["checkpoint_count"] = core.Itoa(len(result.Checkpoints))
	meta["evaluation_count"] = core.Itoa(len(result.Evaluations))

	training := distillProbeTrainingPool.Get().(*probe.Training)
	training.Step = result.Metrics.Steps
	training.Epoch = epoch
	training.Loss = loss.Value
	training.LearningRate = cfg.LearningRate

	cfg.ProbeSink.EmitProbe(probe.Event{
		Kind:     probe.KindTraining,
		Phase:    probe.PhaseTraining,
		Step:     result.Metrics.Steps,
		Meta:     meta,
		Training: training,
	})
	// Public Sink contract — by the time EmitProbe returns, the sink
	// has either consumed-by-value (in-process listener) or cloned
	// (Recorder.EmitProbe → CloneEvent does a deep-copy of meta +
	// Training). Either way the pool can take the map and pointer
	// back without aliasing risk.
	distillProbeTrainingPool.Put(training)
	distillProbeMetaPool.Put(metaPtr)
}

// DistillationBatchLoss computes KL and soft cross-entropy over masked
// tokens between teacher and student logits — delegates the maths to the
// shared dappco.re/go/inference/distill engine (distillinf.BatchLoss), a
// byte-identical port of what this function computed directly before the
// delegation (scratch-pooled log-softmax + prob accumulation, the same
// shape validation, the same finite guards). Only cfg.Temperature and
// cfg.Loss feed the shared call — the rest of DistillConfig
// (CheckpointDir, ProbeSink, ...) is orchestration-loop state the shared
// engine never reads.
func DistillationBatchLoss(teacher, student DistillLogits, mask [][]float32, cfg DistillConfig) (DistillLoss, error) {
	return distillinf.BatchLoss(teacher, student, mask, distillinf.Config{
		Temperature: cfg.Temperature,
		Loss:        cfg.Loss,
	})
}

// DistillBatchCacheKey returns a stable hash for teacher-logit cache
// lookup — delegates the hashing to the shared
// dappco.re/go/inference/distill engine (distillinf.BatchCacheKey), a
// byte-identical port of this function's hand-rolled JSON emitter. The
// cache KEY bytes MUST stay unchanged across the delegation — the
// teacher-logit cache is keyed on this hash, so any drift would silently
// invalidate every cached entry;
// TestDistillLoss_BatchCacheKeyParity/_EndToEnd pin that contract against
// core.JSONMarshal over adversarial fixtures.
func DistillBatchCacheKey(batch SFTBatch) string {
	return distillinf.BatchCacheKey(batch.Batch.Tokens, batch.Targets, batch.Batch.LossMask)
}
