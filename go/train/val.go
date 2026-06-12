// SPDX-Licence-Identifier: EUPL-1.2

// Validation loss (#97): the other half of the Chladni instrument. The
// train-loss curve alone is half-blind — the cascade read on a run is an
// amplitude oscillation across the TRAIN AND VAL loss curves together, so
// the native loop forwards a fixed validation subset (no gradients, no
// optimizer movement) on a cadence and records the curve beside the
// train losses. Points land on the result, in checkpoint metadata, and on
// the probe sink as loss_type=val — the v0 LEM instrument's schema.
//
// The subset is built ONCE at run start and reused for every pass: the
// curve must compare like with like, exactly as the eval probes stay
// fixed for the score cascade.

package train

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/spine"
)

// SFTValLoss is one validation point: the mean no-grad loss over the fixed
// validation subset at an optimizer step.
type SFTValLoss struct {
	Step int     `json:"step"`
	Loss float64 `json:"loss"`
}

// sftValidSamplesDefault caps the fixed validation subset. Validation is a
// full forward per batch per pass — big enough to smooth single-sample
// noise, small enough to stay free against the training step.
const sftValidSamplesDefault = 32

// SFTValEvery resolves the validation cadence: ValEvery when set, else the
// eval cadence (the probes and the val pass usually want the same clock),
// else 0 — baseline-only.
func SFTValEvery(cfg SFTConfig) int {
	if cfg.ValEvery > 0 {
		return cfg.ValEvery
	}
	return cfg.EvalEvery
}

// BuildSFTValidationBatches drains the fixed validation subset from
// cfg.ValidData (up to ValidSamples) and tokenizes it with the run's
// batching settings. Sequence packing is forced off — the subset must
// stay sample-stable across runs and passes.
func BuildSFTValidationBatches(tok *spine.Tokenizer, cfg SFTConfig) ([]SFTBatch, error) {
	if cfg.ValidData == nil {
		return nil, nil
	}
	limit := cfg.ValidSamples
	if limit <= 0 {
		limit = sftValidSamplesDefault
	}
	samples := make([]dataset.Sample, 0, limit)
	for len(samples) < limit {
		sample, ok, err := cfg.ValidData.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		samples = append(samples, sample)
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: validation set produced no samples")
	}
	valCfg := cfg
	valCfg.SequencePacking = false
	return BuildSFTBatches(tok, dataset.NewSliceDataset(samples), valCfg)
}

// ArmSFTValidation installs the validation lane on a result: the fixed
// batches, the cadence, and the loss function (the adapter's no-grad
// forward, injected so the scheduling machinery stays testable without
// Metal). The orchestrator arms once before the epoch loop.
func ArmSFTValidation(result *SFTResult, batches []SFTBatch, every int, lossFn func(SFTBatch) (float64, bool)) {
	if result == nil || len(batches) == 0 || lossFn == nil {
		return
	}
	result.valBatches = batches
	result.valEvery = every
	result.valLossFn = lossFn
}

// RunSFTValidationPass forwards the fixed subset under the current adapter
// weights and records the mean loss at the result's current step — the
// step-0 call is the run's baseline point. A loss failure is loud: the
// operator armed the instrument, so a silent gap would forge the curve.
func RunSFTValidationPass(cfg SFTConfig, result *SFTResult) error {
	if result == nil || len(result.valBatches) == 0 || result.valLossFn == nil {
		return nil
	}
	sum := 0.0
	for _, batch := range result.valBatches {
		loss, ok := result.valLossFn(batch)
		if !ok {
			return core.NewError("mlx: validation loss pass failed")
		}
		sum += loss
	}
	mean := sum / float64(len(result.valBatches))
	result.LastValLoss = mean
	result.ValLosses = append(result.ValLosses, SFTValLoss{Step: result.Steps, Loss: mean})

	if sink := sftProbeSink(cfg); sink != nil {
		sink.EmitProbe(probe.Event{
			Kind:  probe.KindTraining,
			Phase: probe.PhaseTraining,
			Step:  result.Steps,
			Training: &probe.Training{
				Step:     result.Steps,
				Loss:     mean,
				LossType: probe.LossTypeVal,
			},
		})
	}
	return nil
}

// maybeRunSFTValidation is the in-loop gate: a pass at every valEvery-th
// optimizer step when the lane is armed.
func maybeRunSFTValidation(cfg SFTConfig, result *SFTResult) error {
	if result == nil || result.valEvery <= 0 || len(result.valBatches) == 0 {
		return nil
	}
	if result.Steps%result.valEvery != 0 {
		return nil
	}
	return RunSFTValidationPass(cfg, result)
}
