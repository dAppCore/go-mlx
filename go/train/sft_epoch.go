// SPDX-Licence-Identifier: EUPL-1.2

// Package train holds the native training machinery — SFT batch building,
// sequence packing, checkpoint metadata, the LoRA epoch loop, and the SSD
// (sampling-and-fine-tuning) pipeline with its code benchmark. The root mlx
// package aliases the exported types and keeps the Model-bound entry points
// (Model.TrainSFT / Model.RunSSD), which delegate here.
package train

import (
	"context"
	"strconv"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/spine"
)

// TrainSFT runs native supervised LoRA fine-tuning against a loaded MLX model.
func RunSFTDatasetEpoch(ctx context.Context, m Model, tok *spine.Tokenizer, ds dataset.Dataset, adapter *metal.LoRAAdapter, optimizer *metal.AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
	current := make([]sftExample, 0, cfg.BatchSize)
	accumulated := make([]SFTBatch, 0, cfg.GradientAccumulationSteps)
	flushAccumulated := func() error {
		if len(accumulated) == 0 {
			return nil
		}
		if err := runSFTBatchGroup(ctx, m, accumulated, adapter, optimizer, cfg, result, epoch); err != nil {
			return err
		}
		accumulated = accumulated[:0]
		return nil
	}
	flushCurrent := func() error {
		if len(current) == 0 {
			return nil
		}
		accumulated = append(accumulated, sftBatchFromExamples(current))
		current = current[:0]
		if len(accumulated) >= cfg.GradientAccumulationSteps {
			return flushAccumulated()
		}
		return nil
	}
	emit := func(example sftExample) error {
		current = append(current, example)
		if len(current) >= cfg.BatchSize {
			return flushCurrent()
		}
		return nil
	}

	var packer *sftStreamingPacker
	if cfg.SequencePacking {
		packer = newSFTStreamingPacker(cfg.MaxSeqLen, emit)
	}
	// Narrowed per-sample SFTConfig — buildSFTExample only reads
	// MaxSeqLen + NoEOS so we strip the rest. Avoids copying the full
	// SFTConfig (including embedded LoRAConfig with two []string
	// slices) on every dataset row across every epoch. Same trick
	// BuildDatasetBatches uses for the same call.
	exampleCfg := SFTConfig{MaxSeqLen: cfg.MaxSeqLen, NoEOS: cfg.NoEOS}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		sample, ok, err := ds.Next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		example, usable, err := buildSFTExample(tok, sample, exampleCfg)
		if err != nil {
			return err
		}
		if !usable {
			continue
		}
		result.Samples++
		if packer != nil {
			if err := packer.add(example); err != nil {
				return err
			}
			continue
		}
		if err := emit(example); err != nil {
			return err
		}
	}
	if packer != nil {
		if err := packer.finish(); err != nil {
			return err
		}
	}
	if err := flushCurrent(); err != nil {
		return err
	}
	return flushAccumulated()
}

func runSFTBatch(ctx context.Context, m Model, batch SFTBatch, adapter *metal.LoRAAdapter, optimizer *metal.AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
	return runSFTBatchGroup(ctx, m, []SFTBatch{batch}, adapter, optimizer, cfg, result, epoch)
}

func runSFTBatchGroup(ctx context.Context, m Model, batches []SFTBatch, adapter *metal.LoRAAdapter, optimizer *metal.AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loss := sftAdapterStep(adapter, batches, optimizer)
	if loss == nil {
		return core.NewError("mlx: LoRA SFT step returned nil loss")
	}
	metal.Materialize(loss)
	lossValue := loss.Float()
	metal.Free(loss)

	result.Steps++
	result.OptimizerSteps = result.Steps
	result.LastLoss = lossValue
	result.Losses = append(result.Losses, lossValue)

	// Validation precedes checkpointing so a checkpoint saved at this
	// step carries the val loss measured under exactly these weights.
	if err := maybeRunSFTValidation(cfg, result); err != nil {
		return err
	}

	if cfg.CheckpointDir != "" && cfg.CheckpointEvery > 0 && result.Steps%cfg.CheckpointEvery == 0 {
		path := core.PathJoin(cfg.CheckpointDir, sftStepName(result.Steps))
		if err := adapter.Save(path); err != nil {
			return err
		}
		meta := NewSFTCheckpointMetadata(path, m.ModelType(), cfg, result, epoch)
		if err := SaveSFTCheckpointMetadata(path, meta); err != nil {
			return err
		}
		result.Checkpoints = append(result.Checkpoints, path)
		result.CheckpointMetadata = append(result.CheckpointMetadata, meta)
	}

	if err := runSFTEvaluations(ctx, m, cfg, result); err != nil {
		return err
	}

	if sink := sftProbeSink(cfg); sink != nil {
		tokens := 0
		for i := range batches {
			for _, row := range batches[i].Batch.Tokens {
				tokens += len(row)
			}
		}
		meta := make(map[string]string, 6)
		meta["batch_size"] = strconv.Itoa(cfg.BatchSize)
		meta["effective_batch_size"] = strconv.Itoa(SFTEffectiveBatchSize(cfg))
		meta["gradient_accumulation_steps"] = strconv.Itoa(cfg.GradientAccumulationSteps)
		meta["sequence_packing"] = strconv.FormatBool(cfg.SequencePacking)
		meta["optimizer_step"] = strconv.Itoa(result.OptimizerSteps)
		meta["sft_checkpoint_metadata_ver"] = strconv.Itoa(SFTCheckpointMetadataVersion)
		sink.EmitProbe(probe.Event{
			Kind:  probe.KindTraining,
			Phase: probe.PhaseTraining,
			Step:  result.Steps,
			Meta:  meta,
			Training: &probe.Training{
				Step:         result.Steps,
				Epoch:        epoch,
				Loss:         lossValue,
				LearningRate: cfg.LearningRate,
				LossType:     probe.LossTypeTrain,
				Tokens:       tokens,
			},
		})
	}
	return nil
}

func runSFTEvaluations(ctx context.Context, m Model, cfg SFTConfig, result *SFTResult) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil {
		return errSFTModelNil
	}
	if result == nil {
		return core.NewError("mlx: SFT result is nil")
	}
	if cfg.EvalEvery <= 0 || len(cfg.EvalPrompts) == 0 || result.Steps%cfg.EvalEvery != 0 {
		return nil
	}
	info := m.Info()
	opts := sftEvalGenerateOptions(cfg)
	passStart := len(result.Evaluations)
	for _, prompt := range cfg.EvalPrompts {
		if err := ctx.Err(); err != nil {
			return err
		}
		text, err := m.Generate(sftEvalPromptForModel(prompt, info), opts...)
		if err != nil {
			return err
		}
		result.Evaluations = append(result.Evaluations, SFTEvalResult{
			Step:   result.Steps,
			Prompt: prompt,
			Text:   text,
		})
	}
	// Capture-first: the pass's raw generations land on disk the moment
	// they exist, with or without the cascade.
	appendCaptureRows(cfg.CaptureSidecarPath, result.Evaluations[passStart:])
	// The score cascade rides the same pass: every generation above is
	// scored NOW (the score is part of the data point) and appended to
	// the sidecar. ~68µs per pair — free against any training step.
	if cfg.ScoreCascade {
		if result.cascade == nil {
			sidecar := cfg.ScoreSidecarPath
			if sidecar == "" && cfg.CheckpointDir != "" {
				sidecar = core.PathJoin(cfg.CheckpointDir, "score-cascade.jsonl")
			}
			result.cascade = newSFTScoreCascade(sidecar, cfg.ScoreWindow)
			result.ScoreSidecarPath = sidecar
		}
		result.cascade.recordPass(result.Steps, result.Evaluations)
		// The pass aggregates ride the probe sink on the same iteration
		// clock as the loss curves — quality and loss-amplitude patterns
		// land on one dashboard.
		if sink := sftProbeSink(cfg); sink != nil {
			if values, ok := result.cascade.passMetrics(result.Steps); ok {
				sink.EmitProbe(probe.Event{
					Kind:  probe.KindScore,
					Phase: probe.PhaseTraining,
					Step:  result.Steps,
					Score: &probe.Score{Label: "sft-eval", Values: values},
				})
			}
		}
	}
	return nil
}

// FinaliseScoreCascade copies the cascade's verdict onto the result — the
// best eval step by windowed composite and the run's full score records.
// Call once after the epoch loop; a run without the cascade is a no-op.
func FinaliseScoreCascade(result *SFTResult) {
	if result == nil || result.cascade == nil {
		return
	}
	result.ScoreRecords = append([]SFTScoreRecord(nil), result.cascade.records...)
	if step, mean, ok := result.cascade.best(); ok {
		result.BestScoreStep = step
		result.BestScoreComposite = mean
	}
}

func sftEvalGenerateOptions(cfg SFTConfig) []spine.GenerateOption {
	opts := []spine.GenerateOption{func(c *spine.GenerateConfig) { c.MaxTokens = cfg.EvalMaxTokens }}
	if cfg.EvalTemperature != 0 {
		opts = append(opts, func(c *spine.GenerateConfig) { c.Temperature = cfg.EvalTemperature })
	}
	return opts
}

func sftAdapterStep(adapter *metal.LoRAAdapter, batches []SFTBatch, optimizer *metal.AdamW) *metal.Array {
	if len(batches) == 0 {
		return nil
	}
	if len(batches) == 1 {
		return adapter.Step(batches[0].Batch, batches[0].Targets, optimizer)
	}
	metalBatches := make([]metal.Batch, len(batches))
	targets := make([][][]int, len(batches))
	// Index iteration — range over []SFTBatch copies the whole struct
	// (Batch's three slice headers + Targets' slice header = 96 B) per
	// iteration just to forward two field reads. Indexing keeps the
	// loop body to two field loads off the underlying array.
	for i := range batches {
		metalBatches[i] = batches[i].Batch
		targets[i] = batches[i].Targets
	}
	return adapter.StepAccumulated(metalBatches, targets, optimizer)
}

func sftProbeSink(cfg SFTConfig) probe.Sink {
	if cfg.ProbeSink != nil {
		return cfg.ProbeSink
	}
	return cfg.LoRA.ProbeSink
}

type sftStreamingPacker struct {
	maxSeqLen int
	emit      func(sftExample) error
	current   sftExample
}

func newSFTStreamingPacker(maxSeqLen int, emit func(sftExample) error) *sftStreamingPacker {
	return &sftStreamingPacker{maxSeqLen: maxSeqLen, emit: emit}
}

func (p *sftStreamingPacker) add(example sftExample) error {
	if p == nil || p.emit == nil || len(example.inputs) == 0 {
		return nil
	}
	if p.maxSeqLen > 0 && len(p.current.inputs) > 0 && len(p.current.inputs)+len(example.inputs) > p.maxSeqLen {
		if err := p.flush(); err != nil {
			return err
		}
	}
	// Truncate by narrowing the source range — the subsequent appends
	// already copy into p.current, so the prior SliceClone trio was
	// wasted intermediate allocation. Mirrors the same pattern adopted
	// in datasetPacker.add.
	srcInputs := example.inputs
	srcTargets := example.targets
	srcMask := example.mask
	if p.maxSeqLen > 0 && len(srcInputs) > p.maxSeqLen {
		start := len(srcInputs) - p.maxSeqLen
		srcInputs = srcInputs[start:]
		srcTargets = srcTargets[start:]
		srcMask = srcMask[start:]
	}
	// First add into an empty accumulator: pre-size to maxSeqLen (when
	// known) so the doubling cascade across subsequent appends collapses
	// into a single allocation per accumulator field. Inputs + Targets +
	// Mask all share ONE backing of 2*maxSeqLen+(maxSeqLen+1)/2 ints. The
	// inputs/targets are two cap-maxSeqLen []int views over the first
	// 2*maxSeqLen ints; mask is a cap-maxSeqLen []float32 view reinterpreted
	// over the trailing (maxSeqLen+1)/2 ints. All three grow in lockstep —
	// mask appends the same count as inputs on every add (see the appends
	// below) and the caller flushes before any add would push past maxSeqLen
	// — so none ever grows past its cap and the in-place appends never
	// realloc or stomp a neighbour's region. []int is 8-byte aligned
	// (exceeds float32's 4-byte requirement) and neither type contains
	// pointers, so the float32 reinterpret is sound and GC scanning of the
	// combined allocation stays straightforward. Folding mask into the
	// shared backing drops the separate []float32 make — one allocation per
	// first-add instead of two. Mirrors datasetPacker.add and the per-example
	// shared-backing-with-mask pattern in buildSFTExamplePromptResponse.
	if p.maxSeqLen > 0 && cap(p.current.inputs) == 0 {
		maskInts := (p.maxSeqLen + 1) / 2
		intBacking := make([]int, 2*p.maxSeqLen+maskInts)
		p.current.inputs = intBacking[:0:p.maxSeqLen]
		p.current.targets = intBacking[p.maxSeqLen : p.maxSeqLen : 2*p.maxSeqLen]
		p.current.mask = unsafe.Slice((*float32)(unsafe.Pointer(&intBacking[2*p.maxSeqLen])), p.maxSeqLen)[:0:p.maxSeqLen]
	}
	p.current.inputs = append(p.current.inputs, srcInputs...)
	p.current.targets = append(p.current.targets, srcTargets...)
	p.current.mask = append(p.current.mask, srcMask...)
	return nil
}

func (p *sftStreamingPacker) finish() error {
	if p == nil {
		return nil
	}
	return p.flush()
}

func (p *sftStreamingPacker) flush() error {
	if p == nil || p.emit == nil || len(p.current.inputs) == 0 {
		return nil
	}
	// Hand the emitted example p.current's backing arrays directly —
	// the immediately-following p.current = sftExample{} drops our
	// last reference to them, so the example is the sole owner. The
	// previous form cloned all three slices then nuked the originals,
	// paying three pointless allocations per flush. The next add()
	// re-allocates fresh buffers via the cap(...) == 0 branch, same
	// cost it pays today.
	example := p.current
	p.current = sftExample{}
	return p.emit(example)
}
