// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/probe"
)

// TrainSFT runs native supervised LoRA fine-tuning against a loaded MLX model.
func (m *Model) TrainSFT(ctx context.Context, dataset SFTDataset, cfg SFTConfig) (*SFTResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	if dataset == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}
	tok := m.Tokenizer()
	if tok == nil || tok.tok == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}

	cfg = normalizeSFTConfig(cfg)
	adapter, err := m.sftAdapter(cfg)
	if err != nil {
		return nil, err
	}
	if adapter == nil {
		return nil, core.NewError("mlx: LoRA adapter is nil")
	}

	adamCfg := sftAdamWConfig(cfg)
	optimizer := NewAdamW(&adamCfg)
	result := &SFTResult{Adapter: adapter}
	if err := ApplySFTResumeMetadata(result, cfg); err != nil {
		return result, err
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if epoch > 1 {
			if resetter, ok := dataset.(SFTResetter); ok {
				if err := resetter.Reset(); err != nil {
					return result, err
				}
			} else {
				return result, core.NewError("mlx: SFT dataset must implement Reset for multiple epochs")
			}
		}

		if err := m.runSFTDatasetEpoch(ctx, tok, dataset, adapter, optimizer, cfg, result, epoch); err != nil {
			return result, err
		}
		result.Epochs = epoch
	}

	if result.Steps == 0 {
		return result, core.NewError("mlx: SFT dataset produced no trainable batches")
	}
	if cfg.SavePath != "" {
		if err := adapter.Save(cfg.SavePath); err != nil {
			return result, err
		}
		result.AdapterPath = cfg.SavePath
		meta := NewSFTArtifactMetadata(cfg.SavePath, m.ModelType(), cfg, result)
		if err := SaveSFTCheckpointMetadata(cfg.SavePath, meta); err != nil {
			return result, err
		}
		result.AdapterMetadata = &meta
	}
	if cfg.Merge {
		adapter.Merge()
	}
	return result, nil
}

func (m *Model) sftAdapter(cfg SFTConfig) (*LoRAAdapter, error) {
	if cfg.ResumePath != "" {
		adapter, err := m.LoadLoRA(cfg.ResumePath)
		if err != nil {
			return nil, err
		}
		adapter.Config.ProbeSink = nil
		if cfg.LoRA.Lambda != 0 {
			adapter.Config.Lambda = cfg.LoRA.Lambda
		}
		return adapter, nil
	}
	loraCfg := cfg.LoRA
	loraCfg.ProbeSink = nil
	return NewLoRA(m, &loraCfg), nil
}

func (m *Model) runSFTDatasetEpoch(ctx context.Context, tok *Tokenizer, dataset SFTDataset, adapter *LoRAAdapter, optimizer *AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
	current := make([]sftExample, 0, cfg.BatchSize)
	accumulated := make([]SFTBatch, 0, cfg.GradientAccumulationSteps)
	flushAccumulated := func() error {
		if len(accumulated) == 0 {
			return nil
		}
		if err := m.runSFTBatchGroup(ctx, accumulated, adapter, optimizer, cfg, result, epoch); err != nil {
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
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		example, usable, err := buildSFTExample(tok, sample, cfg)
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

func (m *Model) runSFTBatch(ctx context.Context, batch SFTBatch, adapter *LoRAAdapter, optimizer *AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
	return m.runSFTBatchGroup(ctx, []SFTBatch{batch}, adapter, optimizer, cfg, result, epoch)
}

func (m *Model) runSFTBatchGroup(ctx context.Context, batches []SFTBatch, adapter *LoRAAdapter, optimizer *AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	loss := sftAdapterStep(adapter, batches, optimizer)
	if loss == nil {
		return core.NewError("mlx: LoRA SFT step returned nil loss")
	}
	Materialize(loss)
	lossValue := loss.Float()
	Free(loss)

	result.Steps++
	result.OptimizerSteps = result.Steps
	result.LastLoss = lossValue
	result.Losses = append(result.Losses, lossValue)

	if cfg.CheckpointDir != "" && cfg.CheckpointEvery > 0 && result.Steps%cfg.CheckpointEvery == 0 {
		path := core.PathJoin(cfg.CheckpointDir, core.Sprintf("step-%06d", result.Steps))
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

	if cfg.EvalEvery > 0 && len(cfg.EvalPrompts) > 0 && result.Steps%cfg.EvalEvery == 0 {
		for _, prompt := range cfg.EvalPrompts {
			if err := ctx.Err(); err != nil {
				return err
			}
			text, err := m.Generate(prompt, WithMaxTokens(cfg.EvalMaxTokens))
			if err != nil {
				return err
			}
			result.Evaluations = append(result.Evaluations, SFTEvalResult{
				Step:   result.Steps,
				Prompt: prompt,
				Text:   text,
			})
		}
	}

	if sink := sftProbeSink(cfg); sink != nil {
		sink.EmitProbe(probe.Event{
			Kind:  probe.KindTraining,
			Phase: probe.PhaseTraining,
			Step:  result.Steps,
			Meta: map[string]string{
				"batch_size":                  core.Sprintf("%d", cfg.BatchSize),
				"effective_batch_size":        core.Sprintf("%d", SFTEffectiveBatchSize(cfg)),
				"gradient_accumulation_steps": core.Sprintf("%d", cfg.GradientAccumulationSteps),
				"sequence_packing":            core.Sprintf("%t", cfg.SequencePacking),
				"optimizer_step":              core.Sprintf("%d", result.OptimizerSteps),
				"sft_checkpoint_metadata_ver": core.Sprintf("%d", SFTCheckpointMetadataVersion),
			},
			Training: &probe.Training{
				Step:         result.Steps,
				Epoch:        epoch,
				Loss:         lossValue,
				LearningRate: cfg.LearningRate,
			},
		})
	}
	return nil
}

func sftAdapterStep(adapter *LoRAAdapter, batches []SFTBatch, optimizer *AdamW) *Array {
	if len(batches) == 0 {
		return nil
	}
	if len(batches) == 1 {
		return adapter.Step(batches[0].Batch, batches[0].Targets, optimizer)
	}
	metalBatches := make([]Batch, len(batches))
	targets := make([][][]int, len(batches))
	for i, batch := range batches {
		metalBatches[i] = batch.Batch
		targets[i] = batch.Targets
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
	if p.maxSeqLen > 0 && len(example.inputs) > p.maxSeqLen {
		start := len(example.inputs) - p.maxSeqLen
		example.inputs = append([]int(nil), example.inputs[start:]...)
		example.targets = append([]int(nil), example.targets[start:]...)
		example.mask = append([]float32(nil), example.mask[start:]...)
	}
	p.current.inputs = append(p.current.inputs, example.inputs...)
	p.current.targets = append(p.current.targets, example.targets...)
	p.current.mask = append(p.current.mask, example.mask...)
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
	example := sftExample{
		inputs:  append([]int(nil), p.current.inputs...),
		targets: append([]int(nil), p.current.targets...),
		mask:    append([]float32(nil), p.current.mask...),
	}
	p.current = sftExample{}
	return p.emit(example)
}
