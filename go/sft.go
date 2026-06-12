// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/train"
)

// sft.go: the Model-bound SFT entry point. The training machinery (batch
// building, packing, checkpoint metadata, the epoch loop) lives in
// dappco.re/go/mlx/train; root keeps the orchestrator that owns adapter
// creation and aliases the public surface.

// SFTConfig configures native supervised LoRA fine-tuning.
type SFTConfig = train.SFTConfig

// SFTBatch is a tokenized training batch with shifted targets.
type SFTBatch = train.SFTBatch

// SFTEvalResult records one eval prompt output captured during training.
type SFTEvalResult = train.SFTEvalResult

// SFTLoRAMetadata records the adapter identity needed to reproduce an SFT run.
type SFTLoRAMetadata = train.SFTLoRAMetadata

// SFTAdamWMetadata records optimizer hyperparameters for checkpoint replay.
type SFTAdamWMetadata = train.SFTAdamWMetadata

// SFTCheckpointMetadata is the portable JSON sidecar for checkpoints and final adapters.
type SFTCheckpointMetadata = train.SFTCheckpointMetadata

// SFTMetrics is the JSON-friendly training summary for dashboards and probes.
type SFTMetrics = train.SFTMetrics

// SFTResult records the outcome of a native SFT LoRA run.
type SFTResult = train.SFTResult

// SFTCheckpointMetadataVersion versions the checkpoint metadata sidecar.
const SFTCheckpointMetadataVersion = train.SFTCheckpointMetadataVersion

// SFTEffectiveBatchSize returns the optimizer batch size after accumulation.
func SFTEffectiveBatchSize(cfg SFTConfig) int { return train.SFTEffectiveBatchSize(cfg) }

// BuildSFTTrainingBatches tokenizes an SFT dataset using runner-level batching settings.
func BuildSFTTrainingBatches(tok *Tokenizer, ds dataset.Dataset, cfg SFTConfig) ([]SFTBatch, error) {
	return train.BuildSFTTrainingBatches(tok, ds, cfg)
}

// BuildSFTBatches tokenizes an SFT dataset into response-masked training batches.
func BuildSFTBatches(tok *Tokenizer, ds dataset.Dataset, cfg SFTConfig) ([]SFTBatch, error) {
	return train.BuildSFTBatches(tok, ds, cfg)
}

// BuildDatasetBatches tokenizes a dataset with optional sequence packing.
//
//	batches, err := mlx.BuildDatasetBatches(tok, ds, dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 1024})
func BuildDatasetBatches(tok *Tokenizer, ds dataset.Dataset, cfg dataset.BatchConfig) ([]SFTBatch, error) {
	return train.BuildDatasetBatches(tok, ds, cfg)
}

// NewSFTCheckpointMetadata builds the sidecar metadata for a mid-run checkpoint.
func NewSFTCheckpointMetadata(path string, model string, cfg SFTConfig, result *SFTResult, epoch int) SFTCheckpointMetadata {
	return train.NewSFTCheckpointMetadata(path, model, cfg, result, epoch)
}

// NewSFTArtifactMetadata builds the sidecar metadata for the final adapter.
func NewSFTArtifactMetadata(path string, model string, cfg SFTConfig, result *SFTResult) SFTCheckpointMetadata {
	return train.NewSFTArtifactMetadata(path, model, cfg, result)
}

// SaveSFTCheckpointMetadata writes the JSON sidecar next to an adapter file.
func SaveSFTCheckpointMetadata(path string, meta SFTCheckpointMetadata) error {
	return train.SaveSFTCheckpointMetadata(path, meta)
}

// LoadSFTCheckpointMetadata reads the JSON sidecar next to an adapter file.
func LoadSFTCheckpointMetadata(path string) (*SFTCheckpointMetadata, error) {
	return train.LoadSFTCheckpointMetadata(path)
}

// ApplySFTResumeMetadata seeds a result with the checkpoint being resumed.
func ApplySFTResumeMetadata(result *SFTResult, cfg SFTConfig) error {
	return train.ApplySFTResumeMetadata(result, cfg)
}

func (m *Model) TrainSFT(ctx context.Context, ds dataset.Dataset, cfg SFTConfig) (*SFTResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}
	tok := m.Tokenizer()
	if !tok.Valid() {
		return nil, core.NewError("mlx: tokenizer is nil")
	}

	cfg = train.NormalizeSFTConfigForModel(cfg, m.Info())
	adapter, err := m.sftAdapter(cfg)
	if err != nil {
		return nil, err
	}
	if adapter == nil {
		return nil, core.NewError("mlx: LoRA adapter is nil")
	}

	adamCfg := train.SFTAdamWConfig(cfg)
	optimizer := NewAdamW(&adamCfg)
	result := &SFTResult{Adapter: adapter}
	defer train.FinaliseScoreCascade(result)
	if err := ApplySFTResumeMetadata(result, cfg); err != nil {
		return result, err
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if epoch > 1 {
			if resetter, ok := ds.(dataset.Resetter); ok {
				if err := resetter.Reset(); err != nil {
					return result, err
				}
			} else {
				return result, core.NewError("mlx: SFT dataset must implement Reset for multiple epochs")
			}
		}

		if err := train.RunSFTDatasetEpoch(ctx, m, tok, ds, adapter, optimizer, cfg, result, epoch); err != nil {
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
