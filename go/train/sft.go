// SPDX-Licence-Identifier: EUPL-1.2

// Package train holds the native training machinery — SFT batch building,
// sequence packing, checkpoint metadata, the LoRA epoch loop, and the SSD
// (sampling-and-fine-tuning) pipeline with its code benchmark. The root mlx
// package aliases the exported types and keeps the Model-bound entry points
// (Model.TrainSFT / Model.RunSSD), which delegate here.
package train

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/spine"
)

// Model is the slice of the root mlx.Model the SFT epoch machinery needs:
// checkpoint metadata wants the model type, and the eval hook generates
// text. *mlx.Model satisfies it structurally.
type Model interface {
	ModelType() string
	Info() spine.ModelInfo
	Generate(prompt string, opts ...spine.GenerateOption) (string, error)
}

var errSFTModelNil = core.NewError("mlx: model is nil")

// sftEvalPromptForModel wraps an eval prompt in the model's chat template
// for the families that require it (gemma4 answers raw prompts poorly).
func sftEvalPromptForModel(prompt string, info spine.ModelInfo) string {
	if !profile.IsGemma4TargetArchitecture(info.Architecture) {
		return prompt
	}
	return chat.Format([]chat.Message{{Role: "user", Content: prompt}}, chat.ConfigForArchitecture(info.Architecture, info.NumHeads))
}

type SFTConfig struct {
	LoRA                      spine.LoRAConfig
	BatchSize                 int
	GradientAccumulationSteps int
	Epochs                    int
	LearningRate              float64
	AdamW                     metal.AdamWConfig
	MaxSeqLen                 int
	SequencePacking           bool
	CheckpointDir             string
	CheckpointEvery           int
	EvalEvery                 int
	EvalPrompts               []string
	EvalMaxTokens             int
	EvalTemperature           float32
	SavePath                  string
	ResumePath                string
	Merge                     bool
	NoEOS                     bool
	ProbeSink                 probe.Sink
	// ScoreCascade arms the lem-scorer over every eval pass (#50): each
	// (probe, output) pair is scored AT GENERATION TIME into a JSONL
	// sidecar, saved checkpoints carry the windowed composite, and the
	// run reports the best checkpoint by the cascade read — the
	// checkpoint is not a guess, it's from semantic analysis. Requires
	// EvalEvery + EvalPrompts (the fixed probe set).
	ScoreCascade     bool
	ScoreSidecarPath string // default <CheckpointDir>/score-cascade.jsonl
	ScoreWindow      int    // eval passes per windowed composite (default 3)
	// Validation (#97) — the other half of the Chladni instrument: a
	// fixed subset of ValidData forwarded with no gradients at ValEvery
	// steps; the mean loss lands in Result.ValLosses, checkpoint
	// metadata, and the probe sink as loss_type=val.
	ValidData    dataset.Dataset // validation set; nil disables the val pass
	ValidSamples int             // samples in the fixed subset (default 32)
	ValEvery     int             // steps between passes (default EvalEvery)
	// CaptureSidecarPath (#97, capture-first): every eval generation
	// appended as a raw JSONL row at the moment it exists, independent of
	// the score cascade — scoring later is archaeology, a missed capture
	// never existed.
	CaptureSidecarPath string
}

// SFTBatch is a tokenized training batch with shifted targets.
type SFTBatch struct {
	Batch   metal.Batch
	Targets [][]int
}

// SFTEvalResult records one eval prompt output captured during training.
type SFTEvalResult struct {
	Step   int
	Prompt string
	Text   string
}

const SFTCheckpointMetadataVersion = 1

// SFTLoRAMetadata records the adapter identity needed to reproduce an SFT run.
type SFTLoRAMetadata struct {
	Rank                 int      `json:"rank"`
	Alpha                float32  `json:"alpha"`
	Scale                float32  `json:"scale,omitempty"`
	TargetKeys           []string `json:"target_keys,omitempty"`
	TargetLayers         []string `json:"target_layers,omitempty"`
	Lambda               float32  `json:"lambda,omitempty"`
	DType                string   `json:"dtype,omitempty"`
	AllowExtendedTargets bool     `json:"allow_extended_targets,omitempty"`
}

// SFTAdamWMetadata records optimizer hyperparameters for checkpoint replay.
type SFTAdamWMetadata struct {
	LearningRate float64 `json:"learning_rate"`
	Beta1        float64 `json:"beta1"`
	Beta2        float64 `json:"beta2"`
	Eps          float64 `json:"eps"`
	WeightDecay  float64 `json:"weight_decay"`
	PackedState  bool    `json:"packed_state"`
}

// SFTCheckpointMetadata is the portable JSON sidecar for checkpoints and final adapters.
type SFTCheckpointMetadata struct {
	Version                   int              `json:"version"`
	Path                      string           `json:"path"`
	AdapterPath               string           `json:"adapter_path,omitempty"`
	ResumePath                string           `json:"resume_path,omitempty"`
	Model                     string           `json:"model,omitempty"`
	Step                      int              `json:"step"`
	OptimizerStep             int              `json:"optimizer_step"`
	Epoch                     int              `json:"epoch"`
	Samples                   int              `json:"samples"`
	Loss                      float64          `json:"loss"`
	ValLoss                   float64          `json:"val_loss,omitempty"`
	LearningRate              float64          `json:"learning_rate"`
	BatchSize                 int              `json:"batch_size"`
	GradientAccumulationSteps int              `json:"gradient_accumulation_steps"`
	EffectiveBatchSize        int              `json:"effective_batch_size"`
	MaxSeqLen                 int              `json:"max_seq_len,omitempty"`
	SequencePacking           bool             `json:"sequence_packing,omitempty"`
	EvalPrompts               []string         `json:"eval_prompts,omitempty"`
	ScoreComposite            float64          `json:"score_composite,omitempty"`
	EvalTemperature           float32          `json:"eval_temperature,omitempty"`
	LoRA                      SFTLoRAMetadata  `json:"lora"`
	AdamW                     SFTAdamWMetadata `json:"adamw"`
}

// SFTMetrics is the JSON-friendly training summary for dashboards and probes.
type SFTMetrics struct {
	Steps                     int     `json:"steps"`
	OptimizerSteps            int     `json:"optimizer_steps"`
	Epochs                    int     `json:"epochs"`
	Samples                   int     `json:"samples"`
	LastLoss                  float64 `json:"last_loss"`
	LearningRate              float64 `json:"learning_rate"`
	BatchSize                 int     `json:"batch_size"`
	GradientAccumulationSteps int     `json:"gradient_accumulation_steps"`
	EffectiveBatchSize        int     `json:"effective_batch_size"`
	CheckpointCount           int     `json:"checkpoint_count"`
	EvaluationCount           int     `json:"evaluation_count"`
}

// SFTResult records the outcome of a native SFT LoRA run.
type SFTResult struct {
	Adapter            *metal.LoRAAdapter
	Steps              int
	OptimizerSteps     int
	Epochs             int
	Samples            int
	LastLoss           float64
	Losses             []float64
	Checkpoints        []string
	CheckpointMetadata []SFTCheckpointMetadata
	Evaluations        []SFTEvalResult
	AdapterPath        string
	AdapterMetadata    *SFTCheckpointMetadata
	ResumePath         string
	ResumedFrom        *SFTCheckpointMetadata
	// Score cascade (#50) — populated when SFTConfig.ScoreCascade is set:
	// every eval scored at generation time, best checkpoint by windowed
	// composite. ScoreRecords carries the run's full vectors in memory;
	// the sidecar holds the durable copy.
	ScoreRecords       []SFTScoreRecord
	BestScoreStep      int
	BestScoreComposite float64
	ScoreSidecarPath   string
	cascade            *sftScoreCascade
	// Validation lane (#97) — armed by ArmSFTValidation: the val-loss
	// curve beside Losses, sparse (one point per pass, keyed by step).
	ValLosses   []SFTValLoss
	LastValLoss float64
	valBatches  []SFTBatch
	valEvery    int
	valLossFn   func(SFTBatch) (float64, bool)
}

// Metrics returns a stable JSON-friendly summary of an SFT run.
func (r *SFTResult) Metrics(cfg SFTConfig) SFTMetrics {
	// Inline the four scalar defaults Metrics actually reads —
	// normalizeSFTConfig calls normalizeSFTLoRAConfig which clones
	// TargetKeys+TargetLayers (two SliceClones) every call. Metrics
	// touches none of that. The trio of helpers Metrics calls below
	// (SFTEffectiveBatchSize, etc.) all read only the already-normalised
	// scalars now hoisted into local vars.
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	gradAccum := cfg.GradientAccumulationSteps
	if gradAccum <= 0 {
		gradAccum = 1
	}
	learningRate := cfg.LearningRate
	if learningRate == 0 {
		if cfg.AdamW.LearningRate != 0 || cfg.AdamW.LearningRateSet {
			learningRate = cfg.AdamW.LearningRate
		} else {
			learningRate = 1e-5
		}
	}
	effectiveBatchSize := batchSize * gradAccum
	if r == nil {
		return SFTMetrics{
			LearningRate:              learningRate,
			BatchSize:                 batchSize,
			GradientAccumulationSteps: gradAccum,
			EffectiveBatchSize:        effectiveBatchSize,
		}
	}
	optimizerSteps := r.OptimizerSteps
	if optimizerSteps == 0 {
		optimizerSteps = r.Steps
	}
	return SFTMetrics{
		Steps:                     r.Steps,
		OptimizerSteps:            optimizerSteps,
		Epochs:                    r.Epochs,
		Samples:                   r.Samples,
		LastLoss:                  r.LastLoss,
		LearningRate:              learningRate,
		BatchSize:                 batchSize,
		GradientAccumulationSteps: gradAccum,
		EffectiveBatchSize:        effectiveBatchSize,
		CheckpointCount:           len(r.Checkpoints),
		EvaluationCount:           len(r.Evaluations),
	}
}

func normalizeSFTConfig(cfg SFTConfig) SFTConfig {
	cfg = normalizeSFTScalarConfig(cfg)
	cfg.LoRA = normalizeSFTLoRAConfig(cfg.LoRA)
	return cfg
}

func NormalizeSFTConfigForModel(cfg SFTConfig, info spine.ModelInfo) SFTConfig {
	cfg = normalizeSFTScalarConfig(cfg)
	cfg.LoRA = normalizeSFTLoRAConfigForModel(cfg.LoRA, info)
	return cfg
}

func normalizeSFTScalarConfig(cfg SFTConfig) SFTConfig {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	if cfg.GradientAccumulationSteps <= 0 {
		cfg.GradientAccumulationSteps = 1
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.LearningRate == 0 {
		if cfg.AdamW.LearningRate != 0 || cfg.AdamW.LearningRateSet {
			cfg.LearningRate = cfg.AdamW.LearningRate
		} else {
			cfg.LearningRate = 1e-5
		}
	}
	if cfg.EvalMaxTokens <= 0 {
		cfg.EvalMaxTokens = 96
	}
	return cfg
}

// SFTEffectiveBatchSize returns the optimizer batch size after accumulation.
func SFTEffectiveBatchSize(cfg SFTConfig) int {
	// Inline only the two field defaults we need — avoids the
	// six SliceClone operations normalizeSFTLoRAConfig performs on
	// TargetKeys/TargetLayers backfills.
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	gradAccum := cfg.GradientAccumulationSteps
	if gradAccum <= 0 {
		gradAccum = 1
	}
	return batchSize * gradAccum
}

func sftResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
