// SPDX-Licence-Identifier: EUPL-1.2

// Package train holds the native training machinery — SFT batch building,
// sequence packing, checkpoint metadata, the LoRA epoch loop, and the SSD
// (sampling-and-fine-tuning) pipeline with its code benchmark. The root mlx
// package aliases the exported types and keeps the Model-bound entry points
// (Model.TrainSFT / Model.RunSSD), which delegate here.
package train

import (
	core "dappco.re/go"
	traininf "dappco.re/go/inference/train"
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
	// normalizeSFTScalarConfig (not the combined normalizeSFTConfig) —
	// normalizeSFTConfig also runs normalizeSFTLoRAConfig, which clones
	// TargetKeys+TargetLayers (two SliceClones) every call, and Metrics
	// touches none of that.
	cfg = normalizeSFTScalarConfig(cfg)
	effectiveBatchSize := cfg.BatchSize * cfg.GradientAccumulationSteps
	if r == nil {
		return SFTMetrics{
			LearningRate:              cfg.LearningRate,
			BatchSize:                 cfg.BatchSize,
			GradientAccumulationSteps: cfg.GradientAccumulationSteps,
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
		LearningRate:              cfg.LearningRate,
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulationSteps,
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

// normalizeSFTScalarConfig delegates the BatchSize/GradientAccumulationSteps/
// Epochs/EvalMaxTokens/LearningRate defaulting to the shared
// dappco.re/go/inference/train engine (traininf.NormalizeConfig), which
// carries the identical floor/default rules. AdamW.LearningRate is an
// engine-side fallback source with no shared equivalent (traininf.Config
// has no AdamW field — LoRA/AdamW are engine types with no portable
// counterpart, see traininf.Config's doc), so it is resolved here first:
// when the caller left LearningRate unset but supplied an AdamW rate
// (including an explicit zero via AdamW.LearningRateSet), that AdamW value
// wins and the shared engine's generic 1e-5 default is never consulted.
func normalizeSFTScalarConfig(cfg SFTConfig) SFTConfig {
	adamWFallback := cfg.LearningRate == 0 && (cfg.AdamW.LearningRate != 0 || cfg.AdamW.LearningRateSet)
	shared := traininf.NormalizeConfig(traininf.Config{
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulationSteps,
		Epochs:                    cfg.Epochs,
		LearningRate:              cfg.LearningRate,
		EvalMaxTokens:             cfg.EvalMaxTokens,
	})
	cfg.BatchSize = shared.BatchSize
	cfg.GradientAccumulationSteps = shared.GradientAccumulationSteps
	cfg.Epochs = shared.Epochs
	cfg.EvalMaxTokens = shared.EvalMaxTokens
	if adamWFallback {
		cfg.LearningRate = cfg.AdamW.LearningRate
	} else {
		cfg.LearningRate = shared.LearningRate
	}
	return cfg
}

// SFTEffectiveBatchSize returns the optimizer batch size after accumulation
// — delegates to the shared dappco.re/go/inference/train engine so the
// batch-size-floor-times-gradient-accumulation rule cannot drift from
// distill/grpo's equivalents.
func SFTEffectiveBatchSize(cfg SFTConfig) int {
	return traininf.EffectiveBatchSize(traininf.Config{
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulationSteps,
	})
}
