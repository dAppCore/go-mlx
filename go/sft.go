// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
)

// SFTConfig configures native LoRA supervised fine-tuning.
type SFTConfig struct {
	LoRA                      LoRAConfig
	BatchSize                 int
	GradientAccumulationSteps int
	Epochs                    int
	LearningRate              float64
	AdamW                     AdamWConfig
	MaxSeqLen                 int
	SequencePacking           bool
	CheckpointDir             string
	CheckpointEvery           int
	EvalEvery                 int
	EvalPrompts               []string
	EvalMaxTokens             int
	SavePath                  string
	ResumePath                string
	Merge                     bool
	NoEOS                     bool
	ProbeSink                 probe.Sink
}

// SFTBatch is a tokenized training batch with shifted targets.
type SFTBatch struct {
	Batch   Batch
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
	Rank         int      `json:"rank"`
	Alpha        float32  `json:"alpha"`
	Scale        float32  `json:"scale,omitempty"`
	TargetKeys   []string `json:"target_keys,omitempty"`
	TargetLayers []string `json:"target_layers,omitempty"`
	Lambda       float32  `json:"lambda,omitempty"`
	DType        string   `json:"dtype,omitempty"`
}

// SFTAdamWMetadata records optimizer hyperparameters for checkpoint replay.
type SFTAdamWMetadata struct {
	LearningRate float64 `json:"learning_rate"`
	Beta1        float64 `json:"beta1"`
	Beta2        float64 `json:"beta2"`
	Eps          float64 `json:"eps"`
	WeightDecay  float64 `json:"weight_decay"`
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
	LearningRate              float64          `json:"learning_rate"`
	BatchSize                 int              `json:"batch_size"`
	GradientAccumulationSteps int              `json:"gradient_accumulation_steps"`
	EffectiveBatchSize        int              `json:"effective_batch_size"`
	MaxSeqLen                 int              `json:"max_seq_len,omitempty"`
	SequencePacking           bool             `json:"sequence_packing,omitempty"`
	EvalPrompts               []string         `json:"eval_prompts,omitempty"`
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
	Adapter            *LoRAAdapter
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
}

// Metrics returns a stable JSON-friendly summary of an SFT run.
func (r *SFTResult) Metrics(cfg SFTConfig) SFTMetrics {
	cfg = normalizeSFTConfig(cfg)
	if r == nil {
		return SFTMetrics{
			LearningRate:              cfg.LearningRate,
			BatchSize:                 cfg.BatchSize,
			GradientAccumulationSteps: cfg.GradientAccumulationSteps,
			EffectiveBatchSize:        SFTEffectiveBatchSize(cfg),
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
		EffectiveBatchSize:        SFTEffectiveBatchSize(cfg),
		CheckpointCount:           len(r.Checkpoints),
		EvaluationCount:           len(r.Evaluations),
	}
}

type sftExample struct {
	inputs  []int
	targets []int
	mask    []float32
}

func normalizeSFTConfig(cfg SFTConfig) SFTConfig {
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
	cfg.LoRA = normalizeSFTLoRAConfig(cfg.LoRA)
	return cfg
}

// SFTEffectiveBatchSize returns the optimizer batch size after accumulation.
func SFTEffectiveBatchSize(cfg SFTConfig) int {
	cfg = normalizeSFTConfig(cfg)
	return cfg.BatchSize * cfg.GradientAccumulationSteps
}

// BuildSFTTrainingBatches tokenizes an SFT dataset using runner-level batching settings.
func BuildSFTTrainingBatches(tok *Tokenizer, ds dataset.Dataset, cfg SFTConfig) ([]SFTBatch, error) {
	if tok == nil || tok.tok == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}
	cfg = normalizeSFTConfig(cfg)
	return BuildDatasetBatches(tok, ds, dataset.BatchConfig{
		BatchSize:       SFTEffectiveBatchSize(cfg),
		MaxSeqLen:       cfg.MaxSeqLen,
		SequencePacking: cfg.SequencePacking,
		NoEOS:           cfg.NoEOS,
	})
}

// BuildSFTBatches tokenizes an SFT dataset into response-masked training batches.
func BuildSFTBatches(tok *Tokenizer, ds dataset.Dataset, cfg SFTConfig) ([]SFTBatch, error) {
	if tok == nil || tok.tok == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}

	cfg = normalizeSFTConfig(cfg)
	builder := newSFTBatchBuilder(cfg.BatchSize)
	for {
		sample, ok, err := ds.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		example, usable, err := buildSFTExample(tok, sample, cfg)
		if err != nil {
			return nil, err
		}
		if !usable {
			continue
		}
		builder.add(example)
	}
	return builder.finish(), nil
}

// NewSFTCheckpointMetadata captures the reproducible state for one checkpoint.
func NewSFTCheckpointMetadata(path string, model string, cfg SFTConfig, result *SFTResult, epoch int) SFTCheckpointMetadata {
	return newSFTMetadata(path, path, model, cfg, result, epoch)
}

// NewSFTArtifactMetadata captures the reproducible state for a final adapter artifact.
func NewSFTArtifactMetadata(path string, model string, cfg SFTConfig, result *SFTResult) SFTCheckpointMetadata {
	epoch := 0
	if result != nil {
		epoch = result.Epochs
	}
	return newSFTMetadata(path, path, model, cfg, result, epoch)
}

// SaveSFTCheckpointMetadata writes checkpoint metadata beside an adapter package.
func SaveSFTCheckpointMetadata(path string, meta SFTCheckpointMetadata) error {
	if path == "" {
		return core.NewError("mlx: SFT checkpoint metadata path is required")
	}
	if meta.Version == 0 {
		meta.Version = SFTCheckpointMetadataVersion
	}
	if meta.Path == "" {
		meta.Path = path
	}
	metadataPath := sftCheckpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("SFTCheckpointMetadata.Save", "ensure metadata dir", sftResultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("SFTCheckpointMetadata.Save", "marshal metadata", sftResultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("SFTCheckpointMetadata.Save", "write metadata", sftResultError(result))
	}
	return nil
}

// LoadSFTCheckpointMetadata reads checkpoint metadata written by SaveSFTCheckpointMetadata.
func LoadSFTCheckpointMetadata(path string) (*SFTCheckpointMetadata, error) {
	if path == "" {
		return nil, core.NewError("mlx: SFT checkpoint metadata path is required")
	}
	read := core.ReadFile(sftCheckpointMetadataPath(path))
	if !read.OK {
		return nil, sftResultError(read)
	}
	var meta SFTCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadSFTCheckpointMetadata", "parse metadata", sftResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = SFTCheckpointMetadataVersion
	}
	return &meta, nil
}

// ApplySFTResumeMetadata attaches optional checkpoint metadata from ResumePath to a result.
func ApplySFTResumeMetadata(result *SFTResult, cfg SFTConfig) error {
	if result == nil {
		return core.NewError("mlx: SFT result is nil")
	}
	if cfg.ResumePath == "" {
		return nil
	}
	result.ResumePath = cfg.ResumePath
	meta, err := loadSFTResumeMetadata(cfg.ResumePath)
	if err != nil {
		return err
	}
	result.ResumedFrom = meta
	return nil
}

func newSFTMetadata(path string, adapterPath string, model string, cfg SFTConfig, result *SFTResult, epoch int) SFTCheckpointMetadata {
	cfg = normalizeSFTConfig(cfg)
	step := 0
	optimizerStep := 0
	samples := 0
	loss := 0.0
	if result != nil {
		step = result.Steps
		optimizerStep = result.OptimizerSteps
		if optimizerStep == 0 {
			optimizerStep = step
		}
		samples = result.Samples
		loss = result.LastLoss
	}
	return SFTCheckpointMetadata{
		Version:                   SFTCheckpointMetadataVersion,
		Path:                      path,
		AdapterPath:               adapterPath,
		ResumePath:                cfg.ResumePath,
		Model:                     model,
		Step:                      step,
		OptimizerStep:             optimizerStep,
		Epoch:                     epoch,
		Samples:                   samples,
		Loss:                      loss,
		LearningRate:              cfg.LearningRate,
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulationSteps,
		EffectiveBatchSize:        SFTEffectiveBatchSize(cfg),
		MaxSeqLen:                 cfg.MaxSeqLen,
		SequencePacking:           cfg.SequencePacking,
		EvalPrompts:               append([]string(nil), cfg.EvalPrompts...),
		LoRA:                      sftLoRAMetadata(cfg.LoRA),
		AdamW:                     sftAdamWMetadata(sftAdamWConfig(cfg)),
	}
}

func sftLoRAMetadata(cfg LoRAConfig) SFTLoRAMetadata {
	cfg = normalizeSFTLoRAConfig(cfg)
	return SFTLoRAMetadata{
		Rank:         cfg.Rank,
		Alpha:        cfg.Alpha,
		Scale:        cfg.Scale,
		TargetKeys:   append([]string(nil), cfg.TargetKeys...),
		TargetLayers: append([]string(nil), cfg.TargetLayers...),
		Lambda:       cfg.Lambda,
		DType:        cfg.DType.String(),
	}
}

func sftAdamWMetadata(cfg AdamWConfig) SFTAdamWMetadata {
	return SFTAdamWMetadata{
		LearningRate: cfg.LearningRate,
		Beta1:        cfg.Beta1,
		Beta2:        cfg.Beta2,
		Eps:          cfg.Eps,
		WeightDecay:  cfg.WeightDecay,
	}
}

func sftAdamWConfig(cfg SFTConfig) AdamWConfig {
	cfg = normalizeSFTConfig(cfg)
	adam := DefaultAdamWConfig()
	if cfg.AdamW.LearningRate != 0 || cfg.AdamW.LearningRateSet {
		adam.LearningRate = cfg.AdamW.LearningRate
	}
	if cfg.AdamW.Beta1 != 0 || cfg.AdamW.Beta1Set {
		adam.Beta1 = cfg.AdamW.Beta1
	}
	if cfg.AdamW.Beta2 != 0 || cfg.AdamW.Beta2Set {
		adam.Beta2 = cfg.AdamW.Beta2
	}
	if cfg.AdamW.Eps != 0 || cfg.AdamW.EpsSet {
		adam.Eps = cfg.AdamW.Eps
	}
	if cfg.AdamW.WeightDecay != 0 || cfg.AdamW.WeightDecaySet {
		adam.WeightDecay = cfg.AdamW.WeightDecay
	}
	if cfg.LearningRate != 0 {
		adam.LearningRate = cfg.LearningRate
	}
	return adam
}

func normalizeSFTLoRAConfig(cfg LoRAConfig) LoRAConfig {
	if cfg.Rank <= 0 {
		cfg.Rank = 8
	}
	if cfg.Alpha == 0 {
		if cfg.Scale != 0 {
			cfg.Alpha = cfg.Scale * float32(cfg.Rank)
		} else {
			cfg.Alpha = 16
		}
	}
	if cfg.Scale == 0 && cfg.Rank > 0 {
		cfg.Scale = cfg.Alpha / float32(cfg.Rank)
	}
	if len(cfg.TargetKeys) == 0 && len(cfg.TargetLayers) > 0 {
		cfg.TargetKeys = append([]string(nil), cfg.TargetLayers...)
	}
	if len(cfg.TargetKeys) == 0 {
		cfg.TargetKeys = []string{"q_proj", "v_proj"}
	}
	if len(cfg.TargetLayers) == 0 {
		cfg.TargetLayers = append([]string(nil), cfg.TargetKeys...)
	}
	if cfg.DType == 0 {
		cfg.DType = DTypeFloat32
	}
	return cfg
}

func loadSFTResumeMetadata(path string) (*SFTCheckpointMetadata, error) {
	read := core.ReadFile(sftCheckpointMetadataPath(path))
	if !read.OK {
		err := sftResultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta SFTCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadSFTResumeMetadata", "parse metadata", sftResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = SFTCheckpointMetadataVersion
	}
	return &meta, nil
}

func sftCheckpointMetadataPath(path string) string {
	if core.HasSuffix(path, ".safetensors") {
		return core.PathJoin(core.PathDir(path), "sft_checkpoint.json")
	}
	return core.PathJoin(path, "sft_checkpoint.json")
}

type sftBatchBuilder struct {
	batchSize int
	current   []sftExample
	out       []SFTBatch
}

func newSFTBatchBuilder(batchSize int) *sftBatchBuilder {
	if batchSize <= 0 {
		batchSize = 1
	}
	return &sftBatchBuilder{batchSize: batchSize}
}

func (b *sftBatchBuilder) add(example sftExample) {
	b.current = append(b.current, example)
	if len(b.current) >= b.batchSize {
		b.flush()
	}
}

func (b *sftBatchBuilder) finish() []SFTBatch {
	b.flush()
	return append([]SFTBatch(nil), b.out...)
}

func (b *sftBatchBuilder) flush() {
	if len(b.current) == 0 {
		return
	}
	b.out = append(b.out, sftBatchFromExamples(b.current))
	b.current = b.current[:0]
}

func sftBatchFromExamples(examples []sftExample) SFTBatch {
	batch := SFTBatch{
		Batch: Batch{
			Tokens:   make([][]int, 0, len(examples)),
			Length:   make([]int, 0, len(examples)),
			LossMask: make([][]float32, 0, len(examples)),
		},
		Targets: make([][]int, 0, len(examples)),
	}
	for _, example := range examples {
		batch.Batch.Tokens = append(batch.Batch.Tokens, append([]int(nil), example.inputs...))
		batch.Batch.Length = append(batch.Batch.Length, len(example.inputs))
		batch.Batch.LossMask = append(batch.Batch.LossMask, append([]float32(nil), example.mask...))
		batch.Targets = append(batch.Targets, append([]int(nil), example.targets...))
	}
	return batch
}

func buildSFTExample(tok *Tokenizer, sample dataset.Sample, cfg SFTConfig) (sftExample, bool, error) {
	var seq []int32
	var promptLen int
	trainWholeText := sample.Text != ""
	if trainWholeText {
		ids, err := tok.Encode(sample.Text)
		if err != nil {
			return sftExample{}, false, err
		}
		seq = append(seq, ids...)
	} else {
		promptIDs, err := tok.Encode(sample.Prompt)
		if err != nil {
			return sftExample{}, false, err
		}
		responseIDs, err := tok.Encode(sample.Response)
		if err != nil {
			return sftExample{}, false, err
		}
		promptLen = len(promptIDs)
		seq = append(seq, promptIDs...)
		seq = append(seq, responseIDs...)
	}
	if !cfg.NoEOS {
		seq = append(seq, tok.EOS())
	}
	if len(seq) < 2 {
		return sftExample{}, false, nil
	}

	inputs := int32ToIntSlice(seq[:len(seq)-1])
	targets := int32ToIntSlice(seq[1:])
	mask := make([]float32, len(inputs))
	if trainWholeText {
		for i := range mask {
			mask[i] = 1
		}
	} else {
		for i := range mask {
			if i+1 >= promptLen {
				mask[i] = 1
			}
		}
	}

	if cfg.MaxSeqLen > 0 && len(inputs) > cfg.MaxSeqLen {
		start := len(inputs) - cfg.MaxSeqLen
		inputs = append([]int(nil), inputs[start:]...)
		targets = append([]int(nil), targets[start:]...)
		mask = append([]float32(nil), mask[start:]...)
	}
	if !hasTrainingTarget(mask) {
		return sftExample{}, false, nil
	}
	return sftExample{inputs: inputs, targets: targets, mask: mask}, true, nil
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

func int32ToIntSlice(values []int32) []int {
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}

func hasTrainingTarget(mask []float32) bool {
	for _, value := range mask {
		if value != 0 {
			return true
		}
	}
	return false
}
