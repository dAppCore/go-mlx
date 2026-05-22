// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"strconv"

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
	Rank                       int      `json:"rank"`
	Alpha                      float32  `json:"alpha"`
	Scale                      float32  `json:"scale,omitempty"`
	TargetKeys                 []string `json:"target_keys,omitempty"`
	TargetLayers               []string `json:"target_layers,omitempty"`
	Lambda                     float32  `json:"lambda,omitempty"`
	DType                      string   `json:"dtype,omitempty"`
	AllowGemma4ExtendedTargets bool     `json:"allow_gemma4_extended_targets,omitempty"`
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
		EvalPrompts:               core.SliceClone(cfg.EvalPrompts),
		LoRA:                      sftLoRAMetadata(cfg.LoRA),
		AdamW:                     sftAdamWMetadata(sftAdamWConfig(cfg)),
	}
}

func sftLoRAMetadata(cfg LoRAConfig) SFTLoRAMetadata {
	cfg = normalizeSFTLoRAConfig(cfg)
	return SFTLoRAMetadata{
		Rank:                       cfg.Rank,
		Alpha:                      cfg.Alpha,
		Scale:                      cfg.Scale,
		TargetKeys:                 core.SliceClone(cfg.TargetKeys),
		TargetLayers:               core.SliceClone(cfg.TargetLayers),
		Lambda:                     cfg.Lambda,
		DType:                      cfg.DType.String(),
		AllowGemma4ExtendedTargets: cfg.AllowGemma4ExtendedTargets,
	}
}

func sftAdamWMetadata(cfg AdamWConfig) SFTAdamWMetadata {
	return SFTAdamWMetadata{
		LearningRate: cfg.LearningRate,
		Beta1:        cfg.Beta1,
		Beta2:        cfg.Beta2,
		Eps:          cfg.Eps,
		WeightDecay:  cfg.WeightDecay,
		PackedState:  cfg.PackedState,
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
	if cfg.AdamW.PackedState || cfg.AdamW.PackedStateSet {
		adam.PackedState = cfg.AdamW.PackedState
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
		cfg.TargetKeys = core.SliceClone(cfg.TargetLayers)
	}
	if len(cfg.TargetKeys) == 0 {
		cfg.TargetKeys = []string{"q_proj", "v_proj"}
	}
	if len(cfg.TargetLayers) == 0 {
		cfg.TargetLayers = core.SliceClone(cfg.TargetKeys)
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

// sftStepName renders the step-NNNNNN directory name used for SFT
// checkpoints — same output as fmt.Sprintf("step-%06d", step). Built
// with strconv.AppendInt so no fmt format-parser and no interface
// boxing of the int arg, with a pre-sized scratch buffer keeping the
// alloc count at one.
func sftStepName(step int) string {
	const prefix = "step-"
	const padTo = 6
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	if step >= 0 && step < 100000 {
		digits := 1
		for n := step / 10; n > 0; n /= 10 {
			digits++
		}
		for i := digits; i < padTo; i++ {
			buf = append(buf, '0')
		}
	}
	buf = strconv.AppendInt(buf, int64(step), 10)
	return string(buf)
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
	return core.SliceClone(b.out)
}

func (b *sftBatchBuilder) flush() {
	if len(b.current) == 0 {
		return
	}
	b.out = append(b.out, sftBatchFromExamples(b.current))
	b.current = b.current[:0]
}

func sftBatchFromExamples(examples []sftExample) SFTBatch {
	n := len(examples)
	batch := SFTBatch{
		Batch: Batch{
			Tokens:   make([][]int, n),
			Length:   make([]int, n),
			LossMask: make([][]float32, n),
		},
		Targets: make([][]int, n),
	}
	for i, example := range examples {
		batch.Batch.Tokens[i] = core.SliceClone(example.inputs)
		batch.Batch.Length[i] = len(example.inputs)
		batch.Batch.LossMask[i] = core.SliceClone(example.mask)
		batch.Targets[i] = core.SliceClone(example.targets)
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
		// Pre-size for ids + optional EOS so no growth occurs.
		extra := 0
		if !cfg.NoEOS {
			extra = 1
		}
		seq = make([]int32, 0, len(ids)+extra)
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
		extra := 0
		if !cfg.NoEOS {
			extra = 1
		}
		seq = make([]int32, 0, len(promptIDs)+len(responseIDs)+extra)
		seq = append(seq, promptIDs...)
		seq = append(seq, responseIDs...)
	}
	if !cfg.NoEOS {
		seq = append(seq, tok.EOS())
	}
	if len(seq) < 2 {
		return sftExample{}, false, nil
	}

	// inputs[i] = int(seq[i]); targets[i] = int(seq[i+1]) — same length,
	// shifted by one. Building both in a single index walk lets the loop
	// amortise bounds-check elision across the two writes instead of
	// paying once per int32ToIntSlice call (each of which performs its
	// own range loop + int widening).
	n := len(seq) - 1
	inputs := make([]int, n)
	targets := make([]int, n)
	for i := 0; i < n; i++ {
		inputs[i] = int(seq[i])
		targets[i] = int(seq[i+1])
	}
	mask := make([]float32, n)
	if trainWholeText {
		for i := range mask {
			mask[i] = 1
		}
	} else {
		// mask is zero-initialised by make — only write the trailing 1s
		// starting where the response begins (i+1 >= promptLen).
		start := promptLen - 1
		if start < 0 {
			start = 0
		}
		if start < len(mask) {
			tail := mask[start:]
			for i := range tail {
				tail[i] = 1
			}
		}
	}

	if cfg.MaxSeqLen > 0 && len(inputs) > cfg.MaxSeqLen {
		start := len(inputs) - cfg.MaxSeqLen
		inputs = core.SliceClone(inputs[start:])
		targets = core.SliceClone(targets[start:])
		mask = core.SliceClone(mask[start:])
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
	// Scan backward — the SFT response mask is zero across the prompt
	// region and one across the response region, with the response
	// region at the tail. A backward scan finds the first 1 in O(1)
	// for typical inputs; the original forward scan walked the entire
	// prompt prefix every time. For trainWholeText the mask is all
	// ones so direction doesn't matter (O(1) either way). The
	// no-training-target case still costs O(N) but that's the rare
	// path filtered out by the caller.
	for i := len(mask) - 1; i >= 0; i-- {
		if mask[i] != 0 {
			return true
		}
	}
	return false
}

// TrainSFT runs native supervised LoRA fine-tuning against a loaded MLX model.
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
			if resetter, ok := ds.(dataset.Resetter); ok {
				if err := resetter.Reset(); err != nil {
					return result, err
				}
			} else {
				return result, core.NewError("mlx: SFT dataset must implement Reset for multiple epochs")
			}
		}

		if err := m.runSFTDatasetEpoch(ctx, tok, ds, adapter, optimizer, cfg, result, epoch); err != nil {
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

func (m *Model) runSFTDatasetEpoch(ctx context.Context, tok *Tokenizer, ds dataset.Dataset, adapter *LoRAAdapter, optimizer *AdamW, cfg SFTConfig, result *SFTResult, epoch int) error {
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
		sample, ok, err := ds.Next()
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
		example.inputs = core.SliceClone(example.inputs[start:])
		example.targets = core.SliceClone(example.targets[start:])
		example.mask = core.SliceClone(example.mask[start:])
	}
	// First add into an empty accumulator: pre-size to maxSeqLen (when
	// known) so the doubling cascade across subsequent appends collapses
	// into a single allocation per accumulator field.
	if p.maxSeqLen > 0 && cap(p.current.inputs) == 0 {
		p.current.inputs = make([]int, 0, p.maxSeqLen)
		p.current.targets = make([]int, 0, p.maxSeqLen)
		p.current.mask = make([]float32, 0, p.maxSeqLen)
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
		inputs:  core.SliceClone(p.current.inputs),
		targets: core.SliceClone(p.current.targets),
		mask:    core.SliceClone(p.current.mask),
	}
	p.current = sftExample{}
	return p.emit(example)
}
