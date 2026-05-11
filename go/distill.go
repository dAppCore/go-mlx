// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/dataset"
	"context"
	"math"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/probe"
)

const DistillCheckpointMetadataVersion = 1

// DistillLossKind selects the scalar used to train the student.
type DistillLossKind string

const (
	DistillLossKL               DistillLossKind = "kl"
	DistillLossSoftCrossEntropy DistillLossKind = "soft_cross_entropy"
)

// DistillLogits is a batch x sequence x vocabulary tensor in Go-native form.
type DistillLogits [][][]float32

// DistillConfig controls native knowledge distillation over dataset streams.
type DistillConfig struct {
	Batch           dataset.BatchConfig `json:"batch"`
	Epochs          int                `json:"epochs,omitempty"`
	Temperature     float64            `json:"temperature,omitempty"`
	Loss            DistillLossKind    `json:"loss,omitempty"`
	LearningRate    float64            `json:"learning_rate,omitempty"`
	CheckpointDir   string             `json:"checkpoint_dir,omitempty"`
	CheckpointEvery int                `json:"checkpoint_every,omitempty"`
	EvalEvery       int                `json:"eval_every,omitempty"`
	ResumePath      string             `json:"resume_path,omitempty"`
	MaxSamples      int                `json:"max_samples,omitempty"`
	ProbeSink       probe.Sink          `json:"-"`
}

// DistillRunner supplies the model-specific operations for distillation.
type DistillRunner struct {
	TeacherInfo func(context.Context) ModelInfo
	StudentInfo func(context.Context) ModelInfo
	Tokenizer   func(context.Context) *Tokenizer

	BuildBatches   func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error)
	TeacherLogits  func(context.Context, DistillBatch) (DistillLogits, error)
	StudentLogits  func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error)
	ApplyLoss      func(context.Context, DistillBatch, DistillLoss) error
	Evaluate       func(context.Context, DistillEvalContext) (DistillEvalResult, error)
	SaveCheckpoint func(context.Context, DistillCheckpointContext) error

	TeacherCache DistillTeacherLogitCache
}

// DistillBatch is passed to model callbacks for one tokenized training step.
type DistillBatch struct {
	Step        int
	Epoch       int
	SFT         SFTBatch
	Temperature float64
	CacheKey    string
}

// DistillLoss records per-batch distillation loss components.
type DistillLoss struct {
	Value            float64         `json:"value"`
	KL               float64         `json:"kl"`
	SoftCrossEntropy float64         `json:"soft_cross_entropy"`
	TeacherEntropy   float64         `json:"teacher_entropy"`
	Tokens           int             `json:"tokens"`
	Temperature      float64         `json:"temperature"`
	Kind             DistillLossKind `json:"kind"`
}

// DistillMetrics aggregates distillation counters and loss values.
type DistillMetrics struct {
	Steps              int     `json:"steps"`
	Epochs             int     `json:"epochs"`
	Samples            int     `json:"samples"`
	Batches            int     `json:"batches"`
	Tokens             int     `json:"tokens"`
	Loss               float64 `json:"loss"`
	LastLoss           float64 `json:"last_loss"`
	KL                 float64 `json:"kl"`
	SoftCrossEntropy   float64 `json:"soft_cross_entropy"`
	TeacherEntropy     float64 `json:"teacher_entropy"`
	Temperature        float64 `json:"temperature"`
	CheckpointCount    int     `json:"checkpoint_count"`
	EvaluationCount    int     `json:"evaluation_count"`
	TeacherCacheHits   int     `json:"teacher_cache_hits,omitempty"`
	TeacherCacheMisses int     `json:"teacher_cache_misses,omitempty"`
}

// DistillResult records one distillation run.
type DistillResult struct {
	Teacher            ModelInfo                   `json:"teacher"`
	Student            ModelInfo                   `json:"student"`
	Config             DistillConfig               `json:"config"`
	Metrics            DistillMetrics              `json:"metrics"`
	Losses             []DistillLoss               `json:"losses,omitempty"`
	Checkpoints        []string                    `json:"checkpoints,omitempty"`
	CheckpointMetadata []DistillCheckpointMetadata `json:"checkpoint_metadata,omitempty"`
	Evaluations        []DistillEvalResult         `json:"evaluations,omitempty"`
	ResumePath         string                      `json:"resume_path,omitempty"`
	ResumedFrom        *DistillCheckpointMetadata  `json:"resumed_from,omitempty"`
	Duration           time.Duration               `json:"duration,omitempty"`
}

// DistillCheckpointMetadata is the portable JSON sidecar for distillation checkpoints.
type DistillCheckpointMetadata struct {
	Version            int                `json:"version"`
	Path               string             `json:"path"`
	ResumePath         string             `json:"resume_path,omitempty"`
	Step               int                `json:"step"`
	Epoch              int                `json:"epoch"`
	Samples            int                `json:"samples"`
	Tokens             int                `json:"tokens"`
	Loss               float64            `json:"loss"`
	KL                 float64            `json:"kl"`
	SoftCrossEntropy   float64            `json:"soft_cross_entropy"`
	TeacherEntropy     float64            `json:"teacher_entropy"`
	Temperature        float64            `json:"temperature"`
	LossKind           DistillLossKind    `json:"loss_kind"`
	Batch              dataset.BatchConfig `json:"batch"`
	Teacher            ModelInfo          `json:"teacher"`
	Student            ModelInfo          `json:"student"`
	TeacherCacheHits   int                `json:"teacher_cache_hits,omitempty"`
	TeacherCacheMisses int                `json:"teacher_cache_misses,omitempty"`
}

// DistillCheckpointContext is passed to optional checkpoint writers.
type DistillCheckpointContext struct {
	Path     string
	Batch    DistillBatch
	Loss     DistillLoss
	Metadata DistillCheckpointMetadata
}

// DistillEvalContext is passed to optional eval hooks.
type DistillEvalContext struct {
	Step    int
	Epoch   int
	Config  DistillConfig
	Metrics DistillMetrics
	Teacher ModelInfo
	Student ModelInfo
}

// DistillEvalResult records one eval hook result during distillation.
type DistillEvalResult struct {
	Step    int         `json:"step"`
	Epoch   int         `json:"epoch,omitempty"`
	Name    string      `json:"name,omitempty"`
	Metrics eval.Metrics `json:"metrics,omitempty"`
	Report  *eval.Report `json:"report,omitempty"`
}

// DistillTeacherLogitCache provides cache hooks for offline teacher logits.
type DistillTeacherLogitCache interface {
	GetTeacherLogits(context.Context, string) (DistillLogits, bool, error)
	PutTeacherLogits(context.Context, string, DistillLogits) error
}

// MemoryDistillLogitCache is a small in-process teacher-logit cache for tests and local runs.
type MemoryDistillLogitCache struct {
	mu     sync.RWMutex
	logits map[string]DistillLogits
}

// NewMemoryDistillLogitCache creates an in-memory teacher-logit cache.
func NewMemoryDistillLogitCache() *MemoryDistillLogitCache {
	return &MemoryDistillLogitCache{logits: map[string]DistillLogits{}}
}

// GetTeacherLogits returns cached teacher logits for key.
func (c *MemoryDistillLogitCache) GetTeacherLogits(_ context.Context, key string) (DistillLogits, bool, error) {
	if c == nil {
		return nil, false, nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	logits, ok := c.logits[key]
	return cloneDistillLogits(logits), ok, nil
}

// PutTeacherLogits stores teacher logits for key.
func (c *MemoryDistillLogitCache) PutTeacherLogits(_ context.Context, key string, logits DistillLogits) error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.logits == nil {
		c.logits = map[string]DistillLogits{}
	}
	c.logits[key] = cloneDistillLogits(logits)
	return nil
}

// RunDistillation is an alias for RunKnowledgeDistillation.
func RunDistillation(ctx context.Context, runner DistillRunner, ds dataset.Dataset, cfg DistillConfig) (*DistillResult, error) {
	return RunKnowledgeDistillation(ctx, runner, ds, cfg)
}

// RunKnowledgeDistillation trains a student from teacher logits over a dataset stream.
func RunKnowledgeDistillation(ctx context.Context, runner DistillRunner, ds dataset.Dataset, cfg DistillConfig) (*DistillResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if ds == nil {
		return nil, core.NewError("mlx: distillation dataset is nil")
	}
	if runner.StudentLogits == nil {
		return nil, core.NewError("mlx: distillation runner requires StudentLogits")
	}
	cfg = normalizeDistillConfig(cfg)

	result := &DistillResult{Config: cfg}
	if runner.TeacherInfo != nil {
		result.Teacher = runner.TeacherInfo(ctx)
	}
	if runner.StudentInfo != nil {
		result.Student = runner.StudentInfo(ctx)
	}
	if cfg.ResumePath != "" {
		result.ResumePath = cfg.ResumePath
		meta, err := loadDistillResumeMetadata(cfg.ResumePath)
		if err != nil {
			return result, err
		}
		result.ResumedFrom = meta
	}

	start := time.Now()
	accumulator := &distillMetricAccumulator{}
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if epoch > 1 {
			resetter, ok := ds.(dataset.Resetter)
			if !ok {
				return result, core.NewError("mlx: distillation dataset must implement Reset for multiple epochs")
			}
			if err := resetter.Reset(); err != nil {
				return result, err
			}
		}
		if err := runDistillEpoch(ctx, runner, ds, cfg, result, accumulator, epoch); err != nil {
			return result, err
		}
		result.Metrics.Epochs = epoch
	}
	if result.Metrics.Steps == 0 {
		return result, core.NewError("mlx: distillation dataset produced no trainable batches")
	}
	result.Duration = nonZeroDuration(time.Since(start))
	return result, nil
}

func runDistillEpoch(ctx context.Context, runner DistillRunner, ds dataset.Dataset, cfg DistillConfig, result *DistillResult, accumulator *distillMetricAccumulator, epoch int) error {
	batches, err := distillBatches(ctx, runner, ds, cfg)
	if err != nil {
		return err
	}
	if len(batches) == 0 {
		return core.NewError("mlx: distillation dataset produced no tokenized batches")
	}
	for _, sftBatch := range batches {
		if err := ctx.Err(); err != nil {
			return err
		}
		step := result.Metrics.Steps + 1
		cacheKey := DistillBatchCacheKey(sftBatch)
		batch := DistillBatch{
			Step:        step,
			Epoch:       epoch,
			SFT:         sftBatch,
			Temperature: cfg.Temperature,
			CacheKey:    cacheKey,
		}
		teacher, cacheStatus, err := teacherLogitsForDistillBatch(ctx, runner, batch)
		if err != nil {
			return err
		}
		student, err := runner.StudentLogits(ctx, batch, teacher)
		if err != nil {
			return err
		}
		loss, err := DistillationBatchLoss(teacher, student, sftBatch.Batch.LossMask, cfg)
		if err != nil {
			return err
		}
		if runner.ApplyLoss != nil {
			if err := runner.ApplyLoss(ctx, batch, loss); err != nil {
				return err
			}
		}
		updateDistillResult(result, accumulator, sftBatch, loss, cacheStatus)
		result.Losses = append(result.Losses, loss)

		if err := maybeSaveDistillCheckpoint(ctx, runner, cfg, result, batch, loss); err != nil {
			return err
		}
		if err := maybeRunDistillEval(ctx, runner, cfg, result, epoch); err != nil {
			return err
		}
		emitDistillProbe(cfg, result, loss, cacheStatus, epoch)
	}
	return nil
}

func distillBatches(ctx context.Context, runner DistillRunner, ds dataset.Dataset, cfg DistillConfig) ([]SFTBatch, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	source := ds
	if cfg.MaxSamples > 0 {
		samples, err := distillCollectSamples(ctx, ds, cfg.MaxSamples)
		if err != nil {
			return nil, err
		}
		source = dataset.NewSliceDataset(samples)
	}
	if runner.BuildBatches != nil {
		return runner.BuildBatches(ctx, source, cfg.Batch)
	}
	if runner.Tokenizer == nil {
		return nil, core.NewError("mlx: distillation runner requires Tokenizer or BuildBatches")
	}
	tok := runner.Tokenizer(ctx)
	return BuildDatasetBatches(tok, source, cfg.Batch)
}

func teacherLogitsForDistillBatch(ctx context.Context, runner DistillRunner, batch DistillBatch) (DistillLogits, string, error) {
	if runner.TeacherCache != nil && batch.CacheKey != "" {
		logits, ok, err := runner.TeacherCache.GetTeacherLogits(ctx, batch.CacheKey)
		if err != nil {
			return nil, "", err
		}
		if ok {
			return logits, "hit", nil
		}
	}
	if runner.TeacherLogits == nil {
		return nil, "", core.NewError("mlx: distillation runner requires TeacherLogits on teacher cache miss")
	}
	logits, err := runner.TeacherLogits(ctx, batch)
	if err != nil {
		return nil, "", err
	}
	if runner.TeacherCache != nil && batch.CacheKey != "" {
		if err := runner.TeacherCache.PutTeacherLogits(ctx, batch.CacheKey, logits); err != nil {
			return nil, "", err
		}
	}
	return logits, "miss", nil
}

func updateDistillResult(result *DistillResult, accumulator *distillMetricAccumulator, batch SFTBatch, loss DistillLoss, cacheStatus string) {
	samples := len(batch.Batch.Tokens)
	result.Metrics.Steps++
	result.Metrics.Batches++
	result.Metrics.Samples += samples
	result.Metrics.Tokens += loss.Tokens
	result.Metrics.LastLoss = loss.Value
	result.Metrics.Temperature = loss.Temperature
	switch cacheStatus {
	case "hit":
		result.Metrics.TeacherCacheHits++
	case "miss":
		result.Metrics.TeacherCacheMisses++
	}
	accumulator.add(loss)
	result.Metrics.Loss = accumulator.loss()
	result.Metrics.KL = accumulator.kl()
	result.Metrics.SoftCrossEntropy = accumulator.softCrossEntropy()
	result.Metrics.TeacherEntropy = accumulator.teacherEntropy()
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	result.Metrics.EvaluationCount = len(result.Evaluations)
}

func maybeSaveDistillCheckpoint(ctx context.Context, runner DistillRunner, cfg DistillConfig, result *DistillResult, batch DistillBatch, loss DistillLoss) error {
	if cfg.CheckpointDir == "" || cfg.CheckpointEvery <= 0 || result.Metrics.Steps%cfg.CheckpointEvery != 0 {
		return nil
	}
	path := core.PathJoin(cfg.CheckpointDir, core.Sprintf("step-%06d", result.Metrics.Steps))
	meta := NewDistillCheckpointMetadata(path, cfg, result, loss, batch.Epoch)
	if runner.SaveCheckpoint != nil {
		if err := runner.SaveCheckpoint(ctx, DistillCheckpointContext{
			Path:     path,
			Batch:    batch,
			Loss:     loss,
			Metadata: meta,
		}); err != nil {
			return err
		}
	}
	if err := SaveDistillCheckpointMetadata(path, meta); err != nil {
		return err
	}
	result.Checkpoints = append(result.Checkpoints, path)
	result.CheckpointMetadata = append(result.CheckpointMetadata, meta)
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	return nil
}

func maybeRunDistillEval(ctx context.Context, runner DistillRunner, cfg DistillConfig, result *DistillResult, epoch int) error {
	if cfg.EvalEvery <= 0 || runner.Evaluate == nil || result.Metrics.Steps%cfg.EvalEvery != 0 {
		return nil
	}
	eval, err := runner.Evaluate(ctx, DistillEvalContext{
		Step:    result.Metrics.Steps,
		Epoch:   epoch,
		Config:  cfg,
		Metrics: result.Metrics,
		Teacher: result.Teacher,
		Student: result.Student,
	})
	if err != nil {
		return err
	}
	if eval.Step == 0 {
		eval.Step = result.Metrics.Steps
	}
	if eval.Epoch == 0 {
		eval.Epoch = epoch
	}
	result.Evaluations = append(result.Evaluations, eval)
	result.Metrics.EvaluationCount = len(result.Evaluations)
	return nil
}

func emitDistillProbe(cfg DistillConfig, result *DistillResult, loss DistillLoss, cacheStatus string, epoch int) {
	if cfg.ProbeSink == nil {
		return
	}
	cfg.ProbeSink.EmitProbe(probe.Event{
		Kind:  probe.KindTraining,
		Phase: probe.PhaseTraining,
		Step:  result.Metrics.Steps,
		Meta: map[string]string{
			"distillation":     "true",
			"loss_kind":        string(loss.Kind),
			"temperature":      core.Sprintf("%.6f", loss.Temperature),
			"tokens":           core.Sprintf("%d", loss.Tokens),
			"teacher_cache":    cacheStatus,
			"checkpoint_count": core.Sprintf("%d", len(result.Checkpoints)),
			"evaluation_count": core.Sprintf("%d", len(result.Evaluations)),
		},
		Training: &probe.Training{
			Step:         result.Metrics.Steps,
			Epoch:        epoch,
			Loss:         loss.Value,
			LearningRate: cfg.LearningRate,
		},
	})
}

// DistillationBatchLoss computes KL and soft cross-entropy over masked tokens.
func DistillationBatchLoss(teacher, student DistillLogits, mask [][]float32, cfg DistillConfig) (DistillLoss, error) {
	cfg = normalizeDistillConfig(cfg)
	switch cfg.Loss {
	case DistillLossKL, DistillLossSoftCrossEntropy:
	default:
		return DistillLoss{}, core.NewError("mlx: unsupported distillation loss kind: " + string(cfg.Loss))
	}
	if err := validateDistillLogitShapes(teacher, student); err != nil {
		return DistillLoss{}, err
	}
	var softCE float64
	var entropy float64
	var tokens int
	for i := range teacher {
		for j := range teacher[i] {
			if !distillMaskIncludes(mask, i, j) {
				continue
			}
			teacherLogProbs, err := logSoftmaxTemperature(teacher[i][j], cfg.Temperature)
			if err != nil {
				return DistillLoss{}, err
			}
			studentLogProbs, err := logSoftmaxTemperature(student[i][j], cfg.Temperature)
			if err != nil {
				return DistillLoss{}, err
			}
			for k, teacherLogProb := range teacherLogProbs {
				prob := math.Exp(teacherLogProb)
				softCE += -prob * studentLogProbs[k]
				entropy += -prob * teacherLogProb
			}
			tokens++
		}
	}
	if tokens == 0 {
		return DistillLoss{}, core.NewError("mlx: distillation loss has no masked tokens")
	}
	softCE /= float64(tokens)
	entropy /= float64(tokens)
	kl := softCE - entropy
	if kl < 0 && math.Abs(kl) < 1e-12 {
		kl = 0
	}
	if kl < 0 || math.IsNaN(kl) || math.IsInf(kl, 0) {
		return DistillLoss{}, core.NewError("mlx: distillation KL loss is not finite")
	}
	lossValue := kl
	if cfg.Loss == DistillLossSoftCrossEntropy {
		lossValue = softCE
	}
	return DistillLoss{
		Value:            lossValue,
		KL:               kl,
		SoftCrossEntropy: softCE,
		TeacherEntropy:   entropy,
		Tokens:           tokens,
		Temperature:      cfg.Temperature,
		Kind:             cfg.Loss,
	}, nil
}

// DistillBatchCacheKey returns a stable hash for teacher-logit cache lookup.
func DistillBatchCacheKey(batch SFTBatch) string {
	payload := struct {
		Tokens  [][]int     `json:"tokens"`
		Targets [][]int     `json:"targets"`
		Mask    [][]float32 `json:"mask"`
	}{
		Tokens:  batch.Batch.Tokens,
		Targets: batch.Targets,
		Mask:    batch.Batch.LossMask,
	}
	data := core.JSONMarshal(payload)
	if data.OK {
		return core.SHA256Hex(data.Value.([]byte))
	}
	return core.SHA256HexString(core.Sprintf("%+v", payload))
}

// NewDistillCheckpointMetadata captures reproducible distillation state.
func NewDistillCheckpointMetadata(path string, cfg DistillConfig, result *DistillResult, loss DistillLoss, epoch int) DistillCheckpointMetadata {
	cfg = normalizeDistillConfig(cfg)
	meta := DistillCheckpointMetadata{
		Version:     DistillCheckpointMetadataVersion,
		Path:        path,
		ResumePath:  cfg.ResumePath,
		Epoch:       epoch,
		Temperature: cfg.Temperature,
		LossKind:    cfg.Loss,
		Batch:       cfg.Batch,
	}
	if result != nil {
		meta.Step = result.Metrics.Steps
		meta.Samples = result.Metrics.Samples
		meta.Tokens = result.Metrics.Tokens
		meta.Teacher = result.Teacher
		meta.Student = result.Student
		meta.TeacherCacheHits = result.Metrics.TeacherCacheHits
		meta.TeacherCacheMisses = result.Metrics.TeacherCacheMisses
	}
	meta.Loss = loss.Value
	meta.KL = loss.KL
	meta.SoftCrossEntropy = loss.SoftCrossEntropy
	meta.TeacherEntropy = loss.TeacherEntropy
	return meta
}

// SaveDistillCheckpointMetadata writes checkpoint metadata beside student artifacts.
func SaveDistillCheckpointMetadata(path string, meta DistillCheckpointMetadata) error {
	if path == "" {
		return core.NewError("mlx: distillation checkpoint metadata path is required")
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	if meta.Path == "" {
		meta.Path = path
	}
	metadataPath := distillCheckpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("DistillCheckpointMetadata.Save", "ensure metadata dir", distillResultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("DistillCheckpointMetadata.Save", "marshal metadata", distillResultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("DistillCheckpointMetadata.Save", "write metadata", distillResultError(result))
	}
	return nil
}

// LoadDistillCheckpointMetadata reads checkpoint metadata written by SaveDistillCheckpointMetadata.
func LoadDistillCheckpointMetadata(path string) (*DistillCheckpointMetadata, error) {
	if path == "" {
		return nil, core.NewError("mlx: distillation checkpoint metadata path is required")
	}
	read := core.ReadFile(distillCheckpointMetadataPath(path))
	if !read.OK {
		return nil, distillResultError(read)
	}
	var meta DistillCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadDistillCheckpointMetadata", "parse metadata", distillResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	return &meta, nil
}

func loadDistillResumeMetadata(path string) (*DistillCheckpointMetadata, error) {
	read := core.ReadFile(distillCheckpointMetadataPath(path))
	if !read.OK {
		err := distillResultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta DistillCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadDistillResumeMetadata", "parse metadata", distillResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = DistillCheckpointMetadataVersion
	}
	return &meta, nil
}

func distillCheckpointMetadataPath(path string) string {
	return core.PathJoin(path, "distill_checkpoint.json")
}

func normalizeDistillConfig(cfg DistillConfig) DistillConfig {
	cfg.Batch = normalizeDatasetBatchConfig(cfg.Batch)
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 1
	}
	if cfg.Temperature < 0 || math.IsNaN(cfg.Temperature) || math.IsInf(cfg.Temperature, 0) {
		cfg.Temperature = math.NaN()
	}
	if cfg.Loss == "" {
		cfg.Loss = DistillLossKL
	}
	return cfg
}

func validateDistillLogitShapes(teacher, student DistillLogits) error {
	if len(teacher) == 0 {
		return core.NewError("mlx: teacher logits are empty")
	}
	if len(teacher) != len(student) {
		return core.NewError("mlx: distillation logit shape mismatch: batch")
	}
	for i := range teacher {
		if len(teacher[i]) != len(student[i]) {
			return core.NewError("mlx: distillation logit shape mismatch: sequence")
		}
		for j := range teacher[i] {
			if len(teacher[i][j]) == 0 {
				return core.NewError("mlx: distillation logit shape mismatch: empty vocabulary")
			}
			if len(teacher[i][j]) != len(student[i][j]) {
				return core.NewError("mlx: distillation logit shape mismatch: vocabulary")
			}
		}
	}
	return nil
}

func logSoftmaxTemperature(logits []float32, temperature float64) ([]float64, error) {
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return nil, core.NewError("mlx: distillation temperature must be finite and positive")
	}
	if len(logits) == 0 {
		return nil, core.NewError("mlx: distillation logits are empty")
	}
	maxLogit := math.Inf(-1)
	scaled := make([]float64, len(logits))
	for i, logit := range logits {
		value := float64(logit) / temperature
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, core.NewError("mlx: distillation logit is not finite")
		}
		scaled[i] = value
		if value > maxLogit {
			maxLogit = value
		}
	}
	var sumExp float64
	for _, value := range scaled {
		sumExp += math.Exp(value - maxLogit)
	}
	logDenom := maxLogit + math.Log(sumExp)
	for i, value := range scaled {
		scaled[i] = value - logDenom
	}
	return scaled, nil
}

func distillMaskIncludes(mask [][]float32, row, col int) bool {
	if len(mask) == 0 {
		return true
	}
	if row >= len(mask) || col >= len(mask[row]) {
		return false
	}
	return mask[row][col] > 0
}

type distillMetricAccumulator struct {
	tokens     int
	lossSum    float64
	klSum      float64
	softCE     float64
	entropySum float64
}

func (a *distillMetricAccumulator) add(loss DistillLoss) {
	if a == nil || loss.Tokens <= 0 {
		return
	}
	weight := float64(loss.Tokens)
	a.tokens += loss.Tokens
	a.lossSum += loss.Value * weight
	a.klSum += loss.KL * weight
	a.softCE += loss.SoftCrossEntropy * weight
	a.entropySum += loss.TeacherEntropy * weight
}

func (a *distillMetricAccumulator) loss() float64 {
	if a == nil || a.tokens == 0 {
		return 0
	}
	return a.lossSum / float64(a.tokens)
}

func (a *distillMetricAccumulator) kl() float64 {
	if a == nil || a.tokens == 0 {
		return 0
	}
	return a.klSum / float64(a.tokens)
}

func (a *distillMetricAccumulator) softCrossEntropy() float64 {
	if a == nil || a.tokens == 0 {
		return 0
	}
	return a.softCE / float64(a.tokens)
}

func (a *distillMetricAccumulator) teacherEntropy() float64 {
	if a == nil || a.tokens == 0 {
		return 0
	}
	return a.entropySum / float64(a.tokens)
}

func cloneDistillLogits(logits DistillLogits) DistillLogits {
	if len(logits) == 0 {
		return nil
	}
	out := make(DistillLogits, len(logits))
	for i := range logits {
		out[i] = make([][]float32, len(logits[i]))
		for j := range logits[i] {
			out[i][j] = append([]float32(nil), logits[i][j]...)
		}
	}
	return out
}

func distillResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

func distillCollectSamples(ctx context.Context, ds dataset.Dataset, maxSamples int) ([]dataset.Sample, error) {
	var samples []dataset.Sample
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if maxSamples > 0 && len(samples) >= maxSamples {
			break
		}
		sample, ok, err := ds.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		samples = append(samples, dataset.CloneSample(sample))
	}
	return samples, nil
}
