// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"
	"math"
	"sync"
	"time"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/probe"
)

const DistillCheckpointMetadataVersion = 1

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. errDistillLogitNotFinite fires twice (per-batch finite
// guard); errDistillCheckpointPath twice (Save/Resume paths).
var (
	errDistillLogitNotFinite     = core.NewError("mlx: distillation logit is not finite")
	errDistillCheckpointPath     = core.NewError("mlx: distillation checkpoint metadata path is required")
	errTeacherLogitsEmpty        = core.NewError("mlx: teacher logits are empty")
	errDistillTempInvalid        = core.NewError("mlx: distillation temperature must be finite and positive")
	errDistillNeedTokenizer      = core.NewError("mlx: distillation runner requires Tokenizer or BuildBatches")
	errDistillNeedTeacherLogits  = core.NewError("mlx: distillation runner requires TeacherLogits on teacher cache miss")
	errDistillNeedStudentLogits  = core.NewError("mlx: distillation runner requires StudentLogits")
	errDistillNoMaskedTokens     = core.NewError("mlx: distillation loss has no masked tokens")
	errDistillLogitVocab         = core.NewError("mlx: distillation logit shape mismatch: vocabulary")
	errDistillLogitSeq           = core.NewError("mlx: distillation logit shape mismatch: sequence")
	errDistillLogitEmptyVocab    = core.NewError("mlx: distillation logit shape mismatch: empty vocabulary")
	errDistillLogitBatch         = core.NewError("mlx: distillation logit shape mismatch: batch")
	errDistillKLNotFinite        = core.NewError("mlx: distillation KL loss is not finite")
	errDistillNoTrainableBatches = core.NewError("mlx: distillation dataset produced no trainable batches")
	errDistillNoTokenizedBatches = core.NewError("mlx: distillation dataset produced no tokenized batches")
	errDistillDatasetNeedsReset  = core.NewError("mlx: distillation dataset must implement Reset for multiple epochs")
	errDistillDatasetNil         = core.NewError("mlx: distillation dataset is nil")
	errDistillCoreResultFailed   = core.NewError("core result failed")
)

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
	Epochs          int                 `json:"epochs,omitempty"`
	Temperature     float64             `json:"temperature,omitempty"`
	Loss            DistillLossKind     `json:"loss,omitempty"`
	LearningRate    float64             `json:"learning_rate,omitempty"`
	CheckpointDir   string              `json:"checkpoint_dir,omitempty"`
	CheckpointEvery int                 `json:"checkpoint_every,omitempty"`
	EvalEvery       int                 `json:"eval_every,omitempty"`
	ResumePath      string              `json:"resume_path,omitempty"`
	MaxSamples      int                 `json:"max_samples,omitempty"`
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
	Version            int                 `json:"version"`
	Path               string              `json:"path"`
	ResumePath         string              `json:"resume_path,omitempty"`
	Step               int                 `json:"step"`
	Epoch              int                 `json:"epoch"`
	Samples            int                 `json:"samples"`
	Tokens             int                 `json:"tokens"`
	Loss               float64             `json:"loss"`
	KL                 float64             `json:"kl"`
	SoftCrossEntropy   float64             `json:"soft_cross_entropy"`
	TeacherEntropy     float64             `json:"teacher_entropy"`
	Temperature        float64             `json:"temperature"`
	LossKind           DistillLossKind     `json:"loss_kind"`
	Batch              dataset.BatchConfig `json:"batch"`
	Teacher            ModelInfo           `json:"teacher"`
	Student            ModelInfo           `json:"student"`
	TeacherCacheHits   int                 `json:"teacher_cache_hits,omitempty"`
	TeacherCacheMisses int                 `json:"teacher_cache_misses,omitempty"`
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
	Step    int          `json:"step"`
	Epoch   int          `json:"epoch,omitempty"`
	Name    string       `json:"name,omitempty"`
	Metrics eval.Metrics `json:"metrics"`
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
	logits, ok := c.logits[key]
	c.mu.RUnlock()
	// Skip the clone on miss — defer + clone overhead is wasted when
	// there's nothing to copy. Releasing the read lock manually also
	// shrinks the critical section: the clone now runs lock-free, which
	// matters when teacher logits are large (B*S*V float32).
	if !ok {
		return nil, false, nil
	}
	return cloneDistillLogits(logits), true, nil
}

// PutTeacherLogits stores teacher logits for key.
func (c *MemoryDistillLogitCache) PutTeacherLogits(_ context.Context, key string, logits DistillLogits) error {
	if c == nil {
		return nil
	}
	// Clone outside the write lock — the clone is a pure copy of caller
	// data with no shared state, so it can race freely with other
	// goroutines. Acquiring the lock only for the map assignment shrinks
	// the critical section from O(B*S*V) to O(1).
	cloned := cloneDistillLogits(logits)
	c.mu.Lock()
	if c.logits == nil {
		c.logits = map[string]DistillLogits{}
	}
	c.logits[key] = cloned
	c.mu.Unlock()
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
		return nil, errDistillDatasetNil
	}
	if runner.StudentLogits == nil {
		return nil, errDistillNeedStudentLogits
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
				return result, errDistillDatasetNeedsReset
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
		return result, errDistillNoTrainableBatches
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
		return errDistillNoTokenizedBatches
	}
	// Pre-grow result.Losses for this epoch's worth of appends to skip
	// the per-append capacity-grow cascade. On the first epoch the slice
	// is nil; on later epochs len/cap may already cover this epoch's
	// batches and the make is skipped by the cap check.
	if cap(result.Losses)-len(result.Losses) < len(batches) {
		grown := make([]DistillLoss, len(result.Losses), len(result.Losses)+len(batches))
		copy(grown, result.Losses)
		result.Losses = grown
	}
	// Pre-grow checkpoint slices when we know the rate — predictable
	// shape per epoch ((len(batches)+rate-1)/rate checkpoints), so size
	// is cheap to compute and skips repeated grows when many checkpoints
	// fire per epoch.
	if cfg.CheckpointDir != "" && cfg.CheckpointEvery > 0 {
		expected := (len(batches) + cfg.CheckpointEvery - 1) / cfg.CheckpointEvery
		if cap(result.Checkpoints)-len(result.Checkpoints) < expected {
			grown := make([]string, len(result.Checkpoints), len(result.Checkpoints)+expected)
			copy(grown, result.Checkpoints)
			result.Checkpoints = grown
		}
		if cap(result.CheckpointMetadata)-len(result.CheckpointMetadata) < expected {
			grown := make([]DistillCheckpointMetadata, len(result.CheckpointMetadata), len(result.CheckpointMetadata)+expected)
			copy(grown, result.CheckpointMetadata)
			result.CheckpointMetadata = grown
		}
	}
	// Same shape for evaluations.
	if cfg.EvalEvery > 0 {
		expected := (len(batches) + cfg.EvalEvery - 1) / cfg.EvalEvery
		if cap(result.Evaluations)-len(result.Evaluations) < expected {
			grown := make([]DistillEvalResult, len(result.Evaluations), len(result.Evaluations)+expected)
			copy(grown, result.Evaluations)
			result.Evaluations = grown
		}
	}
	// Index iteration — range over []SFTBatch copies the whole struct
	// per iteration (Batch's three slice headers + Targets' header =
	// 96 B). Indexing keeps the body to direct field reads and the
	// single assignment into batch.SFT.
	for i := range batches {
		if err := ctx.Err(); err != nil {
			return err
		}
		sftBatch := &batches[i]
		step := result.Metrics.Steps + 1
		// Only compute CacheKey when there's a teacher cache to look it
		// up in — the key is a JSON-marshal + SHA256 over the entire
		// SFTBatch (tokens + targets + mask), which can be several KB of
		// JSON encode per batch. Runners without TeacherCache attached
		// would otherwise pay this scan on every step for a value that
		// gets thrown away inside teacherLogitsForDistillBatch.
		var cacheKey string
		if runner.TeacherCache != nil {
			cacheKey = DistillBatchCacheKey(*sftBatch)
		}
		batch := DistillBatch{
			Step:        step,
			Epoch:       epoch,
			SFT:         *sftBatch,
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
		updateDistillResult(result, accumulator, len(sftBatch.Batch.Tokens), &loss, cacheStatus)
		result.Losses = append(result.Losses, loss)

		if err := maybeSaveDistillCheckpoint(ctx, runner, cfg, result, &batch, &loss); err != nil {
			return err
		}
		if err := maybeRunDistillEval(ctx, runner, cfg, result, epoch); err != nil {
			return err
		}
		emitDistillProbe(cfg, result, &loss, cacheStatus, epoch)
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
		return nil, errDistillNeedTokenizer
	}
	tok := runner.Tokenizer(ctx)
	return BuildDatasetBatches(tok, source, cfg.Batch)
}

func teacherLogitsForDistillBatch(ctx context.Context, runner DistillRunner, batch DistillBatch) (DistillLogits, string, error) {
	// Evaluate cache eligibility once — both the Get and the Put paths
	// share the same gate (cache present and a non-empty key).
	cacheable := runner.TeacherCache != nil && batch.CacheKey != ""
	if cacheable {
		logits, ok, err := runner.TeacherCache.GetTeacherLogits(ctx, batch.CacheKey)
		if err != nil {
			return nil, "", err
		}
		if ok {
			return logits, "hit", nil
		}
	}
	if runner.TeacherLogits == nil {
		return nil, "", errDistillNeedTeacherLogits
	}
	logits, err := runner.TeacherLogits(ctx, batch)
	if err != nil {
		return nil, "", err
	}
	if cacheable {
		if err := runner.TeacherCache.PutTeacherLogits(ctx, batch.CacheKey, logits); err != nil {
			return nil, "", err
		}
	}
	return logits, "miss", nil
}

func updateDistillResult(result *DistillResult, accumulator *distillMetricAccumulator, samples int, loss *DistillLoss, cacheStatus string) {
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
	// snapshot returns all four metric averages in a single nil/zero
	// guard with one float division — replacing four separate method
	// calls each with their own guard + divide.
	avg := accumulator.snapshot()
	result.Metrics.Loss = avg.loss
	result.Metrics.KL = avg.kl
	result.Metrics.SoftCrossEntropy = avg.softCE
	result.Metrics.TeacherEntropy = avg.entropy
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	result.Metrics.EvaluationCount = len(result.Evaluations)
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

// distillProbeMetaPool recycles the per-step meta map fed to
// probe.Sink.EmitProbe. The Sink contract requires synchronous clone
// on any retention path (Recorder uses CloneEvent which deep-copies
// the map), so by the time EmitProbe returns the map is no longer
// referenced by the sink and is safe to return to the pool. The
// map's value-set is the same seven keys on every iteration, so the
// pool entries are warm with the right bucket-count from the second
// step onwards.
var distillProbeMetaPool = sync.Pool{
	New: func() any {
		m := make(map[string]string, 7)
		return &m
	},
}

// distillProbeTrainingPool recycles the per-step probe.Training
// payload. Same Sink-contract argument as the meta pool: the sink
// either copies-by-value into its own storage (Recorder via
// CloneEvent), or it's an in-process listener that has finished
// reading by the time EmitProbe returns.
var distillProbeTrainingPool = sync.Pool{
	New: func() any {
		return &probe.Training{}
	},
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

type distillMetricAccumulator struct {
	tokens     int
	lossSum    float64
	klSum      float64
	softCE     float64
	entropySum float64
}

func (a *distillMetricAccumulator) add(loss *DistillLoss) {
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

// distillMetricsSnapshot is the all-in-one return shape for snapshot —
// every field is the per-token average of the corresponding accumulator
// sum, or 0 when the accumulator has no tokens yet.
type distillMetricsSnapshot struct {
	loss, kl, softCE, entropy float64
}

// snapshot returns the per-token averages for all four metrics in a
// single nil/zero guard with one float division — replaces four
// separate accessor calls in updateDistillResult.
func (a *distillMetricAccumulator) snapshot() distillMetricsSnapshot {
	if a == nil || a.tokens == 0 {
		return distillMetricsSnapshot{}
	}
	invTokens := 1.0 / float64(a.tokens)
	return distillMetricsSnapshot{
		loss:    a.lossSum * invTokens,
		kl:      a.klSum * invTokens,
		softCE:  a.softCE * invTokens,
		entropy: a.entropySum * invTokens,
	}
}

func distillResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errDistillCoreResultFailed
}

func distillCollectSamples(ctx context.Context, ds dataset.Dataset, maxSamples int) ([]dataset.Sample, error) {
	var samples []dataset.Sample
	if maxSamples > 0 {
		samples = make([]dataset.Sample, 0, maxSamples)
	}
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
