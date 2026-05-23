// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/dataset"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

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

func maybeSaveDistillCheckpoint(ctx context.Context, runner DistillRunner, cfg DistillConfig, result *DistillResult, batch *DistillBatch, loss *DistillLoss) error {
	if cfg.CheckpointDir == "" || cfg.CheckpointEvery <= 0 || result.Metrics.Steps%cfg.CheckpointEvery != 0 {
		return nil
	}
	path := core.PathJoin(cfg.CheckpointDir, formatDistillStepDir(result.Metrics.Steps))
	meta := NewDistillCheckpointMetadata(path, cfg, result, *loss, batch.Epoch)
	if runner.SaveCheckpoint != nil {
		if err := runner.SaveCheckpoint(ctx, DistillCheckpointContext{
			Path:     path,
			Batch:    *batch,
			Loss:     *loss,
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

// distillTempStringCache holds the most recently formatted
// temperature → string mapping. The temperature is per-config
// invariant — every gradient step in a run sees the same value — so
// caching by float64 bits skips strconv.FormatFloat's per-call
// allocation on every step after the first. Uses atomic for the
// cache cell so concurrent emits don't race (also matches the
// lock-free read pattern eval.go uses for its per-call invariants).
type distillTempCacheCell struct {
	bits      uint64
	formatted string
}

var distillTempStringCache atomic.Pointer[distillTempCacheCell]

// distillLossScratchPool recycles the three vocab-sized float64
// scratch buffers consumed by the per-token log-softmax + prob
// accumulators in DistillationBatchLoss. Vocab is essentially
// process-invariant (tokenizer-fixed), so pool entries warm to the
// correct capacity after the first call and every subsequent
// DistillationBatchLoss invocation lifts pre-sized buffers off the
// pool instead of paying three vocab-sized makes per call. For a
// 32k vocab that's 3 × 256KB = 768KB saved per call.
//
// Three separate pools rather than one wrapper struct — the buffers
// are independent (no shared lifecycle), and a wrapper struct would
// just add a pointer indirection per access on the hot per-token
// loop without saving any pool churn.
var (
	distillTeacherScratchPool sync.Pool
	distillTeacherProbPool    sync.Pool
	distillStudentScratchPool sync.Pool
)

// distillGetFloat64Scratch returns a *[]float64 from the pool sized
// to hold at least vocab elements. The pointer wrapper is stable
// across grow — callers pass the same *[]float64 to the matching
// pool.Put when done, which preserves any grown cap (no second
// wrapper alloc per call). Pool entries pre-sized to the running
// vocab amortise to zero per-call alloc cost across an entire
// distillation run.
//
// Per W10-G *Array pool routing: wrap the slice header in *[]T so
// sync.Pool retains a pointer (no per-Get/Put interface escape) and
// any cap grow via `*ptr = make(...)` flows back into the pool on
// the next Put.
func distillGetFloat64Scratch(pool *sync.Pool, vocab int) *[]float64 {
	if v := pool.Get(); v != nil {
		ptr := v.(*[]float64)
		if cap(*ptr) < vocab {
			*ptr = make([]float64, vocab)
		} else {
			*ptr = (*ptr)[:vocab]
		}
		return ptr
	}
	buf := make([]float64, vocab)
	return &buf
}

// distillPutScratchBuffers returns the three log-softmax scratch
// pointers to their respective pools. Grouped helper so the multiple
// error-return paths in DistillationBatchLoss stay one-liners
// instead of three lines per terminus.
func distillPutScratchBuffers(teacherPtr, teacherProbPtr, studentPtr *[]float64) {
	if teacherPtr != nil {
		distillTeacherScratchPool.Put(teacherPtr)
	}
	if teacherProbPtr != nil {
		distillTeacherProbPool.Put(teacherProbPtr)
	}
	if studentPtr != nil {
		distillStudentScratchPool.Put(studentPtr)
	}
}

func formatDistillTemperature(temp float64) string {
	bits := math.Float64bits(temp)
	if cached := distillTempStringCache.Load(); cached != nil && cached.bits == bits {
		return cached.formatted
	}
	formatted := strconv.FormatFloat(temp, 'f', 6, 64)
	distillTempStringCache.Store(&distillTempCacheCell{bits: bits, formatted: formatted})
	return formatted
}

func emitDistillProbe(cfg DistillConfig, result *DistillResult, loss *DistillLoss, cacheStatus string, epoch int) {
	if cfg.ProbeSink == nil {
		return
	}
	metaPtr := distillProbeMetaPool.Get().(*map[string]string)
	meta := *metaPtr
	// Don't bother clear()-ing — every key is reassigned each call,
	// so any stale value is overwritten before the map is read by the
	// sink. Pool entries land here with their bucket array already
	// warm (cap 8) from a previous iteration.
	meta["distillation"] = "true"
	meta["loss_kind"] = string(loss.Kind)
	meta["temperature"] = formatDistillTemperature(loss.Temperature)
	meta["tokens"] = core.Itoa(loss.Tokens)
	meta["teacher_cache"] = cacheStatus
	meta["checkpoint_count"] = core.Itoa(len(result.Checkpoints))
	meta["evaluation_count"] = core.Itoa(len(result.Evaluations))

	training := distillProbeTrainingPool.Get().(*probe.Training)
	training.Step = result.Metrics.Steps
	training.Epoch = epoch
	training.Loss = loss.Value
	training.LearningRate = cfg.LearningRate

	cfg.ProbeSink.EmitProbe(probe.Event{
		Kind:     probe.KindTraining,
		Phase:    probe.PhaseTraining,
		Step:     result.Metrics.Steps,
		Meta:     meta,
		Training: training,
	})
	// Public Sink contract — by the time EmitProbe returns, the sink
	// has either consumed-by-value (in-process listener) or cloned
	// (Recorder.EmitProbe → CloneEvent does a deep-copy of meta +
	// Training). Either way the pool can take the map and pointer
	// back without aliasing risk.
	distillProbeTrainingPool.Put(training)
	distillProbeMetaPool.Put(metaPtr)
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
	// Validate temperature once at the call boundary — the per-token inner
	// loop invokes logSoftmax{,AndProb}TemperatureInto thousands of times,
	// and the helpers' per-call `temperature <= 0 || NaN || Inf` check is
	// the same gate every iteration. Hoist + pass the pre-computed invTemp
	// so the helpers skip both the per-call validation and the per-call
	// reciprocal division.
	if cfg.Temperature <= 0 || math.IsNaN(cfg.Temperature) || math.IsInf(cfg.Temperature, 0) {
		return DistillLoss{}, errDistillTempInvalid
	}
	invTemp := 1.0 / cfg.Temperature
	var softCE float64
	var entropy float64
	var tokens int
	// Scratch buffers reused across every masked token — vocab size is
	// constant (shape-checked above), so three pre-allocated float64 slices
	// replace per-token allocations inside logSoftmaxInvTempInto +
	// logSoftmaxAndProbInvTempInto. For a 32k vocab and 1000 tokens
	// this skips ~2000 256KB allocations per call.
	// teacherProbScratch holds prob(x) = exp(log_prob(x)) computed once
	// inside the log-softmax loop — the inner accumulator below would
	// otherwise call math.Exp per element to recover it.
	//
	// The buffers themselves are now pooled across distillation calls —
	// vocab is process-invariant (tokenizer-fixed), so pool entries hold
	// the right cap from the first call onwards and DistillationBatchLoss
	// itself amortises down to zero per-call alloc cost (3 × vocab × 8 B
	// saved per call, e.g. ~768 KB for 32k vocab). Avoiding `defer` here
	// is deliberate — a deferred Put closure heap-allocates the defer
	// record on every call, which would re-introduce the alloc the pool
	// is trying to eliminate. Pool puts run on the explicit return paths
	// below (one per terminal branch).
	var teacherScratch, teacherProbScratch, studentScratch []float64
	var teacherScratchPtr, teacherProbPtr, studentScratchPtr *[]float64
	// Hoist mask-empty once — an empty mask means "all tokens included",
	// so per-cell calls were wasted when the mask is absent or zero-length.
	// maskRows is non-nil only when we need per-row inspection.
	var maskRows [][]float32
	if len(mask) > 0 {
		maskRows = mask
	}
	for i := range teacher {
		// Per-row mask access — fetch maskRow once, then per-column the
		// check is a single len + element compare with no extra branches.
		// Hoist tRow + sRow once per i: the inner loop previously paid for
		// three teacher[i] / two student[i] slice-header loads per token
		// the compiler can't fold because mask/teacher/student aliasing
		// can't be proven away through the function call boundary.
		tRow := teacher[i]
		sRow := student[i]
		upper := len(tRow)
		var maskRow []float32
		if maskRows != nil {
			if i >= len(maskRows) {
				continue
			}
			maskRow = maskRows[i]
			if maskRow == nil {
				continue
			}
			// Cap the inner loop at len(maskRow) — j values past the
			// mask length all hit the original `j >= len(maskRow)`
			// guard and were skipped anyway. Bounding upper eliminates
			// the per-j length check inside the loop.
			if len(maskRow) < upper {
				upper = len(maskRow)
			}
		}
		// Split mask-present vs mask-absent paths — the per-j `if maskRow
		// != nil && maskRow[j] <= 0` check fires every iteration even when
		// the entire batch was called without a mask, which is the common
		// pre-tokenized teacher-forcing path. Mask-absent branch drops the
		// per-token branch + bounds-check entirely.
		if maskRow == nil {
			for j := 0; j < upper; j++ {
				tCell := tRow[j]
				sCell := sRow[j]
				vocab := len(tCell)
				if cap(teacherScratch) < vocab {
					// First-call cap grow (pool warm-up) or vocab-growth
					// across the per-cell variation case. Lift the pool
					// pointer once and grow in place — subsequent cap
					// trips inside this call grow the existing pointer
					// without re-Get'ing a fresh wrapper.
					if teacherScratchPtr == nil {
						teacherScratchPtr = distillGetFloat64Scratch(&distillTeacherScratchPool, vocab)
						teacherProbPtr = distillGetFloat64Scratch(&distillTeacherProbPool, vocab)
						studentScratchPtr = distillGetFloat64Scratch(&distillStudentScratchPool, vocab)
					} else {
						*teacherScratchPtr = make([]float64, vocab)
						*teacherProbPtr = make([]float64, vocab)
						*studentScratchPtr = make([]float64, vocab)
					}
					teacherScratch = *teacherScratchPtr
					teacherProbScratch = *teacherProbPtr
					studentScratch = *studentScratchPtr
				}
				teacherScratch = teacherScratch[:vocab]
				teacherProbScratch = teacherProbScratch[:vocab]
				studentScratch = studentScratch[:vocab]
				if err := logSoftmaxAndProbInvTempInto(tCell, invTemp, teacherScratch, teacherProbScratch); err != nil {
					distillPutScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
					return DistillLoss{}, err
				}
				if err := logSoftmaxInvTempInto(sCell, invTemp, studentScratch); err != nil {
					distillPutScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
					return DistillLoss{}, err
				}
				// Teacher probabilities are already in teacherProbScratch —
				// the inner loop skips the per-element math.Exp the original
				// form paid to recover prob from log-prob. For 32k vocab this
				// saves ~32k math.Exp calls per masked token. Subtracting
				// directly (softCE -= prob*X) folds the negation into the
				// accumulator update so no per-iteration temporary is
				// needed.
				for k, teacherProb := range teacherProbScratch {
					softCE -= teacherProb * studentScratch[k]
					entropy -= teacherProb * teacherScratch[k]
				}
				tokens++
			}
			continue
		}
		for j := 0; j < upper; j++ {
			if maskRow[j] <= 0 {
				continue
			}
			tCell := tRow[j]
			sCell := sRow[j]
			vocab := len(tCell)
			if cap(teacherScratch) < vocab {
				if teacherScratchPtr == nil {
					teacherScratchPtr = distillGetFloat64Scratch(&distillTeacherScratchPool, vocab)
					teacherProbPtr = distillGetFloat64Scratch(&distillTeacherProbPool, vocab)
					studentScratchPtr = distillGetFloat64Scratch(&distillStudentScratchPool, vocab)
				} else {
					*teacherScratchPtr = make([]float64, vocab)
					*teacherProbPtr = make([]float64, vocab)
					*studentScratchPtr = make([]float64, vocab)
				}
				teacherScratch = *teacherScratchPtr
				teacherProbScratch = *teacherProbPtr
				studentScratch = *studentScratchPtr
			}
			teacherScratch = teacherScratch[:vocab]
			teacherProbScratch = teacherProbScratch[:vocab]
			studentScratch = studentScratch[:vocab]
			if err := logSoftmaxAndProbInvTempInto(tCell, invTemp, teacherScratch, teacherProbScratch); err != nil {
				distillPutScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
				return DistillLoss{}, err
			}
			if err := logSoftmaxInvTempInto(sCell, invTemp, studentScratch); err != nil {
				distillPutScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
				return DistillLoss{}, err
			}
			for k, teacherProb := range teacherProbScratch {
				softCE -= teacherProb * studentScratch[k]
				entropy -= teacherProb * teacherScratch[k]
			}
			tokens++
		}
	}
	distillPutScratchBuffers(teacherScratchPtr, teacherProbPtr, studentScratchPtr)
	if tokens == 0 {
		return DistillLoss{}, errDistillNoMaskedTokens
	}
	softCE /= float64(tokens)
	entropy /= float64(tokens)
	kl := softCE - entropy
	if kl < 0 && math.Abs(kl) < 1e-12 {
		kl = 0
	}
	if kl < 0 || math.IsNaN(kl) || math.IsInf(kl, 0) {
		return DistillLoss{}, errDistillKLNotFinite
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
		return errDistillCheckpointPath
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
		return nil, errDistillCheckpointPath
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
		return errTeacherLogitsEmpty
	}
	if len(teacher) != len(student) {
		return errDistillLogitBatch
	}
	for i := range teacher {
		// Hoist the per-row [][]float32 slice headers once so the inner
		// loop re-indexing pays one pointer load instead of two double-
		// indexes per token.
		tRow := teacher[i]
		sRow := student[i]
		if len(tRow) != len(sRow) {
			return errDistillLogitSeq
		}
		for j := range tRow {
			tVocab := len(tRow[j])
			if tVocab == 0 {
				return errDistillLogitEmptyVocab
			}
			if tVocab != len(sRow[j]) {
				return errDistillLogitVocab
			}
		}
	}
	return nil
}

// logSoftmaxAndProbInvTempInto writes both log_prob and prob for
// each logit, given pre-computed invTemp (1/temperature). logOut[i] =
// log(softmax(logits/temp))[i] and probOut[i] = exp(logOut[i]). The
// DistillationBatchLoss inner loop needs both teacher log-probs (for
// the entropy term) and teacher probs (as the weight on the softCE /
// entropy accumulators). The previous form called math.Exp inside the
// inner accumulator loop to recover prob from log_prob; capturing prob
// during the renormalize pass here skips that per-element math.Exp
// entirely. The invTemp + buffer-shape preconditions are caller-owned
// (validated once in DistillationBatchLoss), so the per-token call
// pays no validation overhead.
func logSoftmaxAndProbInvTempInto(logits []float32, invTemp float64, logOut, probOut []float64) error {
	maxLogit := math.Inf(-1)
	for i, logit := range logits {
		value := float64(logit) * invTemp
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errDistillLogitNotFinite
		}
		logOut[i] = value
		if value > maxLogit {
			maxLogit = value
		}
	}
	// Compute exp(value - maxLogit) and accumulate the partition fn.
	// Store the unnormalised exp in probOut so we don't need to
	// recompute math.Exp during the normalise pass below.
	var sumExp float64
	for i, value := range logOut {
		e := math.Exp(value - maxLogit)
		probOut[i] = e
		sumExp += e
	}
	logDenom := maxLogit + math.Log(sumExp)
	invSum := 1.0 / sumExp
	for i, value := range logOut {
		logOut[i] = value - logDenom
		probOut[i] *= invSum
	}
	return nil
}

// logSoftmaxInvTempInto writes len(logits) log-softmax values into out,
// given pre-computed invTemp (1/temperature). out must be pre-sized to
// len(logits); callers in the distillation hot loop reuse the same
// scratch buffer across every masked token to skip per-token allocation
// of vocab-sized float64 slices. invTemp + buffer-shape preconditions
// are caller-owned (validated once in DistillationBatchLoss), so the
// per-token call pays no validation overhead.
func logSoftmaxInvTempInto(logits []float32, invTemp float64, out []float64) error {
	maxLogit := math.Inf(-1)
	for i, logit := range logits {
		value := float64(logit) * invTemp
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errDistillLogitNotFinite
		}
		out[i] = value
		if value > maxLogit {
			maxLogit = value
		}
	}
	var sumExp float64
	for _, value := range out {
		sumExp += math.Exp(value - maxLogit)
	}
	logDenom := maxLogit + math.Log(sumExp)
	for i, value := range out {
		out[i] = value - logDenom
	}
	return nil
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

func cloneDistillLogits(logits DistillLogits) DistillLogits {
	if len(logits) == 0 {
		return nil
	}
	// Three-flat-buffer clone — first count rows + cells across the
	// batch, then allocate THREE flat buffers (the outer DistillLogits,
	// one shared [][]float32 for the middle row-slice-headers, one
	// shared []float32 for all cell data). Each per-batch middle slice
	// + per-cell []float32 are carved as 3-index slice views into the
	// shared backings instead of paying their own malloc.
	//
	// For a 4×128×32000 teacher tensor:
	//   pre:   513 allocs (1 outer + 4 middle + 4×128 inner)
	//   2-pass:  6 allocs (1 outer + 4 middle + 1 flat cell buffer)
	//   3-pass:  3 allocs (1 outer + 1 flat middle + 1 flat cell)
	//
	// The flat-backing form also gives the resulting clone better cache
	// locality (sequential float32 + sequential slice-header stride)
	// versus the per-cell-alloc form where each row could land on a
	// distinct page.
	var totalRows, totalCells int
	for i := range logits {
		row := logits[i]
		totalRows += len(row)
		for j := range row {
			totalCells += len(row[j])
		}
	}
	out := make(DistillLogits, len(logits))
	if totalRows == 0 {
		return out
	}
	rowBacking := make([][]float32, totalRows)
	flat := make([]float32, totalCells)
	rowCursor := 0
	cellCursor := 0
	for i := range logits {
		row := logits[i]
		rowsHere := len(row)
		rowEnd := rowCursor + rowsHere
		outRow := rowBacking[rowCursor:rowEnd:rowEnd]
		for j := range row {
			src := row[j]
			next := cellCursor + len(src)
			dst := flat[cellCursor:next:next]
			copy(dst, src)
			outRow[j] = dst
			cellCursor = next
		}
		out[i] = outRow
		rowCursor = rowEnd
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

// formatDistillStepDir builds the "step-NNNNNN" checkpoint dirname using
// strconv.AppendInt with explicit zero padding, avoiding fmt's reflection
// path on the per-checkpoint hot loop. Digit count is computed in place
// instead of via a throwaway strconv.AppendInt(nil, ...) so the function
// allocates exactly once — the returned string itself.
func formatDistillStepDir(step int) string {
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
