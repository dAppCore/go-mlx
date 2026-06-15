// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"math"
	"strconv"
	"time"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
	"dappco.re/go/mlx/probe"
)

const GRPOCheckpointMetadataVersion = 1

// GRPOConfig controls experimental grouped reasoning policy optimisation.
type GRPOConfig struct {
	GroupSize        int              `json:"group_size,omitempty"`
	Epochs           int              `json:"epochs,omitempty"`
	KLCoefficient    float64          `json:"kl_coefficient,omitempty"`
	AdvantageEpsilon float64          `json:"advantage_epsilon,omitempty"`
	LearningRate     float64          `json:"learning_rate,omitempty"`
	CheckpointDir    string           `json:"checkpoint_dir,omitempty"`
	CheckpointEvery  int              `json:"checkpoint_every,omitempty"`
	EvalEvery        int              `json:"eval_every,omitempty"`
	ResumePath       string           `json:"resume_path,omitempty"`
	MaxSamples       int              `json:"max_samples,omitempty"`
	RewardFuncs      []GRPORewardFunc `json:"-"`
	ProbeSink        probe.Sink       `json:"-"`
}

// GRPORunner supplies the model-specific operations for experimental GRPO.
type GRPORunner struct {
	PolicyInfo func(context.Context) ModelInfo
	Tokenizer  func(context.Context) *Tokenizer

	Rollout          func(context.Context, GRPORolloutRequest) ([]GRPORollout, error)
	ReferenceLogProb func(context.Context, GRPORolloutRequest, GRPORollout) (float64, error)
	ApplyUpdate      func(context.Context, GRPOUpdate) error
	Evaluate         func(context.Context, GRPOEvalContext) (GRPOEvalResult, error)
	SaveCheckpoint   func(context.Context, GRPOCheckpointContext) error
}

// GRPOSample is a reasoning prompt extracted from an SFT/JSONL sample.
type GRPOSample struct {
	Prompt          string            `json:"prompt"`
	ReferenceAnswer string            `json:"reference_answer,omitempty"`
	ExpectedAnswer  string            `json:"expected_answer,omitempty"`
	Reasoning       string            `json:"reasoning,omitempty"`
	Meta            map[string]string `json:"meta,omitempty"`
}

// GRPORolloutRequest asks the policy for a group of completions.
type GRPORolloutRequest struct {
	Step      int        `json:"step"`
	Epoch     int        `json:"epoch"`
	GroupSize int        `json:"group_size"`
	Sample    GRPOSample `json:"sample"`
	Config    GRPOConfig `json:"config"`
}

// GRPORollout is one sampled reasoning completion plus training annotations.
type GRPORollout struct {
	Text             string       `json:"text,omitempty"`
	Reasoning        string       `json:"reasoning,omitempty"`
	Answer           string       `json:"answer,omitempty"`
	TokenIDs         []int32      `json:"token_ids,omitempty"`
	LogProb          float64      `json:"log_prob,omitempty"`
	ReferenceLogProb float64      `json:"reference_log_prob,omitempty"`
	Reward           float64      `json:"reward,omitempty"`
	RewardParts      []GRPOReward `json:"reward_parts,omitempty"`
	Advantage        float64      `json:"advantage,omitempty"`
	KL               float64      `json:"kl,omitempty"`
	LossContribution float64      `json:"loss_contribution,omitempty"`
}

// GRPOReward is one named reward contribution.
type GRPOReward struct {
	Name   string  `json:"name"`
	Score  float64 `json:"score"`
	Weight float64 `json:"weight,omitempty"`
	Detail string  `json:"detail,omitempty"`
}

// GRPORewardContext is passed to reward functions.
type GRPORewardContext struct {
	Sample  GRPOSample
	Rollout GRPORollout
	Index   int
}

// GRPORewardFunc scores one rollout.
type GRPORewardFunc func(GRPORewardContext) (GRPOReward, error)

// GRPOUpdate is the grouped policy update consumed by a LoRA/autograd backend.
type GRPOUpdate struct {
	Step          int           `json:"step"`
	Epoch         int           `json:"epoch"`
	Sample        GRPOSample    `json:"sample"`
	Rollouts      []GRPORollout `json:"rollouts"`
	RewardMean    float64       `json:"reward_mean"`
	RewardStd     float64       `json:"reward_std"`
	KLMean        float64       `json:"kl_mean,omitempty"`
	Loss          float64       `json:"loss"`
	KLCoefficient float64       `json:"kl_coefficient,omitempty"`
}

// GRPOMetrics aggregates experimental GRPO counters.
type GRPOMetrics struct {
	Steps           int     `json:"steps"`
	Epochs          int     `json:"epochs"`
	Samples         int     `json:"samples"`
	Rollouts        int     `json:"rollouts"`
	RewardMean      float64 `json:"reward_mean"`
	RewardStd       float64 `json:"reward_std"`
	KLMean          float64 `json:"kl_mean,omitempty"`
	Loss            float64 `json:"loss"`
	LastLoss        float64 `json:"last_loss"`
	KLCoefficient   float64 `json:"kl_coefficient,omitempty"`
	CheckpointCount int     `json:"checkpoint_count"`
	EvaluationCount int     `json:"evaluation_count"`
}

// GRPOResult records one experimental GRPO run.
type GRPOResult struct {
	Experimental       bool                     `json:"experimental"`
	Policy             ModelInfo                `json:"policy"`
	Config             GRPOConfig               `json:"config"`
	Metrics            GRPOMetrics              `json:"metrics"`
	Updates            []GRPOUpdate             `json:"updates,omitempty"`
	Checkpoints        []string                 `json:"checkpoints,omitempty"`
	CheckpointMetadata []GRPOCheckpointMetadata `json:"checkpoint_metadata,omitempty"`
	Evaluations        []GRPOEvalResult         `json:"evaluations,omitempty"`
	ResumePath         string                   `json:"resume_path,omitempty"`
	ResumedFrom        *GRPOCheckpointMetadata  `json:"resumed_from,omitempty"`
	Duration           time.Duration            `json:"duration,omitempty"`
}

// GRPOCheckpointMetadata is the portable sidecar for experimental GRPO checkpoints.
type GRPOCheckpointMetadata struct {
	Version       int       `json:"version"`
	Experimental  bool      `json:"experimental"`
	Path          string    `json:"path"`
	ResumePath    string    `json:"resume_path,omitempty"`
	Step          int       `json:"step"`
	Epoch         int       `json:"epoch"`
	Samples       int       `json:"samples"`
	Rollouts      int       `json:"rollouts"`
	GroupSize     int       `json:"group_size"`
	RewardMean    float64   `json:"reward_mean"`
	RewardStd     float64   `json:"reward_std"`
	KLMean        float64   `json:"kl_mean,omitempty"`
	Loss          float64   `json:"loss"`
	KLCoefficient float64   `json:"kl_coefficient,omitempty"`
	LearningRate  float64   `json:"learning_rate,omitempty"`
	Policy        ModelInfo `json:"policy"`
}

// GRPOCheckpointContext is passed to optional native checkpoint writers.
type GRPOCheckpointContext struct {
	Path     string
	Update   GRPOUpdate
	Metadata GRPOCheckpointMetadata
}

// GRPOEvalContext is passed to optional eval hooks.
type GRPOEvalContext struct {
	Step    int
	Epoch   int
	Config  GRPOConfig
	Metrics GRPOMetrics
	Policy  ModelInfo
}

// GRPOEvalResult records one eval hook result.
type GRPOEvalResult struct {
	Step       int     `json:"step"`
	Epoch      int     `json:"epoch,omitempty"`
	Name       string  `json:"name,omitempty"`
	RewardMean float64 `json:"reward_mean,omitempty"`
	Loss       float64 `json:"loss,omitempty"`
}

// RunGRPOReasoningTraining runs an explicit experimental GRPO-style reasoning loop.
func RunGRPOReasoningTraining(ctx context.Context, runner GRPORunner, ds dataset.Dataset, cfg GRPOConfig) (*GRPOResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if runner.Rollout == nil {
		return nil, core.NewError("mlx: experimental GRPO runner requires Rollout")
	}
	if ds == nil {
		return nil, core.NewError("mlx: experimental GRPO dataset is nil")
	}
	cfg = normalizeGRPOConfig(cfg)

	result := &GRPOResult{
		Experimental: true,
		Config:       cfg,
	}
	// Pre-size Updates when the caller capped the run length — every
	// successful step appends exactly one update, so we know the upper
	// bound and can dodge the standard append 1→2→4→8…N alloc cascade
	// that would otherwise back-and-forth across Updates as steps land.
	if cfg.MaxSamples > 0 && cfg.Epochs > 0 {
		result.Updates = make([]GRPOUpdate, 0, cfg.MaxSamples*cfg.Epochs)
	}
	if runner.PolicyInfo != nil {
		result.Policy = runner.PolicyInfo(ctx)
	}
	if cfg.ResumePath != "" {
		result.ResumePath = cfg.ResumePath
		meta, err := loadGRPOResumeMetadata(cfg.ResumePath)
		if err != nil {
			return result, err
		}
		result.ResumedFrom = meta
	}

	start := time.Now()
	accumulator := &grpoMetricAccumulator{}
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if epoch > 1 {
			resetter, ok := ds.(dataset.Resetter)
			if !ok {
				return result, core.NewError("mlx: experimental GRPO dataset must implement Reset for multiple epochs")
			}
			if err := resetter.Reset(); err != nil {
				return result, err
			}
		}
		if err := runGRPOEpoch(ctx, runner, ds, cfg, result, accumulator, epoch); err != nil {
			return result, err
		}
		result.Metrics.Epochs = epoch
	}
	if result.Metrics.Steps == 0 {
		return result, core.NewError("mlx: experimental GRPO dataset produced no trainable samples")
	}
	result.Duration = nonZeroDuration(time.Since(start))
	return result, nil
}

func runGRPOEpoch(ctx context.Context, runner GRPORunner, ds dataset.Dataset, cfg GRPOConfig, result *GRPOResult, accumulator *grpoMetricAccumulator, epoch int) error {
	samples := 0
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if cfg.MaxSamples > 0 && samples >= cfg.MaxSamples {
			break
		}
		raw, ok, err := ds.Next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		sample := GRPOSampleFromSFT(raw)
		// sample.Prompt is already trimmed by GRPOSampleFromSFT — the
		// previous core.Trim re-scan was wasted work on every dataset
		// row in every epoch.
		if sample.Prompt == "" {
			continue
		}
		samples++
		step := result.Metrics.Steps + 1
		request := GRPORolloutRequest{
			Step:      step,
			Epoch:     epoch,
			GroupSize: cfg.GroupSize,
			Sample:    sample,
			Config:    cfg,
		}
		rollouts, err := runner.Rollout(ctx, request)
		if err != nil {
			return err
		}
		update, err := buildGRPOUpdate(ctx, runner, request, rollouts, cfg)
		if err != nil {
			return err
		}
		if runner.ApplyUpdate != nil {
			if err := runner.ApplyUpdate(ctx, update); err != nil {
				return err
			}
		}
		updateGRPOResult(result, accumulator, &update)
		result.Updates = append(result.Updates, update)
		if err := maybeSaveGRPOCheckpoint(ctx, runner, cfg, result, &update); err != nil {
			return err
		}
		if err := maybeRunGRPOEval(ctx, runner, cfg, result, epoch); err != nil {
			return err
		}
		emitGRPOProbe(cfg, result, &update, epoch)
	}
	return nil
}

func buildGRPOUpdate(ctx context.Context, runner GRPORunner, request GRPORolloutRequest, rollouts []GRPORollout, cfg GRPOConfig) (GRPOUpdate, error) {
	if len(rollouts) == 0 {
		return GRPOUpdate{}, core.NewError("mlx: experimental GRPO rollout returned no completions")
	}
	if len(rollouts) != request.GroupSize {
		return GRPOUpdate{}, core.NewError(core.Sprintf("mlx: experimental GRPO rollout group size mismatch: got %d want %d", len(rollouts), request.GroupSize))
	}
	rewardFuncs := cfg.RewardFuncs
	if len(rewardFuncs) == 0 {
		// Default reward funcs slice is shared package-wide — the
		// closure has no per-call state (weight=1 is captured at init)
		// and scoreGRPORollout only reads from the slice. Previously a
		// fresh closure + 1-element slice fired once per buildGRPOUpdate
		// call (per training step) for callers using the default config.
		rewardFuncs = defaultGRPORewardFuncs
	}
	// Hoist invariants out of the rollout loop — the KL branch flag and
	// the cfg-side values never change across rollouts. The compiler
	// can't prove that for an interface-method field (runner.Reference-
	// LogProb), so it re-checks both per iteration unless we lift them.
	computeKL := cfg.KLCoefficient != 0 && runner.ReferenceLogProb != nil
	klCoef := cfg.KLCoefficient
	advEps := cfg.AdvantageEpsilon
	n := len(rollouts)
	// Reuse a single GRPORewardContext across rollouts — the user-facing
	// reward func still receives it by value (scoreGRPORollout derefs
	// before each fn call), so we just refresh the Rollout + Index
	// fields per iteration instead of building a fresh ctx struct
	// (GRPOSample with map header + GRPORollout with strings + slices)
	// every time. Sample is invariant across the group.
	rewardCtx := GRPORewardContext{Sample: request.Sample}
	// Pre-allocate one shared []GRPOReward backing for all rollouts'
	// parts in this step. scoreGRPORollout carves a per-rollout view
	// out of it instead of paying its own make per call. Capacity =
	// n × len(funcs) is the upper bound (every fn produces one entry);
	// the actual len consumed depends on how many funcs are non-nil.
	// cloneGRPORollouts later copies these views OUT into the cloned
	// rollouts' own flat backing, so the shared partsBacking can be
	// GC'd at the end of buildGRPOUpdate without retaining anything.
	partsBacking := make([]GRPOReward, 0, n*len(rewardFuncs))
	for i := range n {
		rewardCtx.Rollout = rollouts[i]
		rewardCtx.Index = i
		// Hand the running tail of partsBacking to scoreGRPORollout so
		// it appends into the shared backing rather than allocating its
		// own parts slice per rollout.
		start := len(partsBacking)
		filled, total, err := scoreGRPORollout(&rewardCtx, rewardFuncs, partsBacking)
		if err != nil {
			return GRPOUpdate{}, err
		}
		partsBacking = filled
		// Slice rollouts[i].RewardParts as a 3-index view bounded to
		// what scoreGRPORollout actually appended — capacity is locked
		// so a subsequent append on this view can't overwrite the next
		// rollout's range.
		end := len(partsBacking)
		rollouts[i].RewardParts = partsBacking[start:end:end]
		rollouts[i].Reward = total
		if computeKL {
			reference, err := runner.ReferenceLogProb(ctx, request, rollouts[i])
			if err != nil {
				return GRPOUpdate{}, err
			}
			rollouts[i].ReferenceLogProb = reference
			rollouts[i].KL = rollouts[i].LogProb - reference
		}
	}
	rewardMean, rewardStd := grpoRewardStats(rollouts)
	// Reciprocal mul, single division, single std-vs-eps branch outside
	// the inner loop — when rewardStd ≤ advEps every rollout's advantage
	// is zero so the (reward-mean)/std arithmetic can be skipped entirely.
	invStd := 0.0
	useStd := rewardStd > advEps
	if useStd {
		invStd = 1.0 / rewardStd
	}
	var loss float64
	var klSum float64
	for i := range n {
		if useStd {
			rollouts[i].Advantage = (rollouts[i].Reward - rewardMean) * invStd
		} else {
			rollouts[i].Advantage = 0
		}
		rollouts[i].LossContribution = -rollouts[i].Advantage*rollouts[i].LogProb + klCoef*rollouts[i].KL
		loss += rollouts[i].LossContribution
		klSum += rollouts[i].KL
	}
	invN := 1.0 / float64(n)
	loss *= invN
	klMean := klSum * invN
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return GRPOUpdate{}, core.NewError("mlx: experimental GRPO loss is not finite")
	}
	return GRPOUpdate{
		Step:          request.Step,
		Epoch:         request.Epoch,
		Sample:        request.Sample,
		Rollouts:      cloneGRPORollouts(rollouts),
		RewardMean:    rewardMean,
		RewardStd:     rewardStd,
		KLMean:        klMean,
		Loss:          loss,
		KLCoefficient: cfg.KLCoefficient,
	}, nil
}

// scoreGRPORollout walks every reward func against ctx and appends a
// GRPOReward per non-nil func into out. The caller passes in the
// shared partsBacking and gets the grown slice back so it can carve a
// per-rollout view at known offsets. Returning out instead of a fresh
// allocation lets buildGRPOUpdate amortise N per-rollout allocations
// down to a single n*len(funcs) make at the top of the step.
func scoreGRPORollout(ctx *GRPORewardContext, funcs []GRPORewardFunc, out []GRPOReward) ([]GRPOReward, float64, error) {
	var total float64
	for _, fn := range funcs {
		if fn == nil {
			continue
		}
		reward, err := fn(*ctx)
		if err != nil {
			return out, 0, err
		}
		if reward.Name == "" {
			reward.Name = "reward"
		}
		if math.IsNaN(reward.Score) || math.IsInf(reward.Score, 0) {
			return out, 0, core.NewError("mlx: experimental GRPO reward is not finite")
		}
		out = append(out, reward)
		total += reward.Score
	}
	return out, total, nil
}

func updateGRPOResult(result *GRPOResult, accumulator *grpoMetricAccumulator, update *GRPOUpdate) {
	result.Metrics.Steps++
	result.Metrics.Samples++
	result.Metrics.Rollouts += len(update.Rollouts)
	result.Metrics.LastLoss = update.Loss
	result.Metrics.KLCoefficient = update.KLCoefficient
	accumulator.add(update)
	// snapshot returns all four metric averages in a single nil/zero
	// guard with one float division — replacing four separate method
	// calls each with their own guard + divide. Mirrors the same
	// pattern adopted for the distill metric accumulator.
	avg := accumulator.snapshot()
	result.Metrics.RewardMean = avg.rewardMean
	result.Metrics.RewardStd = avg.rewardStd
	result.Metrics.KLMean = avg.klMean
	result.Metrics.Loss = avg.loss
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	result.Metrics.EvaluationCount = len(result.Evaluations)
}

func maybeRunGRPOEval(ctx context.Context, runner GRPORunner, cfg GRPOConfig, result *GRPOResult, epoch int) error {
	if cfg.EvalEvery <= 0 || runner.Evaluate == nil || result.Metrics.Steps%cfg.EvalEvery != 0 {
		return nil
	}
	eval, err := runner.Evaluate(ctx, GRPOEvalContext{
		Step:    result.Metrics.Steps,
		Epoch:   epoch,
		Config:  cfg,
		Metrics: result.Metrics,
		Policy:  result.Policy,
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

func emitGRPOProbe(cfg GRPOConfig, result *GRPOResult, update *GRPOUpdate, epoch int) {
	if cfg.ProbeSink == nil {
		return
	}
	// Direct strconv.Itoa / strconv.FormatFloat — escape the
	// fmt.Sprintf format-parser path that interface-boxes each arg
	// and runs the (small) format machinery on every probe event.
	// emitGRPOProbe fires once per training step, so the per-event
	// alloc/CPU saving compounds across an epoch.
	meta := make(map[string]string, 8)
	meta["grpo_experimental"] = "true"
	meta["group_size"] = strconv.Itoa(cfg.GroupSize)
	meta["rollouts"] = strconv.Itoa(len(update.Rollouts))
	meta["reward_mean"] = strconv.FormatFloat(update.RewardMean, 'f', 6, 64)
	meta["reward_std"] = strconv.FormatFloat(update.RewardStd, 'f', 6, 64)
	meta["kl_mean"] = strconv.FormatFloat(update.KLMean, 'f', 6, 64)
	meta["checkpoint_count"] = strconv.Itoa(len(result.Checkpoints))
	meta["evaluation_count"] = strconv.Itoa(len(result.Evaluations))
	cfg.ProbeSink.EmitProbe(probe.Event{
		Kind:  probe.KindTraining,
		Phase: probe.PhaseTraining,
		Step:  result.Metrics.Steps,
		Meta:  meta,
		Training: &probe.Training{
			Step:         result.Metrics.Steps,
			Epoch:        epoch,
			Loss:         update.Loss,
			LearningRate: cfg.LearningRate,
		},
	})
}

func normalizeGRPOConfig(cfg GRPOConfig) GRPOConfig {
	if cfg.GroupSize <= 0 {
		cfg.GroupSize = 4
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.AdvantageEpsilon <= 0 {
		cfg.AdvantageEpsilon = 1e-8
	}
	return cfg
}

type grpoMetricAccumulator struct {
	groups    int
	rollouts  int
	rewardSum float64
	stdSum    float64
	klSum     float64
	lossSum   float64
}

func (a *grpoMetricAccumulator) add(update *GRPOUpdate) {
	if a == nil {
		return
	}
	a.groups++
	a.rollouts += len(update.Rollouts)
	a.rewardSum += update.RewardMean
	a.stdSum += update.RewardStd
	a.klSum += update.KLMean
	a.lossSum += update.Loss
}

// grpoMetricsSnapshot is the all-in-one return shape for snapshot —
// every field is the per-group average of the corresponding
// accumulator sum, or 0 when the accumulator has no groups yet.
type grpoMetricsSnapshot struct {
	rewardMean, rewardStd, klMean, loss float64
}

// snapshot returns the per-group averages for all four metrics in a
// single nil/zero guard with one float division — replaces the four
// individual accessor methods (rewardMean, rewardStd, klMean, loss),
// each of which paid its own nil-guard + divide.
func (a *grpoMetricAccumulator) snapshot() grpoMetricsSnapshot {
	if a == nil || a.groups == 0 {
		return grpoMetricsSnapshot{}
	}
	invGroups := 1.0 / float64(a.groups)
	return grpoMetricsSnapshot{
		rewardMean: a.rewardSum * invGroups,
		rewardStd:  a.stdSum * invGroups,
		klMean:     a.klSum * invGroups,
		loss:       a.lossSum * invGroups,
	}
}

func cloneGRPORollouts(rollouts []GRPORollout) []GRPORollout {
	out := make([]GRPORollout, len(rollouts))
	// Bulk copy the struct slice first — copy() lowers to memmove for
	// contiguous element memory, replacing the per-iteration struct
	// copy (GRPORollout is ~10 fields wide so each per-iter copy is
	// a non-trivial pile of moves). Inner slice fields are then
	// re-sliced into per-field flat backings so out's TokenIDs /
	// RewardParts don't alias rollouts' but only allocate two big
	// buffers instead of 2*N (one per rollout per field).
	copy(out, rollouts)
	// Two-pass clone for the inner slice fields — sum once for sizing,
	// then carve per-rollout views out of two shared backing buffers.
	// For a default group of 4 rollouts with 128 tokens + 1 reward each
	// this collapses 8 inner allocs down to 2 (one per shared backing).
	var totalTokens, totalRewards int
	for i := range rollouts {
		totalTokens += len(rollouts[i].TokenIDs)
		totalRewards += len(rollouts[i].RewardParts)
	}
	var tokenBacking []int32
	if totalTokens > 0 {
		tokenBacking = make([]int32, totalTokens)
	}
	var rewardBacking []GRPOReward
	if totalRewards > 0 {
		rewardBacking = make([]GRPOReward, totalRewards)
	}
	var tokenCursor, rewardCursor int
	for i := range rollouts {
		if src := rollouts[i].TokenIDs; len(src) > 0 {
			next := tokenCursor + len(src)
			dst := tokenBacking[tokenCursor:next:next]
			copy(dst, src)
			out[i].TokenIDs = dst
			tokenCursor = next
		} else {
			out[i].TokenIDs = nil
		}
		if src := rollouts[i].RewardParts; len(src) > 0 {
			next := rewardCursor + len(src)
			dst := rewardBacking[rewardCursor:next:next]
			copy(dst, src)
			out[i].RewardParts = dst
			rewardCursor = next
		} else {
			out[i].RewardParts = nil
		}
	}
	return out
}

func grpoResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
