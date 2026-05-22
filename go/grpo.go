// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/dataset"
	"math"
	"strconv"
	"time"

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
	for i := 0; i < n; i++ {
		parts, total, err := scoreGRPORollout(GRPORewardContext{Sample: request.Sample, Rollout: rollouts[i], Index: i}, rewardFuncs)
		if err != nil {
			return GRPOUpdate{}, err
		}
		rollouts[i].RewardParts = parts
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
	for i := 0; i < n; i++ {
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

func scoreGRPORollout(ctx GRPORewardContext, funcs []GRPORewardFunc) ([]GRPOReward, float64, error) {
	parts := make([]GRPOReward, 0, len(funcs))
	var total float64
	for _, fn := range funcs {
		if fn == nil {
			continue
		}
		reward, err := fn(ctx)
		if err != nil {
			return nil, 0, err
		}
		if reward.Name == "" {
			reward.Name = "reward"
		}
		if math.IsNaN(reward.Score) || math.IsInf(reward.Score, 0) {
			return nil, 0, core.NewError("mlx: experimental GRPO reward is not finite")
		}
		parts = append(parts, reward)
		total += reward.Score
	}
	return parts, total, nil
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

func maybeSaveGRPOCheckpoint(ctx context.Context, runner GRPORunner, cfg GRPOConfig, result *GRPOResult, update *GRPOUpdate) error {
	if cfg.CheckpointDir == "" || cfg.CheckpointEvery <= 0 || result.Metrics.Steps%cfg.CheckpointEvery != 0 {
		return nil
	}
	path := core.PathJoin(cfg.CheckpointDir, grpoStepName(result.Metrics.Steps))
	meta := NewGRPOCheckpointMetadata(path, cfg, result, *update)
	if runner.SaveCheckpoint != nil {
		if err := runner.SaveCheckpoint(ctx, GRPOCheckpointContext{Path: path, Update: *update, Metadata: meta}); err != nil {
			return err
		}
	}
	if err := SaveGRPOCheckpointMetadata(path, meta); err != nil {
		return err
	}
	result.Checkpoints = append(result.Checkpoints, path)
	result.CheckpointMetadata = append(result.CheckpointMetadata, meta)
	result.Metrics.CheckpointCount = len(result.Checkpoints)
	return nil
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

// GRPOSampleFromSFT extracts a reasoning prompt and expected answer.
func GRPOSampleFromSFT(sample dataset.Sample) GRPOSample {
	prompt := core.Trim(sample.Prompt)
	if prompt == "" {
		prompt = core.Trim(sample.Text)
	}
	// Trim Response once and feed the trimmed string back into the
	// (by-value) sample copy so the inner ExtractGRPOExpectedAnswer +
	// extractGRPOReasoningWithAnswer both see a pre-trimmed Response.
	// strings.TrimSpace is a no-op on already-trimmed input so the
	// inner re-trims become free; we save the two extra whitespace
	// scans the original form paid on every reasoning sample.
	sample.Response = core.Trim(sample.Response)
	// Extract the answer once and forward it to the reasoning step —
	// extractGRPOReasoning would otherwise re-run the full meta-key
	// sweep + line scan to recover the same value.
	expected := ExtractGRPOExpectedAnswer(sample)
	return GRPOSample{
		Prompt:          prompt,
		ReferenceAnswer: sample.Response,
		ExpectedAnswer:  expected,
		Reasoning:       extractGRPOReasoningWithAnswer(sample, expected),
		Meta:            cloneStringMap(sample.Meta),
	}
}

// grpoAnswerMetaKeys are the SFT-meta keys ExtractGRPOExpectedAnswer
// consults when the dataset carries an explicit answer field. Hoisted
// to package-level so we don't rebuild the four-entry backing array
// on every reasoning sample.
var grpoAnswerMetaKeys = [...]string{"answer", "expected_answer", "solution", "output"}

// ExtractGRPOExpectedAnswer returns the answer target from reasoning-style samples.
func ExtractGRPOExpectedAnswer(sample dataset.Sample) string {
	if sample.Meta != nil {
		// Lift the nil check out of the loop — meta is invariant across
		// the key sweep.
		for _, key := range grpoAnswerMetaKeys {
			if value := core.Trim(sample.Meta[key]); value != "" {
				return value
			}
		}
	}
	text := core.Trim(sample.Response)
	if text == "" {
		text = core.Trim(sample.Text)
	}
	// Fast path — when the text has no CR we skip the strings.Count
	// scan that ReplaceAll runs to size the result builder. The typical
	// SFT sample is LF-only, so this short-circuits the (small but
	// real) per-call Count walk for the common case.
	normalised := text
	if core.Index(text, "\r") >= 0 {
		normalised = core.Replace(text, "\r\n", "\n")
	}
	// Single-line fast path — when the response is a single line (no
	// "\n"), Split would allocate a one-element []string just to feed it
	// straight to cleanGRPOAnswerLine. Skip the slice entirely. Short
	// SFT answers ("42", "Paris", a sentence) hit this branch.
	if core.Index(normalised, "\n") < 0 {
		return cleanGRPOAnswerLine(normalised)
	}
	lines := core.Split(normalised, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := cleanGRPOAnswerLine(lines[i])
		if line != "" {
			return line
		}
	}
	return ""
}

func extractGRPOReasoning(sample dataset.Sample) string {
	return extractGRPOReasoningWithAnswer(sample, ExtractGRPOExpectedAnswer(sample))
}

// extractGRPOReasoningWithAnswer is the inner form that takes the
// already-extracted expected answer so callers (the dominant one being
// GRPOSampleFromSFT) don't run ExtractGRPOExpectedAnswer twice — once
// for the answer field and once again here for the suffix-strip.
func extractGRPOReasoningWithAnswer(sample dataset.Sample, answer string) string {
	if sample.Meta != nil {
		if value := core.Trim(sample.Meta["reasoning"]); value != "" {
			return value
		}
		if value := core.Trim(sample.Meta["thinking"]); value != "" {
			return value
		}
	}
	if answer == "" {
		return ""
	}
	response := core.Trim(sample.Response)
	if response == "" {
		return ""
	}
	return core.Trim(core.TrimSuffix(response, answer))
}

// grpoAnswerPrefixes are the reasoning-style answer prefixes
// cleanGRPOAnswerLine looks for. Hoisted to a package-level var so
// every call doesn't re-allocate the three-element backing array
// (cleanGRPOAnswerLine fires for every line in every reasoning
// sample on the GRPOSampleFromSFT / ExtractGRPOExpectedAnswer path).
var grpoAnswerPrefixes = [...]string{"final answer:", "answer:", "solution:"}

func cleanGRPOAnswerLine(line string) string {
	line = core.Trim(line)
	if line == "" {
		return ""
	}
	// First-byte gate — the three answer prefixes all start with one of
	// {a, f, s}. Anything else skips the core.Lower allocation entirely.
	// On free-form text the dominant outcome is "no match", so we cash
	// in the per-line lowercase string for nothing in the common case.
	switch line[0] {
	case 'a', 'A', 'f', 'F', 's', 'S':
	default:
		return line
	}
	lower := core.Lower(line)
	for _, prefix := range grpoAnswerPrefixes {
		if core.HasPrefix(lower, prefix) {
			return core.Trim(line[len(prefix):])
		}
	}
	return line
}

// defaultGRPORewardFuncs is the fallback []GRPORewardFunc used by
// buildGRPOUpdate when GRPOConfig.RewardFuncs is empty. Package-level
// so we don't allocate a fresh closure + 1-element slice once per
// training step on the default-config path. The captured weight (1)
// is fixed at init.
var defaultGRPORewardFuncs = []GRPORewardFunc{GRPORewardContainsAnswer(1)}

// GRPORewardContainsAnswer rewards a rollout when it contains the expected answer.
func GRPORewardContainsAnswer(weight float64) GRPORewardFunc {
	if weight == 0 {
		weight = 1
	}
	return func(ctx GRPORewardContext) (GRPOReward, error) {
		expected := core.Lower(core.Trim(ctx.Sample.ExpectedAnswer))
		if expected == "" {
			return GRPOReward{Name: "contains_answer", Weight: weight, Detail: "no expected answer"}, nil
		}
		text := core.Lower(core.Join("\n", ctx.Rollout.Answer, ctx.Rollout.Text, ctx.Rollout.Reasoning))
		score := 0.0
		detail := "missing"
		if core.Contains(text, expected) {
			score = weight
			detail = "matched"
		}
		return GRPOReward{Name: "contains_answer", Score: score, Weight: weight, Detail: detail}, nil
	}
}

// GRPORewardExactAnswer rewards exact normalized answer matches.
func GRPORewardExactAnswer(weight float64) GRPORewardFunc {
	if weight == 0 {
		weight = 1
	}
	return func(ctx GRPORewardContext) (GRPOReward, error) {
		expected := core.Lower(core.Trim(ctx.Sample.ExpectedAnswer))
		answer := core.Lower(core.Trim(ctx.Rollout.Answer))
		score := 0.0
		detail := "missing"
		if expected != "" && answer == expected {
			score = weight
			detail = "matched"
		}
		return GRPOReward{Name: "exact_answer", Score: score, Weight: weight, Detail: detail}, nil
	}
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

func grpoRewardStats(rollouts []GRPORollout) (float64, float64) {
	n := len(rollouts)
	if n == 0 {
		return 0, 0
	}
	// Index iteration — range over []GRPORollout copies the whole struct
	// (Text/Reasoning/Answer strings, TokenIDs + RewardParts slice
	// headers, all the float fields) on each iteration even though we
	// only ever read the Reward float. Indexing skips the copy.
	var sum float64
	for i := 0; i < n; i++ {
		sum += rollouts[i].Reward
	}
	invN := 1.0 / float64(n)
	mean := sum * invN
	var variance float64
	for i := 0; i < n; i++ {
		delta := rollouts[i].Reward - mean
		variance += delta * delta
	}
	variance *= invN
	return mean, math.Sqrt(variance)
}

// NewGRPOCheckpointMetadata captures reproducible experimental GRPO state.
func NewGRPOCheckpointMetadata(path string, cfg GRPOConfig, result *GRPOResult, update GRPOUpdate) GRPOCheckpointMetadata {
	cfg = normalizeGRPOConfig(cfg)
	meta := GRPOCheckpointMetadata{
		Version:       GRPOCheckpointMetadataVersion,
		Experimental:  true,
		Path:          path,
		ResumePath:    cfg.ResumePath,
		Step:          update.Step,
		Epoch:         update.Epoch,
		GroupSize:     cfg.GroupSize,
		RewardMean:    update.RewardMean,
		RewardStd:     update.RewardStd,
		KLMean:        update.KLMean,
		Loss:          update.Loss,
		KLCoefficient: cfg.KLCoefficient,
		LearningRate:  cfg.LearningRate,
	}
	if result != nil {
		meta.Samples = result.Metrics.Samples
		meta.Rollouts = result.Metrics.Rollouts
		meta.Policy = result.Policy
	}
	return meta
}

// SaveGRPOCheckpointMetadata writes checkpoint metadata beside policy artifacts.
func SaveGRPOCheckpointMetadata(path string, meta GRPOCheckpointMetadata) error {
	if path == "" {
		return core.NewError("mlx: experimental GRPO checkpoint metadata path is required")
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	meta.Experimental = true
	if meta.Path == "" {
		meta.Path = path
	}
	metadataPath := grpoCheckpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("GRPOCheckpointMetadata.Save", "ensure metadata dir", grpoResultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("GRPOCheckpointMetadata.Save", "marshal metadata", grpoResultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("GRPOCheckpointMetadata.Save", "write metadata", grpoResultError(result))
	}
	return nil
}

// LoadGRPOCheckpointMetadata reads checkpoint metadata written by SaveGRPOCheckpointMetadata.
func LoadGRPOCheckpointMetadata(path string) (*GRPOCheckpointMetadata, error) {
	if path == "" {
		return nil, core.NewError("mlx: experimental GRPO checkpoint metadata path is required")
	}
	read := core.ReadFile(grpoCheckpointMetadataPath(path))
	if !read.OK {
		return nil, grpoResultError(read)
	}
	var meta GRPOCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadGRPOCheckpointMetadata", "parse metadata", grpoResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	return &meta, nil
}

func loadGRPOResumeMetadata(path string) (*GRPOCheckpointMetadata, error) {
	read := core.ReadFile(grpoCheckpointMetadataPath(path))
	if !read.OK {
		err := grpoResultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta GRPOCheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadGRPOResumeMetadata", "parse metadata", grpoResultError(result))
	}
	if meta.Version == 0 {
		meta.Version = GRPOCheckpointMetadataVersion
	}
	return &meta, nil
}

func grpoCheckpointMetadataPath(path string) string {
	return core.PathJoin(path, "grpo_checkpoint.json")
}

// grpoStepName renders the step-NNNNNN directory name used for GRPO
// checkpoints. Same output as fmt.Sprintf("step-%06d", step) — six-
// digit zero-pad below 1e6, untruncated digit count above. Built with
// strconv.AppendInt so no fmt format-parser + no interface-boxing of
// the int arg; pre-sized output keeps the alloc count at one.
func grpoStepName(step int) string {
	const prefix = "step-"
	const padTo = 6
	// Allocate room for the prefix plus enough digits — 20 covers the
	// max int64 width.
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	if step >= 0 && step < 100000 {
		// Hand-rolled zero-pad — strconv.Itoa lacks a Printf-style
		// width modifier, so for the typical sub-1e5 range we count
		// leading zeros ourselves. Above 1e5 strconv emits the full
		// width naturally.
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
	// re-cloned so out's TokenIDs/RewardParts don't alias rollouts'.
	copy(out, rollouts)
	for i := range rollouts {
		// core.SliceClone is slices.Clone — pre-sized make+copy, one
		// alloc per slice instead of append-grow on a nil head. Also
		// returns nil for nil input so empty TokenIDs / RewardParts
		// don't allocate at all.
		out[i].TokenIDs = core.SliceClone(rollouts[i].TokenIDs)
		out[i].RewardParts = core.SliceClone(rollouts[i].RewardParts)
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
