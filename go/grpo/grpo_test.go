// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"math"
	"strings"
	"testing"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
	"dappco.re/go/mlx/probe"
)

// TestGrpo_RunGRPOReasoningTraining_Good covers the successful training
// loops end-to-end: a fully-wired grouped run with rewards/KL/checkpoint/
// eval/probe, a resume + max-samples + exact-reward run, a multi-epoch
// replay over a resettable dataset, and the eval-result step/epoch
// backfill. Each is a genuinely different happy-path shape.
func TestGrpo_RunGRPOReasoningTraining_Good(t *testing.T) {
	t.Run("group_rollouts_reward_kl_checkpoint_probe", func(t *testing.T) {
		dataset, err := dataset.LoadJSONL(strings.NewReader(`{"question":"What is 2+2?","reasoning":"Add two and two.","answer":"4"}`), dataset.Config{})
		if err != nil {
			t.Fatalf("dataset.LoadJSONL() error = %v", err)
		}
		recorder := probe.NewRecorder()
		checkpointDir := core.PathJoin(t.TempDir(), "checkpoints")
		var updates []GRPOUpdate
		evalCalls := 0

		result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			PolicyInfo: func(context.Context) ModelInfo {
				return ModelInfo{Architecture: "qwen3", VocabSize: 16}
			},
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				if req.GroupSize != 3 || req.Sample.ExpectedAnswer != "4" || req.Sample.Prompt == "" {
					t.Fatalf("rollout request = %+v, want grouped reasoning prompt with expected answer", req)
				}
				return []GRPORollout{
					{Text: "2+2 is 5", Answer: "5", TokenIDs: []int32{5}, LogProb: -1.50},
					{Text: "2+2 is 4", Reasoning: "two pairs make four", Answer: "4", TokenIDs: []int32{4}, LogProb: -0.50},
					{Text: "<think>2+2</think> final 4", Answer: "4", TokenIDs: []int32{4, 4}, LogProb: -0.75},
				}, nil
			},
			ReferenceLogProb: func(_ context.Context, _ GRPORolloutRequest, rollout GRPORollout) (float64, error) {
				return rollout.LogProb - 0.20, nil
			},
			ApplyUpdate: func(_ context.Context, update GRPOUpdate) error {
				updates = append(updates, update)
				return nil
			},
			Evaluate: func(_ context.Context, ctx GRPOEvalContext) (GRPOEvalResult, error) {
				evalCalls++
				return GRPOEvalResult{Step: ctx.Step, RewardMean: ctx.Metrics.RewardMean}, nil
			},
		}, dataset, GRPOConfig{
			GroupSize:       3,
			KLCoefficient:   0.2,
			CheckpointDir:   checkpointDir,
			CheckpointEvery: 1,
			EvalEvery:       1,
			RewardFuncs:     []GRPORewardFunc{GRPORewardContainsAnswer(1)},
			ProbeSink:       recorder,
		})
		if err != nil {
			t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
		}
		if result.Metrics.Steps != 1 || result.Metrics.Samples != 1 || result.Metrics.Rollouts != 3 {
			t.Fatalf("metrics = %+v, want one grouped GRPO step", result.Metrics)
		}
		if math.Abs(result.Metrics.RewardMean-(2.0/3.0)) > 1e-9 {
			t.Fatalf("reward mean = %.9f, want 2/3", result.Metrics.RewardMean)
		}
		if result.Metrics.KLMean <= 0 || result.Metrics.Loss == 0 {
			t.Fatalf("metrics = %+v, want KL-controlled non-zero policy objective", result.Metrics)
		}
		if len(updates) != 1 || len(updates[0].Rollouts) != 3 {
			t.Fatalf("updates = %+v, want one update with three rollouts", updates)
		}
		if math.Abs(updates[0].Rollouts[0].Advantage+updates[0].Rollouts[1].Advantage+updates[0].Rollouts[2].Advantage) > 1e-6 {
			t.Fatalf("advantages = %+v, want zero-mean group normalization", updates[0].Rollouts)
		}
		if updates[0].Rollouts[0].Reward >= updates[0].Rollouts[1].Reward {
			t.Fatalf("rewards = %+v, want answer reward to separate incorrect rollout", updates[0].Rollouts)
		}
		if len(result.Checkpoints) != 1 || len(result.CheckpointMetadata) != 1 {
			t.Fatalf("checkpoints = %+v metadata=%+v, want one checkpoint", result.Checkpoints, result.CheckpointMetadata)
		}
		meta, err := LoadGRPOCheckpointMetadata(result.Checkpoints[0])
		if err != nil {
			t.Fatalf("LoadGRPOCheckpointMetadata() error = %v", err)
		}
		if !meta.Experimental || meta.Step != 1 || meta.GroupSize != 3 || meta.Policy.Architecture != "qwen3" {
			t.Fatalf("checkpoint metadata = %+v, want experimental GRPO identity", meta)
		}
		if evalCalls != 1 || len(result.Evaluations) != 1 {
			t.Fatalf("evalCalls=%d evaluations=%+v, want one eval result", evalCalls, result.Evaluations)
		}
		events := recorder.Events()
		if len(events) != 1 || events[0].Training == nil || events[0].Training.Loss == 0 {
			t.Fatalf("probe events = %+v, want GRPO training probe", events)
		}
		if events[0].Meta["grpo_experimental"] != "true" || events[0].Meta["group_size"] != "3" {
			t.Fatalf("probe meta = %+v, want GRPO experimental metadata", events[0].Meta)
		}
	})

	t.Run("resume_max_samples_exact_reward", func(t *testing.T) {
		resume := core.PathJoin(t.TempDir(), "resume")
		if err := SaveGRPOCheckpointMetadata(resume, GRPOCheckpointMetadata{Step: 9, GroupSize: 1}); err != nil {
			t.Fatalf("SaveGRPOCheckpointMetadata() error = %v", err)
		}

		rolloutCalls := 0
		result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				rolloutCalls++
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, TokenIDs: []int32{1}, LogProb: -0.2}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{
			{Prompt: "first", Response: "alpha"},
			{Prompt: "second", Response: "beta"},
		}), GRPOConfig{
			GroupSize:   1,
			MaxSamples:  1,
			ResumePath:  resume,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(3)},
		})
		if err != nil {
			t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
		}
		if result.ResumedFrom == nil || result.ResumedFrom.Step != 9 || rolloutCalls != 1 {
			t.Fatalf("resume=%+v rolloutCalls=%d, want resume step 9 and one bounded rollout", result.ResumedFrom, rolloutCalls)
		}
		if result.Metrics.RewardMean != 3 || len(result.Updates) != 1 || result.Updates[0].Rollouts[0].Reward != 3 {
			t.Fatalf("result = %+v update=%+v, want exact-answer reward", result.Metrics, result.Updates)
		}
	})

	t.Run("multi_epoch_replays_dataset", func(t *testing.T) {
		// Two epochs over a two-row replayable dataset: the loop must Reset
		// between epochs and accumulate steps across both passes.
		rolloutCalls := 0
		result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				rolloutCalls++
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.3}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{
			{Prompt: "first", Response: "alpha"},
			{Prompt: "second", Response: "beta"},
		}), GRPOConfig{
			GroupSize:   1,
			Epochs:      2,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err != nil {
			t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
		}
		if result.Metrics.Epochs != 2 {
			t.Fatalf("epochs = %d, want 2", result.Metrics.Epochs)
		}
		// 2 rows × 2 epochs = 4 steps; rollout fired once per step.
		if result.Metrics.Steps != 4 || rolloutCalls != 4 {
			t.Fatalf("steps=%d rolloutCalls=%d, want 4 each", result.Metrics.Steps, rolloutCalls)
		}
		if len(result.Updates) != 4 {
			t.Fatalf("updates = %d, want 4", len(result.Updates))
		}
	})

	t.Run("maybe_run_eval_backfills_step_and_epoch", func(t *testing.T) {
		// When an Evaluate hook returns a result with Step==0 and Epoch==0,
		// the loop stamps them from the current training step and epoch
		// rather than recording a zero-keyed eval.
		result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.3}}, nil
			},
			Evaluate: func(_ context.Context, _ GRPOEvalContext) (GRPOEvalResult, error) {
				// Return Step==0 and Epoch==0 to force the backfill branch.
				return GRPOEvalResult{Name: "acc", RewardMean: 0.9}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
			GroupSize:   1,
			EvalEvery:   1,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err != nil {
			t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
		}
		if len(result.Evaluations) != 1 {
			t.Fatalf("evaluations = %+v, want one eval", result.Evaluations)
		}
		if result.Evaluations[0].Step != 1 || result.Evaluations[0].Epoch != 1 {
			t.Fatalf("eval = %+v, want backfilled Step=1 Epoch=1", result.Evaluations[0])
		}
	})
}

// TestGrpo_RunGRPOReasoningTraining_Bad covers the explicit rejection
// paths: a missing Rollout func, a nil dataset, a multi-epoch run over a
// non-resettable dataset, and an already-cancelled context plus a dataset
// that yields no trainable samples. Each path must error with a
// distinguishing message rather than producing a poisoned result.
func TestGrpo_RunGRPOReasoningTraining_Bad(t *testing.T) {
	t.Run("requires_rollout", func(t *testing.T) {
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}}), GRPOConfig{
			RewardFuncs: []GRPORewardFunc{GRPORewardContainsAnswer(1)},
		})
		if err == nil {
			t.Fatal("expected missing rollout error")
		}
		if !core.Contains(core.Lower(err.Error()), "rollout") {
			t.Fatalf("error = %v, want rollout context", err)
		}
	})

	t.Run("requires_dataset", func(t *testing.T) {
		// A nil dataset is a caller contract violation rejected before any
		// rollout work — distinct from the missing-Rollout guard.
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(context.Context, GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{}}, nil
			},
		}, nil, GRPOConfig{GroupSize: 1, RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)}})
		if err == nil || !core.Contains(core.Lower(err.Error()), "dataset") {
			t.Fatalf("error = %v, want nil-dataset rejection", err)
		}
	})

	t.Run("multi_epoch_requires_resetter", func(t *testing.T) {
		// A non-replayable (Func) dataset with Epochs>1 must fail with a Reset
		// requirement after the first epoch consumes it.
		served := false
		ds := dataset.Func(func() (dataset.Sample, bool, error) {
			if served {
				return dataset.Sample{}, false, nil
			}
			served = true
			return dataset.Sample{Prompt: "only", Response: "row"}, true, nil
		})
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.1}}, nil
			},
		}, ds, GRPOConfig{
			GroupSize:   1,
			Epochs:      2,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "reset") {
			t.Fatalf("error = %v, want Reset requirement for multi-epoch non-resetter", err)
		}
	})

	t.Run("cancelled_context_and_no_samples", func(t *testing.T) {
		// An already-cancelled context is rejected before any work happens.
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := RunGRPOReasoningTraining(cancelled, GRPORunner{
			Rollout: func(context.Context, GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}}), GRPOConfig{GroupSize: 1}); err == nil {
			t.Fatal("RunGRPOReasoningTraining(cancelled ctx) error = nil")
		}
		// A dataset whose rows are all empty prompts produces no trainable
		// samples and the loop reports that explicitly.
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(context.Context, GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "   "}}), GRPOConfig{GroupSize: 1})
		if err == nil || !core.Contains(core.Lower(err.Error()), "no trainable") {
			t.Fatalf("error = %v, want no-trainable-samples", err)
		}
	})
}

// TestGrpo_RunGRPOReasoningTraining_Ugly covers the awkward-but-tolerated
// inputs: a nil context (silently replaced with Background) paired with a
// resume path that does not yet exist (a nil ResumedFrom, no error), and a
// group whose rollouts all earn equal rewards so every advantage is a
// finite zero rather than a NaN from a zero-std division.
func TestGrpo_RunGRPOReasoningTraining_Ugly(t *testing.T) {
	t.Run("nil_context_and_missing_resume", func(t *testing.T) {
		// nil context is tolerated (replaced with Background) and a resume
		// path that does not exist yields a nil ResumedFrom without erroring.
		result, err := RunGRPOReasoningTraining(nil, GRPORunner{ //nolint:staticcheck // exercising nil-ctx tolerance
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.1}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}}), GRPOConfig{
			GroupSize:   1,
			ResumePath:  core.PathJoin(t.TempDir(), "does-not-exist"),
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err != nil {
			t.Fatalf("RunGRPOReasoningTraining(nil ctx) error = %v", err)
		}
		if result.ResumedFrom != nil {
			t.Fatalf("resumedFrom = %+v, want nil for missing resume metadata", result.ResumedFrom)
		}
		if result.Metrics.Steps != 1 {
			t.Fatalf("steps = %d, want 1", result.Metrics.Steps)
		}
	})

	t.Run("equal_rewards_have_finite_zero_advantages", func(t *testing.T) {
		var update GRPOUpdate
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{
					{Text: "same", Answer: req.Sample.ExpectedAnswer, LogProb: -1},
					{Text: "same again", Answer: req.Sample.ExpectedAnswer, LogProb: -1},
				}, nil
			},
			ApplyUpdate: func(_ context.Context, got GRPOUpdate) error {
				update = got
				return nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
			GroupSize:   2,
			RewardFuncs: []GRPORewardFunc{GRPORewardContainsAnswer(1)},
		})
		if err != nil {
			t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
		}
		for _, rollout := range update.Rollouts {
			if rollout.Advantage != 0 || math.IsNaN(rollout.LossContribution) || math.IsInf(rollout.LossContribution, 0) {
				t.Fatalf("rollout = %+v, want finite zero-advantage update", rollout)
			}
		}
	})
}

// TestGrpo_buildGRPOUpdate_Good pins the GRPO objective arithmetic to
// hand-computable values and covers the default-reward fallback. The
// advantage/loss math is asserted to 1e-9 for a [1,1,0] reward group, the
// KL term is driven by a fixed reference-logprob delta, and an empty
// RewardFuncs config falls back to the package default contains-answer
// rubric.
func TestGrpo_buildGRPOUpdate_Good(t *testing.T) {
	t.Run("exact_advantage_reward_loss_math", func(t *testing.T) {
		// A group of three exact-answer rollouts scores rewards [1, 1, 0];
		// mean = 2/3, std = sqrt(2)/3, advantages sum to zero, and with KL
		// disabled the mean loss works out to -1/sqrt(2). Every value is
		// asserted to 1e-9 so a regression in the formula is caught exactly.
		const eps = 1e-9
		invSqrt2 := 1.0 / math.Sqrt2 // 0.70710678...
		request := GRPORolloutRequest{
			Step:      1,
			Epoch:     1,
			GroupSize: 3,
			Sample:    GRPOSample{Prompt: "p", ExpectedAnswer: "42"},
		}
		rollouts := []GRPORollout{
			{Answer: "42", LogProb: -0.5},    // match  → reward 1
			{Answer: "42", LogProb: -0.5},    // match  → reward 1
			{Answer: "wrong", LogProb: -2.0}, // miss   → reward 0
		}
		cfg := normalizeGRPOConfig(GRPOConfig{
			GroupSize:   3,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		update, err := buildGRPOUpdate(context.Background(), GRPORunner{}, request, rollouts, cfg)
		if err != nil {
			t.Fatalf("buildGRPOUpdate() error = %v", err)
		}
		if math.Abs(update.RewardMean-2.0/3.0) > eps {
			t.Fatalf("reward mean = %.12f, want 2/3", update.RewardMean)
		}
		if math.Abs(update.RewardStd-math.Sqrt(2.0)/3.0) > eps {
			t.Fatalf("reward std = %.12f, want sqrt(2)/3", update.RewardStd)
		}
		wantAdv := []float64{invSqrt2, invSqrt2, -math.Sqrt2}
		for i, want := range wantAdv {
			if math.Abs(update.Rollouts[i].Advantage-want) > eps {
				t.Fatalf("advantage[%d] = %.12f, want %.12f", i, update.Rollouts[i].Advantage, want)
			}
		}
		// Advantages are zero-mean by construction (group-normalised).
		advSum := update.Rollouts[0].Advantage + update.Rollouts[1].Advantage + update.Rollouts[2].Advantage
		if math.Abs(advSum) > 1e-6 {
			t.Fatalf("advantage sum = %.12f, want ~0", advSum)
		}
		// LossContribution_i = -adv_i * logprob_i (KL disabled).
		wantLC := []float64{-invSqrt2 * -0.5, -invSqrt2 * -0.5, -(-math.Sqrt2) * -2.0}
		for i, want := range wantLC {
			if math.Abs(update.Rollouts[i].LossContribution-want) > eps {
				t.Fatalf("loss contribution[%d] = %.12f, want %.12f", i, update.Rollouts[i].LossContribution, want)
			}
		}
		if math.Abs(update.Loss-(-invSqrt2)) > eps {
			t.Fatalf("loss = %.12f, want -1/sqrt(2) = %.12f", update.Loss, -invSqrt2)
		}
		if update.KLMean != 0 {
			t.Fatalf("kl mean = %v, want 0 (no reference logprob configured)", update.KLMean)
		}
	})

	t.Run("kl_contribution_math", func(t *testing.T) {
		// With a reference logprob exactly 0.25 below each rollout's logprob,
		// KL_i = 0.25 for every rollout, so KLMean = 0.25 and each
		// LossContribution carries kl_coefficient * 0.25. Rewards are equal
		// (every rollout answers correctly) so advantages are zero and the
		// loss is purely the KL term: kl_coefficient * 0.25.
		const (
			eps   = 1e-9
			klDel = 0.25
			klCo  = 0.2
		)
		request := GRPORolloutRequest{
			Step:      1,
			Epoch:     1,
			GroupSize: 2,
			Sample:    GRPOSample{Prompt: "p", ExpectedAnswer: "ok"},
		}
		rollouts := []GRPORollout{
			{Answer: "ok", LogProb: -1.0},
			{Answer: "ok", LogProb: -0.5},
		}
		cfg := normalizeGRPOConfig(GRPOConfig{
			GroupSize:     2,
			KLCoefficient: klCo,
			RewardFuncs:   []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		update, err := buildGRPOUpdate(context.Background(), GRPORunner{
			ReferenceLogProb: func(_ context.Context, _ GRPORolloutRequest, r GRPORollout) (float64, error) {
				return r.LogProb - klDel, nil
			},
		}, request, rollouts, cfg)
		if err != nil {
			t.Fatalf("buildGRPOUpdate() error = %v", err)
		}
		for i := range update.Rollouts {
			if math.Abs(update.Rollouts[i].KL-klDel) > eps {
				t.Fatalf("kl[%d] = %.12f, want %.2f", i, update.Rollouts[i].KL, klDel)
			}
			// Equal rewards → zero advantage → loss contribution is the KL term only.
			if update.Rollouts[i].Advantage != 0 {
				t.Fatalf("advantage[%d] = %v, want 0 (equal rewards)", i, update.Rollouts[i].Advantage)
			}
			if math.Abs(update.Rollouts[i].LossContribution-klCo*klDel) > eps {
				t.Fatalf("loss contribution[%d] = %.12f, want klCoef*KL = %.12f", i, update.Rollouts[i].LossContribution, klCo*klDel)
			}
		}
		if math.Abs(update.KLMean-klDel) > eps {
			t.Fatalf("kl mean = %.12f, want %.2f", update.KLMean, klDel)
		}
		if math.Abs(update.Loss-klCo*klDel) > eps {
			t.Fatalf("loss = %.12f, want klCoef*KL = %.12f", update.Loss, klCo*klDel)
		}
	})

	t.Run("default_reward_funcs", func(t *testing.T) {
		// An empty RewardFuncs config falls back to the package default
		// (contains-answer, weight 1), so a matching rollout still scores
		// without the caller wiring a reward func.
		request := GRPORolloutRequest{Step: 1, Epoch: 1, GroupSize: 2, Sample: GRPOSample{Prompt: "p", ExpectedAnswer: "42"}}
		good, err := buildGRPOUpdate(context.Background(), GRPORunner{}, request, []GRPORollout{
			{Answer: "42", Text: "the answer is 42", LogProb: -0.5},
			{Answer: "no", Text: "not here", LogProb: -1.0},
		}, normalizeGRPOConfig(GRPOConfig{GroupSize: 2}))
		if err != nil {
			t.Fatalf("buildGRPOUpdate(default rewards) error = %v", err)
		}
		if good.Rollouts[0].Reward != 1 || good.Rollouts[1].Reward != 0 {
			t.Fatalf("rewards = [%v %v], want default contains_answer to separate the match", good.Rollouts[0].Reward, good.Rollouts[1].Reward)
		}
		if good.Rollouts[0].RewardParts[0].Name != "contains_answer" {
			t.Fatalf("reward part = %+v, want default contains_answer rubric", good.Rollouts[0].RewardParts[0])
		}
	})
}

// TestGrpo_buildGRPOUpdate_Bad covers the error branches: an empty rollout
// slice, a group-size mismatch, a reward func that errors, a non-finite
// reward score, and a +Inf logprob that makes the loss non-finite. Each
// must abort the build with a distinguishing message rather than emitting
// a poisoned update.
func TestGrpo_buildGRPOUpdate_Bad(t *testing.T) {
	request := GRPORolloutRequest{
		Step:      1,
		Epoch:     1,
		GroupSize: 2,
		Sample:    GRPOSample{Prompt: "p", ExpectedAnswer: "a"},
	}
	cases := []struct {
		name     string
		rollouts []GRPORollout
		cfg      GRPOConfig
		want     string
	}{
		{
			name: "empty",
			want: "no completions",
		},
		{
			name:     "group_mismatch",
			rollouts: []GRPORollout{{Answer: "a"}},
			want:     "group size",
		},
		{
			name:     "reward_error",
			rollouts: []GRPORollout{{Answer: "a"}, {Answer: "a"}},
			cfg: GRPOConfig{RewardFuncs: []GRPORewardFunc{func(GRPORewardContext) (GRPOReward, error) {
				return GRPOReward{}, core.NewError("reward failed")
			}}},
			want: "reward failed",
		},
		{
			name:     "nonfinite_reward",
			rollouts: []GRPORollout{{Answer: "a"}, {Answer: "a"}},
			cfg: GRPOConfig{RewardFuncs: []GRPORewardFunc{func(GRPORewardContext) (GRPOReward, error) {
				return GRPOReward{Score: math.Inf(1)}, nil
			}}},
			want: "finite",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := buildGRPOUpdate(context.Background(), GRPORunner{}, request, tc.rollouts, normalizeGRPOConfig(tc.cfg))
			if err == nil || !core.Contains(core.Lower(err.Error()), tc.want) {
				t.Fatalf("buildGRPOUpdate() error = %v, want %q", err, tc.want)
			}
		})
	}

	// A +Inf logprob makes -advantage*logprob non-finite and the build
	// rejects the resulting poisoned loss.
	if _, err := buildGRPOUpdate(context.Background(), GRPORunner{}, request, []GRPORollout{
		{Answer: "42", LogProb: math.Inf(1)},
		{Answer: "no", LogProb: -1.0},
	}, normalizeGRPOConfig(GRPOConfig{GroupSize: 2, RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)}})); err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
		t.Fatalf("buildGRPOUpdate(+Inf logprob) error = %v, want non-finite loss rejection", err)
	}
}

// TestGrpo_buildGRPOUpdate_Ugly covers the boundary group shapes: a
// single-rollout group (zero variance → zero advantage, the update is the
// pure KL/zero-advantage degenerate case) which must still build a finite
// update rather than dividing by a zero std.
func TestGrpo_buildGRPOUpdate_Ugly(t *testing.T) {
	request := GRPORolloutRequest{Step: 1, Epoch: 1, GroupSize: 1, Sample: GRPOSample{Prompt: "p", ExpectedAnswer: "42"}}
	update, err := buildGRPOUpdate(context.Background(), GRPORunner{}, request, []GRPORollout{
		{Answer: "42", LogProb: -0.5},
	}, normalizeGRPOConfig(GRPOConfig{GroupSize: 1, RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)}}))
	if err != nil {
		t.Fatalf("buildGRPOUpdate(single rollout) error = %v", err)
	}
	// One rollout: variance is zero so std ≤ epsilon and the advantage is a
	// finite zero, not a NaN.
	if update.RewardStd != 0 {
		t.Fatalf("reward std = %v, want 0 for a single rollout", update.RewardStd)
	}
	if update.Rollouts[0].Advantage != 0 || math.IsNaN(update.Loss) || math.IsInf(update.Loss, 0) {
		t.Fatalf("update = %+v, want finite zero-advantage single-rollout build", update)
	}
}

// TestGrpo_scoreGRPORollout_Ugly checks the reward-loop nil guard: a
// RewardFuncs slice carrying a nil entry alongside a real one scores only
// the real func and silently skips the nil rather than panicking. The
// surviving func's score is the whole reward.
func TestGrpo_scoreGRPORollout_Ugly(t *testing.T) {
	ctx := GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "42"},
		Rollout: GRPORollout{Answer: "42"},
	}
	out, total, err := scoreGRPORollout(&ctx, []GRPORewardFunc{nil, GRPORewardExactAnswer(2), nil}, nil)
	if err != nil {
		t.Fatalf("scoreGRPORollout() error = %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("reward parts = %+v, want only the non-nil func to contribute", out)
	}
	if total != 2 || out[0].Name != "exact_answer" {
		t.Fatalf("total=%v parts=%+v, want exact_answer weight 2", total, out)
	}
}

// TestGrpo_grpoMetricAccumulator_Ugly covers the nil/empty guards on the
// metric accumulator: add on a nil receiver is a no-op and snapshot on a
// nil or group-less accumulator returns the zero snapshot (no
// divide-by-zero), while a single update averages over one group.
func TestGrpo_grpoMetricAccumulator_Ugly(t *testing.T) {
	var nilAcc *grpoMetricAccumulator
	nilAcc.add(&GRPOUpdate{RewardMean: 1}) // must not panic
	if snap := nilAcc.snapshot(); snap != (grpoMetricsSnapshot{}) {
		t.Fatalf("nil accumulator snapshot = %+v, want zero", snap)
	}
	empty := &grpoMetricAccumulator{}
	if snap := empty.snapshot(); snap != (grpoMetricsSnapshot{}) {
		t.Fatalf("empty accumulator snapshot = %+v, want zero", snap)
	}
	// One update then snapshot averages over a single group.
	empty.add(&GRPOUpdate{RewardMean: 0.5, RewardStd: 0.25, KLMean: 0.1, Loss: 1.5})
	snap := empty.snapshot()
	if snap.rewardMean != 0.5 || snap.rewardStd != 0.25 || snap.klMean != 0.1 || snap.loss != 1.5 {
		t.Fatalf("single-group snapshot = %+v, want the lone update's values", snap)
	}
}

// TestGrpo_cloneGRPORollouts_Good covers the per-rollout branch selection
// in cloneGRPORollouts when only one of the two inner slices is populated.
// It must deep-copy the present slice into a fresh backing (no aliasing)
// and leave the absent one nil — both the TokenIDs-only and
// RewardParts-only shapes in one group.
func TestGrpo_cloneGRPORollouts_Good(t *testing.T) {
	src := []GRPORollout{
		{TokenIDs: []int32{1, 2, 3}},                                      // tokens, no reward parts
		{RewardParts: []GRPOReward{{Name: "r", Score: 1}}},                // reward parts, no tokens
		{TokenIDs: []int32{9}, RewardParts: []GRPOReward{{Name: "both"}}}, // both
		{}, // neither
	}
	out := cloneGRPORollouts(src)
	if len(out) != 4 {
		t.Fatalf("clone length = %d, want 4", len(out))
	}
	if out[0].RewardParts != nil || out[1].TokenIDs != nil {
		t.Fatalf("absent inner slices must clone to nil: out[0].RewardParts=%v out[1].TokenIDs=%v", out[0].RewardParts, out[1].TokenIDs)
	}
	if len(out[0].TokenIDs) != 3 || len(out[1].RewardParts) != 1 {
		t.Fatalf("present inner slices must be copied: out[0].TokenIDs=%v out[1].RewardParts=%v", out[0].TokenIDs, out[1].RewardParts)
	}
	if out[3].TokenIDs != nil || out[3].RewardParts != nil {
		t.Fatalf("empty rollout must clone to nil/nil, got %+v", out[3])
	}
	// Mutating the clone must not touch the source (no shared backing).
	out[0].TokenIDs[0] = 99
	if src[0].TokenIDs[0] != 1 {
		t.Fatalf("clone aliases source TokenIDs: src mutated to %v", src[0].TokenIDs[0])
	}
}

// TestGrpo_maybeRunGRPOEval_Good covers the eval cadence and result
// backfill via a full run: an Evaluate hook returning Step==0/Epoch==0 is
// stamped from the current training step and epoch rather than recorded
// with a zero key.
func TestGrpo_maybeRunGRPOEval_Good(t *testing.T) {
	result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.3}}, nil
		},
		Evaluate: func(_ context.Context, _ GRPOEvalContext) (GRPOEvalResult, error) {
			// Return Step==0 and Epoch==0 to force the backfill branch in
			// maybeRunGRPOEval.
			return GRPOEvalResult{Name: "acc", RewardMean: 0.9}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
		GroupSize:   1,
		EvalEvery:   1,
		RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	if err != nil {
		t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
	}
	if len(result.Evaluations) != 1 {
		t.Fatalf("evaluations = %+v, want one eval", result.Evaluations)
	}
	if result.Evaluations[0].Step != 1 || result.Evaluations[0].Epoch != 1 {
		t.Fatalf("eval = %+v, want backfilled Step=1 Epoch=1", result.Evaluations[0])
	}
}
