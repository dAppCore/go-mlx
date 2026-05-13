// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/dataset"
	"math"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/probe"
)

func TestRunGRPOReasoningTraining_GroupRolloutsRewardKLCheckpointProbe_Good(t *testing.T) {
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
}

func TestGRPORewardContainsAnswer_ExtractsReasoningAnswer_Good(t *testing.T) {
	sample := GRPOSample{
		Prompt:          "Solve",
		ReferenceAnswer: "reasoning trace\n\n42",
		ExpectedAnswer:  ExtractGRPOExpectedAnswer(dataset.Sample{Response: "reasoning trace\n\n42"}),
	}
	reward, err := GRPORewardContainsAnswer(2)(GRPORewardContext{
		Sample:  sample,
		Rollout: GRPORollout{Text: "The final answer is 42."},
	})
	if err != nil {
		t.Fatalf("GRPORewardContainsAnswer() error = %v", err)
	}
	if reward.Score != 2 || reward.Name == "" {
		t.Fatalf("reward = %+v, want weighted answer match", reward)
	}
}

func TestRunGRPOReasoningTraining_ResumeMaxSamplesExactReward_Good(t *testing.T) {
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
}

func TestRunGRPOReasoningTraining_RequiresRollout_Bad(t *testing.T) {
	_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}}), GRPOConfig{
		RewardFuncs: []GRPORewardFunc{GRPORewardContainsAnswer(1)},
	})
	if err == nil {
		t.Fatal("expected missing rollout error")
	}
	if !core.Contains(core.Lower(err.Error()), "rollout") {
		t.Fatalf("error = %v, want rollout context", err)
	}
}

func TestBuildGRPOUpdate_ErrorBranches_Bad(t *testing.T) {
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
}

func TestGRPORewardExactAnswerAndMetadataErrors_Bad(t *testing.T) {
	reward, err := GRPORewardExactAnswer(0)(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "alpha"},
		Rollout: GRPORollout{Answer: "beta"},
	})
	if err != nil {
		t.Fatalf("GRPORewardExactAnswer() error = %v", err)
	}
	if reward.Score != 0 || reward.Weight != 1 || reward.Detail != "missing" {
		t.Fatalf("reward = %+v, want default weight miss", reward)
	}
	if err := SaveGRPOCheckpointMetadata("", GRPOCheckpointMetadata{}); err == nil {
		t.Fatal("SaveGRPOCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadGRPOCheckpointMetadata(""); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(empty) error = nil")
	}
	dir := t.TempDir()
	writeModelPackFile(t, grpoCheckpointMetadataPath(dir), "{")
	if _, err := LoadGRPOCheckpointMetadata(dir); err == nil {
		t.Fatal("LoadGRPOCheckpointMetadata(invalid JSON) error = nil")
	}
	if _, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(context.Context, GRPORolloutRequest) ([]GRPORollout, error) {
			return nil, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{ResumePath: dir}); err == nil {
		t.Fatal("RunGRPOReasoningTraining(invalid resume metadata) error = nil")
	}
}

func TestRunGRPOReasoningTraining_EqualRewardsHaveFiniteZeroAdvantages_Ugly(t *testing.T) {
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
}
