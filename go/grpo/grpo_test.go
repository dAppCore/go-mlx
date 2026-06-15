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

func TestExtractGRPOExpectedAnswer_MetaPrefixMultiLine_Good(t *testing.T) {
	// Explicit meta answer key wins over the response body.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "lots of reasoning here\nfinal answer: 99",
		Meta:     map[string]string{"solution": " 1729 "},
	}); got != "1729" {
		t.Fatalf("meta key answer = %q, want trimmed 1729", got)
	}
	// No meta — last non-empty line carries an "answer:" prefix that must
	// be stripped case-insensitively (drives cleanGRPOAnswerLine +
	// asciiHasPrefixFold against the mixed-case prefix).
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "Add the numbers.\nFinal Answer: 42",
	}); got != "42" {
		t.Fatalf("multi-line answer prefix = %q, want stripped 42", got)
	}
	// CRLF normalisation path + a blank trailing line: the backward walk
	// must skip the empty tail and recover the real answer line.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "step one\r\nstep two\r\nSOLUTION: seven\r\n",
	}); got != "seven" {
		t.Fatalf("CRLF + trailing-blank answer = %q, want seven", got)
	}
	// Falls back to Text when Response is empty; single-line fast path.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Text: "answer: Paris"}); got != "Paris" {
		t.Fatalf("text fallback single line = %q, want Paris", got)
	}
}

func TestExtractGRPOExpectedAnswer_EmptyAndUnprefixed_Bad(t *testing.T) {
	// Nothing to extract from — empty meta, empty response, empty text.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{}); got != "" {
		t.Fatalf("empty sample answer = %q, want empty", got)
	}
	// A single-char line "a" is shorter than every prefix, so
	// asciiHasPrefixFold returns false on the len(s)<len(prefix) gate and
	// the raw line is returned verbatim — not mistaken for "answer:".
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "a"}); got != "a" {
		t.Fatalf("one-char trigger line = %q, want verbatim a", got)
	}
	// A multi-line body whose only non-blank line has no prefix returns
	// that line unchanged via the backward walk.
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "\n\njust a sentence"}); got != "just a sentence" {
		t.Fatalf("unprefixed multi-line = %q, want the sentence", got)
	}
}

func TestExtractGRPOExpectedAnswer_AllBlankLines_Ugly(t *testing.T) {
	// Every line is whitespace-only — the backward walk must exhaust all
	// boundaries and return empty rather than emitting a blank "answer".
	if got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "   \n\t\n  "}); got != "" {
		t.Fatalf("all-blank multi-line answer = %q, want empty", got)
	}
}

func TestGRPOSampleFromSFT_ReasoningMetaAndSuffixStrip_Good(t *testing.T) {
	// meta["reasoning"] is taken verbatim and short-circuits the
	// suffix-strip path in extractGRPOReasoningWithAnswer.
	got := GRPOSampleFromSFT(dataset.Sample{
		Prompt:   "  Solve 1+1  ",
		Response: "the answer is 2",
		Meta:     map[string]string{"reasoning": "carry the one", "answer": "2"},
	})
	if got.Prompt != "Solve 1+1" {
		t.Fatalf("prompt = %q, want trimmed", got.Prompt)
	}
	if got.ExpectedAnswer != "2" || got.Reasoning != "carry the one" {
		t.Fatalf("sample = %+v, want meta answer + meta reasoning", got)
	}
	if got.Meta["answer"] != "2" {
		t.Fatalf("meta = %+v, want cloned meta carried through", got.Meta)
	}
}

func TestGRPOSampleFromSFT_ThinkingKeyAndComputedSuffix_Good(t *testing.T) {
	// No reasoning key but a thinking key — the second meta branch.
	thinking := GRPOSampleFromSFT(dataset.Sample{
		Prompt: "p", Response: "trace then 7", Meta: map[string]string{"thinking": "ponder"},
	})
	if thinking.Reasoning != "ponder" {
		t.Fatalf("reasoning = %q, want thinking-key value", thinking.Reasoning)
	}
	// No meta at all: the answer is the last line ("42"), and reasoning is
	// the full response with that answer suffix trimmed off (drives
	// extractGRPOReasoningWithAnswer's TrimSuffix branch).
	computed := GRPOSampleFromSFT(dataset.Sample{Prompt: "p", Response: "reasoning trace\n42"})
	if computed.ExpectedAnswer != "42" {
		t.Fatalf("expected answer = %q, want 42", computed.ExpectedAnswer)
	}
	if computed.Reasoning != "reasoning trace" {
		t.Fatalf("reasoning = %q, want answer suffix trimmed", computed.Reasoning)
	}
}

func TestGRPOSampleFromSFT_PromptlessAndAnswerlessReasoning_Bad(t *testing.T) {
	// Prompt falls back to Text when Prompt is empty.
	fromText := GRPOSampleFromSFT(dataset.Sample{Text: "  prompt in text  "})
	if fromText.Prompt != "prompt in text" {
		t.Fatalf("prompt = %q, want text fallback", fromText.Prompt)
	}
	// Empty response → no answer → empty reasoning (the answer=="" early
	// return inside extractGRPOReasoningWithAnswer).
	noAnswer := GRPOSampleFromSFT(dataset.Sample{Prompt: "p", Response: ""})
	if noAnswer.ExpectedAnswer != "" || noAnswer.Reasoning != "" {
		t.Fatalf("sample = %+v, want empty answer and reasoning", noAnswer)
	}
}

func TestGRPORewardContainsAnswer_EmptyAndUnicodeFallback_Bad(t *testing.T) {
	fn := GRPORewardContainsAnswer(1)
	// Empty expected answer → neutral reward with the explanatory detail.
	empty, err := fn(GRPORewardContext{Sample: GRPOSample{ExpectedAnswer: "  "}})
	if err != nil {
		t.Fatalf("empty expected error = %v", err)
	}
	if empty.Score != 0 || empty.Detail != "no expected answer" {
		t.Fatalf("reward = %+v, want neutral no-answer reward", empty)
	}
	// Non-ASCII expected answer forces the unicode core.Join+Lower
	// fallback; a match still scores the weight.
	hit, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "café"},
		Rollout: GRPORollout{Text: "Le CAFÉ est ouvert"},
	})
	if err != nil {
		t.Fatalf("unicode reward error = %v", err)
	}
	if hit.Score != 1 || hit.Detail != "matched" {
		t.Fatalf("reward = %+v, want unicode-fallback match", hit)
	}
	// Non-ASCII expected absent from all fragments → miss via the same
	// fallback path.
	miss, err := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "café"},
		Rollout: GRPORollout{Text: "no match here"},
	})
	if err != nil {
		t.Fatalf("unicode miss error = %v", err)
	}
	if miss.Score != 0 || miss.Detail != "missing" {
		t.Fatalf("reward = %+v, want unicode-fallback miss", miss)
	}
}

func TestRunGRPOReasoningTraining_MultiEpochReplaysDataset_Good(t *testing.T) {
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
}

func TestRunGRPOReasoningTraining_NilContextAndMissingResume_Ugly(t *testing.T) {
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
}

func TestRunGRPOReasoningTraining_MultiEpochRequiresResetter_Bad(t *testing.T) {
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
}

func TestRunGRPOReasoningTraining_CancelledContextAndNoSamples_Bad(t *testing.T) {
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
}

func TestSaveLoadGRPOCheckpointMetadata_RoundTrip_Good(t *testing.T) {
	dir := core.PathJoin(t.TempDir(), "ckpt")
	meta := GRPOCheckpointMetadata{
		Step: 7, Epoch: 2, GroupSize: 4, RewardMean: 0.5, Loss: 1.25,
		Policy: ModelInfo{Architecture: "qwen3", VocabSize: 32},
	}
	if err := SaveGRPOCheckpointMetadata(dir, meta); err != nil {
		t.Fatalf("SaveGRPOCheckpointMetadata() error = %v", err)
	}
	loaded, err := LoadGRPOCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata() error = %v", err)
	}
	// Save backfills Version + Experimental even when the caller left them
	// zero, and the round-trip preserves the substantive fields.
	if loaded.Version != GRPOCheckpointMetadataVersion || !loaded.Experimental {
		t.Fatalf("loaded = %+v, want version + experimental backfilled", loaded)
	}
	if loaded.Step != 7 || loaded.GroupSize != 4 || loaded.Policy.Architecture != "qwen3" {
		t.Fatalf("loaded = %+v, want round-tripped fields", loaded)
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
