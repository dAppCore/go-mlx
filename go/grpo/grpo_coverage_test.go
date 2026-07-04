// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"
	"math"
	"testing"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

// errDataset is a dataset.Dataset whose Next() yields a fixed number of
// rows then returns an injected error, letting tests drive the
// ds.Next() error branch in runGRPOEpoch. With reset != nil it also
// implements dataset.Resetter, returning the reset error to drive the
// epoch>1 Reset failure branch in RunGRPOReasoningTraining.
type errDataset struct {
	rows     []dataset.Sample
	pos      int
	nextErr  error // returned by Next() once rows are exhausted
	resetErr error // returned by Reset(); presence makes this a Resetter
	resets   int
}

func (d *errDataset) Next() (dataset.Sample, bool, error) {
	if d.pos < len(d.rows) {
		s := d.rows[d.pos]
		d.pos++
		return s, true, nil
	}
	if d.nextErr != nil {
		return dataset.Sample{}, false, d.nextErr
	}
	return dataset.Sample{}, false, nil
}

// errResetDataset embeds errDataset and always advertises Reset so the
// multi-epoch Reset path is exercised. Reset rewinds pos and returns the
// configured resetErr (nil = success).
type errResetDataset struct {
	errDataset
}

func (d *errResetDataset) Reset() error {
	d.resets++
	d.pos = 0
	return d.resetErr
}

// TestGrpo_SaveGRPOCheckpointMetadata_FSErrors covers the three
// filesystem failure branches of SaveGRPOCheckpointMetadata that the
// happy-path tests never reach: a non-finite scalar that fails JSON
// marshalling, a parent component that is a regular file (MkdirAll
// fails ENOTDIR), and a metadata target path that is already a directory
// (WriteFile fails EISDIR). Each must wrap the underlying failure with
// its own operation context rather than leaking a bare OS error.
func TestGrpo_SaveGRPOCheckpointMetadata_FSErrors(t *testing.T) {
	t.Run("marshal_error_on_non_finite", func(t *testing.T) {
		// +Inf in a float field is rejected by encoding/json, so the
		// marshal step fails before any write happens.
		dir := core.PathJoin(t.TempDir(), "ckpt")
		err := SaveGRPOCheckpointMetadata(dir, GRPOCheckpointMetadata{Loss: math.Inf(1)})
		if err == nil || !core.Contains(core.Lower(err.Error()), "marshal metadata") {
			t.Fatalf("error = %v, want marshal failure on +Inf loss", err)
		}
	})

	t.Run("mkdir_error_under_file", func(t *testing.T) {
		// A regular file used as a parent directory makes MkdirAll fail
		// with ENOTDIR, exercising the ensure-metadata-dir branch.
		base := t.TempDir()
		fileAsParent := core.PathJoin(base, "afile")
		if r := core.WriteFile(fileAsParent, []byte("x"), 0o644); !r.OK {
			t.Fatalf("setup write file: %v", r.Value)
		}
		err := SaveGRPOCheckpointMetadata(core.PathJoin(fileAsParent, "deeper"), GRPOCheckpointMetadata{Step: 1})
		if err == nil || !core.Contains(core.Lower(err.Error()), "metadata dir") {
			t.Fatalf("error = %v, want ensure-metadata-dir failure", err)
		}
	})

	t.Run("write_error_onto_directory", func(t *testing.T) {
		// Pre-create the metadata FILE path as a DIRECTORY so the final
		// WriteFile fails EISDIR — the dir-creation step succeeds, only
		// the write fails, isolating the write-metadata branch.
		target := core.PathJoin(t.TempDir(), "ckpt")
		if r := core.MkdirAll(grpoCheckpointMetadataPath(target), 0o755); !r.OK {
			t.Fatalf("setup mkdir metadata-as-dir: %v", r.Value)
		}
		err := SaveGRPOCheckpointMetadata(target, GRPOCheckpointMetadata{Step: 1})
		if err == nil || !core.Contains(core.Lower(err.Error()), "write metadata") {
			t.Fatalf("error = %v, want write-metadata failure", err)
		}
	})
}

// TestGrpo_maybeSaveGRPOCheckpoint_Good covers the runner-callback save
// path: with a non-nil SaveCheckpoint hook on the cadence boundary, the
// hook fires with the checkpoint context and the sidecar is written
// alongside, both recorded on the result.
func TestGrpo_maybeSaveGRPOCheckpoint_Good(t *testing.T) {
	saved := 0
	var gotPath string
	result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
		},
		SaveCheckpoint: func(_ context.Context, ckpt GRPOCheckpointContext) error {
			saved++
			gotPath = ckpt.Path
			if ckpt.Metadata.Step == 0 || ckpt.Metadata.Path == "" {
				t.Fatalf("checkpoint context = %+v, want populated metadata", ckpt)
			}
			return nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
		GroupSize:       1,
		CheckpointDir:   core.PathJoin(t.TempDir(), "ckpt"),
		CheckpointEvery: 1,
		RewardFuncs:     []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	if err != nil {
		t.Fatalf("RunGRPOReasoningTraining() error = %v", err)
	}
	if saved != 1 {
		t.Fatalf("SaveCheckpoint calls = %d, want 1", saved)
	}
	if len(result.Checkpoints) != 1 || result.Checkpoints[0] != gotPath {
		t.Fatalf("checkpoints = %v, want the saved path %q recorded", result.Checkpoints, gotPath)
	}
	// The sidecar Save also ran; the metadata is loadable.
	if _, err := LoadGRPOCheckpointMetadata(result.Checkpoints[0]); err != nil {
		t.Fatalf("LoadGRPOCheckpointMetadata() error = %v, want sidecar written", err)
	}
}

// TestGrpo_maybeSaveGRPOCheckpoint_Bad covers the two checkpoint failure
// branches inside the training loop: the runner's SaveCheckpoint hook
// returning an error, and the sidecar SaveGRPOCheckpointMetadata failing
// because the checkpoint directory cannot be created. Either aborts the
// run with the underlying failure surfaced.
func TestGrpo_maybeSaveGRPOCheckpoint_Bad(t *testing.T) {
	t.Run("save_checkpoint_hook_error", func(t *testing.T) {
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
			},
			SaveCheckpoint: func(context.Context, GRPOCheckpointContext) error {
				return core.NewError("checkpoint hook failed")
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
			GroupSize:       1,
			CheckpointDir:   core.PathJoin(t.TempDir(), "ckpt"),
			CheckpointEvery: 1,
			RewardFuncs:     []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "checkpoint hook failed") {
			t.Fatalf("error = %v, want SaveCheckpoint hook failure surfaced", err)
		}
	})

	t.Run("sidecar_save_error", func(t *testing.T) {
		// CheckpointDir is itself a regular file, so PathJoin(dir, step)
		// can't be created and the sidecar Save fails the run.
		base := t.TempDir()
		fileDir := core.PathJoin(base, "notadir")
		if r := core.WriteFile(fileDir, []byte("x"), 0o644); !r.OK {
			t.Fatalf("setup write file: %v", r.Value)
		}
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
			GroupSize:       1,
			CheckpointDir:   fileDir,
			CheckpointEvery: 1,
			RewardFuncs:     []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "metadata dir") {
			t.Fatalf("error = %v, want sidecar save failure on unwritable checkpoint dir", err)
		}
	})
}

// TestGrpo_runGRPOEpoch_Errors covers the error-propagation branches
// inside the per-epoch loop that the happy-path runs never trip: a
// dataset Next() error, a rollout error, a buildGRPOUpdate rejection (via
// a group-size mismatch), an ApplyUpdate error, a ReferenceLogProb error
// on the KL path, and a context cancelled mid-epoch. Each must abort the
// run and surface the originating failure verbatim.
func TestGrpo_runGRPOEpoch_Errors(t *testing.T) {
	okRollout := func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
		return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
	}
	rows := []dataset.Sample{{Prompt: "p", Response: "a"}}

	t.Run("dataset_next_error", func(t *testing.T) {
		// No rows, Next() returns an injected error immediately.
		ds := &errDataset{nextErr: core.NewError("next exploded")}
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{Rollout: okRollout}, ds, GRPOConfig{
			GroupSize:   1,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "next exploded") {
			t.Fatalf("error = %v, want dataset Next() error surfaced", err)
		}
	})

	t.Run("rollout_error", func(t *testing.T) {
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(context.Context, GRPORolloutRequest) ([]GRPORollout, error) {
				return nil, core.NewError("rollout exploded")
			},
		}, dataset.NewSliceDataset(rows), GRPOConfig{
			GroupSize:   1,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "rollout exploded") {
			t.Fatalf("error = %v, want rollout error surfaced", err)
		}
	})

	t.Run("build_update_error", func(t *testing.T) {
		// Rollout returns two completions for a group size of one, so
		// buildGRPOUpdate rejects the group-size mismatch mid-loop.
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer}, {Answer: req.Sample.ExpectedAnswer}}, nil
			},
		}, dataset.NewSliceDataset(rows), GRPOConfig{
			GroupSize:   1,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "group size") {
			t.Fatalf("error = %v, want buildGRPOUpdate group-size rejection", err)
		}
	})

	t.Run("apply_update_error", func(t *testing.T) {
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: okRollout,
			ApplyUpdate: func(context.Context, GRPOUpdate) error {
				return core.NewError("apply exploded")
			},
		}, dataset.NewSliceDataset(rows), GRPOConfig{
			GroupSize:   1,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "apply exploded") {
			t.Fatalf("error = %v, want ApplyUpdate error surfaced", err)
		}
	})

	t.Run("reference_logprob_error", func(t *testing.T) {
		// A non-zero KL coefficient plus a ReferenceLogProb hook drives the
		// KL branch in buildGRPOUpdate, where the hook's error aborts.
		_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
			Rollout: okRollout,
			ReferenceLogProb: func(context.Context, GRPORolloutRequest, GRPORollout) (float64, error) {
				return 0, core.NewError("reference exploded")
			},
		}, dataset.NewSliceDataset(rows), GRPOConfig{
			GroupSize:     1,
			KLCoefficient: 0.1,
			RewardFuncs:   []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil || !core.Contains(core.Lower(err.Error()), "reference exploded") {
			t.Fatalf("error = %v, want ReferenceLogProb error surfaced", err)
		}
	})

	t.Run("context_cancelled_mid_epoch", func(t *testing.T) {
		// A two-row dataset with a context cancelled inside the first
		// rollout: the second loop iteration observes ctx.Err() and aborts
		// before consuming the second row.
		ctx, cancel := context.WithCancel(context.Background())
		_, err := RunGRPOReasoningTraining(ctx, GRPORunner{
			Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
				cancel() // cancel after the first sample is in flight
				return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{
			{Prompt: "first", Response: "a"},
			{Prompt: "second", Response: "b"},
		}), GRPOConfig{
			GroupSize:   1,
			RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
		})
		if err == nil {
			t.Fatal("error = nil, want context-cancelled abort mid-epoch")
		}
	})
}

// TestGrpo_RunGRPOReasoningTraining_ResetError covers the epoch>1 Reset
// failure branch: a resettable dataset whose Reset returns an error must
// abort the run at the start of the second epoch rather than retrying or
// swallowing it.
func TestGrpo_RunGRPOReasoningTraining_ResetError(t *testing.T) {
	ds := &errResetDataset{
		errDataset: errDataset{rows: []dataset.Sample{{Prompt: "p", Response: "a"}}},
	}
	ds.resetErr = core.NewError("reset exploded")
	_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
		},
	}, ds, GRPOConfig{
		GroupSize:   1,
		Epochs:      2,
		RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	if err == nil || !core.Contains(core.Lower(err.Error()), "reset exploded") {
		t.Fatalf("error = %v, want Reset error surfaced at epoch 2", err)
	}
	if ds.resets != 1 {
		t.Fatalf("resets = %d, want exactly one Reset attempt before abort", ds.resets)
	}
}

// TestGrpo_maybeRunGRPOEval_Error covers the eval-hook error branch: an
// Evaluate hook that returns an error on the eval cadence boundary must
// abort the run with that error rather than recording a partial eval.
func TestGrpo_maybeRunGRPOEval_Error(t *testing.T) {
	_, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
		},
		Evaluate: func(context.Context, GRPOEvalContext) (GRPOEvalResult, error) {
			return GRPOEvalResult{}, core.NewError("eval exploded")
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
		GroupSize:   1,
		EvalEvery:   1,
		RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	if err == nil || !core.Contains(core.Lower(err.Error()), "eval exploded") {
		t.Fatalf("error = %v, want Evaluate hook error surfaced", err)
	}
}

// TestGrpo_loadGRPOResumeMetadata_NonNotExistError covers the resume
// loader's distinction between a missing sidecar (tolerated, nil/nil) and
// any other read failure (surfaced). A resume path whose sidecar location
// is a directory makes ReadFile fail EISDIR — not a not-exist — so the
// run must abort rather than treating it as a fresh start.
func TestGrpo_loadGRPOResumeMetadata_NonNotExistError(t *testing.T) {
	// Direct unit form: the sidecar path is a directory → EISDIR, not
	// IsNotExist, so the loader returns the error.
	dir := t.TempDir()
	if r := core.MkdirAll(grpoCheckpointMetadataPath(dir), 0o755); !r.OK {
		t.Fatalf("setup mkdir sidecar-as-dir: %v", r.Value)
	}
	meta, err := loadGRPOResumeMetadata(dir)
	if err == nil {
		t.Fatal("loadGRPOResumeMetadata(sidecar-is-dir) error = nil, want EISDIR surfaced")
	}
	if core.IsNotExist(err) {
		t.Fatalf("error = %v, must not be classified as not-exist", err)
	}
	if meta != nil {
		t.Fatalf("meta = %+v, want nil on read failure", meta)
	}

	// And through the full run: a resume path with an unreadable sidecar
	// aborts the run before any training happens.
	_, runErr := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{{Answer: req.Sample.ExpectedAnswer, LogProb: -0.2}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "a"}}), GRPOConfig{
		GroupSize:   1,
		ResumePath:  dir,
		RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	if runErr == nil {
		t.Fatal("RunGRPOReasoningTraining(unreadable resume) error = nil, want abort")
	}
}

// TestGrpo_ExtractGRPOExpectedAnswer_MultiLineWalk covers the multi-line
// backward-walk path of ExtractGRPOExpectedAnswer that the single-line
// fast path skips: a response whose trailing line is a bare answer prefix
// (cleaning to empty) forces the walk to step back to an earlier line,
// and the early `start < 0` exhaustion guard fires when every line cleans
// away. These exercise the loop body, the empty-line continue, and the
// exhaustion return.
func TestGrpo_ExtractGRPOExpectedAnswer_MultiLineWalk(t *testing.T) {
	t.Run("trailing_empty_line_steps_back", func(t *testing.T) {
		// Last line "answer:" cleans to "" so the walk steps back to the
		// first line "foo", which is returned as the answer.
		got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "foo\nanswer:"})
		if got != "foo" {
			t.Fatalf("ExtractGRPOExpectedAnswer = %q, want %q from the multi-line walk", got, "foo")
		}
	})

	t.Run("answer_prefix_on_last_line", func(t *testing.T) {
		// The last non-empty line carries a recognised prefix, so the
		// suffix after "final answer:" is returned.
		got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "reasoning here\nfinal answer: 42"})
		if got != "42" {
			t.Fatalf("ExtractGRPOExpectedAnswer = %q, want %q", got, "42")
		}
	})

	t.Run("all_lines_clean_to_empty_exhausts", func(t *testing.T) {
		// Every line is a bare prefix keyword cleaning to "", so the
		// backward walk runs out of lines and returns "" via the
		// start<0 exhaustion guard.
		got := ExtractGRPOExpectedAnswer(dataset.Sample{Response: "answer:\nsolution:"})
		if got != "" {
			t.Fatalf("ExtractGRPOExpectedAnswer = %q, want empty when every line cleans away", got)
		}
	})
}

// TestGrpo_GRPORewardContainsAnswer_FallbackPaths covers the non-ASCII
// and newline-bearing expected-answer fallback in GRPORewardContainsAnswer
// that the fast ASCII scan skips: a unicode expected answer and a
// newline-spanning expected answer both route through the join +
// case-fold path. The unicode case must still match case-insensitively,
// and the newline case must match across the fragment join.
func TestGrpo_GRPORewardContainsAnswer_FallbackPaths(t *testing.T) {
	reward := GRPORewardContainsAnswer(2)

	t.Run("unicode_expected_uses_unicode_fold", func(t *testing.T) {
		// "café" is non-ASCII, forcing the core.Lower fallback; the rollout
		// carries the same word with different case, so it matches.
		got, err := reward(GRPORewardContext{
			Sample:  GRPOSample{ExpectedAnswer: "Café"},
			Rollout: GRPORollout{Text: "the answer is a CAFÉ downtown"},
		})
		if err != nil {
			t.Fatalf("reward() error = %v", err)
		}
		if got.Score != 2 || got.Detail != "matched" {
			t.Fatalf("reward = %+v, want unicode-fold match scoring the weight", got)
		}
	})

	t.Run("unicode_expected_miss", func(t *testing.T) {
		// Same non-ASCII fallback path but the rollout does not contain it.
		got, err := reward(GRPORewardContext{
			Sample:  GRPOSample{ExpectedAnswer: "naïve"},
			Rollout: GRPORollout{Text: "nothing relevant here"},
		})
		if err != nil {
			t.Fatalf("reward() error = %v", err)
		}
		if got.Score != 0 || got.Detail != "missing" {
			t.Fatalf("reward = %+v, want unicode-fold miss scoring zero", got)
		}
	})

	t.Run("newline_expected_spans_fragments", func(t *testing.T) {
		// An expected answer containing "\n" can only be matched against
		// the joined Answer/Text/Reasoning, so it routes through the
		// fallback. Here it spans the Answer→Text join boundary.
		got, err := reward(GRPORewardContext{
			Sample:  GRPOSample{ExpectedAnswer: "first\nsecond"},
			Rollout: GRPORollout{Answer: "first", Text: "second"},
		})
		if err != nil {
			t.Fatalf("reward() error = %v", err)
		}
		if got.Score != 2 || got.Detail != "matched" {
			t.Fatalf("reward = %+v, want newline-spanning match via the join path", got)
		}
	})
}
