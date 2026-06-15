// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"
	"math"
	"testing"

	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/probe"
)

func TestRunKnowledgeDistillation_OfflineTeacherCacheCheckpointEvalProbe_Good(t *testing.T) {
	tokenizer := mlx.NewTokenizer(fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {1},
			"response": {2},
		},
		eos: 3,
	})
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})
	recorder := probe.NewRecorder()
	cache := NewMemoryDistillLogitCache()
	checkpointDir := core.PathJoin(t.TempDir(), "checkpoints")
	teacherCalls := 0
	studentCalls := 0
	evalCalls := 0

	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		TeacherInfo: func(context.Context) ModelInfo {
			return ModelInfo{Architecture: "qwen3", VocabSize: 2}
		},
		StudentInfo: func(context.Context) ModelInfo {
			return ModelInfo{Architecture: "qwen3", VocabSize: 2}
		},
		Tokenizer: func(context.Context) *Tokenizer {
			return tokenizer
		},
		TeacherCache: cache,
		TeacherLogits: func(_ context.Context, batch DistillBatch) (DistillLogits, error) {
			teacherCalls++
			return distillTestLogits(batch.SFT, 2, 1, 4), nil
		},
		StudentLogits: func(_ context.Context, batch DistillBatch, teacher DistillLogits) (DistillLogits, error) {
			studentCalls++
			if len(teacher) == 0 {
				return nil, core.NewError("teacher logits missing")
			}
			return distillTestLogits(batch.SFT, 2, 0, 2), nil
		},
		Evaluate: func(_ context.Context, ev DistillEvalContext) (DistillEvalResult, error) {
			evalCalls++
			return DistillEvalResult{
				Step: ev.Step,
				Metrics: eval.Metrics{
					Samples: ev.Metrics.Samples,
					Tokens:  ev.Metrics.Tokens,
					Loss:    ev.Metrics.Loss,
				},
			}, nil
		},
	}, ds, DistillConfig{
		Batch:           dataset.BatchConfig{BatchSize: 1},
		Temperature:     2,
		CheckpointDir:   checkpointDir,
		CheckpointEvery: 1,
		EvalEvery:       1,
		ProbeSink:       recorder,
	})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if result.Metrics.Steps != 2 || result.Metrics.Samples != 2 || result.Metrics.Tokens != 4 {
		t.Fatalf("metrics = %+v, want two repeated batches and four masked tokens", result.Metrics)
	}
	if teacherCalls != 1 || result.Metrics.TeacherCacheHits != 1 || result.Metrics.TeacherCacheMisses != 1 {
		t.Fatalf("teacher cache calls=%d metrics=%+v, want one hit and one miss", teacherCalls, result.Metrics)
	}
	if studentCalls != 2 || evalCalls != 2 {
		t.Fatalf("studentCalls=%d evalCalls=%d, want 2/2", studentCalls, evalCalls)
	}
	if len(result.Checkpoints) != 2 || len(result.CheckpointMetadata) != 2 {
		t.Fatalf("checkpoints = %+v metadata=%+v, want per-step checkpoint metadata", result.Checkpoints, result.CheckpointMetadata)
	}
	meta, err := LoadDistillCheckpointMetadata(result.Checkpoints[0])
	if err != nil {
		t.Fatalf("LoadDistillCheckpointMetadata() error = %v", err)
	}
	if meta.Step != 1 || meta.Temperature != 2 || meta.Teacher.Architecture != "qwen3" || meta.Student.Architecture != "qwen3" {
		t.Fatalf("checkpoint metadata = %+v, want reproducible distillation identity", meta)
	}
	if len(result.Evaluations) != 2 {
		t.Fatalf("evaluations = %+v, want per-step eval results", result.Evaluations)
	}
	events := recorder.Events()
	if len(events) != 2 || events[0].Training == nil || events[0].Training.Loss <= 0 {
		t.Fatalf("probe events = %+v, want training loss probes", events)
	}
	if events[0].Meta["teacher_cache"] != "miss" || events[1].Meta["teacher_cache"] != "hit" {
		t.Fatalf("probe cache metadata = %+v / %+v", events[0].Meta, events[1].Meta)
	}
}

func TestDistillationBatchLoss_SoftCrossEntropyUsesMask_Good(t *testing.T) {
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}, {0, 0}}},
		DistillLogits{{{0, 0}, {10, -10}}},
		[][]float32{{1, 0}},
		DistillConfig{Loss: DistillLossSoftCrossEntropy, Temperature: 1},
	)
	if err != nil {
		t.Fatalf("DistillationBatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("tokens = %d, want mask to include one token", loss.Tokens)
	}
	if math.Abs(loss.SoftCrossEntropy-math.Log(2)) > 1e-6 {
		t.Fatalf("soft CE = %.9f, want ln(2)", loss.SoftCrossEntropy)
	}
	if math.Abs(loss.Value-loss.SoftCrossEntropy) > 1e-9 {
		t.Fatalf("loss value = %.9f, want soft CE %.9f", loss.Value, loss.SoftCrossEntropy)
	}
}

func TestRunDistillation_ResumeMaxSamplesBuildBatches_Good(t *testing.T) {
	resume := core.PathJoin(t.TempDir(), "resume")
	if err := SaveDistillCheckpointMetadata(resume, DistillCheckpointMetadata{Step: 7, Loss: 0.25}); err != nil {
		t.Fatalf("SaveDistillCheckpointMetadata() error = %v", err)
	}

	seenSamples := 0
	result, err := RunDistillation(context.Background(), DistillRunner{
		BuildBatches: func(_ context.Context, ds dataset.Dataset, _ dataset.BatchConfig) ([]SFTBatch, error) {
			for {
				_, ok, err := ds.Next()
				if err != nil {
					return nil, err
				}
				if !ok {
					break
				}
				seenSamples++
			}
			return []SFTBatch{{
				Batch:   Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}},
				Targets: [][]int{{1}},
			}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "a"}, {Text: "b"}}), DistillConfig{
		MaxSamples: 1,
		ResumePath: resume,
	})
	if err != nil {
		t.Fatalf("RunDistillation() error = %v", err)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Step != 7 || seenSamples != 1 {
		t.Fatalf("resume=%+v seenSamples=%d, want resume step 7 and one bounded sample", result.ResumedFrom, seenSamples)
	}
	if result.Metrics.Steps != 1 || result.Metrics.Tokens != 1 {
		t.Fatalf("metrics = %+v, want one distilled token", result.Metrics)
	}
}

func TestRunKnowledgeDistillation_RequiresTeacherLogits_Bad(t *testing.T) {
	tokenizer := mlx.NewTokenizer(fakeSFTTokenizer{encoded: map[string][]int32{"x": {1, 2}}, eos: 3})

	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		Tokenizer: func(context.Context) *Tokenizer { return tokenizer },
		StudentLogits: func(_ context.Context, batch DistillBatch, _ DistillLogits) (DistillLogits, error) {
			return distillTestLogits(batch.SFT, 2, 0, 1), nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
	if err == nil {
		t.Fatal("expected missing teacher logits error")
	}
	if !core.Contains(core.Lower(err.Error()), "teacher") {
		t.Fatalf("error = %v, want teacher context", err)
	}
}

func TestDistillationBatchLoss_ValidationErrors_Bad(t *testing.T) {
	cases := []struct {
		name    string
		teacher DistillLogits
		student DistillLogits
		mask    [][]float32
		cfg     DistillConfig
		want    string
	}{
		{
			name:    "unsupported_loss",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{Loss: DistillLossKind("bad")},
			want:    "unsupported",
		},
		{
			name:    "empty_teacher",
			teacher: DistillLogits{},
			student: DistillLogits{},
			cfg:     DistillConfig{},
			want:    "empty",
		},
		{
			name:    "no_masked_tokens",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}},
			mask:    [][]float32{{0}},
			cfg:     DistillConfig{},
			want:    "no masked",
		},
		{
			name:    "bad_temperature",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{Temperature: -1},
			want:    "temperature",
		},
		{
			name:    "nonfinite_logit",
			teacher: DistillLogits{{{float32(math.Inf(1))}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{},
			want:    "finite",
		},
		{
			name:    "batch_count_mismatch",
			teacher: DistillLogits{{{0}}},
			student: DistillLogits{{{0}}, {{0}}},
			cfg:     DistillConfig{},
			want:    "batch",
		},
		{
			name:    "sequence_length_mismatch",
			teacher: DistillLogits{{{0}, {0}}},
			student: DistillLogits{{{0}}},
			cfg:     DistillConfig{},
			want:    "sequence",
		},
		{
			name:    "empty_vocabulary",
			teacher: DistillLogits{{{}}},
			student: DistillLogits{{{}}},
			cfg:     DistillConfig{},
			want:    "vocabulary",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DistillationBatchLoss(tc.teacher, tc.student, tc.mask, tc.cfg)
			if err == nil || !core.Contains(core.Lower(err.Error()), tc.want) {
				t.Fatalf("DistillationBatchLoss() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestDistillCheckpointMetadataErrors_Bad(t *testing.T) {
	if err := SaveDistillCheckpointMetadata("", DistillCheckpointMetadata{}); err == nil {
		t.Fatal("SaveDistillCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadDistillCheckpointMetadata(""); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(empty) error = nil")
	}
	if _, err := LoadDistillCheckpointMetadata(core.PathJoin(t.TempDir(), "absent")); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(missing file) error = nil")
	}
	dir := t.TempDir()
	writeModelPackFile(t, distillCheckpointMetadataPath(dir), "{")
	if _, err := LoadDistillCheckpointMetadata(dir); err == nil {
		t.Fatal("LoadDistillCheckpointMetadata(invalid JSON) error = nil")
	}
	if _, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return nil, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return nil, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{ResumePath: dir}); err == nil {
		t.Fatal("RunKnowledgeDistillation(invalid resume metadata) error = nil")
	}
}

func TestRunKnowledgeDistillation_RejectsLogitShapeMismatch_Ugly(t *testing.T) {
	tokenizer := mlx.NewTokenizer(fakeSFTTokenizer{encoded: map[string][]int32{"x": {1, 2}}, eos: 3})

	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		Tokenizer: func(context.Context) *Tokenizer { return tokenizer },
		TeacherLogits: func(_ context.Context, batch DistillBatch) (DistillLogits, error) {
			return distillTestLogits(batch.SFT, 2, 0, 1), nil
		},
		StudentLogits: func(_ context.Context, batch DistillBatch, _ DistillLogits) (DistillLogits, error) {
			return distillTestLogits(batch.SFT, 3, 0, 1), nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
	if err == nil {
		t.Fatal("expected logit shape mismatch error")
	}
	if !core.Contains(core.Lower(err.Error()), "shape") {
		t.Fatalf("error = %v, want shape context", err)
	}
}

// nonResetDataset is a dataset that never implements dataset.Resetter,
// used to drive the multi-epoch "needs Reset" error path.
type nonResetDataset struct {
	samples []dataset.Sample
	pos     int
}

func (d *nonResetDataset) Next() (dataset.Sample, bool, error) {
	if d.pos >= len(d.samples) {
		return dataset.Sample{}, false, nil
	}
	s := d.samples[d.pos]
	d.pos++
	return s, true, nil
}

// failingResetDataset implements dataset.Resetter but fails its Reset,
// used to drive the epoch>1 Reset-error propagation path.
type failingResetDataset struct {
	samples []dataset.Sample
	pos     int
}

func (d *failingResetDataset) Next() (dataset.Sample, bool, error) {
	if d.pos >= len(d.samples) {
		return dataset.Sample{}, false, nil
	}
	s := d.samples[d.pos]
	d.pos++
	return s, true, nil
}

func (d *failingResetDataset) Reset() error { return core.NewError("reset boom") }

func TestRunKnowledgeDistillation_ApplyLossAndSaveCheckpointHooks_Good(t *testing.T) {
	checkpointDir := core.PathJoin(t.TempDir(), "ckpt")
	applied := 0
	saved := 0

	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{
				Batch:   Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}},
				Targets: [][]int{{1}},
			}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 2}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{2, 0}}}, nil
		},
		ApplyLoss: func(_ context.Context, batch DistillBatch, loss DistillLoss) error {
			applied++
			if loss.Tokens != 1 || batch.Step != 1 {
				return core.NewError("unexpected apply-loss context")
			}
			return nil
		},
		SaveCheckpoint: func(_ context.Context, cc DistillCheckpointContext) error {
			saved++
			if cc.Path == "" || cc.Metadata.Step != 1 {
				return core.NewError("unexpected checkpoint context")
			}
			return nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{
		CheckpointDir:   checkpointDir,
		CheckpointEvery: 1,
	})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if applied != 1 || saved != 1 {
		t.Fatalf("applied=%d saved=%d, want both hooks fired once", applied, saved)
	}
	if len(result.Checkpoints) != 1 || result.Metrics.CheckpointCount != 1 {
		t.Fatalf("checkpoints=%+v count=%d, want one saved checkpoint", result.Checkpoints, result.Metrics.CheckpointCount)
	}
}

func TestRunKnowledgeDistillation_MultiEpochResetsDataset_Good(t *testing.T) {
	epochsSeen := map[int]int{}

	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(_ context.Context, ds dataset.Dataset, _ dataset.BatchConfig) ([]SFTBatch, error) {
			count := 0
			for {
				_, ok, err := ds.Next()
				if err != nil {
					return nil, err
				}
				if !ok {
					break
				}
				count++
			}
			if count != 1 {
				return nil, core.NewError("dataset was not reset between epochs")
			}
			return []SFTBatch{{
				Batch:   Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}},
				Targets: [][]int{{1}},
			}}, nil
		},
		TeacherLogits: func(_ context.Context, batch DistillBatch) (DistillLogits, error) {
			epochsSeen[batch.Epoch]++
			return DistillLogits{{{0, 2}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{2, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{Epochs: 2})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if result.Metrics.Epochs != 2 || result.Metrics.Steps != 2 {
		t.Fatalf("metrics = %+v, want two epochs / two steps", result.Metrics)
	}
	if epochsSeen[1] != 1 || epochsSeen[2] != 1 {
		t.Fatalf("epochsSeen = %+v, want one batch per epoch", epochsSeen)
	}
}

func TestRunKnowledgeDistillation_ResumeMissingMetadataIsClean_Good(t *testing.T) {
	// A ResumePath pointing at a directory with no checkpoint sidecar is
	// a clean cold start (loadDistillResumeMetadata's IsNotExist arm) —
	// not an error, and ResumedFrom stays nil.
	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{
				Batch:   Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}},
				Targets: [][]int{{1}},
			}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{
		ResumePath: core.PathJoin(t.TempDir(), "absent"),
	})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if result.ResumedFrom != nil {
		t.Fatalf("ResumedFrom = %+v, want nil for a missing-metadata cold start", result.ResumedFrom)
	}
	if result.ResumePath == "" || result.Metrics.Steps != 1 {
		t.Fatalf("resume=%q steps=%d, want recorded resume path and one step", result.ResumePath, result.Metrics.Steps)
	}
}

func TestRunKnowledgeDistillation_HookAndContextFailures_Bad(t *testing.T) {
	okBatches := func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
		return []SFTBatch{{
			Batch:   Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}},
			Targets: [][]int{{1}},
		}}, nil
	}
	okTeacher := func(context.Context, DistillBatch) (DistillLogits, error) {
		return DistillLogits{{{0, 1}}}, nil
	}
	okStudent := func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
		return DistillLogits{{{1, 0}}}, nil
	}

	t.Run("nil_dataset", func(t *testing.T) {
		if _, err := RunKnowledgeDistillation(context.Background(), DistillRunner{StudentLogits: okStudent}, nil, DistillConfig{}); err == nil {
			t.Fatal("RunKnowledgeDistillation(nil dataset) error = nil")
		}
	})

	t.Run("missing_student_logits", func(t *testing.T) {
		if _, err := RunKnowledgeDistillation(context.Background(), DistillRunner{}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{}); err == nil {
			t.Fatal("RunKnowledgeDistillation(no StudentLogits) error = nil")
		}
	})

	t.Run("cancelled_context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := RunKnowledgeDistillation(ctx, DistillRunner{
			BuildBatches: okBatches, TeacherLogits: okTeacher, StudentLogits: okStudent,
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{}); err == nil {
			t.Fatal("RunKnowledgeDistillation(cancelled ctx) error = nil")
		}
	})

	t.Run("apply_loss_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches:  okBatches,
			TeacherLogits: okTeacher,
			StudentLogits: okStudent,
			ApplyLoss:     func(context.Context, DistillBatch, DistillLoss) error { return core.NewError("apply boom") },
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
		if err == nil || !core.Contains(err.Error(), "apply boom") {
			t.Fatalf("ApplyLoss error not surfaced: %v", err)
		}
	})

	t.Run("save_checkpoint_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches:   okBatches,
			TeacherLogits:  okTeacher,
			StudentLogits:  okStudent,
			SaveCheckpoint: func(context.Context, DistillCheckpointContext) error { return core.NewError("save boom") },
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{
			CheckpointDir:   core.PathJoin(t.TempDir(), "ckpt"),
			CheckpointEvery: 1,
		})
		if err == nil || !core.Contains(err.Error(), "save boom") {
			t.Fatalf("SaveCheckpoint error not surfaced: %v", err)
		}
	})

	t.Run("evaluate_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches:  okBatches,
			TeacherLogits: okTeacher,
			StudentLogits: okStudent,
			Evaluate:      func(context.Context, DistillEvalContext) (DistillEvalResult, error) { return DistillEvalResult{}, core.NewError("eval boom") },
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{EvalEvery: 1})
		if err == nil || !core.Contains(err.Error(), "eval boom") {
			t.Fatalf("Evaluate error not surfaced: %v", err)
		}
	})

	t.Run("reset_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches:  okBatches,
			TeacherLogits: okTeacher,
			StudentLogits: okStudent,
		}, &failingResetDataset{samples: []dataset.Sample{{Text: "x"}}}, DistillConfig{Epochs: 2})
		if err == nil || !core.Contains(err.Error(), "reset boom") {
			t.Fatalf("Reset error not surfaced: %v", err)
		}
	})
}

func TestRunKnowledgeDistillation_MultiEpochNonResetDataset_Ugly(t *testing.T) {
	// Epochs > 1 against a dataset that cannot Reset must fail loudly
	// rather than silently re-running an exhausted stream.
	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(_ context.Context, ds dataset.Dataset, _ dataset.BatchConfig) ([]SFTBatch, error) {
			for {
				if _, ok, e := ds.Next(); e != nil || !ok {
					break
				}
			}
			return []SFTBatch{{
				Batch:   Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}},
				Targets: [][]int{{1}},
			}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, &nonResetDataset{samples: []dataset.Sample{{Text: "x"}}}, DistillConfig{Epochs: 2})
	if err == nil || !core.Contains(core.Lower(err.Error()), "reset") {
		t.Fatalf("error = %v, want Reset requirement on multi-epoch non-resetter", err)
	}
}

func TestDistillationBatchLoss_SoftCrossEntropyMaskShapes_Good(t *testing.T) {
	// SoftCrossEntropy loss-kind selects loss.Value = softCE (the
	// non-KL lossValue arm), and a mask shorter than the sequence
	// bounds the inner loop to the masked prefix.
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}, {0, 0}}},
		DistillLogits{{{1, -1}, {5, -5}}},
		[][]float32{{1}}, // mask covers only the first of two positions
		DistillConfig{Loss: DistillLossSoftCrossEntropy, Temperature: 1},
	)
	if err != nil {
		t.Fatalf("DistillationBatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("tokens = %d, want short mask to bound to one token", loss.Tokens)
	}
	if loss.Kind != DistillLossSoftCrossEntropy || loss.Value != loss.SoftCrossEntropy {
		t.Fatalf("loss = %+v, want value == softCE for the soft-CE kind", loss)
	}
}

func TestDistillationBatchLoss_NilAndShortMaskRowsSkipped_Good(t *testing.T) {
	// A mask with fewer rows than the batch (second row absent) and a
	// nil mask row exercise the i>=len(maskRows) continue and the
	// maskRow==nil continue without producing a no-masked-tokens error.
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}, {{0, 0}}, {{0, 0}}},
		DistillLogits{{{2, -2}}, {{2, -2}}, {{2, -2}}},
		[][]float32{{1}, nil}, // row 0 masked, row 1 nil, row 2 absent
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err != nil {
		t.Fatalf("DistillationBatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("tokens = %d, want only the first masked row counted", loss.Tokens)
	}
}

func TestDistillationBatchLoss_StudentNonFinite_Ugly(t *testing.T) {
	// The teacher side is finite; the student side carries an Inf,
	// which must be rejected by the student log-softmax finite guard.
	_, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}},
		DistillLogits{{{float32(math.Inf(1)), 0}}},
		[][]float32{{1}},
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
		t.Fatalf("error = %v, want non-finite student logit rejection", err)
	}
}

func TestMemoryDistillLogitCache_NilReceiverGuards_Good(t *testing.T) {
	var cache *MemoryDistillLogitCache // nil receiver

	logits, ok, err := cache.GetTeacherLogits(context.Background(), "k")
	if err != nil || ok || logits != nil {
		t.Fatalf("nil-cache Get = (%v,%v,%v), want (nil,false,nil)", logits, ok, err)
	}
	if err := cache.PutTeacherLogits(context.Background(), "k", DistillLogits{{{1}}}); err != nil {
		t.Fatalf("nil-cache Put error = %v, want nil no-op", err)
	}

	// A zero-value cache (logits map nil) must lazily initialise on Put
	// and then round-trip the stored logits on Get.
	zero := &MemoryDistillLogitCache{}
	if err := zero.PutTeacherLogits(context.Background(), "k", DistillLogits{{{7, 8}}}); err != nil {
		t.Fatalf("zero-cache Put error = %v", err)
	}
	got, ok, err := zero.GetTeacherLogits(context.Background(), "k")
	if err != nil || !ok || len(got) != 1 || got[0][0][1] != 8 {
		t.Fatalf("zero-cache round-trip = (%v,%v,%v), want stored logits", got, ok, err)
	}
}

func TestSaveDistillCheckpointMetadata_UnwritablePath_Bad(t *testing.T) {
	// A metadata dir whose parent is a regular file cannot be created,
	// so the MkdirAll arm of SaveDistillCheckpointMetadata must error.
	fileAsParent := core.PathJoin(t.TempDir(), "not-a-dir")
	writeModelPackFile(t, fileAsParent, "x")
	target := core.PathJoin(fileAsParent, "child")
	if err := SaveDistillCheckpointMetadata(target, DistillCheckpointMetadata{Step: 1}); err == nil {
		t.Fatal("SaveDistillCheckpointMetadata(file-as-parent) error = nil")
	}
}

func distillTestLogits(batch SFTBatch, vocab int, preferred int, scale float32) DistillLogits {
	out := make(DistillLogits, len(batch.Batch.Tokens))
	for i, row := range batch.Batch.Tokens {
		out[i] = make([][]float32, len(row))
		for j := range row {
			out[i][j] = make([]float32, vocab)
			for k := range out[i][j] {
				out[i][j][k] = -scale
			}
			if preferred >= 0 && preferred < vocab {
				out[i][j][preferred] = scale
			}
		}
	}
	return out
}

// writeModelPackFile is a small test helper that writes a file under
// the test's temp dir. Lives here (rather than in a separate
// `*_test_helpers_test.go`) per the test-file-per-source convention —
// distill_test.go and grpo_test.go both call it from the same package.
func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}
