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
			Evaluate: func(context.Context, DistillEvalContext) (DistillEvalResult, error) {
				return DistillEvalResult{}, core.NewError("eval boom")
			},
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

// failingDistillLogitCache implements DistillLogitCache but fails on the
// chosen operation, driving the cache Get/Put error-propagation arms of
// teacherLogitsForDistillBatch without a live teacher.
type failingDistillLogitCache struct {
	failGet bool
	failPut bool
}

func (c failingDistillLogitCache) GetTeacherLogits(context.Context, string) (DistillLogits, bool, error) {
	if c.failGet {
		return nil, false, core.NewError("cache get boom")
	}
	return nil, false, nil
}

func (c failingDistillLogitCache) PutTeacherLogits(context.Context, string, DistillLogits) error {
	if c.failPut {
		return core.NewError("cache put boom")
	}
	return nil
}

// nextErrorDataset returns an error from Next, driving the ds.Next error
// arm of distillCollectSamples (the MaxSamples collection path).
type nextErrorDataset struct{}

func (nextErrorDataset) Next() (dataset.Sample, bool, error) {
	return dataset.Sample{}, false, core.NewError("next boom")
}

func TestDistillMetricAccumulator_NilAndZeroTokenGuards_Good(t *testing.T) {
	// A nil accumulator and a zero-token loss are both no-ops; snapshot on
	// an empty accumulator returns the zero average shape.
	var nilAcc *distillMetricAccumulator
	nilAcc.add(&DistillLoss{Tokens: 1, Value: 1}) // nil receiver: no panic, no-op
	if snap := nilAcc.snapshot(); snap != (distillMetricsSnapshot{}) {
		t.Fatalf("nil snapshot = %+v, want zero snapshot", snap)
	}

	acc := &distillMetricAccumulator{}
	acc.add(&DistillLoss{Tokens: 0, Value: 99}) // zero tokens: skipped
	if acc.tokens != 0 {
		t.Fatalf("tokens = %d, want zero-token loss skipped", acc.tokens)
	}
	if snap := acc.snapshot(); snap != (distillMetricsSnapshot{}) {
		t.Fatalf("empty snapshot = %+v, want zero snapshot", snap)
	}
}

func TestDistillResultError_OKAndNonErrorValue_Good(t *testing.T) {
	// An OK result carries no error; a failed result whose Value is not an
	// error falls back to the package sentinel rather than panicking on
	// the type assertion.
	if err := distillResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("distillResultError(ok) = %v, want nil", err)
	}
	err := distillResultError(core.Result{OK: false, Value: "not-an-error"})
	if err == nil {
		t.Fatal("distillResultError(non-error value) = nil, want sentinel")
	}
}

func TestDistillCollectSamples_ContextAndNextErrors_Bad(t *testing.T) {
	// A pre-cancelled context aborts the run at the entry ctx guard
	// before any work; a dataset whose Next errors during MaxSamples
	// collection surfaces that error.
	okStudent := func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
		return DistillLogits{{{1, 0}}}, nil
	}
	okTeacher := func(context.Context, DistillBatch) (DistillLogits, error) {
		return DistillLogits{{{0, 1}}}, nil
	}

	t.Run("cancelled_context_during_collect", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := RunKnowledgeDistillation(ctx, DistillRunner{TeacherLogits: okTeacher, StudentLogits: okStudent},
			dataset.NewSliceDataset([]dataset.Sample{{Text: "a"}}), DistillConfig{MaxSamples: 1})
		if err == nil {
			t.Fatal("RunKnowledgeDistillation(cancelled, MaxSamples) error = nil")
		}
	})

	t.Run("dataset_next_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{TeacherLogits: okTeacher, StudentLogits: okStudent},
			nextErrorDataset{}, DistillConfig{MaxSamples: 2})
		if err == nil || !core.Contains(err.Error(), "next boom") {
			t.Fatalf("error = %v, want dataset Next error surfaced", err)
		}
	})
}

func TestSaveDistillCheckpointMetadata_WritePathIsDir_Bad(t *testing.T) {
	// The metadata dir is created fine, but the metadata FILE path is
	// itself an existing directory, so the WriteFile arm of Save errors
	// (distinct from the MkdirAll arm).
	dir := t.TempDir()
	if result := core.MkdirAll(distillCheckpointMetadataPath(dir), 0o755); !result.OK {
		t.Fatalf("setup mkdir: %v", result.Value)
	}
	if err := SaveDistillCheckpointMetadata(dir, DistillCheckpointMetadata{Step: 1}); err == nil {
		t.Fatal("SaveDistillCheckpointMetadata(file-path-is-dir) error = nil")
	}
}

func TestRunKnowledgeDistillation_ResumeReadAndParseErrors_Bad(t *testing.T) {
	okStudent := func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
		return DistillLogits{{{1, 0}}}, nil
	}

	t.Run("resume_parse_error", func(t *testing.T) {
		// A resume sidecar with invalid JSON surfaces a parse error from
		// loadDistillResumeMetadata.
		dir := t.TempDir()
		writeModelPackFile(t, distillCheckpointMetadataPath(dir), "{not json")
		if _, err := RunKnowledgeDistillation(context.Background(), DistillRunner{StudentLogits: okStudent},
			dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{ResumePath: dir}); err == nil {
			t.Fatal("RunKnowledgeDistillation(invalid resume JSON) error = nil")
		}
	})

	t.Run("resume_read_error_not_missing", func(t *testing.T) {
		// A resume path whose sidecar location is an unreadable directory
		// (not a clean missing-file) returns the read error rather than a
		// cold start.
		dir := t.TempDir()
		if result := core.MkdirAll(distillCheckpointMetadataPath(dir), 0o755); !result.OK {
			t.Fatalf("setup mkdir: %v", result.Value)
		}
		if _, err := RunKnowledgeDistillation(context.Background(), DistillRunner{StudentLogits: okStudent},
			dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{ResumePath: dir}); err == nil {
			t.Fatal("RunKnowledgeDistillation(resume read error) error = nil")
		}
	})
}

func TestRunKnowledgeDistillation_ResumeVersionDefaulted_Good(t *testing.T) {
	// A valid resume sidecar without a version loads cleanly and stamps
	// the current version (the zero-version default arm of resume load).
	dir := t.TempDir()
	writeModelPackFile(t, distillCheckpointMetadataPath(dir), `{"step":4,"loss":0.5}`)
	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{ResumePath: dir})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Step != 4 || result.ResumedFrom.Version != DistillCheckpointMetadataVersion {
		t.Fatalf("ResumedFrom = %+v, want step 4 with defaulted version", result.ResumedFrom)
	}
}

func TestRunKnowledgeDistillation_NilContextNormalised_Good(t *testing.T) {
	// A nil context is normalised to context.Background() rather than
	// panicking on the first ctx.Err() check.
	result, err := RunKnowledgeDistillation(nil, DistillRunner{ //nolint:staticcheck // exercising the nil-ctx normalise arm
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation(nil ctx) error = %v", err)
	}
	if result.Metrics.Steps != 1 {
		t.Fatalf("steps = %d, want one step under normalised context", result.Metrics.Steps)
	}
}

func TestRunKnowledgeDistillation_BatchAndStudentErrors_Bad(t *testing.T) {
	t.Run("build_batches_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
				return nil, core.NewError("build boom")
			},
			StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
				return DistillLogits{{{1, 0}}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
		if err == nil || !core.Contains(err.Error(), "build boom") {
			t.Fatalf("error = %v, want BuildBatches error surfaced", err)
		}
	})

	t.Run("empty_batches", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
				return nil, nil
			},
			StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
				return DistillLogits{{{1, 0}}}, nil
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
		if err == nil {
			t.Fatal("RunKnowledgeDistillation(empty batches) error = nil")
		}
	})

	t.Run("student_logits_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
				return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
			},
			TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
				return DistillLogits{{{0, 1}}}, nil
			},
			StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
				return nil, core.NewError("student boom")
			},
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
		if err == nil || !core.Contains(err.Error(), "student boom") {
			t.Fatalf("error = %v, want StudentLogits error surfaced", err)
		}
	})
}

func TestTeacherLogitsForDistillBatch_CacheGetAndPutErrors_Bad(t *testing.T) {
	okBatches := func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
		return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
	}
	okTeacher := func(context.Context, DistillBatch) (DistillLogits, error) {
		return DistillLogits{{{0, 1}}}, nil
	}
	okStudent := func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
		return DistillLogits{{{1, 0}}}, nil
	}

	t.Run("cache_get_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches: okBatches, TeacherLogits: okTeacher, StudentLogits: okStudent,
			TeacherCache: failingDistillLogitCache{failGet: true},
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
		if err == nil || !core.Contains(err.Error(), "cache get boom") {
			t.Fatalf("error = %v, want cache Get error surfaced", err)
		}
	})

	t.Run("cache_put_error", func(t *testing.T) {
		_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
			BuildBatches: okBatches, TeacherLogits: okTeacher, StudentLogits: okStudent,
			TeacherCache: failingDistillLogitCache{failPut: true},
		}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
		if err == nil || !core.Contains(err.Error(), "cache put boom") {
			t.Fatalf("error = %v, want cache Put error surfaced", err)
		}
	})
}

func TestDistillationBatchLoss_MaskPresentTeacherNonFinite_Ugly(t *testing.T) {
	// The masked inner loop validates the teacher side too: a non-finite
	// teacher logit under an active mask must be rejected by the teacher
	// log-softmax finite guard (the mask-present counterpart of the
	// student-side rejection).
	_, err := DistillationBatchLoss(
		DistillLogits{{{float32(math.Inf(1)), 0}}},
		DistillLogits{{{0, 0}}},
		[][]float32{{1}},
		DistillConfig{Loss: DistillLossKL, Temperature: 1},
	)
	if err == nil || !core.Contains(core.Lower(err.Error()), "finite") {
		t.Fatalf("error = %v, want non-finite teacher rejection on the masked path", err)
	}
}

func TestDistillBatches_NoBuilderNoTokenizer_Bad(t *testing.T) {
	// With neither a BuildBatches hook nor a Tokenizer, distillBatches has
	// no way to tokenize the stream and must return the need-tokenizer
	// guard rather than falling through to a nil dereference.
	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
	if err == nil {
		t.Fatal("RunKnowledgeDistillation(no builder, no tokenizer) error = nil")
	}
}

func TestTeacherLogitsForDistillBatch_TeacherReturnsError_Bad(t *testing.T) {
	// teacherLogitsForDistillBatch surfaces an error returned by the
	// TeacherLogits hook itself (distinct from the nil-hook guard).
	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return nil, core.NewError("teacher boom")
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{})
	if err == nil || !core.Contains(err.Error(), "teacher boom") {
		t.Fatalf("error = %v, want TeacherLogits hook error surfaced", err)
	}
}

func TestMaybeSaveDistillCheckpoint_MetadataSaveFails_Bad(t *testing.T) {
	// With no SaveCheckpoint hook, maybeSaveDistillCheckpoint still writes
	// the metadata sidecar. A CheckpointDir that is a regular file makes
	// the per-step directory uncreatable, so SaveDistillCheckpointMetadata
	// fails and the run surfaces it (the metadata-save arm, reached only
	// when the hook does not short-circuit first).
	fileAsCheckpointDir := core.PathJoin(t.TempDir(), "ckpt-file")
	writeModelPackFile(t, fileAsCheckpointDir, "x")
	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{
		CheckpointDir:   fileAsCheckpointDir,
		CheckpointEvery: 1,
	})
	if err == nil {
		t.Fatal("RunKnowledgeDistillation(checkpoint dir is a file) error = nil")
	}
}

func TestMaybeRunDistillEval_DefaultsStepFromMetrics_Good(t *testing.T) {
	// An Evaluate hook that returns a zero-Step result has its Step (and
	// Epoch) back-filled from the current run metrics so each recorded
	// evaluation is attributable to a training step.
	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
		Evaluate: func(context.Context, DistillEvalContext) (DistillEvalResult, error) {
			return DistillEvalResult{}, nil // zero Step -> defaulted from metrics
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), DistillConfig{EvalEvery: 1})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if len(result.Evaluations) != 1 || result.Evaluations[0].Step != 1 || result.Evaluations[0].Epoch != 1 {
		t.Fatalf("evaluations = %+v, want step/epoch back-filled to 1", result.Evaluations)
	}
}

func TestDistillCollectSamples_ExhaustsUnderMaxSamples_Good(t *testing.T) {
	// MaxSamples larger than the dataset drains the stream and stops on
	// exhaustion (the !ok break in distillCollectSamples), not on the
	// cap — the run still trains on the collected subset.
	seen := 0
	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(_ context.Context, ds dataset.Dataset, _ dataset.BatchConfig) ([]SFTBatch, error) {
			for {
				_, ok, e := ds.Next()
				if e != nil {
					return nil, e
				}
				if !ok {
					break
				}
				seen++
			}
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}, LossMask: [][]float32{{1}}}, Targets: [][]int{{1}}}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{0, 1}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{1, 0}}}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Text: "a"}, {Text: "b"}}), DistillConfig{MaxSamples: 5})
	if err != nil {
		t.Fatalf("RunKnowledgeDistillation() error = %v", err)
	}
	if seen != 2 {
		t.Fatalf("seen = %d, want both samples collected before exhaustion", seen)
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
