// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/dataset"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/probe"
)

func TestRunKnowledgeDistillation_OfflineTeacherCacheCheckpointEvalProbe_Good(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {1},
			"response": {2},
		},
		eos: 3,
	}}
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
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{encoded: map[string][]int32{"x": {1, 2}}, eos: 3}}

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
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{encoded: map[string][]int32{"x": {1, 2}}, eos: 3}}

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
