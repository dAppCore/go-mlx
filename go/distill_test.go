// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestRunKnowledgeDistillation_OfflineTeacherCacheCheckpointEvalProbe_Good(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {1},
			"response": {2},
		},
		eos: 3,
	}}
	dataset := NewSFTSliceDataset([]SFTSample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})
	recorder := NewProbeRecorder()
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
		Evaluate: func(_ context.Context, eval DistillEvalContext) (DistillEvalResult, error) {
			evalCalls++
			return DistillEvalResult{
				Step: eval.Step,
				Metrics: EvalMetrics{
					Samples: eval.Metrics.Samples,
					Tokens:  eval.Metrics.Tokens,
					Loss:    eval.Metrics.Loss,
				},
			}, nil
		},
	}, dataset, DistillConfig{
		Batch:           DatasetBatchConfig{BatchSize: 1},
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

func TestRunKnowledgeDistillation_RequiresTeacherLogits_Bad(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{encoded: map[string][]int32{"x": {1, 2}}, eos: 3}}

	_, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		Tokenizer: func(context.Context) *Tokenizer { return tokenizer },
		StudentLogits: func(_ context.Context, batch DistillBatch, _ DistillLogits) (DistillLogits, error) {
			return distillTestLogits(batch.SFT, 2, 0, 1), nil
		},
	}, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), DistillConfig{})
	if err == nil {
		t.Fatal("expected missing teacher logits error")
	}
	if !core.Contains(core.Lower(err.Error()), "teacher") {
		t.Fatalf("error = %v, want teacher context", err)
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
	}, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), DistillConfig{})
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
