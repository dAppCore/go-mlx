// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

// ExampleRunKnowledgeDistillation shows the minimal runner: a teacher
// logit source, a student logit source, and a pre-tokenized batch built
// by BuildBatches. The student is trained toward the teacher's soft
// targets and the run reports how many gradient steps fired.
func ExampleRunKnowledgeDistillation() {
	ds := dataset.NewSliceDataset([]dataset.Sample{{Text: "hello"}})

	result, err := RunKnowledgeDistillation(context.Background(), DistillRunner{
		BuildBatches: func(context.Context, dataset.Dataset, dataset.BatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{
				Batch:   Batch{Tokens: [][]int{{1, 2}}, LossMask: [][]float32{{1, 1}}},
				Targets: [][]int{{2, 3}},
			}}, nil
		},
		TeacherLogits: func(context.Context, DistillBatch) (DistillLogits, error) {
			return DistillLogits{{{2, 0}, {0, 2}}}, nil
		},
		StudentLogits: func(context.Context, DistillBatch, DistillLogits) (DistillLogits, error) {
			return DistillLogits{{{0, 0}, {0, 0}}}, nil
		},
	}, ds, DistillConfig{Temperature: 2})
	if err != nil {
		core.Println(err.Error())
		return
	}

	core.Println(result.Metrics.Steps, result.Metrics.Tokens)
	// Output: 1 2
}

// ExampleMemoryDistillLogitCache shows the offline teacher-logit cache:
// store a batch's teacher logits under a key, then retrieve them on a
// later step to skip recomputing the teacher forward pass.
func ExampleMemoryDistillLogitCache() {
	cache := NewMemoryDistillLogitCache()
	ctx := context.Background()

	if err := cache.PutTeacherLogits(ctx, "batch-1", DistillLogits{{{1, 2, 3}}}); err != nil {
		core.Println(err.Error())
		return
	}
	logits, ok, err := cache.GetTeacherLogits(ctx, "batch-1")
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(ok, len(logits[0][0]))
	// Output: true 3
}
