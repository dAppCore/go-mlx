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

// ExampleRunDistillation shows RunDistillation is the drop-in alias of
// RunKnowledgeDistillation: the same runner shape trains the student and
// reports the same step/token counts.
func ExampleRunDistillation() {
	ds := dataset.NewSliceDataset([]dataset.Sample{{Text: "hello"}})

	result, err := RunDistillation(context.Background(), DistillRunner{
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
			return DistillLogits{{{0, 0}}}, nil
		},
	}, ds, DistillConfig{Temperature: 2})
	if err != nil {
		core.Println(err.Error())
		return
	}

	core.Println(result.Metrics.Steps, result.Metrics.Tokens)
	// Output: 1 1
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

// ExampleNewMemoryDistillLogitCache shows the constructor returns a ready
// cache whose backing map is initialised, so the first Get on an unknown
// key is a clean miss with no panic.
func ExampleNewMemoryDistillLogitCache() {
	cache := NewMemoryDistillLogitCache()
	_, ok, err := cache.GetTeacherLogits(context.Background(), "unknown")
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(ok)
	// Output: false
}

// ExampleMemoryDistillLogitCache_GetTeacherLogits shows Get returns a hit
// flag and the stored logits for a previously stored key.
func ExampleMemoryDistillLogitCache_GetTeacherLogits() {
	cache := NewMemoryDistillLogitCache()
	ctx := context.Background()
	if err := cache.PutTeacherLogits(ctx, "k", DistillLogits{{{5, 6}}}); err != nil {
		core.Println(err.Error())
		return
	}
	logits, ok, err := cache.GetTeacherLogits(ctx, "k")
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(ok, logits[0][0][1])
	// Output: true 6
}

// ExampleMemoryDistillLogitCache_PutTeacherLogits shows Put stores a clone:
// mutating the caller's slice after Put does not change the cached copy a
// later Get returns.
func ExampleMemoryDistillLogitCache_PutTeacherLogits() {
	cache := NewMemoryDistillLogitCache()
	ctx := context.Background()
	src := DistillLogits{{{1, 2}}}
	if err := cache.PutTeacherLogits(ctx, "k", src); err != nil {
		core.Println(err.Error())
		return
	}
	src[0][0][1] = 99 // mutate after Put — the cache holds a clone
	logits, _, _ := cache.GetTeacherLogits(ctx, "k")
	core.Println(logits[0][0][1])
	// Output: 2
}
