// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	core "dappco.re/go"
)

// ExampleDistillationBatchLoss computes the soft cross-entropy between a
// uniform teacher and a peaked student over one masked token, and
// reports the token count and the loss kind it selected.
func ExampleDistillationBatchLoss() {
	loss, err := DistillationBatchLoss(
		DistillLogits{{{0, 0}}},    // uniform teacher
		DistillLogits{{{10, -10}}}, // peaked student
		[][]float32{{1}},           // one masked token
		DistillConfig{Loss: DistillLossSoftCrossEntropy, Temperature: 1},
	)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(loss.Tokens, string(loss.Kind))
	// Output: 1 soft_cross_entropy
}

// ExampleDistillBatchCacheKey shows that the cache key is stable: the
// same batch contents always hash to the same key, so a teacher-logit
// cache lookup is reproducible across runs.
func ExampleDistillBatchCacheKey() {
	batch := SFTBatch{
		Batch:   Batch{Tokens: [][]int{{1, 2}}, LossMask: [][]float32{{1, 1}}},
		Targets: [][]int{{2, 3}},
	}
	core.Println(DistillBatchCacheKey(batch) == DistillBatchCacheKey(batch))
	// Output: true
}
