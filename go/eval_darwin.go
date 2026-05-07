// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
)

type nativeEvalInternalModel interface {
	Internal() metal.InternalModel
}

// NewModelEvalRunner adapts a loaded native Model to dataset evaluation.
func NewModelEvalRunner(model *Model) EvalRunner {
	return EvalRunner{
		Info: func(ctx context.Context) ModelInfo {
			if err := ctx.Err(); err != nil || model == nil {
				return ModelInfo{}
			}
			return model.Info()
		},
		Tokenizer: func(ctx context.Context) *Tokenizer {
			if err := ctx.Err(); err != nil || model == nil {
				return nil
			}
			return model.Tokenizer()
		},
		LoadAdapter: func(ctx context.Context, path string) (LoRAAdapterInfo, error) {
			if err := ctx.Err(); err != nil {
				return LoRAAdapterInfo{}, err
			}
			if model == nil {
				return LoRAAdapterInfo{}, core.NewError("mlx: model is nil")
			}
			if _, err := model.LoadLoRA(path); err != nil {
				return LoRAAdapterInfo{}, err
			}
			return model.Adapter(), nil
		},
		EvaluateBatch: func(ctx context.Context, batch SFTBatch) (EvalBatchMetrics, error) {
			if model == nil {
				return EvalBatchMetrics{}, core.NewError("mlx: model is nil")
			}
			return model.evaluateDatasetBatch(ctx, batch)
		},
	}
}

func (m *Model) evaluateDatasetBatch(ctx context.Context, batch SFTBatch) (EvalBatchMetrics, error) {
	if err := ctx.Err(); err != nil {
		return EvalBatchMetrics{}, err
	}
	if m == nil || m.model == nil {
		return EvalBatchMetrics{}, core.NewError("mlx: model is nil")
	}

	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		return EvalBatchMetrics{}, err
	}
	inputs := FromValues(evalBatchTokenData(batch.Batch.Tokens, lengths, maxLen), len(lengths), maxLen)
	targets := FromValues(evalBatchTokenData(batch.Targets, lengths, maxLen), len(lengths), maxLen)
	lossMask := FromValues(evalBatchLossMaskData(batch, lengths, maxLen), len(lengths), maxLen)
	attnMask := evalBatchAttentionMask(lengths, maxLen)
	defer Free(inputs, targets, lossMask, attnMask)

	native, ok := m.model.(nativeEvalInternalModel)
	if !ok {
		return EvalBatchMetrics{}, core.NewError("mlx: native model does not expose eval forward")
	}
	internal := native.Internal()
	caches := internal.NewCache()
	defer freeEvalCaches(caches)

	logits := internal.ForwardMasked(inputs, attnMask, caches)
	if logits == nil {
		return EvalBatchMetrics{}, core.NewError("mlx: eval forward returned nil logits")
	}
	loss := MaskedCrossEntropyLoss(logits, targets, lossMask)
	if loss == nil {
		Free(logits)
		return EvalBatchMetrics{}, core.NewError("mlx: eval loss returned nil")
	}
	Materialize(loss)
	lossValue := loss.Float()
	Free(logits, loss)
	if math.IsNaN(lossValue) || math.IsInf(lossValue, 0) {
		return EvalBatchMetrics{}, core.NewError("mlx: eval loss is not finite")
	}
	return EvalBatchMetrics{
		Samples: len(lengths),
		Tokens:  sftBatchLossTokens(batch),
		Loss:    lossValue,
	}, nil
}

func evalBatchLengths(batch SFTBatch) ([]int32, int, error) {
	if len(batch.Batch.Tokens) == 0 || len(batch.Batch.Tokens) != len(batch.Targets) {
		return nil, 0, core.NewError("mlx: eval batch tokens and targets must be non-empty and aligned")
	}
	lengths := make([]int32, len(batch.Batch.Tokens))
	maxLen := 0
	for i := range batch.Batch.Tokens {
		n := len(batch.Batch.Tokens[i])
		if len(batch.Targets[i]) < n {
			n = len(batch.Targets[i])
		}
		if i < len(batch.Batch.Length) && batch.Batch.Length[i] > 0 && batch.Batch.Length[i] < n {
			n = batch.Batch.Length[i]
		}
		if i < len(batch.Batch.LossMask) && len(batch.Batch.LossMask[i]) < n {
			n = len(batch.Batch.LossMask[i])
		}
		if n <= 0 {
			return nil, 0, core.NewError("mlx: eval batch contains an empty sequence")
		}
		lengths[i] = int32(n)
		if n > maxLen {
			maxLen = n
		}
	}
	return lengths, maxLen, nil
}

func evalBatchTokenData(seqs [][]int, lengths []int32, maxLen int) []int32 {
	data := make([]int32, len(seqs)*maxLen)
	for i, seq := range seqs {
		limit := int(lengths[i])
		base := i * maxLen
		for j := 0; j < limit; j++ {
			data[base+j] = int32(seq[j])
		}
	}
	return data
}

func evalBatchLossMaskData(batch SFTBatch, lengths []int32, maxLen int) []float32 {
	data := make([]float32, len(lengths)*maxLen)
	for i := range lengths {
		limit := int(lengths[i])
		base := i * maxLen
		for j := 0; j < limit; j++ {
			value := float32(1)
			if i < len(batch.Batch.LossMask) && j < len(batch.Batch.LossMask[i]) {
				value = batch.Batch.LossMask[i][j]
			}
			data[base+j] = value
		}
	}
	return data
}

func evalBatchAttentionMask(lengths []int32, maxLen int) *Array {
	negInf := float32(math.Inf(-1))
	batchSize := len(lengths)
	data := make([]float32, batchSize*maxLen*maxLen)
	for b, length := range lengths {
		base := b * maxLen * maxLen
		for i := 0; i < maxLen; i++ {
			for j := 0; j < maxLen; j++ {
				if j <= i && j < int(length) {
					data[base+i*maxLen+j] = 0
				} else {
					data[base+i*maxLen+j] = negInf
				}
			}
		}
	}
	return FromValues(data, batchSize, 1, maxLen, maxLen)
}

func freeEvalCaches(caches []Cache) {
	for _, cache := range caches {
		if cache == nil {
			continue
		}
		Free(cache.State()...)
		cache.Reset()
	}
}
