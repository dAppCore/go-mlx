// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/spine"
)

var (
	errMLXNativeEvalTokenModelNil    = core.NewError("mlx: native eval token model is nil")
	errMLXNativeEvalTargetOutOfRange = core.NewError("mlx: native eval target id out of range")
	errMLXNativeEvalLogitsTooShort   = core.NewError("mlx: native eval logits are shorter than target vocabulary")
)

func (m *nativeTextModel) Evaluate(ctx context.Context, dataset inference.DatasetStream, cfg inference.EvalConfig) (*inference.EvalReport, error) {
	if m == nil {
		return nil, errMLXModelNil
	}
	report, err := eval.RunDataset(ctx, m.nativeEvalRunner(), wrapSFTDataset(inferenceDataset{stream: dataset}), toEvalConfig(cfg))
	if err != nil {
		return nil, err
	}
	return toInferenceEvalReport(report), nil
}

func (m *nativeTextModel) nativeEvalRunner() eval.Runner {
	return eval.Runner{
		Info: func(ctx context.Context) eval.Info {
			if ctx != nil {
				if err := ctx.Err(); err != nil {
					return eval.Info{}
				}
			}
			if m == nil {
				return eval.Info{}
			}
			return nativeInfoToEval(m.Info())
		},
		BuildBatches: func(ctx context.Context, ds eval.Dataset, cfg eval.BatchConfig) ([]eval.Batch, error) {
			if ctx != nil {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if m == nil {
				return nil, errMLXModelNil
			}
			if m.tok == nil {
				return nil, errMLXEvalTokenizerNil
			}
			batchCfg, ok := cfg.(dataset.BatchConfig)
			if !ok {
				batchCfg = dataset.BatchConfig{}
			}
			sftBatches, err := BuildDatasetBatches(spine.NewTokenizer(m.tok), evalDatasetToSFT(ds), batchCfg)
			if err != nil {
				return nil, err
			}
			batches := make([]eval.Batch, len(sftBatches))
			for i := range sftBatches {
				batches[i] = sftBatches[i]
			}
			return batches, nil
		},
		EvaluateBatch: func(ctx context.Context, batch eval.Batch) (eval.BatchMetrics, error) {
			if m == nil {
				return eval.BatchMetrics{}, errMLXModelNil
			}
			sftBatch, ok := batch.(SFTBatch)
			if !ok {
				return eval.BatchMetrics{}, errMLXEvalBatchNotSFTBatch
			}
			return m.evaluateNativeBatch(ctx, sftBatch)
		},
		BatchTokens: sftBatchTokens,
		SampleText:  sftSampleText,
	}
}

func nativeInfoToEval(info inference.ModelInfo) eval.Info {
	return eval.Info{
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	}
}

func (m *nativeTextModel) evaluateNativeBatch(ctx context.Context, batch SFTBatch) (eval.BatchMetrics, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return eval.BatchMetrics{}, err
	}
	if m == nil {
		return eval.BatchMetrics{}, errMLXModelNil
	}
	if m.tm == nil {
		return eval.BatchMetrics{}, errMLXNativeEvalTokenModelNil
	}
	lengths, _, err := evalBatchLengths(batch)
	if err != nil {
		return eval.BatchMetrics{}, err
	}

	tokenCount := 0
	lossSum := 0.0
	if sessionModel, ok := m.tm.(model.SessionModel); ok {
		for row := range batch.Batch.Tokens {
			rowLoss, rowTokens, err := nativeEvalSessionRow(ctx, sessionModel, batch, row, int(lengths[row]))
			if err != nil {
				return eval.BatchMetrics{}, err
			}
			lossSum += rowLoss
			tokenCount += rowTokens
		}
	} else {
		for row := range batch.Batch.Tokens {
			rowLoss, rowTokens, err := nativeEvalWholeSeqRow(ctx, m.tm, batch, row, int(lengths[row]))
			if err != nil {
				return eval.BatchMetrics{}, err
			}
			lossSum += rowLoss
			tokenCount += rowTokens
		}
	}
	if tokenCount <= 0 {
		return eval.BatchMetrics{Samples: len(lengths)}, nil
	}
	loss := lossSum / float64(tokenCount)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return eval.BatchMetrics{}, errMLXEvalLossNonFinite
	}
	return eval.BatchMetrics{
		Samples: len(lengths),
		Tokens:  tokenCount,
		Loss:    loss,
	}, nil
}

func nativeEvalSessionRow(ctx context.Context, tm model.SessionModel, batch SFTBatch, row, length int) (float64, int, error) {
	sess, err := tm.OpenSession()
	if err != nil {
		return 0, 0, err
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	stepID, idAware := sess.(interface {
		StepWithID(id int32, emb []byte) ([]byte, error)
	})

	var lossSum float64
	tokenCount := 0
	for pos := 0; pos < length; pos++ {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		id := int32(batch.Batch.Tokens[row][pos])
		emb, err := tm.Embed(id)
		if err != nil {
			return 0, 0, err
		}
		var hidden []byte
		if idAware {
			hidden, err = stepID.StepWithID(id, emb)
		} else {
			hidden, err = sess.Step(emb)
		}
		if err != nil {
			return 0, 0, err
		}
		mask := nativeEvalLossMask(batch, row, pos)
		if mask <= 0 {
			continue
		}
		logits, err := tm.Head(hidden)
		if err != nil {
			return 0, 0, err
		}
		loss, err := nativeEvalCrossEntropy(logits, int32(batch.Targets[row][pos]))
		if err != nil {
			return 0, 0, err
		}
		lossSum += loss * float64(mask)
		tokenCount++
	}
	return lossSum, tokenCount, nil
}

func nativeEvalWholeSeqRow(ctx context.Context, tm model.TokenModel, batch SFTBatch, row, length int) (float64, int, error) {
	inputs := make([][]byte, 0, length)
	var lossSum float64
	tokenCount := 0
	for pos := 0; pos < length; pos++ {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		id := int32(batch.Batch.Tokens[row][pos])
		emb, err := tm.Embed(id)
		if err != nil {
			return 0, 0, err
		}
		inputs = append(inputs, emb)
		mask := nativeEvalLossMask(batch, row, pos)
		if mask <= 0 {
			continue
		}
		hiddens, err := tm.DecodeForward(inputs)
		if err != nil {
			return 0, 0, err
		}
		if len(hiddens) == 0 {
			return 0, 0, errMLXEvalForwardNilLogits
		}
		logits, err := tm.Head(hiddens[len(hiddens)-1])
		if err != nil {
			return 0, 0, err
		}
		loss, err := nativeEvalCrossEntropy(logits, int32(batch.Targets[row][pos]))
		if err != nil {
			return 0, 0, err
		}
		lossSum += loss * float64(mask)
		tokenCount++
	}
	return lossSum, tokenCount, nil
}

func nativeEvalLossMask(batch SFTBatch, row, pos int) float32 {
	if row >= len(batch.Batch.LossMask) {
		return 1
	}
	maskRow := batch.Batch.LossMask[row]
	if pos >= len(maskRow) {
		return 1
	}
	return maskRow[pos]
}

func nativeEvalCrossEntropy(logits []byte, target int32) (float64, error) {
	vocab := len(logits) / 2
	if vocab <= 0 {
		return 0, errMLXEvalForwardNilLogits
	}
	if target < 0 || int(target) >= vocab {
		return 0, errMLXNativeEvalTargetOutOfRange
	}
	if len(logits) < vocab*2 {
		return 0, errMLXNativeEvalLogitsTooShort
	}

	maxLogit := math.Inf(-1)
	for i := 0; i < vocab; i++ {
		value := nativeEvalBF16(logits, i)
		if value > maxLogit {
			maxLogit = value
		}
	}
	sum := 0.0
	for i := 0; i < vocab; i++ {
		sum += math.Exp(nativeEvalBF16(logits, i) - maxLogit)
	}
	loss := maxLogit + math.Log(sum) - nativeEvalBF16(logits, int(target))
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, errMLXEvalLossNonFinite
	}
	return loss, nil
}

func nativeEvalBF16(buf []byte, idx int) float64 {
	bits := uint32(binary.LittleEndian.Uint16(buf[idx*2:idx*2+2])) << 16
	return float64(math.Float32frombits(bits))
}
