// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
	"math"
)

// RunModelEval evaluates a loaded model over an SFT/JSONL dataset stream.
// The mlx-root wrapper adapts dataset.Dataset/dataset.Sample/SFTBatch to eval's
// opaque types and forwards to eval.RunDataset.
func RunModelEval(ctx context.Context, model *Model, ds dataset.Dataset, cfg eval.Config) (*eval.Report, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	// Pre-size for len+1 so the second append doesn't trigger a regrow —
	// the original cloned via append([]T(nil), ...) then appended the
	// ResponseCoverageProbe, paying the grow twice. One make + two
	// appends fits the final size in a single allocation.
	probes := make([]eval.QualityProbe, len(cfg.QualityProbes), len(cfg.QualityProbes)+1)
	copy(probes, cfg.QualityProbes)
	cfg.QualityProbes = append(probes, eval.ResponseCoverageProbe())
	return eval.RunDataset(ctx, NewModelEvalRunner(model), wrapSFTDataset(ds), cfg)
}

// sftSampleText pulls text/response from a wrapped dataset.Sample for eval's
// quality probes that need to inspect sample content.
func sftSampleText(sample eval.Sample) (string, string) {
	if s, ok := sample.(dataset.Sample); ok {
		return s.Text, s.Response
	}
	return "", ""
}

// sftBatchTokens returns the loss-eligible token count for a wrapped SFTBatch.
func sftBatchTokens(batch eval.Batch) int {
	if b, ok := batch.(SFTBatch); ok {
		return sftBatchLossTokens(b)
	}
	return 0
}

func sftBatchLossTokens(batch SFTBatch) int {
	tokens := 0
	if len(batch.Batch.LossMask) > 0 {
		for _, row := range batch.Batch.LossMask {
			for _, value := range row {
				if value > 0 {
					tokens++
				}
			}
		}
		return tokens
	}
	if len(batch.Batch.Length) > 0 {
		for _, length := range batch.Batch.Length {
			if length > 0 {
				tokens += length
			}
		}
		return tokens
	}
	for _, row := range batch.Batch.Tokens {
		tokens += len(row)
	}
	return tokens
}

// wrapSFTDataset adapts a mlx.SFTDataset to eval.Dataset (opaque samples).
func wrapSFTDataset(d dataset.Dataset) eval.Dataset {
	if d == nil {
		return nil
	}
	return &sftDatasetAdapter{ds: d}
}

type sftDatasetAdapter struct {
	ds dataset.Dataset
}

func (a *sftDatasetAdapter) Next() (eval.Sample, bool, error) {
	sample, ok, err := a.ds.Next()
	if err != nil || !ok {
		return nil, ok, err
	}
	return dataset.CloneSample(sample), true, nil
}

// modelInfoToEval converts an mlx.ModelInfo to the driver-neutral eval.Info.
func modelInfoToEval(info ModelInfo) eval.Info {
	return eval.Info{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       loraToEvalAdapter(info.Adapter),
	}
}

// loraToEvalAdapter converts an mlx-root lora.AdapterInfo to eval.AdapterInfo.
func loraToEvalAdapter(info lora.AdapterInfo) eval.AdapterInfo {
	return eval.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: core.SliceClone(info.TargetKeys),
	}
}

// evalAdapterToLora converts back from eval.AdapterInfo when mlx-root code
// needs the typed mlx.lora form.
func evalAdapterToLora(info eval.AdapterInfo) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: core.SliceClone(info.TargetKeys),
	}
}

// evalInfoToModel converts from driver-neutral eval.Info back to mlx.ModelInfo.
func evalInfoToModel(info eval.Info) ModelInfo {
	return ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       evalAdapterToLora(info.Adapter),
	}
}

type nativeEvalInternalModel interface {
	Internal() metal.InternalModel
}

// NewModelEvalRunner adapts a loaded native Model to driver-neutral
// eval.Runner. The driver provides callbacks for the few accessors
// eval needs (Info, LoadAdapter, BuildBatches, EvaluateBatch, BatchTokens,
// SampleText).
func NewModelEvalRunner(model *Model) eval.Runner {
	return eval.Runner{
		Info: func(ctx context.Context) eval.Info {
			if err := ctx.Err(); err != nil || model == nil {
				return eval.Info{}
			}
			return modelInfoToEval(model.Info())
		},
		LoadAdapter: func(ctx context.Context, path string) (eval.AdapterInfo, error) {
			if err := ctx.Err(); err != nil {
				return eval.AdapterInfo{}, err
			}
			if model == nil {
				return eval.AdapterInfo{}, core.NewError("mlx: model is nil")
			}
			if _, err := model.LoadLoRA(path); err != nil {
				return eval.AdapterInfo{}, err
			}
			return loraToEvalAdapter(model.Adapter()), nil
		},
		BuildBatches: func(ctx context.Context, ds eval.Dataset, cfg eval.BatchConfig) ([]eval.Batch, error) {
			if model == nil {
				return nil, core.NewError("mlx: model is nil")
			}
			batchCfg, ok := cfg.(dataset.BatchConfig)
			if !ok {
				batchCfg = dataset.BatchConfig{}
			}
			tok := model.Tokenizer()
			if tok == nil {
				return nil, core.NewError("mlx: model tokenizer is nil")
			}
			sftDataset := evalDatasetToSFT(ds)
			sftBatches, err := BuildDatasetBatches(tok, sftDataset, batchCfg)
			if err != nil {
				return nil, err
			}
			batches := make([]eval.Batch, len(sftBatches))
			for i, b := range sftBatches {
				batches[i] = b
			}
			return batches, nil
		},
		EvaluateBatch: func(ctx context.Context, batch eval.Batch) (eval.BatchMetrics, error) {
			if model == nil {
				return eval.BatchMetrics{}, core.NewError("mlx: model is nil")
			}
			sftBatch, ok := batch.(SFTBatch)
			if !ok {
				return eval.BatchMetrics{}, core.NewError("mlx: eval batch is not an SFTBatch")
			}
			m, err := model.evaluateDatasetBatch(ctx, sftBatch)
			if err != nil {
				return eval.BatchMetrics{}, err
			}
			return eval.BatchMetrics{Samples: m.Samples, Tokens: m.Tokens, Loss: m.Loss}, nil
		},
		BatchTokens: sftBatchTokens,
		SampleText:  sftSampleText,
	}
}

type evalDatasetSFTAdapter struct {
	src eval.Dataset
}

func (a *evalDatasetSFTAdapter) Next() (dataset.Sample, bool, error) {
	sample, ok, err := a.src.Next()
	if err != nil || !ok {
		return dataset.Sample{}, ok, err
	}
	if s, ok := sample.(dataset.Sample); ok {
		return s, true, nil
	}
	return dataset.Sample{}, false, core.NewError("mlx: eval dataset returned a non-dataset.Sample value")
}

func evalDatasetToSFT(d eval.Dataset) dataset.Dataset {
	return &evalDatasetSFTAdapter{src: d}
}

// evalBatchMetricsDarwin is the driver-internal version used by Model.evaluateDatasetBatch.
type evalBatchMetricsDarwin struct {
	Samples int
	Tokens  int
	Loss    float64
}

func (m *Model) evaluateDatasetBatch(ctx context.Context, batch SFTBatch) (evalBatchMetricsDarwin, error) {
	if err := ctx.Err(); err != nil {
		return evalBatchMetricsDarwin{}, err
	}
	if m == nil || m.model == nil {
		return evalBatchMetricsDarwin{}, core.NewError("mlx: model is nil")
	}

	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		return evalBatchMetricsDarwin{}, err
	}
	inputs := FromValues(evalBatchTokenData(batch.Batch.Tokens, lengths, maxLen), len(lengths), maxLen)
	targets := FromValues(evalBatchTokenData(batch.Targets, lengths, maxLen), len(lengths), maxLen)
	lossMask := FromValues(evalBatchLossMaskData(batch, lengths, maxLen), len(lengths), maxLen)
	attnMask := evalOptionalBatchAttentionMask(lengths, maxLen)
	defer Free(inputs, targets, lossMask, attnMask)

	native, ok := m.model.(nativeEvalInternalModel)
	if !ok {
		return evalBatchMetricsDarwin{}, core.NewError("mlx: native model does not expose eval forward")
	}
	internal := native.Internal()
	caches := internal.NewCache()
	defer freeEvalCaches(caches)

	logits := internal.ForwardMasked(inputs, attnMask, caches)
	if logits == nil {
		return evalBatchMetricsDarwin{}, core.NewError("mlx: eval forward returned nil logits")
	}
	loss := MaskedCrossEntropyLoss(logits, targets, lossMask)
	if loss == nil {
		Free(logits)
		return evalBatchMetricsDarwin{}, core.NewError("mlx: eval loss returned nil")
	}
	Materialize(loss)
	lossValue := loss.Float()
	Free(logits, loss)
	if math.IsNaN(lossValue) || math.IsInf(lossValue, 0) {
		return evalBatchMetricsDarwin{}, core.NewError("mlx: eval loss is not finite")
	}
	return evalBatchMetricsDarwin{
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
		// Local slice + ranged limit lets the compiler hoist the per-iter
		// bounds checks on data[base+j] and seq[j] — the previous form
		// repeated data[base+j] with two-operand index, which the SSA
		// pass treats as needing a fresh bounds check per write.
		dst := data[base : base+limit : base+limit]
		src := seq[:limit:limit]
		for j := range dst {
			dst[j] = int32(src[j])
		}
	}
	return data
}

func evalBatchLossMaskData(batch SFTBatch, lengths []int32, maxLen int) []float32 {
	data := make([]float32, len(lengths)*maxLen)
	masks := batch.Batch.LossMask
	for i, l := range lengths {
		limit := int(l)
		base := i * maxLen
		// Hoist the per-row mask resolution out of the inner loop —
		// the original checked len(masks) and len(masks[i]) on every
		// token, which is the hot path for SFT eval batches.
		var maskRow []float32
		if i < len(masks) {
			maskRow = masks[i]
		}
		if len(maskRow) >= limit {
			// Full mask row available — copy from the explicit values,
			// no per-element fallback needed.
			copy(data[base:base+limit], maskRow[:limit])
		} else {
			// Partial or no mask: copy what we have, then fill the
			// remaining limit slots with the default value of 1.
			n := copy(data[base:base+limit], maskRow)
			row := data[base+n : base+limit]
			for j := range row {
				row[j] = 1
			}
		}
	}
	return data
}

func evalBatchAttentionMask(lengths []int32, maxLen int) *Array {
	negInf := float32(math.Inf(-1))
	batchSize := len(lengths)
	data := make([]float32, batchSize*maxLen*maxLen)
	// data is zero-initialised — only need to set negInf positions.
	// Causal+padding mask: for each (i,j), unmask iff j <= i && j < length.
	// Walk the masked region by row, writing the negInf tail in two
	// runs per row instead of branching per cell. This drops the per-
	// (i,j) compare from O(N²) to one slice write per row.
	for b, length := range lengths {
		base := b * maxLen * maxLen
		limit := int(length)
		for i := 0; i < maxLen; i++ {
			rowStart := base + i*maxLen
			// Unmasked range: j in [0, min(i+1, limit)). All other cells
			// in the row stay non-zero (negInf).
			unmaskedEnd := i + 1
			if unmaskedEnd > limit {
				unmaskedEnd = limit
			}
			if unmaskedEnd < 0 {
				unmaskedEnd = 0
			}
			// Fill the masked tail with negInf — left zeros are already
			// the unmask value, no per-cell store needed there.
			tail := data[rowStart+unmaskedEnd : rowStart+maxLen]
			for j := range tail {
				tail[j] = negInf
			}
		}
	}
	return FromValues(data, batchSize, 1, maxLen, maxLen)
}

func evalOptionalBatchAttentionMask(lengths []int32, maxLen int) *Array {
	if !evalNeedsExplicitAttentionMask(lengths, maxLen) {
		return nil
	}
	return evalBatchAttentionMask(lengths, maxLen)
}

func evalNeedsExplicitAttentionMask(lengths []int32, maxLen int) bool {
	if maxLen <= 0 || len(lengths) == 0 {
		return true
	}
	for _, length := range lengths {
		if int(length) != maxLen {
			return true
		}
	}
	return false
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
