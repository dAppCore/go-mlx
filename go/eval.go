// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/dataset"
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/lora"
)

// RunModelEval evaluates a loaded model over an SFT/JSONL dataset stream.
// The mlx-root wrapper adapts dataset.Dataset/dataset.Sample/SFTBatch to eval's
// opaque types and forwards to eval.RunDataset.
func RunModelEval(ctx context.Context, model *Model, ds dataset.Dataset, cfg eval.Config) (*eval.Report, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	cfg.QualityProbes = append([]eval.QualityProbe(nil), cfg.QualityProbes...)
	cfg.QualityProbes = append(cfg.QualityProbes, eval.ResponseCoverageProbe())
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
		TargetKeys: append([]string(nil), info.TargetKeys...),
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
		TargetKeys: append([]string(nil), info.TargetKeys...),
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
