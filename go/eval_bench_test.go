// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the CPU-only side of eval.go — batch shape helpers,
// adapter/info converters, and the attention-mask builders. Per AX-11 —
// these run per evaluation batch, and evaluation passes routinely chew
// through hundreds of batches in a single quality run. The attention-mask
// builder allocates O(batch × max_len^2) floats, so it's the per-batch
// cost the eval loop is most likely to feel.
//
// Model-bound functions (evaluateDatasetBatch, ForwardMasked, the
// Runner callbacks that depend on a real model) need a loaded *Model
// and are intentionally OUT of scope.
//
// Run:    go test -bench='BenchmarkEval' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/lora"
)

// Sinks defeat compiler DCE. Distinct from other bench files in this package.
var (
	evalBenchSinkLengths   []int32
	evalBenchSinkMaxLen    int
	evalBenchSinkErr       error
	evalBenchSinkTokens    []int32
	evalBenchSinkMask      []float32
	evalBenchSinkBool      bool
	evalBenchSinkEvalInfo  eval.Info
	evalBenchSinkModelInfo ModelInfo
	evalBenchSinkLoraInfo  lora.AdapterInfo
	evalBenchSinkAdapter   eval.AdapterInfo
	evalBenchSinkSample    string
	evalBenchSinkTokenN    int
)

// evalBenchBatch builds a representative SFTBatch with the shape of a
// realistic SFT eval row. batchSize sequences, each containing seqLen
// non-padded tokens plus a sparse loss mask. Targets are the same shape
// as inputs (shifted by one in real flows — here we just reuse the
// numbers so the converter sees aligned slices).
func evalBenchBatch(batchSize, seqLen int) SFTBatch {
	tokens := make([][]int, batchSize)
	targets := make([][]int, batchSize)
	lossMask := make([][]float32, batchSize)
	lengths := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		tokens[i] = make([]int, seqLen)
		targets[i] = make([]int, seqLen)
		lossMask[i] = make([]float32, seqLen)
		lengths[i] = seqLen
		for j := 0; j < seqLen; j++ {
			tokens[i][j] = (i*seqLen + j) % 32000
			targets[i][j] = (i*seqLen + j + 1) % 32000
			if j >= seqLen/2 {
				lossMask[i][j] = 1
			}
		}
	}
	return SFTBatch{
		Batch:   Batch{Tokens: tokens, Length: lengths, LossMask: lossMask},
		Targets: targets,
	}
}

// evalBenchInfo mirrors fastEvalBenchMlxInfo shape but stays inside the
// eval-bench file so the two converters can be exercised independently.
func evalBenchInfo() ModelInfo {
	return ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
		Adapter: lora.AdapterInfo{
			Name:       "eval-bench-lora",
			Path:       "/models/adapters/eval-bench",
			Rank:       16,
			Alpha:      32,
			Scale:      0.5,
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		},
	}
}

// evalBenchEvalInfo is the cross-side mirror used by evalInfoToModel.
func evalBenchEvalInfo() eval.Info {
	return eval.Info{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
		Adapter: eval.AdapterInfo{
			Name:       "eval-bench-lora",
			Path:       "/models/adapters/eval-bench",
			Rank:       16,
			Alpha:      32,
			Scale:      0.5,
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		},
	}
}

// --- evalBatchLengths — per-batch shape derivation ---

func BenchmarkEval_EvalBatchLengths_Batch1_Seq512(b *testing.B) {
	batch := evalBenchBatch(1, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkLengths, evalBenchSinkMaxLen, evalBenchSinkErr = evalBatchLengths(batch)
	}
}

func BenchmarkEval_EvalBatchLengths_Batch4_Seq512(b *testing.B) {
	batch := evalBenchBatch(4, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkLengths, evalBenchSinkMaxLen, evalBenchSinkErr = evalBatchLengths(batch)
	}
}

func BenchmarkEval_EvalBatchLengths_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkLengths, evalBenchSinkMaxLen, evalBenchSinkErr = evalBatchLengths(batch)
	}
}

// --- evalBatchTokenData — per-batch token tensor flatten + cast ---

func BenchmarkEval_EvalBatchTokenData_Batch1_Seq512(b *testing.B) {
	batch := evalBenchBatch(1, 512)
	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkTokens = evalBatchTokenData(batch.Batch.Tokens, lengths, maxLen)
	}
}

func BenchmarkEval_EvalBatchTokenData_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkTokens = evalBatchTokenData(batch.Batch.Tokens, lengths, maxLen)
	}
}

// --- evalBatchLossMaskData — per-batch loss mask flatten ---

func BenchmarkEval_EvalBatchLossMaskData_Batch1_Seq512(b *testing.B) {
	batch := evalBenchBatch(1, 512)
	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkMask = evalBatchLossMaskData(batch, lengths, maxLen)
	}
}

func BenchmarkEval_EvalBatchLossMaskData_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkMask = evalBatchLossMaskData(batch, lengths, maxLen)
	}
}

// --- sftBatchLossTokens — per-batch loss-token counter ---

func BenchmarkEval_SftBatchLossTokens_LossMaskPath_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkTokenN = sftBatchLossTokens(batch)
	}
}

// Length-only path — strip the LossMask to force the Length branch.
func BenchmarkEval_SftBatchLossTokens_LengthPath_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	batch.Batch.LossMask = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkTokenN = sftBatchLossTokens(batch)
	}
}

// Tokens-only path — strip both LossMask and Length.
func BenchmarkEval_SftBatchLossTokens_TokensPath_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	batch.Batch.LossMask = nil
	batch.Batch.Length = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkTokenN = sftBatchLossTokens(batch)
	}
}

// --- sftBatchTokens — eval.Batch wrapper, used by the Runner callback ---

func BenchmarkEval_SftBatchTokens_Batch4_Seq2048(b *testing.B) {
	batch := evalBenchBatch(4, 2048)
	var asEval eval.Batch = batch
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkTokenN = sftBatchTokens(asEval)
	}
}

// --- evalNeedsExplicitAttentionMask — per-batch fast-path check ---

func BenchmarkEval_EvalNeedsExplicitAttentionMask_AllEqual(b *testing.B) {
	lengths := []int32{2048, 2048, 2048, 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkBool = evalNeedsExplicitAttentionMask(lengths, 2048)
	}
}

func BenchmarkEval_EvalNeedsExplicitAttentionMask_Ragged(b *testing.B) {
	lengths := []int32{2048, 1500, 800, 256}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkBool = evalNeedsExplicitAttentionMask(lengths, 2048)
	}
}

// NOTE: evalBatchAttentionMask + evalOptionalBatchAttentionMask wrap
// FromValues, which crosses into the metal cgo layer. They are NOT
// benched here — pure mask-array construction is fine, but the FromValues
// call drags in Metal initialisation and an MLX allocation, which makes
// the bench measure GPU init noise rather than the per-call mask build.
// The pure fast-path predicate (evalNeedsExplicitAttentionMask) above
// already covers the early-exit branch evalOptionalBatchAttentionMask
// checks before allocating.

// --- modelInfoToEval / evalInfoToModel — converter pair ---

func BenchmarkEval_ModelInfoToEval(b *testing.B) {
	info := evalBenchInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkEvalInfo = modelInfoToEval(info)
	}
}

func BenchmarkEval_EvalInfoToModel(b *testing.B) {
	info := evalBenchEvalInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkModelInfo = evalInfoToModel(info)
	}
}

// --- loraToEvalAdapter / evalAdapterToLora ---

func BenchmarkEval_LoraToEvalAdapter(b *testing.B) {
	info := evalBenchInfo().Adapter
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkAdapter = loraToEvalAdapter(info)
	}
}

func BenchmarkEval_EvalAdapterToLora(b *testing.B) {
	info := evalBenchEvalInfo().Adapter
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkLoraInfo = evalAdapterToLora(info)
	}
}

// --- sftSampleText — pulls strings out of dataset.Sample for eval probes ---

func BenchmarkEval_SftSampleText_DatasetSample(b *testing.B) {
	sample := dataset.Sample{Text: "free-form passage", Prompt: "p", Response: "r"}
	var asEval eval.Sample = sample
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalBenchSinkSample, _ = sftSampleText(asEval)
	}
}
