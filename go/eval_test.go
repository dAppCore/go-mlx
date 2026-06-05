// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/dataset"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
)

func requireRealEvalModel(t *testing.T) string {
	t.Helper()
	if core.Getenv("GO_MLX_RUN_MODEL_EVAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_MODEL_EVAL_TESTS=1 to enable real model eval tests")
	}
	modelPath := core.Getenv("GO_MLX_EVAL_MODEL")
	if modelPath == "" {
		t.Skip("set GO_MLX_EVAL_MODEL to a local model pack")
	}
	return modelPath
}

func TestRunModelEval_RealModelSkip_Good(t *testing.T) {
	modelPath := requireRealEvalModel(t)
	model, err := LoadModel(modelPath, WithContextLength(512), WithBatchSize(1))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	t.Cleanup(func() {
		_ = model.Close()
		ClearCache()
	})

	report, err := RunModelEval(context.Background(), model, dataset.NewSliceDataset([]dataset.Sample{
		{Text: "Local evaluation should produce a finite loss."},
	}), eval.Config{Batch: dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 64}})
	if err != nil {
		t.Fatalf("RunModelEval() error = %v", err)
	}
	if report.Metrics.Tokens == 0 || report.Metrics.Perplexity == 0 {
		t.Fatalf("metrics = %+v, want tokens and perplexity", report.Metrics)
	}
}

func TestEvalOptionalBatchAttentionMask_SkipsDenseMaskForUnpaddedBatch_Good(t *testing.T) {
	mask, bufPtr := evalOptionalBatchAttentionMask([]int32{4, 4}, 4)
	if mask != nil {
		t.Fatalf("evalOptionalBatchAttentionMask returned dense mask for unpadded batch")
	}
	if bufPtr != nil {
		t.Fatalf("evalOptionalBatchAttentionMask returned non-nil bufPtr on fast path")
	}
}

func TestEvalOptionalBatchAttentionMask_KeepsMaskForPaddedBatch_Good(t *testing.T) {
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	mask, bufPtr := evalOptionalBatchAttentionMask([]int32{4, 3}, 4)
	if mask == nil {
		t.Fatalf("evalOptionalBatchAttentionMask returned nil for padded batch")
	}
	if bufPtr != nil {
		releaseEvalBatchAttnMaskBuf(bufPtr)
	}
	defer Free(mask)

	Materialize(mask)
	shape := mask.Shape()
	want := []int32{2, 1, 4, 4}
	for i, got := range shape {
		if got != want[i] {
			t.Fatalf("mask shape[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func TestNewModelEvalRunner_NilAndCancelled_Bad(t *testing.T) {
	runner := NewModelEvalRunner(nil)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	if info := runner.Info(cancelled); info.Architecture != "" {
		t.Fatalf("Info(cancelled) = %+v, want zero value", info)
	}
	if _, err := runner.LoadAdapter(cancelled, "adapter"); err != context.Canceled {
		t.Fatalf("LoadAdapter(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := runner.LoadAdapter(context.Background(), "adapter"); err == nil {
		t.Fatal("expected nil model adapter load error")
	}
	if _, err := runner.EvaluateBatch(context.Background(), SFTBatch{}); err == nil {
		t.Fatal("expected nil model evaluate error")
	}

	var model *Model
	if _, err := model.evaluateDatasetBatch(context.Background(), SFTBatch{}); err == nil {
		t.Fatal("expected nil receiver eval error")
	}
	if _, err := (&Model{}).evaluateDatasetBatch(cancelled, SFTBatch{}); err != context.Canceled {
		t.Fatalf("evaluateDatasetBatch(cancelled) = %v, want context.Canceled", err)
	}
}

func TestEvalBatchDataHelpers_Good(t *testing.T) {
	batch := SFTBatch{
		Batch: Batch{
			Tokens:   [][]int{{1, 2, 3, 4}, {5, 6, 7}},
			Length:   []int{3, 0},
			LossMask: [][]float32{{1, 0}, {0.25, 1, 0}},
		},
		Targets: [][]int{{2, 3, 4, 5}, {6, 7, 8}},
	}

	lengths, maxLen, err := evalBatchLengths(batch)
	if err != nil {
		t.Fatalf("evalBatchLengths() error = %v", err)
	}
	if !equalInt32Slices(lengths, []int32{2, 3}) || maxLen != 3 {
		t.Fatalf("lengths=%v max=%d, want [2 3]/3", lengths, maxLen)
	}
	tokensPtr := evalBatchTokenData(batch.Batch.Tokens, lengths, maxLen)
	if !equalInt32Slices(*tokensPtr, []int32{1, 2, 0, 5, 6, 7}) {
		t.Fatalf("token data = %v, want padded rows", *tokensPtr)
	}
	releaseEvalBatchInt32Buf(tokensPtr)
	targetsPtr := evalBatchTokenData(batch.Targets, lengths, maxLen)
	if !equalInt32Slices(*targetsPtr, []int32{2, 3, 0, 6, 7, 8}) {
		t.Fatalf("target data = %v, want padded rows", *targetsPtr)
	}
	releaseEvalBatchInt32Buf(targetsPtr)
	maskPtr := evalBatchLossMaskData(batch, lengths, maxLen)
	if !equalFloat32Slices(*maskPtr, []float32{1, 0, 0, 0.25, 1, 0}) {
		t.Fatalf("loss mask data = %v, want padded mask", *maskPtr)
	}
	releaseEvalBatchFloat32Buf(maskPtr)
	if evalNeedsExplicitAttentionMask([]int32{3, 3}, 3) {
		t.Fatal("equal lengths should not need explicit attention mask")
	}
	if !evalNeedsExplicitAttentionMask(nil, 3) || !evalNeedsExplicitAttentionMask([]int32{2, 3}, 3) || !evalNeedsExplicitAttentionMask([]int32{3}, 0) {
		t.Fatal("padded, empty, or zero max length batch should need explicit attention mask")
	}
	freeEvalCaches([]Cache{nil})
}

func TestEvalBatchLengths_Bad(t *testing.T) {
	if _, _, err := evalBatchLengths(SFTBatch{}); err == nil {
		t.Fatal("expected empty batch error")
	}
	if _, _, err := evalBatchLengths(SFTBatch{
		Batch:   Batch{Tokens: [][]int{{1}}},
		Targets: [][]int{{1}, {2}},
	}); err == nil {
		t.Fatal("expected unaligned batch error")
	}
	if _, _, err := evalBatchLengths(SFTBatch{
		Batch:   Batch{Tokens: [][]int{{}}},
		Targets: [][]int{{}},
	}); err == nil {
		t.Fatal("expected empty sequence error")
	}
	if _, err := (&Model{model: &fakeNativeModel{}}).evaluateDatasetBatch(context.Background(), SFTBatch{}); err == nil {
		t.Fatal("expected invalid batch before native eval")
	}
}

func equalInt32Slices(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
