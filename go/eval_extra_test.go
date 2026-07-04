// SPDX-Licence-Identifier: EUPL-1.2

// Extra coverage for eval.go: the driver-neutral conversion helpers and the
// NewModelEvalRunner closure surface that are reachable WITHOUT a GPU forward
// pass. The deep loss path (Model.evaluateDatasetBatch → native forward) needs
// a real Metal model and is exercised only by the model_eval-gated live suite;
// here we drive the runner with the in-package fakeNativeModel and assert the
// guard/conversion behaviour, plus the pure SFT-batch token accounting.

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
)

func TestEval_SFTSampleText_Good(t *testing.T) {
	text, resp := sftSampleText(dataset.Sample{Text: "the body", Response: "the reply"})
	if text != "the body" || resp != "the reply" {
		t.Fatalf("sftSampleText = (%q,%q), want (the body, the reply)", text, resp)
	}
	// A non-dataset.Sample value yields empty strings, never a panic.
	if a, b := sftSampleText(42); a != "" || b != "" {
		t.Fatalf("sftSampleText(non-sample) = (%q,%q), want empties", a, b)
	}
}

func TestEval_SFTBatchTokens_LossMask_Good(t *testing.T) {
	// LossMask present → count positive mask entries.
	batch := SFTBatch{Batch: Batch{LossMask: [][]float32{{1, 0, 0.5}, {0, 2}}}}
	if got := sftBatchTokens(batch); got != 3 {
		t.Fatalf("sftBatchTokens(lossmask) = %d, want 3", got)
	}
}

func TestEval_SFTBatchTokens_LengthFallback_Good(t *testing.T) {
	// No LossMask, Length present → sum positive lengths.
	batch := SFTBatch{Batch: Batch{Length: []int{3, 0, 4}}}
	if got := sftBatchTokens(batch); got != 7 {
		t.Fatalf("sftBatchTokens(length) = %d, want 7", got)
	}
}

func TestEval_SFTBatchTokens_TokenFallback_Good(t *testing.T) {
	// Neither LossMask nor Length → sum row token counts.
	batch := SFTBatch{Batch: Batch{Tokens: [][]int{{1, 2, 3}, {4, 5}}}}
	if got := sftBatchTokens(batch); got != 5 {
		t.Fatalf("sftBatchTokens(tokens) = %d, want 5", got)
	}
}

func TestEval_SFTBatchTokens_NonSFTBatch_Ugly(t *testing.T) {
	if got := sftBatchTokens("not a batch"); got != 0 {
		t.Fatalf("sftBatchTokens(non-batch) = %d, want 0", got)
	}
}

func TestEval_WrapSFTDataset_NextAndNil_Good(t *testing.T) {
	if wrapSFTDataset(nil) != nil {
		t.Fatal("wrapSFTDataset(nil) != nil")
	}
	wrapped := wrapSFTDataset(dataset.NewSliceDataset([]dataset.Sample{{Text: "row0"}}))
	sample, ok, err := wrapped.Next()
	if err != nil || !ok {
		t.Fatalf("wrapped.Next() = ok=%v err=%v, want ok=true", ok, err)
	}
	if s, isSample := sample.(dataset.Sample); !isSample || s.Text != "row0" {
		t.Fatalf("wrapped sample = %#v, want dataset.Sample{Text:row0}", sample)
	}
	// Exhausted dataset reports ok=false.
	if _, ok, _ := wrapped.Next(); ok {
		t.Fatal("wrapped.Next() after exhaustion ok = true, want false")
	}
}

func TestEval_ModelInfoToEval_RoundTrip_Good(t *testing.T) {
	info := ModelInfo{
		Architecture: "gemma4", VocabSize: 262144, NumLayers: 35,
		HiddenSize: 1536, QuantBits: 6, QuantGroup: 64, ContextLength: 131072,
		Adapter: lora.AdapterInfo{Name: "dom", Path: "/a", Rank: 8, Alpha: 16, Scale: 2, TargetKeys: []string{"q_proj"}},
	}
	e := modelInfoToEval(info)
	if e.Architecture != "gemma4" || e.VocabSize != 262144 || e.NumLayers != 35 || e.QuantBits != 6 {
		t.Fatalf("modelInfoToEval scalar fields wrong: %+v", e)
	}
	if e.Adapter.Name != "dom" || e.Adapter.Rank != 8 || len(e.Adapter.TargetKeys) != 1 {
		t.Fatalf("modelInfoToEval adapter wrong: %+v", e.Adapter)
	}
	// Round-trip back through evalInfoToModel preserves the scalars.
	back := evalInfoToModel(e)
	if back.Architecture != info.Architecture || back.QuantBits != info.QuantBits {
		t.Fatalf("evalInfoToModel round-trip lost fields: %+v", back)
	}
}

func TestEval_LoRAToEvalAdapterRoundTrip_Good(t *testing.T) {
	src := lora.AdapterInfo{Name: "n", Path: "/p", Hash: "h", Rank: 4, Alpha: 8, Scale: 2, TargetKeys: []string{"a", "b"}}
	e := loraToEvalAdapter(src)
	if e.Name != "n" || e.Hash != "h" || e.Rank != 4 || len(e.TargetKeys) != 2 {
		t.Fatalf("loraToEvalAdapter = %+v", e)
	}
	// Mutating the clone must not alias the source slice.
	e.TargetKeys[0] = "mutated"
	if src.TargetKeys[0] != "a" {
		t.Fatalf("loraToEvalAdapter aliased source TargetKeys: %v", src.TargetKeys)
	}
	back := evalAdapterToLora(e)
	if back.Name != "n" || back.Rank != 4 {
		t.Fatalf("evalAdapterToLora = %+v", back)
	}
}

func TestEval_DatasetToSFT_NextAndUnknown_Good(t *testing.T) {
	// A dataset.Sample value passes through.
	good := evalDatasetToSFT(wrapSFTDataset(dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}})))
	s, ok, err := good.Next()
	if err != nil || !ok || s.Text != "x" {
		t.Fatalf("evalDatasetToSFT good Next = (%+v,%v,%v)", s, ok, err)
	}
	// A non-dataset.Sample value is rejected with the typed sentinel.
	bad := evalDatasetToSFT(&fakeEvalDataset{value: 1234})
	if _, _, err := bad.Next(); err != errMLXEvalDatasetSampleNotKnown {
		t.Fatalf("evalDatasetToSFT bad Next err = %v, want sample-not-known", err)
	}
}

// fakeEvalDataset yields a single non-dataset.Sample value to drive the
// unknown-sample branch of evalDatasetSFTAdapter.Next.
type fakeEvalDataset struct {
	value any
	done  bool
}

func (f *fakeEvalDataset) Next() (eval.Sample, bool, error) {
	if f.done {
		return nil, false, nil
	}
	f.done = true
	return f.value, true, nil
}

func TestEval_NewModelEvalRunner_InfoAndGuards_Good(t *testing.T) {
	model := &Model{model: &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4", NumLayers: 7}}}
	runner := NewModelEvalRunner(model)

	// Info forwards the model info on a live context.
	if info := runner.Info(context.Background()); info.Architecture != "gemma4" || info.NumLayers != 7 {
		t.Fatalf("runner.Info = %+v, want gemma4/7", info)
	}

	// BuildBatches with a nil tokenizer (fake default) surfaces the tokenizer sentinel.
	_, err := runner.BuildBatches(context.Background(), wrapSFTDataset(dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}})), dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 8})
	if err != errMLXEvalTokenizerNil {
		t.Fatalf("BuildBatches(nil tokenizer) err = %v, want tokenizer-nil", err)
	}

	// EvaluateBatch rejects a non-SFTBatch with the typed sentinel.
	if _, err := runner.EvaluateBatch(context.Background(), "not-a-batch"); err != errMLXEvalBatchNotSFTBatch {
		t.Fatalf("EvaluateBatch(non-sft) err = %v, want not-sft-batch", err)
	}
}

func TestEval_RunModelEval_NilModel_Bad(t *testing.T) {
	if _, err := RunModelEval(context.Background(), nil, nil, eval.Config{}); err != errMLXModelNil {
		t.Fatalf("RunModelEval(nil model) err = %v, want model-nil", err)
	}
}

func TestEval_RunModelEval_DelegatesAndSurfacesInnerError_Bad(t *testing.T) {
	// With a fake (no real tokenizer) RunModelEval's own body runs — probe
	// sizing + delegation — and eval.RunDataset surfaces the tokenizer-nil
	// failure from BuildBatches. The point is the wrapper body executes.
	model := &Model{model: &fakeNativeModel{}}
	ds := dataset.NewSliceDataset([]dataset.Sample{{Text: "hello"}})
	if _, err := RunModelEval(context.Background(), model, ds, eval.Config{Batch: dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 8}}); err == nil {
		t.Fatal("RunModelEval(fake) err = nil, want inner tokenizer failure surfaced")
	}
}
