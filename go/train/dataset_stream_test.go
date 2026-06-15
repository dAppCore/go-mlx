// SPDX-Licence-Identifier: EUPL-1.2

// Tests for BuildDatasetBatches — the entry point lem.sh train calls to turn
// a Dataset into tokenised SFTBatches. Two production paths converge here:
// the non-packing path (delegates to BuildSFTBatches, one example per row,
// padded inside SFT) and the sequence-packing path (the datasetPacker
// concatenates rows up to MaxSeqLen, flushing when the next would overflow).
// Both go through buildSFTExample → tokenizer encode for each row. No model
// is loaded — the fake tokenizer from dataset_stream_bench_test.go drives
// every case, so this stays inside AX-11 (synthetic data, no multi-GB load).
//
// Reuses datasetStreamBenchTokenizer / newDatasetStreamBenchTokenizer /
// datasetStreamBenchSamples / datasetStreamBenchTextSamples from the bench
// file (same package), plus equalIntSlices from sft_test.go.
//
// Run:    go test -tags metal_runtime -run='TestDatasetStream' ./go

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
)

// totalRowsInBatches sums the per-batch row counts (the Length entries) so a
// test can assert "every usable sample landed somewhere" without caring how
// the packer distributed them across batches.
func totalRowsInBatches(batches []SFTBatch) int {
	rows := 0
	for _, b := range batches {
		rows += len(b.Batch.Length)
	}
	return rows
}

// TestDatasetStream_BuildDatasetBatches_NoPack_Good walks the non-packing
// path: each row becomes its own example, grouped into batches of BatchSize.
// 10 rows at BatchSize 4 → 3 batches (4+4+2), every row present.
func TestDatasetStream_BuildDatasetBatches_NoPack_Good(t *testing.T) {
	tok := newDatasetStreamBenchTokenizer()
	ds := dataset.NewSliceDataset(datasetStreamBenchSamples(10))
	cfg := dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 128}

	batches, err := BuildDatasetBatches(tok, ds, cfg)
	if err != nil {
		t.Fatalf("BuildDatasetBatches() error = %v", err)
	}
	if len(batches) != 3 {
		t.Fatalf("batches = %d, want 3 (4+4+2 at BatchSize 4)", len(batches))
	}
	if got := totalRowsInBatches(batches); got != 10 {
		t.Fatalf("rows across batches = %d, want 10", got)
	}
	// Each row in the non-packing path carries the SFT triple: inputs,
	// targets, mask all the same non-zero length, and a Length entry to match.
	first := batches[0]
	if len(first.Batch.Tokens) != 4 || len(first.Targets) != 4 || len(first.Batch.LossMask) != 4 || len(first.Batch.Length) != 4 {
		t.Fatalf("first batch shape = tokens %d / targets %d / mask %d / length %d, want 4 each",
			len(first.Batch.Tokens), len(first.Targets), len(first.Batch.LossMask), len(first.Batch.Length))
	}
	for i := range first.Batch.Tokens {
		if len(first.Batch.Tokens[i]) == 0 {
			t.Fatalf("row %d tokens empty", i)
		}
		if len(first.Batch.Tokens[i]) != len(first.Targets[i]) || len(first.Batch.Tokens[i]) != len(first.Batch.LossMask[i]) {
			t.Fatalf("row %d ragged: tokens %d targets %d mask %d", i,
				len(first.Batch.Tokens[i]), len(first.Targets[i]), len(first.Batch.LossMask[i]))
		}
		if first.Batch.Length[i] != len(first.Batch.Tokens[i]) {
			t.Fatalf("row %d Length = %d, want %d", i, first.Batch.Length[i], len(first.Batch.Tokens[i]))
		}
	}
}

// TestDatasetStream_BuildDatasetBatches_Packed_Good walks the sequence-packing
// path: rows are concatenated until the next would overflow MaxSeqLen, then a
// new packed sequence starts. With a 48-token prompt+response virtual length
// (32+16) and EOS, a tight MaxSeqLen forces several flushes — so the packed
// batch count is strictly fewer than the row count, and every packed sequence
// respects the cap.
func TestDatasetStream_BuildDatasetBatches_Packed_Good(t *testing.T) {
	tok := newDatasetStreamBenchTokenizer()
	ds := dataset.NewSliceDataset(datasetStreamBenchSamples(20))
	// MaxSeqLen 100 holds ~2 rows per packed sequence (each row ~49 tokens),
	// BatchSize 1 so each packed sequence is its own batch.
	cfg := dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 100, SequencePacking: true}

	batches, err := BuildDatasetBatches(tok, ds, cfg)
	if err != nil {
		t.Fatalf("BuildDatasetBatches() error = %v", err)
	}
	if len(batches) == 0 {
		t.Fatal("packed batches = 0, want at least one packed sequence")
	}
	// Packing must compress: 20 rows packed ~2-per-sequence yields fewer than
	// 20 batches — the whole point of the packer.
	if len(batches) >= 20 {
		t.Fatalf("packed batches = %d, want fewer than the 20 rows (packing compresses)", len(batches))
	}
	// Every packed sequence respects MaxSeqLen.
	for bi, b := range batches {
		for ri, seq := range b.Batch.Tokens {
			if len(seq) == 0 {
				t.Fatalf("batch %d row %d empty", bi, ri)
			}
			if len(seq) > cfg.MaxSeqLen {
				t.Fatalf("batch %d row %d length %d exceeds MaxSeqLen %d", bi, ri, len(seq), cfg.MaxSeqLen)
			}
		}
	}
}

// TestDatasetStream_BuildDatasetBatches_TextOnlyAndNormalizeBatch_Good drives
// the free-form text branch (whole sequence is the target) AND the zero-batch
// defaulting (BatchSize 0 → normalizeDatasetBatchConfig coerces to 1).
func TestDatasetStream_BuildDatasetBatches_TextOnlyAndNormalizeBatch_Good(t *testing.T) {
	tok := newDatasetStreamBenchTokenizer()
	ds := dataset.NewSliceDataset(datasetStreamBenchTextSamples(5))
	// BatchSize 0 exercises normalizeDatasetBatchConfig's coercion on the
	// packing path; text rows exercise the buildSFTExampleText branch.
	cfg := dataset.BatchConfig{BatchSize: 0, MaxSeqLen: 256, SequencePacking: true}

	batches, err := BuildDatasetBatches(tok, ds, cfg)
	if err != nil {
		t.Fatalf("BuildDatasetBatches() error = %v", err)
	}
	if got := totalRowsInBatches(batches); got == 0 {
		t.Fatal("text-only packed rows = 0, want the free-form rows packed")
	}
	// Text-path rows are fully masked (every position is a training target),
	// so each mask must contain at least one 1.
	for _, b := range batches {
		for _, mask := range b.Batch.LossMask {
			has := false
			for _, m := range mask {
				if m != 0 {
					has = true
					break
				}
			}
			if !has {
				t.Fatal("text-path packed mask has no training target")
			}
		}
	}
}

// TestDatasetStream_BuildDatasetBatches_NilGuards_Bad asserts the packing-path
// nil guards: a nil tokenizer and a nil dataset are both rejected with an
// error rather than panicking. (The non-packing path delegates to
// BuildSFTBatches, which has its own guards — exercised separately.)
func TestDatasetStream_BuildDatasetBatches_NilGuards_Bad(t *testing.T) {
	tok := newDatasetStreamBenchTokenizer()
	packCfg := dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 64, SequencePacking: true}

	if _, err := BuildDatasetBatches(nil, dataset.NewSliceDataset(datasetStreamBenchSamples(1)), packCfg); err == nil {
		t.Fatal("BuildDatasetBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildDatasetBatches(tok, nil, packCfg); err == nil {
		t.Fatal("BuildDatasetBatches(nil dataset) error = nil, want rejection")
	}
	// The non-packing path must reject the same way (delegates to BuildSFTBatches).
	noPackCfg := dataset.BatchConfig{BatchSize: 2, MaxSeqLen: 64}
	if _, err := BuildDatasetBatches(nil, dataset.NewSliceDataset(datasetStreamBenchSamples(1)), noPackCfg); err == nil {
		t.Fatal("BuildDatasetBatches(nil tok, no-pack) error = nil, want rejection")
	}
}

// TestDatasetStream_BuildDatasetBatches_EmptyAndTightTruncation_Ugly covers two
// degenerate-but-legal shapes: an empty dataset (no rows → no batches, no
// error) and a MaxSeqLen so tight (24) that every row is truncated to the cap
// in datasetPacker.add — the aggressive-packing branch the bench also stresses.
func TestDatasetStream_BuildDatasetBatches_EmptyAndTightTruncation_Ugly(t *testing.T) {
	tok := newDatasetStreamBenchTokenizer()

	// Empty dataset: legal, yields no batches and no error.
	empty, err := BuildDatasetBatches(tok, dataset.NewSliceDataset(nil), dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 128, SequencePacking: true})
	if err != nil {
		t.Fatalf("empty BuildDatasetBatches() error = %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty batches = %d, want 0", len(empty))
	}

	// Tight MaxSeqLen forces truncation: each row's ~49 tokens > 24, so the
	// packer narrows every example to the tail. Output must still respect 24.
	ds := dataset.NewSliceDataset(datasetStreamBenchSamples(8))
	tight, err := BuildDatasetBatches(tok, ds, dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 24, SequencePacking: true})
	if err != nil {
		t.Fatalf("tight BuildDatasetBatches() error = %v", err)
	}
	if len(tight) == 0 {
		t.Fatal("tight-pack batches = 0, want the truncated rows")
	}
	for bi, b := range tight {
		for ri, seq := range b.Batch.Tokens {
			if len(seq) > 24 {
				t.Fatalf("tight batch %d row %d length %d exceeds MaxSeqLen 24", bi, ri, len(seq))
			}
		}
	}
}
