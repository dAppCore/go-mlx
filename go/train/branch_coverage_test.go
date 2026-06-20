// SPDX-Licence-Identifier: EUPL-1.2

// Pure-Go branch coverage for the train package's guard and error arms that
// need no model and no Metal — config defaulting, result-to-error mapping,
// dataset/tokenizer error propagation, sidecar I/O failures, and the packing
// helpers' truncation and nil-guard paths. The real-loss success paths live in
// the metal_runtime-tagged files.
package train

import (
	"context"
	"errors"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// nilImplTokenizer wraps a nil implementation: every Encode call returns the
// "tokenizer is nil" error, exercising the build-example encode-error arms
// without a real tokenizer.
func nilImplTokenizer() *spine.Tokenizer { return spine.NewTokenizer(nil) }

// errDataset is a dataset whose Next always fails.
func errDataset(err error) dataset.Dataset {
	return dataset.Func(func() (dataset.Sample, bool, error) { return dataset.Sample{}, false, err })
}

func ssdNoopGenerate(context.Context, string, spine.GenerateConfig) (string, error) { return "", nil }

// --- sft.go: Metrics learning-rate arm ---

// TestBranch_Metrics_AdamWLearningRate covers the Metrics learning-rate arm that
// reads cfg.AdamW.LearningRate when the top-level LearningRate is unset.
func TestBranch_Metrics_AdamWLearningRate(t *testing.T) {
	r := &SFTResult{Steps: 2}
	m := r.Metrics(SFTConfig{AdamW: metal.AdamWConfig{LearningRate: 3e-4, LearningRateSet: true}})
	if m.LearningRate != 3e-4 {
		t.Fatalf("Metrics LearningRate = %v, want 3e-4 from AdamW config", m.LearningRate)
	}
}

// --- sftResultError ---

// TestBranch_SftResultError covers all three arms: an OK result maps to nil, a
// failed result carrying an error returns that error, and a failed result with a
// non-error value returns the generic fallback.
func TestBranch_SftResultError(t *testing.T) {
	if err := sftResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("sftResultError(OK) = %v, want nil", err)
	}
	wrapped := errors.New("boom")
	if err := sftResultError(core.Result{OK: false, Value: wrapped}); !errors.Is(err, wrapped) {
		t.Fatalf("sftResultError(error value) = %v, want %v", err, wrapped)
	}
	if err := sftResultError(core.Result{OK: false, Value: "not an error"}); err == nil {
		t.Fatal("sftResultError(non-error value) = nil, want generic failure")
	}
}

// --- val.go ---

// TestBranch_BuildSFTValidationBatches_DatasetError surfaces an erroring
// validation dataset out of the subset drain.
func TestBranch_BuildSFTValidationBatches_DatasetError(t *testing.T) {
	cfg := SFTConfig{ValidData: errDataset(errors.New("val read failed"))}
	if _, err := BuildSFTValidationBatches(newSFTBatchTestTokenizer(), cfg); err == nil {
		t.Fatal("BuildSFTValidationBatches over erroring val data = nil, want surfaced error")
	}
}

// --- ssd.go ---

// TestBranch_RunSSD_GuardArms covers RunSSD's nil-dataset / nil-generate
// rejections and a mid-run dataset read error from buildSSDDataset (with ctx nil
// defaulting to Background).
func TestBranch_RunSSD_GuardArms(t *testing.T) {
	// nil dataset rejected (ctx nil → Background default also taken here).
	if _, err := RunSSD(nil, SSDRunner{Generate: ssdNoopGenerate}, nil, SSDConfig{}); err == nil { //nolint:staticcheck // nil ctx is part of the branch
		t.Fatal("RunSSD(nil ds) = nil, want nil-dataset rejection")
	}
	// nil generate rejected.
	if _, err := RunSSD(context.Background(), SSDRunner{}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{}); err == nil {
		t.Fatal("RunSSD(nil generate) = nil, want nil-generate rejection")
	}
	// buildSSDDataset surfaces a dataset read error.
	_, err := RunSSD(context.Background(), SSDRunner{Generate: ssdNoopGenerate}, errDataset(errors.New("ds boom")), SSDConfig{
		SampleTemperature: 1.5,
		SampleMaxTokens:   8,
	})
	if err == nil {
		t.Fatal("RunSSD over erroring dataset = nil, want surfaced read error")
	}
}

// --- sft_epoch.go: epoch flush-closure error propagation (no real loss) ---

// TestBranch_RunSFTDatasetEpoch_FlushErrorPropagation drives the epoch flush
// closures over usable rows with a Model-nil adapter: every flush bottoms out in
// runSFTBatchGroup's nil-loss error, which propagates back up through
// flushAccumulated / flushCurrent / emit. Three shapes exercise the accumulation
// flush, the sub-batch (emit returns nil) path, and the sequence-packing path —
// all without a real loss array (the empty adapter short-circuits before GPU).
func TestBranch_RunSFTDatasetEpoch_FlushErrorPropagation(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	adapter := &metal.LoRAAdapter{} // Model nil → Step returns nil → nil-loss error

	// BatchSize 1: the single usable row flushes immediately → flushAccumulated
	// reports the nil-loss error.
	cfg1 := normalizeSFTConfig(SFTConfig{BatchSize: 1, GradientAccumulationSteps: 1})
	ds1 := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})
	if err := RunSFTDatasetEpoch(context.Background(), epochBranchModel{}, tok, ds1, adapter, nil, cfg1, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch (batch 1, empty adapter) = nil, want nil-loss error surfaced through flush")
	}

	// BatchSize 2 with one row: emit returns nil (no flush yet, line 52), then the
	// terminal flushCurrent flushes and surfaces the nil-loss error.
	cfg2 := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 1})
	ds2 := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})
	if err := RunSFTDatasetEpoch(context.Background(), epochBranchModel{}, tok, ds2, adapter, nil, cfg2, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch (batch 2, one row) = nil, want terminal flush nil-loss error")
	}

	// Sequence packing, terminal flush: one row feeds packer.add, packer.finish
	// flushes the packed example → emit → flush → nil-loss error.
	cfgP := normalizeSFTConfig(SFTConfig{BatchSize: 1, SequencePacking: true, MaxSeqLen: 16})
	dsP := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})
	if err := RunSFTDatasetEpoch(context.Background(), epochBranchModel{}, tok, dsP, adapter, nil, cfgP, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch (packing, empty adapter) = nil, want nil-loss error through packer flush")
	}

	// Sequence packing, mid-add flush: a tight MaxSeqLen makes the second row's
	// packer.add overflow and flush mid-loop → emit → nil-loss error surfaces out
	// of packer.add (not just the terminal finish).
	cfgMid := normalizeSFTConfig(SFTConfig{BatchSize: 1, SequencePacking: true, MaxSeqLen: 8})
	dsMid := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})
	if err := RunSFTDatasetEpoch(context.Background(), epochBranchModel{}, tok, dsMid, adapter, nil, cfgMid, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch (packing mid-add flush) = nil, want nil-loss error out of packer.add")
	}
}

// --- sft_batch.go: build-example encode errors ---

// TestBranch_BuildSFTExample_EncodeErrors drives the prompt/response and the
// whole-text encode-error arms via a nil-implementation tokenizer. BuildSFTBatches
// over the same tokenizer surfaces the error through buildSFTExample too.
func TestBranch_BuildSFTExample_EncodeErrors(t *testing.T) {
	tok := nilImplTokenizer()

	// Prompt/response path: prompt Encode fails first.
	if _, _, err := buildSFTExample(tok, dataset.Sample{Prompt: "p", Response: "r"}, SFTConfig{}); err == nil {
		t.Fatal("buildSFTExample(prompt/response, nil tok) = nil, want encode error")
	}
	// Whole-text path: Text Encode fails.
	if _, _, err := buildSFTExample(tok, dataset.Sample{Text: "t"}, SFTConfig{}); err == nil {
		t.Fatal("buildSFTExample(text, nil tok) = nil, want encode error")
	}
	// BuildSFTBatches surfaces the same error from inside its loop.
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}})
	if _, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: 1}); err == nil {
		t.Fatal("BuildSFTBatches(nil tok) = nil, want surfaced encode error")
	}
	// BuildSFTBatches over a valid tokenizer but an erroring dataset surfaces the
	// ds.Next read error from its loop.
	if _, err := BuildSFTBatches(newSFTBatchTestTokenizer(), errDataset(errors.New("ds boom")), SFTConfig{BatchSize: 1}); err == nil {
		t.Fatal("BuildSFTBatches(erroring ds) = nil, want surfaced read error")
	}
}

// --- dataset_stream.go ---

// TestBranch_BuildDatasetBatches_PackingErrors covers the sequence-packing
// branch of BuildDatasetBatches: a dataset read error and a build-example encode
// error each abort the packing loop.
func TestBranch_BuildDatasetBatches_PackingErrors(t *testing.T) {
	// Dataset read error in the packing loop.
	if _, err := BuildDatasetBatches(newSFTBatchTestTokenizer(), errDataset(errors.New("ds boom")), dataset.BatchConfig{BatchSize: 1, SequencePacking: true, MaxSeqLen: 8}); err == nil {
		t.Fatal("BuildDatasetBatches(packing, erroring ds) = nil, want surfaced error")
	}
	// Build-example encode error in the packing loop (nil-impl tokenizer).
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}})
	if _, err := BuildDatasetBatches(nilImplTokenizer(), ds, dataset.BatchConfig{BatchSize: 1, SequencePacking: true, MaxSeqLen: 8}); err == nil {
		t.Fatal("BuildDatasetBatches(packing, nil tok) = nil, want surfaced encode error")
	}
}

// TestBranch_DatasetPacker_GuardsAndTruncation covers the datasetPacker's
// nil/empty guards and the oversized-example truncation arm.
func TestBranch_DatasetPacker_GuardsAndTruncation(t *testing.T) {
	// nil packer and nil-builder packer are both no-ops.
	(*datasetPacker)(nil).add(&sftExample{inputs: []int{1}})
	emptyBuilderPacker := &datasetPacker{}
	emptyBuilderPacker.add(&sftExample{inputs: []int{1}})
	// nil example and empty-inputs example are skipped.
	builder := newSFTBatchBuilder(1)
	packer := newDatasetPacker(4, builder)
	packer.add(nil)
	packer.add(&sftExample{})
	// An oversized example (> maxSeqLen) is truncated to its tail.
	packer.add(&sftExample{
		inputs:  []int{1, 2, 3, 4, 5, 6},
		targets: []int{2, 3, 4, 5, 6, 7},
		mask:    []float32{1, 1, 1, 1, 1, 1},
	})
	packer.finish()
	batches := builder.finish()
	if len(batches) == 0 {
		t.Fatal("packer produced no batch after an oversized add")
	}
	if got := len(batches[0].Batch.Tokens[0]); got != 4 {
		t.Fatalf("truncated packed length = %d, want 4 (maxSeqLen tail)", got)
	}
}

// --- score_cascade.go ---

// TestBranch_ScoreCascade_AppendSidecarAndComposite covers appendSidecar's
// empty-path and step-mismatch/empty-output early returns, and compositeAt's
// nil/empty and no-matching-step returns.
func TestBranch_ScoreCascade_AppendSidecarAndComposite(t *testing.T) {
	// appendSidecar with no sidecar path is a no-op (empty-path arm).
	noPath := newSFTScoreCascade("", 0)
	noPath.recordPass(1, []SFTEvalResult{{Step: 1, Prompt: "p", Text: "calm and present"}})

	// With a real sidecar path, recording at step 1 writes those rows. A later
	// explicit appendSidecar(step) for a step that matches no record walks every
	// record hitting the Step-mismatch continue, leaves the output empty, and
	// returns early — exercising the per-record skip and empty-output arms.
	sidecar := core.PathJoin(t.TempDir(), "score-cascade.jsonl")
	withPath := newSFTScoreCascade(sidecar, 0)
	withPath.recordPass(1, []SFTEvalResult{{Step: 1, Prompt: "p", Text: "calm and present"}})
	withPath.appendSidecar(999)

	// compositeAt: nil receiver and empty perStep both read as 0.
	if got := (*sftScoreCascade)(nil).compositeAt(1); got != 0 {
		t.Fatalf("compositeAt(nil) = %v, want 0", got)
	}
	empty := newSFTScoreCascade("", 0)
	if got := empty.compositeAt(1); got != 0 {
		t.Fatalf("compositeAt(empty) = %v, want 0", got)
	}
	// A cascade with a pass at step 5, queried below it (step 1), finds no
	// matching window → 0.
	scored := newSFTScoreCascade("", 0)
	scored.recordPass(5, []SFTEvalResult{{Step: 5, Prompt: "p", Text: "still and aware"}})
	if got := scored.compositeAt(1); got != 0 {
		t.Fatalf("compositeAt(step before any pass) = %v, want 0", got)
	}
	// sftScoreCompositeAt nil-result and nil-cascade guards.
	if got := sftScoreCompositeAt(nil, 1); got != 0 {
		t.Fatalf("sftScoreCompositeAt(nil) = %v, want 0", got)
	}
	if got := sftScoreCompositeAt(&SFTResult{}, 1); got != 0 {
		t.Fatalf("sftScoreCompositeAt(no cascade) = %v, want 0", got)
	}
	// With a cascade armed, sftScoreCompositeAt delegates to compositeAt.
	withCascade := &SFTResult{cascade: scored}
	_ = sftScoreCompositeAt(withCascade, 5) // delegates; value asserted via compositeAt above

	// appendSidecar's Append-failure arm: a sidecar path whose parent is a
	// regular file cannot be opened for append, so recordPass's write is a silent
	// best-effort no-op (no panic, records still held in memory).
	blocker := core.PathJoin(t.TempDir(), "afile")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	badSidecar := newSFTScoreCascade(core.PathJoin(blocker, "score.jsonl"), 0)
	badSidecar.recordPass(1, []SFTEvalResult{{Step: 1, Prompt: "p", Text: "held and steady"}})
	if len(badSidecar.records) != 1 {
		t.Fatalf("records after failed-append recordPass = %d, want 1 (held in memory)", len(badSidecar.records))
	}
}

// --- capture.go ---

// TestBranch_AppendCaptureRows_AppendFailure covers the capture sidecar's
// Append-failure arm: a path whose parent is a regular file cannot be opened for
// append, so the function reports zero rows landed (best-effort, no error).
func TestBranch_AppendCaptureRows_AppendFailure(t *testing.T) {
	blocker := core.PathJoin(t.TempDir(), "afile")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	bad := core.PathJoin(blocker, "captures.jsonl")
	rows := appendCaptureRows(bad, []SFTEvalResult{{Step: 1, Prompt: "p", Text: "t"}})
	if rows != 0 {
		t.Fatalf("appendCaptureRows(unwritable) = %d rows, want 0 (append failed)", rows)
	}
}

// --- sft_checkpoint.go ---

// TestBranch_SaveSFTCheckpointMetadata_DirError covers the metadata save's
// MkdirAll-failure arm: the metadata path's directory cannot be created because
// its parent is a regular file.
func TestBranch_SaveSFTCheckpointMetadata_DirError(t *testing.T) {
	blocker := core.PathJoin(t.TempDir(), "afile")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	// metadata path lives under <blocker>/sub/, which MkdirAll cannot create.
	dest := core.PathJoin(blocker, "sub")
	if err := SaveSFTCheckpointMetadata(dest, SFTCheckpointMetadata{Step: 1}); err == nil {
		t.Fatal("SaveSFTCheckpointMetadata under a file-parent = nil, want MkdirAll failure")
	}

	// WriteFile-failure arm: the metadata file path is itself an existing
	// directory, so the dir is fine but the write fails.
	dir := t.TempDir()
	if r := core.MkdirAll(sftCheckpointMetadataPath(dir), 0o755); !r.OK {
		t.Fatalf("setup metadata-file-as-dir: %v", r.Value)
	}
	if err := SaveSFTCheckpointMetadata(dir, SFTCheckpointMetadata{Step: 1}); err == nil {
		t.Fatal("SaveSFTCheckpointMetadata onto a directory file-path = nil, want WriteFile failure")
	}
}

// TestBranch_SFTAdamWConfig_EpsArm covers the SFTAdamWConfig optimiser-config arm
// that applies a set Eps override.
func TestBranch_SFTAdamWConfig_EpsArm(t *testing.T) {
	cfg := SFTConfig{AdamW: metal.AdamWConfig{Eps: 1e-9, EpsSet: true}}
	adam := SFTAdamWConfig(cfg)
	if adam.Eps != 1e-9 {
		t.Fatalf("SFTAdamWConfig Eps = %v, want 1e-9 override applied", adam.Eps)
	}
}

// TestBranch_LoadSFTResumeMetadata_ReadErrorAndVersionDefault covers
// loadSFTResumeMetadata's non-not-exist read-error arm and the version-zero
// backfill on a successfully parsed metadata file.
func TestBranch_LoadSFTResumeMetadata_ReadErrorAndVersionDefault(t *testing.T) {
	// A directory at the metadata-file location makes ReadFile fail with a
	// non-not-exist error (not silently treated as absent).
	dir := t.TempDir()
	metaAsDir := sftCheckpointMetadataPath(dir)
	if r := core.MkdirAll(metaAsDir, 0o755); !r.OK {
		t.Fatalf("setup metadata-as-dir: %v", r.Value)
	}
	if _, err := loadSFTResumeMetadata(dir); err == nil {
		t.Fatal("loadSFTResumeMetadata over a directory = nil, want non-not-exist read error")
	}

	// A valid metadata file with version 0 is parsed and the version backfilled.
	dir2 := t.TempDir()
	if err := SaveSFTCheckpointMetadata(dir2, SFTCheckpointMetadata{Version: 0, Step: 7}); err != nil {
		t.Fatalf("save metadata: %v", err)
	}
	// Overwrite with an explicit version:0 JSON so the backfill arm fires.
	if w := core.WriteFile(sftCheckpointMetadataPath(dir2), []byte(`{"version":0,"step":7}`), 0o600); !w.OK {
		t.Fatalf("write version-0 metadata: %v", w.Value)
	}
	meta, err := loadSFTResumeMetadata(dir2)
	if err != nil {
		t.Fatalf("loadSFTResumeMetadata(version 0) error = %v", err)
	}
	if meta == nil || meta.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("resume metadata version = %+v, want backfilled to %d", meta, SFTCheckpointMetadataVersion)
	}
}
