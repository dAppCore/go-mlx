// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// These tests drive the LoRA SFT step machinery against a real Metal device —
// runSFTBatchGroup's success path (Materialize loss, step bookkeeping, checkpoint
// save, probe emit, validation, score cascade) and the full RunSFTDatasetEpoch
// chain need a non-nil *metal.Array loss, which only a real adapter produces. The
// adapter here is the smallest possible real one: a single shape-adaptive LoRA
// layer over a toy InternalModel, the same pattern pkg/metal/lora_test.go uses for
// its own Step tests. This is functional verification, not a model load (AX-11) —
// one trainable scalar, a handful of tokens, runs in milliseconds. The file is
// gated so the untagged `go test ./train/` stays pure-Go.
package train

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/spine"
)

// epochToyVocab is the toy model's vocabulary — the smoke tokenizer below emits
// only IDs < this, so MaskedCrossEntropyLoss's gather stays in bounds.
const epochToyVocab = 8

// epochToyModel is a minimal InternalModel whose logits carry one trainable LoRA
// scalar at vocab position 0, broadcast across the input's [B, L] shape. The
// scalar flows into the loss so valueAndGrad returns real gradients, while the
// shape-adaptive logits tolerate any batch the SFT tokenizer produces.
type epochToyModel struct {
	layer *metal.LoRALinear
}

func (m *epochToyModel) Forward(tokens *metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *epochToyModel) ForwardMasked(tokens *metal.Array, _ *metal.Array, _ []metal.Cache) *metal.Array {
	shape := tokens.Shape()
	b, l := shape[0], shape[1]
	// Trainable scalar at vocab slot 0; remaining slots zero. Reshape to a
	// [1,1,V] row then broadcast over the batch's [B, L] positions.
	scalar := metal.Add(m.layer.A, m.layer.B) // [1,1]
	tail := metal.Zeros([]int32{1, epochToyVocab - 1}, metal.DTypeFloat32)
	row := metal.Concatenate([]*metal.Array{scalar, tail}, 1) // [1,V]
	rowLogits := metal.Reshape(row, 1, 1, epochToyVocab)
	logits := metal.BroadcastTo(rowLogits, []int32{b, l, epochToyVocab})
	metal.Free(scalar, tail, row, rowLogits)
	return logits
}

func (m *epochToyModel) NewCache() []metal.Cache                         { return nil }
func (m *epochToyModel) NumLayers() int                                  { return 1 }
func (m *epochToyModel) Tokenizer() *metal.Tokenizer                     { return nil }
func (m *epochToyModel) ModelType() string                               { return "epoch-toy" }
func (m *epochToyModel) ApplyLoRA(_ metal.LoRAConfig) *metal.LoRAAdapter { return nil }

// newEpochToyAdapter returns a real one-layer LoRA adapter plus its trainable
// arrays (for cleanup). The optimiser learning rate defaults are irrelevant —
// the tests assert bookkeeping, not convergence.
func newEpochToyAdapter() (*metal.LoRAAdapter, *metal.LoRALinear) {
	layer := &metal.LoRALinear{
		A:     metal.FromValues([]float32{0.25}, 1, 1),
		B:     metal.FromValues([]float32{0.5}, 1, 1),
		Scale: 1,
		Rank:  1,
		Alpha: 1,
	}
	adapter := &metal.LoRAAdapter{
		Layers: map[string]*metal.LoRALinear{"model.layers.0.self_attn.q_proj": layer},
		Config: metal.LoRAConfig{},
		Model:  &epochToyModel{layer: layer},
	}
	return adapter, layer
}

// smokeEpochModel satisfies the train.Model interface for the eval hook.
type smokeEpochModel struct {
	info spine.ModelInfo
	gen  string
}

func (m smokeEpochModel) ModelType() string     { return "smoke-epoch" }
func (m smokeEpochModel) Info() spine.ModelInfo  { return m.info }
func (m smokeEpochModel) Generate(string, ...spine.GenerateOption) (string, error) {
	return m.gen, nil
}

// epochSmokeTokenizer emits small token IDs (< epochToyVocab) so the toy model's
// gather stays in range across a real example build.
type epochSmokeTokenizer struct{}

func (epochSmokeTokenizer) Encode(s string) []int32 {
	switch s {
	case "p":
		return []int32{0, 1}
	case "r":
		return []int32{2, 3}
	}
	return []int32{1}
}
func (epochSmokeTokenizer) Decode([]int32) string    { return "" }
func (epochSmokeTokenizer) DecodeOne(int32) string   { return "" }
func (epochSmokeTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (epochSmokeTokenizer) IDToken(int32) string     { return "" }
func (epochSmokeTokenizer) BOS() int32               { return 0 }
func (epochSmokeTokenizer) EOS() int32               { return 4 }
func (epochSmokeTokenizer) HasBOSToken() bool        { return false }

func newEpochSmokeTokenizer() *spine.Tokenizer {
	return spine.NewTokenizer(epochSmokeTokenizer{})
}

// TestSftEpochMetal_RunSFTBatchGroup_SuccessPath drives runSFTBatchGroup over a
// hand-built minimal batch with a real adapter — the whole success path past the
// nil-loss guard: a real loss materialises, step bookkeeping advances, a
// checkpoint lands, validation runs, and the probe sink fires a training event.
func TestSftEpochMetal_RunSFTBatchGroup_SuccessPath(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	dir := t.TempDir()
	sink := &collectingSink{}
	cfg := normalizeSFTConfig(SFTConfig{
		BatchSize:       1,
		CheckpointDir:   dir,
		CheckpointEvery: 1,
		ProbeSink:       sink,
	})
	result := &SFTResult{}
	batch := SFTBatch{
		Batch:   metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}},
		Targets: [][]int{{1}},
	}
	m := smokeEpochModel{}
	if err := runSFTBatchGroup(context.Background(), m, []SFTBatch{batch}, adapter, opt, cfg, result, 1); err != nil {
		t.Fatalf("runSFTBatchGroup() success path error = %v", err)
	}
	if result.Steps != 1 {
		t.Fatalf("Steps = %d, want 1", result.Steps)
	}
	if result.OptimizerSteps != 1 {
		t.Fatalf("OptimizerSteps = %d, want 1", result.OptimizerSteps)
	}
	if len(result.Losses) != 1 {
		t.Fatalf("Losses = %d, want 1", len(result.Losses))
	}
	if len(result.Checkpoints) != 1 {
		t.Fatalf("Checkpoints = %d, want 1 (CheckpointEvery=1)", len(result.Checkpoints))
	}
	if len(result.CheckpointMetadata) != 1 {
		t.Fatalf("CheckpointMetadata = %d, want 1", len(result.CheckpointMetadata))
	}
	if len(sink.events) == 0 {
		t.Fatal("probe sink received no events, want a training event")
	}
}

// collectingSink records every probe event for assertions.
type collectingSink struct {
	events []probe.Event
}

func (s *collectingSink) EmitProbe(ev probe.Event) { s.events = append(s.events, ev) }

// TestSftEpochMetal_RunSFTDatasetEpoch_SuccessPath drives the whole epoch chain
// over a usable dataset with a real adapter: tokenize → example → accumulate →
// flush → real Step → bookkeeping. With BatchSize 1 and one row, exactly one
// optimiser step fires and Samples counts the row.
func TestSftEpochMetal_RunSFTDatasetEpoch_SuccessPath(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	tok := newEpochSmokeTokenizer()
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 1, GradientAccumulationSteps: 1})
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p", Response: "r"}})
	result := &SFTResult{}
	if err := RunSFTDatasetEpoch(context.Background(), smokeEpochModel{}, tok, ds, adapter, opt, cfg, result, 1); err != nil {
		t.Fatalf("RunSFTDatasetEpoch() success path error = %v", err)
	}
	if result.Samples != 1 {
		t.Fatalf("Samples = %d, want 1", result.Samples)
	}
	if result.Steps != 1 {
		t.Fatalf("Steps = %d, want 1 (one batch flushed)", result.Steps)
	}
}

// TestSftEpochMetal_RunSFTDatasetEpoch_Packing drives the sequence-packing branch
// of the epoch loop: cfg.SequencePacking constructs the streaming packer, usable
// rows feed packer.add, and packer.finish flushes the packed example through the
// real Step. One packed batch → one optimiser step.
func TestSftEpochMetal_RunSFTDatasetEpoch_Packing(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	tok := newEpochSmokeTokenizer()
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 1, SequencePacking: true, MaxSeqLen: 16})
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p", Response: "r"},
		{Prompt: "p", Response: "r"},
	})
	result := &SFTResult{}
	if err := RunSFTDatasetEpoch(context.Background(), smokeEpochModel{}, tok, ds, adapter, opt, cfg, result, 1); err != nil {
		t.Fatalf("RunSFTDatasetEpoch() packing path error = %v", err)
	}
	if result.Samples != 2 {
		t.Fatalf("Samples = %d, want 2", result.Samples)
	}
	if result.Steps == 0 {
		t.Fatal("Steps = 0, want >=1 (packed batch flushed through Step)")
	}
}

// TestSftEpochMetal_RunSFTDatasetEpoch_GradAccum drives gradient accumulation: two
// rows at BatchSize 1, GradientAccumulationSteps 2 accumulate into one micro-batch
// group, so flushAccumulated routes both batches through StepAccumulated for a
// single optimiser step.
func TestSftEpochMetal_RunSFTDatasetEpoch_GradAccum(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	tok := newEpochSmokeTokenizer()
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 1, GradientAccumulationSteps: 2})
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p", Response: "r"},
		{Prompt: "p", Response: "r"},
	})
	result := &SFTResult{}
	if err := RunSFTDatasetEpoch(context.Background(), smokeEpochModel{}, tok, ds, adapter, opt, cfg, result, 1); err != nil {
		t.Fatalf("RunSFTDatasetEpoch() grad-accum error = %v", err)
	}
	if result.Samples != 2 {
		t.Fatalf("Samples = %d, want 2", result.Samples)
	}
	if result.Steps != 1 {
		t.Fatalf("Steps = %d, want 1 (two micro-batches, one optimiser step)", result.Steps)
	}
}

// TestSftEpochMetal_RunSFTBatchGroup_EvalError reaches the eval-error arm inside
// the batch-group success block: a real loss gets past the nil-loss guard, then
// runSFTEvaluations fires (EvalEvery 1) and its Generate error surfaces out of
// the group.
func TestSftEpochMetal_RunSFTBatchGroup_EvalError(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	cfg := normalizeSFTConfig(SFTConfig{
		BatchSize:   1,
		EvalEvery:   1,
		EvalPrompts: []string{"probe"},
	})
	result := &SFTResult{}
	batch := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}}, Targets: [][]int{{1}}}
	m := smokeEpochErrModel{}
	if err := runSFTBatchGroup(context.Background(), m, []SFTBatch{batch}, adapter, opt, cfg, result, 1); err == nil {
		t.Fatal("runSFTBatchGroup with failing eval = nil, want surfaced generate error")
	}
}

// smokeEpochErrModel is a train.Model whose Generate always errors — for the
// eval-error arm.
type smokeEpochErrModel struct{}

func (smokeEpochErrModel) ModelType() string    { return "smoke-epoch-err" }
func (smokeEpochErrModel) Info() spine.ModelInfo { return spine.ModelInfo{} }
func (smokeEpochErrModel) Generate(string, ...spine.GenerateOption) (string, error) {
	return "", context.DeadlineExceeded
}

// TestSftEpochMetal_RunSFTBatchGroup_CheckpointSaveError reaches the adapter.Save
// error arm: a real loss gets past the nil-loss guard, the checkpoint block runs
// (CheckpointEvery 1), but CheckpointDir is a regular FILE so the per-step subdir
// save fails and the error surfaces.
func TestSftEpochMetal_RunSFTBatchGroup_CheckpointSaveError(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	// A file where a directory is expected: the step-subdir write under it fails.
	fileAsDir := core.PathJoin(t.TempDir(), "notadir")
	if w := core.WriteFile(fileAsDir, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup: write blocker file: %v", w.Value)
	}
	cfg := normalizeSFTConfig(SFTConfig{
		BatchSize:       1,
		CheckpointDir:   fileAsDir,
		CheckpointEvery: 1,
	})
	result := &SFTResult{}
	batch := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}}, Targets: [][]int{{1}}}
	if err := runSFTBatchGroup(context.Background(), smokeEpochModel{}, []SFTBatch{batch}, adapter, opt, cfg, result, 1); err == nil {
		t.Fatal("runSFTBatchGroup with unwritable checkpoint dir = nil, want save error")
	}
}

// TestSftEpochMetal_RunSFTBatchGroup_ValidationError reaches the validation-error
// arm: the validation lane is armed with a failing loss fn (injected, no Metal of
// its own), a real loss gets past the nil-loss guard, and the val pass at step 1
// fails — surfacing before checkpointing.
func TestSftEpochMetal_RunSFTBatchGroup_ValidationError(t *testing.T) {
	adapter, layer := newEpochToyAdapter()
	defer metal.Free(layer.A, layer.B)
	opt := metal.NewAdamW(&metal.AdamWConfig{LearningRate: 0})

	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 1})
	result := &SFTResult{}
	valBatch := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}}, Targets: [][]int{{1}}}
	ArmSFTValidation(result, []SFTBatch{valBatch}, 1, func(SFTBatch) (float64, bool) { return 0, false })
	batch := SFTBatch{Batch: metal.Batch{Tokens: [][]int{{0}}, Length: []int{1}}, Targets: [][]int{{1}}}
	if err := runSFTBatchGroup(context.Background(), smokeEpochModel{}, []SFTBatch{batch}, adapter, opt, cfg, result, 1); err == nil {
		t.Fatal("runSFTBatchGroup with failing validation = nil, want surfaced val error")
	}
}
