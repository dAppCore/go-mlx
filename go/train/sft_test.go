// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"errors"
	"testing"

	core "dappco.re/go"

	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/spine"
)

// sftTestModel is the minimal train.Model fake: records the last Generate
// prompt and returns the seeded text.
type sftTestModel struct {
	info       spine.ModelInfo
	text       string
	lastPrompt string
}

func (m *sftTestModel) ModelType() string     { return "test" }
func (m *sftTestModel) Info() spine.ModelInfo { return m.info }
func (m *sftTestModel) Generate(prompt string, opts ...spine.GenerateOption) (string, error) {
	m.lastPrompt = prompt
	return m.text, nil
}

func equalIntSlices(a, b []int) bool {
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

func equalStringSlices(a, b []string) bool {
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

func TestSFTDatasetEpoch_EmptyErrorAndCancelledBranches_Bad(t *testing.T) {
	result := &SFTResult{}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2})
	if err := RunSFTDatasetEpoch(context.Background(), nil, nil, dataset.NewSliceDataset(nil), nil, nil, cfg, result, 1); err != nil {
		t.Fatalf("empty epoch error = %v", err)
	}
	if result.Samples != 0 {
		t.Fatalf("empty epoch samples = %d, want 0", result.Samples)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := RunSFTDatasetEpoch(cancelled, nil, nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled epoch error = %v, want context.Canceled", err)
	}
	if err := runSFTBatchGroup(cancelled, nil, nil, nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch group error = %v, want context.Canceled", err)
	}
}

func TestSFTEvalPrompts_Gemma4LargeVariantUsesSharedFormatter_Good(t *testing.T) {
	model := &sftTestModel{
		info: spine.ModelInfo{Architecture: "Gemma4ForConditionalGeneration", NumHeads: 16},
		text: "ok",
	}
	result := &SFTResult{Steps: 1}
	cfg := NormalizeSFTConfigForModel(SFTConfig{
		EvalEvery:     1,
		EvalPrompts:   []string{"Write one line."},
		EvalMaxTokens: 8,
	}, model.Info())

	if err := runSFTEvaluations(context.Background(), model, cfg, result); err != nil {
		t.Fatalf("runSFTEvaluations() error = %v", err)
	}

	wantPrompt := chat.Format([]chat.Message{{Role: "user", Content: "Write one line."}}, chat.Config{
		Architecture:   "Gemma4ForConditionalGeneration",
		EnableThinking: true,
		LargeVariant:   true,
	})
	if model.lastPrompt != wantPrompt {
		t.Fatalf("Generate prompt = %q, want shared Gemma4 formatter %q", model.lastPrompt, wantPrompt)
	}
	if len(result.Evaluations) != 1 || result.Evaluations[0].Prompt != "Write one line." || result.Evaluations[0].Text != "ok" {
		t.Fatalf("Evaluations = %+v, want original prompt identity and generated text", result.Evaluations)
	}
}

// --- merged from sft_runner_test.go (Track A: tests match their source file) ---

func TestSFTStreamingPacker_Good(t *testing.T) {
	var emitted []sftExample
	packer := newSFTStreamingPacker(4, func(example sftExample) error {
		emitted = append(emitted, example)
		return nil
	})

	if err := packer.add(sftExample{
		inputs:  []int{1, 2},
		targets: []int{2, 3},
		mask:    []float32{0, 1},
	}); err != nil {
		t.Fatalf("add first: %v", err)
	}
	if err := packer.add(sftExample{
		inputs:  []int{3, 4, 5},
		targets: []int{4, 5, 6},
		mask:    []float32{1, 1, 1},
	}); err != nil {
		t.Fatalf("add second: %v", err)
	}
	if err := packer.add(sftExample{
		inputs:  []int{6, 7, 8, 9, 10},
		targets: []int{7, 8, 9, 10, 11},
		mask:    []float32{1, 1, 1, 1, 1},
	}); err != nil {
		t.Fatalf("add long: %v", err)
	}
	if err := packer.finish(); err != nil {
		t.Fatalf("finish: %v", err)
	}

	if len(emitted) != 3 {
		t.Fatalf("emitted len = %d, want 3", len(emitted))
	}
	if !equalIntSlices(emitted[0].inputs, []int{1, 2}) {
		t.Fatalf("first packed inputs = %v, want [1 2]", emitted[0].inputs)
	}
	if !equalIntSlices(emitted[1].inputs, []int{3, 4, 5}) {
		t.Fatalf("second packed inputs = %v, want [3 4 5]", emitted[1].inputs)
	}
	if !equalIntSlices(emitted[2].inputs, []int{7, 8, 9, 10}) {
		t.Fatalf("trimmed packed inputs = %v, want last four tokens", emitted[2].inputs)
	}
	if len(packer.current.inputs) != 0 {
		t.Fatalf("packer current = %+v, want flushed", packer.current)
	}
}

func TestSFTStreamingPacker_BadAndHelpers(t *testing.T) {
	if err := (*sftStreamingPacker)(nil).finish(); err != nil {
		t.Fatalf("nil finish error = %v", err)
	}
	if err := (*sftStreamingPacker)(nil).add(sftExample{inputs: []int{1}}); err != nil {
		t.Fatalf("nil add error = %v", err)
	}
	packer := newSFTStreamingPacker(8, nil)
	if err := packer.add(sftExample{inputs: []int{1}}); err != nil {
		t.Fatalf("nil emit add error = %v", err)
	}
	if err := packer.flush(); err != nil {
		t.Fatalf("empty flush error = %v", err)
	}

	wantErr := errors.New("emit failed")
	packer = newSFTStreamingPacker(8, func(sftExample) error { return wantErr })
	if err := packer.add(sftExample{inputs: []int{1}, targets: []int{2}, mask: []float32{1}}); err != nil {
		t.Fatalf("add before failing flush error = %v", err)
	}
	if err := packer.finish(); !errors.Is(err, wantErr) {
		t.Fatalf("finish error = %v, want %v", err, wantErr)
	}

	if loss := sftAdapterStep(nil, nil, nil); loss != nil {
		t.Fatalf("sftAdapterStep(empty) = %+v, want nil", loss)
	}
	if sink := sftProbeSink(SFTConfig{ProbeSink: probe.NewRecorder()}); sink == nil {
		t.Fatal("sftProbeSink did not prefer direct SFT probe sink")
	}
	if sink := sftProbeSink(SFTConfig{LoRA: spine.LoRAConfig{ProbeSink: probe.NewRecorder()}}); sink == nil {
		t.Fatal("sftProbeSink did not fall back to LoRA probe sink")
	}
}

func TestSFT_Gemma4ArchitectureUsesProfileArchitectureID_Good(t *testing.T) {
	cases := map[string]bool{
		"gemma4":                                true,
		"gemma4_text":                           true,
		"gemma4_unified":                        true,
		"gemma4_unified_text":                   true,
		"Gemma4ForConditionalGeneration":        true,
		"Gemma4UnifiedForConditionalGeneration": true,
		"Gemma4ForCausalLM":                     true,
		"Gemma4TextForCausalLM":                 true,
		"Gemma4AssistantForCausalLM":            false,
		"gemma4_assistant":                      false,
		"gemma3":                                false,
		"qwen3":                                 false,
		"":                                      false,
	}
	for arch, want := range cases {
		if got := profile.IsGemma4TargetArchitecture(arch); got != want {
			t.Fatalf("profile.IsGemma4TargetArchitecture(%q) = %v, want %v", arch, got, want)
		}
	}
}

func TestSFTEvalGenerateOptions_CarriesTemperature_Good(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{EvalMaxTokens: 64, EvalTemperature: 0.35})
	opts := sftEvalGenerateOptions(cfg)
	applied := spine.ApplyGenerateOptions(opts)
	if applied.MaxTokens != 64 || applied.Temperature != 0.35 {
		t.Fatalf("eval generate config = %+v, want max tokens and temperature", applied)
	}
}

func TestSFTAdapterArtifactMetadata_Good(t *testing.T) {
	result := &SFTResult{Steps: 3, Samples: 5, LastLoss: 0.25}
	cfg := normalizeSFTConfig(SFTConfig{
		SavePath:                  core.PathJoin(t.TempDir(), "adapter"),
		BatchSize:                 2,
		GradientAccumulationSteps: 4,
		LearningRate:              1e-4,
		EvalTemperature:           0.25,
		LoRA: spine.LoRAConfig{
			Rank:                 8,
			Alpha:                16,
			TargetKeys:           []string{"q_proj"},
			AllowExtendedTargets: true,
		},
	})

	meta := NewSFTArtifactMetadata(cfg.SavePath, "gemma4", cfg, result)
	if meta.Path != cfg.SavePath || meta.Step != 3 || meta.Samples != 5 {
		t.Fatalf("artifact metadata = %+v, want final adapter state", meta)
	}
	if meta.GradientAccumulationSteps != 4 || meta.EvalTemperature != 0.25 || meta.LoRA.Rank != 8 || !meta.LoRA.AllowExtendedTargets || meta.Model != "gemma4" {
		t.Fatalf("artifact metadata = %+v, want config attached", meta)
	}
}

func TestSFTAdamWConfig_UsesExplicitOptimizer_Bad(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{
		AdamW: metal.AdamWConfig{
			LearningRate:   3e-4,
			Beta1:          0.85,
			Beta2:          0.98,
			WeightDecay:    0,
			WeightDecaySet: true,
			PackedState:    false,
			PackedStateSet: true,
		},
	})

	adam := SFTAdamWConfig(cfg)
	if adam.LearningRate != 3e-4 || adam.Beta1 != 0.85 || adam.Beta2 != 0.98 || adam.WeightDecay != 0 || adam.PackedState {
		t.Fatalf("adam = %+v, want explicit optimizer config", adam)
	}
	meta := sftAdamWMetadata(adam)
	if meta.PackedState {
		t.Fatalf("adam metadata = %+v, want explicit packed-state setting", meta)
	}
}

func TestNormalizeSFTConfig_DefaultsLoRA_Ugly(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{})
	meta := sftLoRAMetadata(cfg.LoRA)
	if meta.Rank != 8 || meta.Alpha != 16 || !equalStringSlices(meta.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("lora metadata = %+v, want default adapter identity", meta)
	}
}

// --- BuildSFTBatches / BuildSFTTrainingBatches — the data-loading entry points ---

// sftBatchTestTokenizer maps prompt/response/text strings to caller-chosen
// token IDs so a test can drive precise lengths without a model. Mirrors the
// shape of buildExampleTestTokenizer but with a fixed three-key vocab so the
// batch-building tests read clearly. (Distinct name — buildExampleTestTokenizer
// is owned by sft_buildexample_test.go.)
type sftBatchTestTokenizer struct {
	prompt   []int32
	response []int32
	text     []int32
	eos      int32
}

func (t sftBatchTestTokenizer) Encode(s string) []int32 {
	switch s {
	case "prompt":
		return append([]int32(nil), t.prompt...)
	case "response":
		return append([]int32(nil), t.response...)
	case "text":
		return append([]int32(nil), t.text...)
	}
	return nil
}
func (sftBatchTestTokenizer) Decode([]int32) string        { return "" }
func (sftBatchTestTokenizer) DecodeOne(int32) string       { return "" }
func (sftBatchTestTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (sftBatchTestTokenizer) IDToken(int32) string         { return "" }
func (sftBatchTestTokenizer) BOS() int32                   { return 0 }
func (t sftBatchTestTokenizer) EOS() int32                 { return t.eos }
func (sftBatchTestTokenizer) HasBOSToken() bool            { return false }

func newSFTBatchTestTokenizer() *spine.Tokenizer {
	return spine.NewTokenizer(sftBatchTestTokenizer{
		prompt:   makeIDs(100, 4),
		response: makeIDs(500, 3),
		text:     makeIDs(900, 5),
		eos:      9,
	})
}

// TestSFT_BuildSFTBatches_Good builds batches from a small prompt/response
// dataset and asserts the response-masked triple is correct: inputs/targets
// are V[0..n)/V[1..n+1) over prompt|response|EOS, the mask is 0 across the
// prompt region and 1 over the response+EOS region, and rows group into
// BatchSize batches. Drives the unexported batch builder + sftBatchFromExamples
// transitively (output asserted, not memory layout — the unsafe-slice share is
// a deliberate alloc optimisation).
func TestSFT_BuildSFTBatches_Good(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})

	batches, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: 2})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	// 3 rows at BatchSize 2 → batches of 2 + 1.
	if len(batches) != 2 {
		t.Fatalf("batches = %d, want 2 (2+1 at BatchSize 2)", len(batches))
	}
	if len(batches[0].Batch.Tokens) != 2 || len(batches[1].Batch.Tokens) != 1 {
		t.Fatalf("batch sizes = %d/%d, want 2/1", len(batches[0].Batch.Tokens), len(batches[1].Batch.Tokens))
	}

	// Virtual sequence V = [100 101 102 103][500 501 502][EOS=9], len 8 → n=7.
	wantInputs := []int{100, 101, 102, 103, 500, 501, 502}
	wantTargets := []int{101, 102, 103, 500, 501, 502, 9}
	// promptLen=4 → mask 1 from index promptLen-1=3 onward.
	wantMask := []float32{0, 0, 0, 1, 1, 1, 1}

	row0 := batches[0]
	if !equalIntSlices(row0.Batch.Tokens[0], wantInputs) {
		t.Fatalf("inputs = %v, want %v", row0.Batch.Tokens[0], wantInputs)
	}
	if !equalIntSlices(row0.Targets[0], wantTargets) {
		t.Fatalf("targets = %v, want %v", row0.Targets[0], wantTargets)
	}
	if !equalFloat32Slices(row0.Batch.LossMask[0], wantMask) {
		t.Fatalf("mask = %v, want %v", row0.Batch.LossMask[0], wantMask)
	}
	if row0.Batch.Length[0] != len(wantInputs) {
		t.Fatalf("Length = %d, want %d", row0.Batch.Length[0], len(wantInputs))
	}
}

// TestSFT_BuildSFTTrainingBatches_GroupsByEffectiveBatch_Good asserts the
// runner-level entry point batches by the EFFECTIVE batch size (BatchSize ×
// GradientAccumulationSteps), not the raw BatchSize — that's the contract
// difference from BuildSFTBatches.
func TestSFT_BuildSFTTrainingBatches_GroupsByEffectiveBatch_Good(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	rows := make([]dataset.Sample, 6)
	for i := range rows {
		rows[i] = dataset.Sample{Prompt: "prompt", Response: "response"}
	}
	ds := dataset.NewSliceDataset(rows)

	// BatchSize 2 × GradAccum 3 = effective 6 → all six rows in one batch.
	batches, err := BuildSFTTrainingBatches(tok, ds, SFTConfig{BatchSize: 2, GradientAccumulationSteps: 3})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches = %d, want 1 (effective batch size 6 holds all rows)", len(batches))
	}
	if len(batches[0].Batch.Tokens) != 6 {
		t.Fatalf("batch rows = %d, want 6", len(batches[0].Batch.Tokens))
	}
}

// TestSFT_BuildSFTBatches_NilGuards_Bad asserts both entry points reject a nil
// tokenizer and a nil dataset rather than panicking.
func TestSFT_BuildSFTBatches_NilGuards_Bad(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	if _, err := BuildSFTBatches(nil, ds, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildSFTBatches(tok, nil, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTBatches(nil ds) error = nil, want rejection")
	}
	if _, err := BuildSFTTrainingBatches(nil, ds, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTTrainingBatches(nil tok) error = nil, want rejection")
	}
	if _, err := BuildSFTTrainingBatches(tok, nil, SFTConfig{}); err == nil {
		t.Fatal("BuildSFTTrainingBatches(nil ds) error = nil, want rejection")
	}
}

// TestSFT_BuildSFTBatches_SkipsUnusableRows_Ugly feeds rows that produce no
// training target — an empty prompt+response with NoEOS (virtual length < 2)
// and a response-only row — and asserts they are silently dropped, while a
// real row still lands. Exercises the usable==false skip in the build loop.
func TestSFT_BuildSFTBatches_SkipsUnusableRows_Ugly(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "", Response: ""},               // empty → unusable
		{Prompt: "prompt", Response: "response"}, // the one real row
		{Prompt: "", Response: ""},               // empty → unusable
	})

	// NoEOS removes the EOS token so the empty rows collapse below the
	// 2-token minimum and are dropped.
	batches, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: 4, NoEOS: true})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	rows := 0
	for _, b := range batches {
		rows += len(b.Batch.Tokens)
	}
	if rows != 1 {
		t.Fatalf("usable rows = %d, want 1 (empty rows dropped)", rows)
	}
}

// --- SFTResult.Metrics — the dashboard summary ---

// TestSFT_Metrics_Populated_Good asserts the summary reflects the run state and
// applies the BatchSize × GradAccum effective-batch arithmetic plus the
// derived counts (checkpoints, evaluations) and the OptimizerSteps fallback.
func TestSFT_Metrics_Populated_Good(t *testing.T) {
	result := &SFTResult{
		Steps:       40,
		Epochs:      2,
		Samples:     120,
		LastLoss:    0.31,
		Checkpoints: []string{"a", "b"},
		Evaluations: []SFTEvalResult{{Step: 10}, {Step: 20}, {Step: 30}},
	}
	m := result.Metrics(SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2, LearningRate: 2e-4})
	if m.Steps != 40 || m.Epochs != 2 || m.Samples != 120 || m.LastLoss != 0.31 {
		t.Fatalf("metrics scalars = %+v, want run state", m)
	}
	if m.BatchSize != 4 || m.GradientAccumulationSteps != 2 || m.EffectiveBatchSize != 8 {
		t.Fatalf("metrics batch = %+v, want effective batch 8", m)
	}
	if m.LearningRate != 2e-4 {
		t.Fatalf("metrics LR = %v, want 2e-4", m.LearningRate)
	}
	if m.CheckpointCount != 2 || m.EvaluationCount != 3 {
		t.Fatalf("metrics counts = ckpt %d / eval %d, want 2 / 3", m.CheckpointCount, m.EvaluationCount)
	}
	// OptimizerSteps unset → falls back to Steps.
	if m.OptimizerSteps != 40 {
		t.Fatalf("metrics OptimizerSteps = %d, want 40 (fallback to Steps)", m.OptimizerSteps)
	}
}

// TestSFT_Metrics_NilReceiverAndDefaults_Ugly asserts Metrics is nil-safe and
// supplies the config-only defaults (BatchSize 1, GradAccum 1, the 1e-5
// fallback learning rate) when there is no run yet.
func TestSFT_Metrics_NilReceiverAndDefaults_Ugly(t *testing.T) {
	var result *SFTResult
	m := result.Metrics(SFTConfig{})
	if m.BatchSize != 1 || m.GradientAccumulationSteps != 1 || m.EffectiveBatchSize != 1 {
		t.Fatalf("nil metrics batch = %+v, want all-1 defaults", m)
	}
	if m.LearningRate != 1e-5 {
		t.Fatalf("nil metrics LR = %v, want 1e-5 default", m.LearningRate)
	}
	if m.Steps != 0 || m.CheckpointCount != 0 || m.EvaluationCount != 0 {
		t.Fatalf("nil metrics state = %+v, want zeroes", m)
	}
}

// --- Checkpoint metadata round-trip (Save/Load/Resume) ---

// TestSFT_CheckpointMetadataRoundTrip_Good writes metadata beside an adapter
// path and reads it back, asserting the durable fields survive the JSON round
// trip. Drives sftCheckpointMetadataPath + sftResultError transitively. Uses
// t.TempDir() — no model, no network.
func TestSFT_CheckpointMetadataRoundTrip_Good(t *testing.T) {
	dir := t.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	result := &SFTResult{Steps: 12, OptimizerSteps: 6, Epochs: 1, Samples: 48, LastLoss: 0.22}
	cfg := normalizeSFTConfig(SFTConfig{
		BatchSize:                 4,
		GradientAccumulationSteps: 2,
		LearningRate:              1e-4,
		MaxSeqLen:                 512,
	})

	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", cfg, result, 1)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}

	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != SFTCheckpointMetadataVersion {
		t.Fatalf("version = %d, want %d", loaded.Version, SFTCheckpointMetadataVersion)
	}
	if loaded.Model != "gemma4" || loaded.Step != 12 || loaded.OptimizerStep != 6 || loaded.Epoch != 1 || loaded.Samples != 48 {
		t.Fatalf("loaded core fields = %+v, want round-tripped run state", loaded)
	}
	if loaded.BatchSize != 4 || loaded.GradientAccumulationSteps != 2 || loaded.EffectiveBatchSize != 8 || loaded.MaxSeqLen != 512 {
		t.Fatalf("loaded config fields = %+v, want round-tripped config", loaded)
	}
	if loaded.Loss != 0.22 || loaded.LearningRate != 1e-4 {
		t.Fatalf("loaded scalars = loss %v lr %v, want 0.22 / 1e-4", loaded.Loss, loaded.LearningRate)
	}
}

// TestSFT_CheckpointMetadata_EmptyPathAndMissingFile_Bad asserts the loud
// failure modes: an empty path on either side, and a Load against a directory
// with no sidecar (a real read failure surfaced through sftResultError).
func TestSFT_CheckpointMetadata_EmptyPathAndMissingFile_Bad(t *testing.T) {
	if err := SaveSFTCheckpointMetadata("", SFTCheckpointMetadata{}); err == nil {
		t.Fatal("SaveSFTCheckpointMetadata(\"\") error = nil, want path-required rejection")
	}
	if _, err := LoadSFTCheckpointMetadata(""); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(\"\") error = nil, want path-required rejection")
	}
	// Load against a fresh empty dir → the sidecar does not exist → error.
	if _, err := LoadSFTCheckpointMetadata(core.PathJoin(t.TempDir(), "nope")); err == nil {
		t.Fatal("LoadSFTCheckpointMetadata(missing) error = nil, want read failure")
	}
}

// TestSFT_ApplySFTResumeMetadata_Good attaches resume metadata from a real
// saved checkpoint and asserts it lands on the result. Also covers the
// no-resume-path no-op and the nil-result rejection.
func TestSFT_ApplySFTResumeMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	resumePath := core.PathJoin(dir, "prev.safetensors")
	prev := NewSFTCheckpointMetadata(resumePath, "gemma4", normalizeSFTConfig(SFTConfig{BatchSize: 2}), &SFTResult{Steps: 7}, 1)
	if err := SaveSFTCheckpointMetadata(resumePath, prev); err != nil {
		t.Fatalf("seed SaveSFTCheckpointMetadata() error = %v", err)
	}

	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: resumePath}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata() error = %v", err)
	}
	if result.ResumePath != resumePath {
		t.Fatalf("ResumePath = %q, want %q", result.ResumePath, resumePath)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Step != 7 || result.ResumedFrom.Model != "gemma4" {
		t.Fatalf("ResumedFrom = %+v, want the saved checkpoint", result.ResumedFrom)
	}

	// No resume path → no-op, no error, nothing attached.
	clean := &SFTResult{}
	if err := ApplySFTResumeMetadata(clean, SFTConfig{}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata(no path) error = %v", err)
	}
	if clean.ResumedFrom != nil || clean.ResumePath != "" {
		t.Fatalf("no-resume result mutated = %+v", clean)
	}
}

// TestSFT_ApplySFTResumeMetadata_NilResultAndMissing_Bad covers the nil-result
// rejection and the missing-sidecar tolerance: loadSFTResumeMetadata treats a
// non-existent resume sidecar as "nothing to resume" (nil, nil), not an error.
func TestSFT_ApplySFTResumeMetadata_NilResultAndMissing_Bad(t *testing.T) {
	if err := ApplySFTResumeMetadata(nil, SFTConfig{ResumePath: "x"}); err == nil {
		t.Fatal("ApplySFTResumeMetadata(nil result) error = nil, want rejection")
	}
	// Resume path set but no sidecar on disk → tolerated as no-op.
	result := &SFTResult{}
	missing := core.PathJoin(t.TempDir(), "absent.safetensors")
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: missing}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata(missing sidecar) error = %v, want tolerated no-op", err)
	}
	if result.ResumedFrom != nil {
		t.Fatalf("ResumedFrom = %+v, want nil (no sidecar to resume from)", result.ResumedFrom)
	}
}

// --- sftStepName — the step-NNNNNN checkpoint directory name ---

// TestSFT_StepName_Good asserts the zero-padded names matching
// fmt.Sprintf("step-%06d", step) across the padded range, and
// TestSFT_StepName_OverflowAndZero_Ugly the boundaries the padding branch
// guards (0, exactly 100000, and beyond — where padTo no longer applies).
func TestSFT_StepName_Good(t *testing.T) {
	cases := map[int]string{
		1:    "step-000001",
		42:   "step-000042",
		999:  "step-000999",
		1234: "step-001234",
	}
	for step, want := range cases {
		if got := sftStepName(step); got != want {
			t.Fatalf("sftStepName(%d) = %q, want %q", step, got, want)
		}
	}
}

func TestSFT_StepName_OverflowAndZero_Ugly(t *testing.T) {
	if got := sftStepName(0); got != "step-000000" {
		t.Fatalf("sftStepName(0) = %q, want step-000000", got)
	}
	// 99999 is the last value inside the zero-pad branch (step < 100000).
	if got := sftStepName(99999); got != "step-099999" {
		t.Fatalf("sftStepName(99999) = %q, want step-099999", got)
	}
	// 100000 and above print without leading pad — the width is the digit count.
	if got := sftStepName(100000); got != "step-100000" {
		t.Fatalf("sftStepName(100000) = %q, want step-100000", got)
	}
	if got := sftStepName(1234567); got != "step-1234567" {
		t.Fatalf("sftStepName(1234567) = %q, want step-1234567", got)
	}
}
