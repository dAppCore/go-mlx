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
