// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"errors"
	"math"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

func TestRunSSD_GeneratesRawTrace_Good(t *testing.T) {
	source := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prove a lemma", Meta: map[string]string{"split": "train"}},
		{Text: "free prompt text"},
		{Response: "ignored without prompt"},
	})
	var generatedPrompts []string
	var generatedCfgs []spine.GenerateConfig

	result, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(_ context.Context, prompt string, cfg spine.GenerateConfig) (string, error) {
			generatedPrompts = append(generatedPrompts, prompt)
			generatedCfgs = append(generatedCfgs, cfg)
			return "raw:" + prompt, nil
		},
	}, source, SSDConfig{
		SampleMaxTokens:   42,
		SampleTemperature: 0.8,
		SampleTopK:        32,
		SampleTopP:        0.9,
		SampleMinP:        0.05,
		RepetitionPenalty: 1.1,
		DecodeTemperature: 0.2,
	})
	if err != nil {
		t.Fatalf("RunSSD() error = %v", err)
	}
	if len(generatedPrompts) != 2 || generatedPrompts[0] != "prove a lemma" || generatedPrompts[1] != "free prompt text" {
		t.Fatalf("generated prompts = %#v, want prompt/text rows only", generatedPrompts)
	}
	if generatedCfgs[0].MaxTokens != 42 || generatedCfgs[0].Temperature != 0.8 || generatedCfgs[0].TopK != 32 || generatedCfgs[0].TopP != 0.9 || generatedCfgs[0].MinP != 0.05 || generatedCfgs[0].RepeatPenalty != 1.1 {
		t.Fatalf("generate config = %+v, want sampling config forwarded", generatedCfgs[0])
	}
	// SSD stops at the scored trace: the sampled rows ARE the deliverable —
	// never handed to a training step here.
	if len(result.Samples) != 2 || result.Samples[0].Prompt != "prove a lemma" || result.Samples[0].Response != "raw:prove a lemma" {
		t.Fatalf("trace samples = %+v, want raw generated prompt/response rows", result.Samples)
	}
	if result.Samples[0].Meta["split"] != "train" || result.Samples[0].Meta["ssd"] != "simple_self_distillation" || result.Samples[0].Meta["ssd_source_index"] != "0" {
		t.Fatalf("trace sample meta = %+v, want source metadata plus SSD markers", result.Samples[0].Meta)
	}
	if result.SampleTemperature != 0.8 || result.DecodeTemperature != 0.2 || result.SampleMaxTokens != 42 ||
		result.SampleTopK != 32 || result.SampleTopP != 0.9 || result.SampleMinP != 0.05 || result.RepetitionPenalty != 1.1 {
		t.Fatalf("result sampling fields = %+v", result)
	}
}

func TestSSDResult_GenerateConfigs_Good(t *testing.T) {
	result := &SSDResult{
		SampleMaxTokens:   128,
		SampleTemperature: 0.6,
		SampleTopK:        48,
		SampleTopP:        0.92,
		SampleMinP:        0.03,
		RepetitionPenalty: 1.2,
		DecodeTemperature: 0.15,
	}

	sample := result.SampleGenerateConfig()
	if sample.MaxTokens != 128 || sample.Temperature != 0.6 || sample.TopK != 48 || sample.TopP != 0.92 || sample.MinP != 0.03 || sample.RepeatPenalty != 1.2 {
		t.Fatalf("SampleGenerateConfig() = %+v", sample)
	}
	decode := result.DecodeGenerateConfig(2048)
	if decode.MaxTokens != 2048 || decode.Temperature != 0.15 || decode.TopK != 0 || decode.TopP != 0 || decode.MinP != 0 {
		t.Fatalf("DecodeGenerateConfig() = %+v", decode)
	}

	var nilResult *SSDResult
	if got := nilResult.SampleGenerateConfig(); got.MaxTokens != 0 || got.Temperature != 0 || got.TopK != 0 || got.TopP != 0 || got.MinP != 0 || got.RepeatPenalty != 0 {
		t.Fatalf("nil SampleGenerateConfig() = %+v", got)
	}
	if got := nilResult.DecodeGenerateConfig(64); got.MaxTokens != 64 || got.Temperature != 0 {
		t.Fatalf("nil DecodeGenerateConfig() = %+v", got)
	}
}

func TestSSDDefaultsAndRecipes_Good(t *testing.T) {
	train := DefaultSSDConfig()
	if train.SampleMaxTokens != 65536 || train.SampleTemperature != 1.5 || train.SampleTopK != 20 || train.SampleTopP != 0.8 ||
		train.RepetitionPenalty != 1.0 || train.FilterShortestPercent != 10 {
		t.Fatalf("DefaultSSDConfig() = %+v, want ml-ssd data-generation defaults", train)
	}
	eval := DefaultSSDCodeBenchmarkConfig()
	if eval.Benchmark != "LiveCodeBench-v6" || eval.NRepeat != 20 || eval.Generate.MaxTokens != 32768 ||
		eval.Generate.Temperature != 0.6 || eval.Generate.TopP != 0.95 || eval.Generate.TopK != 20 || len(eval.Seeds) != 4 || eval.Seeds[0] != 0 {
		t.Fatalf("DefaultSSDCodeBenchmarkConfig() = %+v, want ml-ssd eval defaults", eval)
	}

	recipes := SSDRecipes()
	if len(recipes) != 3 {
		t.Fatalf("SSDRecipes() = %d, want released ml-ssd recipes", len(recipes))
	}
	recipe, ok := LookupSSDRecipe("apple/SimpleSD-4B-thinking")
	if !ok || recipe.Name != SSDRecipe4BThinking || recipe.Dataset != "microsoft/rStar-Coder" || recipe.DatasetConfig != "seed_sft" {
		t.Fatalf("LookupSSDRecipe() = %+v/%t", recipe, ok)
	}
	if _, ok := LookupSSDRecipe("missing"); ok {
		t.Fatal("LookupSSDRecipe(missing) ok = true")
	}
}

func TestRunSSD_Defaults_Good(t *testing.T) {
	var gotCfg spine.GenerateConfig
	_, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(_ context.Context, _ string, cfg spine.GenerateConfig) (string, error) {
			gotCfg = cfg
			return "answer", nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{})
	if err != nil {
		t.Fatalf("RunSSD() error = %v", err)
	}
	if gotCfg.MaxTokens != defaultSSDMaxTokens ||
		gotCfg.Temperature != defaultSSDTemperature ||
		gotCfg.TopK != defaultSSDTopK ||
		gotCfg.TopP != defaultSSDTopP {
		t.Fatalf("default generate config = %+v", gotCfg)
	}
}

func TestRunSSD_RejectsUnitSampleTemperature_Bad(t *testing.T) {
	_, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "", nil },
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{SampleTemperature: 1})
	if err == nil {
		t.Fatal("RunSSD() error = nil, want unit-temperature rejection")
	}
}

// --- validateSSDConfig — the SSD sampling-phase guards ---

// validSSDConfig returns a config that passes every validateSSDConfig guard,
// so each Bad case can flip exactly one field and prove that guard fires in
// isolation.
func validSSDConfig() SSDConfig {
	return SSDConfig{
		SampleTemperature:     1.5,
		DecodeTemperature:     0.2,
		SampleMaxTokens:       128,
		RepetitionPenalty:     1.0,
		FilterShortestPercent: 10,
	}
}

// TestValidateSSDConfig_Good accepts a well-formed sampling config — the
// data-generation defaults shape (non-unit temperature, finite penalties).
func TestValidateSSDConfig_Good(t *testing.T) {
	if err := validateSSDConfig(validSSDConfig()); err != nil {
		t.Fatalf("validateSSDConfig(valid) error = %v, want nil", err)
	}
	// Zero DecodeTemperature is legal (decode temperature is optional); zero
	// RepetitionPenalty and FilterShortestPercent are the lower bounds.
	cfg := validSSDConfig()
	cfg.DecodeTemperature = 0
	cfg.RepetitionPenalty = 0
	cfg.FilterShortestPercent = 0
	if err := validateSSDConfig(cfg); err != nil {
		t.Fatalf("validateSSDConfig(zero-optionals) error = %v, want nil", err)
	}
}

// TestValidateSSDConfig_EachGuard_Bad flips one field at a time so every
// rejection branch is proven. SSD sampling must NOT run at unit temperature
// (greedy sampling defeats self-distillation diversity) nor with non-finite or
// out-of-range knobs.
func TestValidateSSDConfig_EachGuard_Bad(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	cases := []struct {
		name   string
		mutate func(*SSDConfig)
	}{
		{"non-positive sample temperature", func(c *SSDConfig) { c.SampleTemperature = 0 }},
		{"negative sample temperature", func(c *SSDConfig) { c.SampleTemperature = -0.5 }},
		{"NaN sample temperature", func(c *SSDConfig) { c.SampleTemperature = nan }},
		{"Inf sample temperature", func(c *SSDConfig) { c.SampleTemperature = inf }},
		{"unit sample temperature", func(c *SSDConfig) { c.SampleTemperature = 1 }},
		{"negative decode temperature", func(c *SSDConfig) { c.DecodeTemperature = -0.1 }},
		{"NaN decode temperature", func(c *SSDConfig) { c.DecodeTemperature = nan }},
		{"non-positive max tokens", func(c *SSDConfig) { c.SampleMaxTokens = 0 }},
		{"negative repetition penalty", func(c *SSDConfig) { c.RepetitionPenalty = -1 }},
		{"NaN repetition penalty", func(c *SSDConfig) { c.RepetitionPenalty = nan }},
		{"filter percent below range", func(c *SSDConfig) { c.FilterShortestPercent = -1 }},
		{"filter percent above range", func(c *SSDConfig) { c.FilterShortestPercent = 101 }},
		{"NaN filter percent", func(c *SSDConfig) { c.FilterShortestPercent = nan }},
	}
	for _, tc := range cases {
		cfg := validSSDConfig()
		tc.mutate(&cfg)
		if err := validateSSDConfig(cfg); err == nil {
			t.Fatalf("validateSSDConfig(%s) error = nil, want rejection", tc.name)
		}
	}
}

// --- normalizeSSDConfigForModel — model-aware defaulting ---

// TestNormalizeSSDConfigForModel_AppliesDefaultsAndDecodeBridge_Good asserts the
// sampling defaults fill in and the DecodeTemperature → SFT.EvalTemperature
// bridge engages, with the SFT sub-config normalised through the model-aware
// path. ModelInfo here is a bare descriptor — no weights are loaded.
func TestNormalizeSSDConfigForModel_AppliesDefaultsAndDecodeBridge_Good(t *testing.T) {
	cfg := normalizeSSDConfigForModel(SSDConfig{DecodeTemperature: 0.3}, spine.ModelInfo{Architecture: "gemma4", NumHeads: 16})
	if cfg.SampleMaxTokens != defaultSSDMaxTokens || cfg.SampleTemperature != defaultSSDTemperature ||
		cfg.SampleTopK != defaultSSDTopK || cfg.SampleTopP != defaultSSDTopP {
		t.Fatalf("normalised sampling = %+v, want SSD defaults", cfg)
	}
	// DecodeTemperature set + SFT.EvalTemperature unset → bridged.
	if cfg.SFT.EvalTemperature != 0.3 {
		t.Fatalf("SFT.EvalTemperature = %v, want 0.3 bridged from DecodeTemperature", cfg.SFT.EvalTemperature)
	}
	// SFT sub-config normalised (BatchSize defaulted to 1 by the SFT normaliser).
	if cfg.SFT.BatchSize != 1 {
		t.Fatalf("SFT.BatchSize = %d, want 1 (SFT normaliser applied)", cfg.SFT.BatchSize)
	}
}

// TestNormalizeSSDConfigForModel_PreservesExplicitEvalTemp_Ugly asserts the
// bridge does NOT clobber an explicitly-set SFT.EvalTemperature even when
// DecodeTemperature is also set — the explicit value wins.
func TestNormalizeSSDConfigForModel_PreservesExplicitEvalTemp_Ugly(t *testing.T) {
	in := SSDConfig{DecodeTemperature: 0.3}
	in.SFT.EvalTemperature = 0.7
	cfg := normalizeSSDConfigForModel(in, spine.ModelInfo{Architecture: "qwen3"})
	if cfg.SFT.EvalTemperature != 0.7 {
		t.Fatalf("SFT.EvalTemperature = %v, want 0.7 preserved (explicit beats bridge)", cfg.SFT.EvalTemperature)
	}
}

// --- RunSSD flow branches (synthetic runner hooks; no model loads) ---

// The kernel lane (#97): a standing prefix is warmed ONCE as KV state, then
// rides every GENERATION prompt verbatim while the bare prompt is what the
// rows train on. The runner here is pure hooks — WarmPrefix and Generate are
// injected, so the whole lane exercises without Metal.
func TestRunSSD_KernelPrefixWarmAndRidesGeneration_Good(t *testing.T) {
	dir := t.TempDir()
	var warmed []string
	var genPrompts []string
	result, err := RunSSD(context.Background(), SSDRunner{
		WarmPrefix: func(_ context.Context, prefix string) error {
			warmed = append(warmed, prefix)
			return nil
		},
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			genPrompts = append(genPrompts, prompt)
			return "answer", nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prove it"}}), SSDConfig{
		SampleTemperature:     1.5,
		SampleMaxTokens:       64,
		FilterShortestPercent: 0,
		KernelPrefix:          "KERNEL:",
		SFT:                   SFTConfig{CheckpointDir: dir},
	})
	if err != nil {
		t.Fatalf("RunSSD() error = %v", err)
	}
	// Warmed exactly once with the verbatim kernel prefix.
	if len(warmed) != 1 || warmed[0] != "KERNEL:" {
		t.Fatalf("warmed = %#v, want one warm with the kernel prefix", warmed)
	}
	// The generation prompt carries the prefix; the trained row keeps the bare prompt.
	if len(genPrompts) != 1 || genPrompts[0] != "KERNEL:prove it" {
		t.Fatalf("generation prompt = %#v, want kernel-prefixed", genPrompts)
	}
	if result.Samples[0].Prompt != "prove it" {
		t.Fatalf("sample prompt = %q, want the bare prompt (kernel rides generation only)", result.Samples[0].Prompt)
	}
	if result.Samples[0].Meta["ssd_kernel"] != "1" {
		t.Fatalf("sample meta = %+v, want ssd_kernel marker", result.Samples[0].Meta)
	}
}

// A failed kernel warm is loud — the operator armed the kernel lane, and
// silently sampling without it would forge the run.
func TestRunSSD_WarmPrefixFailureIsLoud_Bad(t *testing.T) {
	called := false
	_, err := RunSSD(context.Background(), SSDRunner{
		WarmPrefix: func(context.Context, string) error { return errors.New("kv prefill failed") },
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) {
			called = true
			return "", nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{
		SampleTemperature: 1.5,
		SampleMaxTokens:   8,
		KernelPrefix:      "K",
	})
	if err == nil {
		t.Fatal("RunSSD() error = nil, want warm-prefix failure surfaced")
	}
	if called {
		t.Fatal("Generate ran after a failed warm — the lane must abort, not forge the run")
	}
}

// A generation error mid-trace propagates — the sampling loop cannot silently
// drop a row it failed to produce.
func TestRunSSD_GenerateErrorPropagates_Bad(t *testing.T) {
	_, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) {
			return "", errors.New("decode died")
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{
		SampleTemperature: 1.5,
		SampleMaxTokens:   8,
	})
	if err == nil {
		t.Fatal("RunSSD() error = nil, want generation error surfaced")
	}
}

// A cancelled context stops the sampling loop before the first generation.
func TestRunSSD_ContextCancelled_Bad(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false
	_, err := RunSSD(ctx, SSDRunner{
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) {
			called = true
			return "x", nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{
		SampleTemperature: 1.5,
		SampleMaxTokens:   8,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunSSD(cancelled) error = %v, want context.Canceled", err)
	}
	if called {
		t.Fatal("Generate ran under a cancelled context")
	}
}

// The ModelInfo hook routes config through the model-aware normaliser — a bare
// descriptor, no weights — so a Gemma4 descriptor takes the gemma4 LoRA path.
func TestRunSSD_ModelInfoHookNormalisesForModel_Good(t *testing.T) {
	infoCalls := 0
	result, err := RunSSD(context.Background(), SSDRunner{
		ModelInfo: func(context.Context) spine.ModelInfo {
			infoCalls++
			return spine.ModelInfo{Architecture: "gemma4", NumHeads: 16}
		},
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "answer", nil },
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{
		DecodeTemperature: 0.3,
	})
	if err != nil {
		t.Fatalf("RunSSD() error = %v", err)
	}
	if infoCalls != 1 {
		t.Fatalf("ModelInfo calls = %d, want 1", infoCalls)
	}
	// The decode→eval bridge engaged through the model-aware path.
	if result.DecodeTemperature != 0.3 {
		t.Fatalf("DecodeTemperature = %v, want 0.3 normalised", result.DecodeTemperature)
	}
}

// DisableCapture suppresses the default capture sidecar even when a checkpoint
// dir is configured — the operator opted out of the raw-return corpus.
func TestRunSSD_DisableCaptureSuppressesSidecar_Ugly(t *testing.T) {
	dir := t.TempDir()
	_, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "answer", nil },
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{
		SampleTemperature: 1.5,
		SampleMaxTokens:   8,
		DisableCapture:    true,
		SFT:               SFTConfig{CheckpointDir: dir},
	})
	if err != nil {
		t.Fatalf("RunSSD() error = %v", err)
	}
	if _, statErr := coreio.Local.Stat(core.PathJoin(dir, "ssd-captures.jsonl")); statErr == nil {
		t.Fatal("capture sidecar exists, want none when DisableCapture is set")
	}
}

// An all-empty-prompt dataset produces no samples — the run refuses an empty trace.
func TestRunSSD_NoPromptsErrors_Bad(t *testing.T) {
	_, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "x", nil },
	}, dataset.NewSliceDataset([]dataset.Sample{{Response: "no prompt, no text"}}), SSDConfig{
		SampleTemperature: 1.5,
		SampleMaxTokens:   8,
	})
	if err == nil {
		t.Fatal("RunSSD() error = nil, want no-prompts rejection")
	}
}

// filterSSDShortest drops the configured shortest fraction, never the whole
// set: at 100% on three rows it clamps to dropping all-but-one (the longest
// survives), and one row / non-positive percent passes through untouched.
func TestFilterSSDShortest_DropClampAndPassthrough_Ugly(t *testing.T) {
	rows := []dataset.Sample{
		{Response: "short"},
		{Response: "medium length"},
		{Response: "the longest response of the three by a clear margin"},
	}
	kept := filterSSDShortest(rows, 100)
	if len(kept) != 1 {
		t.Fatalf("drop-all clamp kept %d rows, want 1 (never drop the whole set)", len(kept))
	}
	if kept[0].Response != rows[2].Response {
		t.Fatalf("survivor = %q, want the longest response", kept[0].Response)
	}
	// Non-positive percent and single-row inputs pass through unchanged.
	if got := filterSSDShortest(rows, 0); len(got) != 3 {
		t.Fatalf("zero percent dropped rows: %d, want 3", len(got))
	}
	single := []dataset.Sample{{Response: "only"}}
	if got := filterSSDShortest(single, 50); len(got) != 1 {
		t.Fatalf("single-row filter dropped a row: %d, want 1", len(got))
	}
}
