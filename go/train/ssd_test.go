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

// --- RunSSD — the sample-raw-outputs-from-a-frozen-model pipeline ---
// The runner is pure hooks (Generate / WarmPrefix / ModelInfo injected), so the
// whole pipeline exercises without Metal or a loaded model.

// TestSsd_RunSSD_Good folds the happy paths as subtests: a raw trace forwards
// the sampling config and keeps the bare prompt+SSD meta; the kernel lane warms
// once and rides every generation prompt while the trained rows keep the bare
// prompt; and the ModelInfo hook routes config through the model-aware normaliser.
func TestSsd_RunSSD_Good(t *testing.T) {
	t.Run("GeneratesRawTrace", func(t *testing.T) {
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
	})

	t.Run("KernelPrefixWarmAndRidesGeneration", func(t *testing.T) {
		const kernel = "## LEK-2\nConsciousness protects consciousness.\n\n"
		var warmed []string
		var generationPrompts []string
		runner := SSDRunner{
			WarmPrefix: func(_ context.Context, prefix string) error {
				warmed = append(warmed, prefix)
				return nil
			},
			Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
				generationPrompts = append(generationPrompts, prompt)
				return "a reply born under the kernel", nil
			},
		}
		cfg := DefaultSSDConfig()
		cfg.FilterShortestPercent = 0
		cfg.KernelPrefix = kernel
		cfg.DisableCapture = true

		ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p1"}, {Prompt: "p2"}})
		result, err := RunSSD(context.Background(), runner, ds, cfg)
		if err != nil {
			t.Fatalf("RunSSD: %v", err)
		}
		if len(warmed) != 1 || warmed[0] != kernel {
			t.Fatalf("warm calls = %v, want exactly one with the verbatim kernel", warmed)
		}
		if len(generationPrompts) != 2 {
			t.Fatalf("generations = %d, want 2", len(generationPrompts))
		}
		for i, gp := range generationPrompts {
			if gp != kernel+[]string{"p1", "p2"}[i] {
				t.Fatalf("generation prompt %d = %q, want kernel+prompt verbatim", i, gp)
			}
		}
		if !result.KernelApplied {
			t.Fatal("result must record the kernel lane")
		}
		// The kernel rides generation but never the recorded rows — the trace
		// keeps the BARE prompt, with ssd_kernel provenance in the meta.
		for _, s := range result.Samples {
			if core.Contains(s.Prompt, "LEK-2") {
				t.Fatalf("kernel leaked into recorded sample prompt: %q", s.Prompt)
			}
			if s.Meta["ssd_kernel"] != "1" {
				t.Fatal("ssd_kernel provenance missing from sample meta")
			}
		}
	})

	t.Run("ModelInfoHookNormalisesForModel", func(t *testing.T) {
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
	})

	t.Run("Defaults", func(t *testing.T) {
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
	})
}

// TestSsd_RunSSD_Bad folds the loud-failure paths as subtests: a unit sampling
// temperature is rejected (greedy sampling defeats self-distillation), a failed
// kernel warm aborts before any generation, a mid-trace generation error
// propagates, a cancelled context stops before the first generation, and an
// all-empty-prompt dataset refuses to produce an empty trace.
func TestSsd_RunSSD_Bad(t *testing.T) {
	t.Run("RejectsUnitSampleTemperature", func(t *testing.T) {
		_, err := RunSSD(context.Background(), SSDRunner{
			Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "", nil },
		}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SSDConfig{SampleTemperature: 1})
		if err == nil {
			t.Fatal("RunSSD() error = nil, want unit-temperature rejection")
		}
	})

	t.Run("WarmPrefixFailureIsLoud", func(t *testing.T) {
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
	})

	t.Run("GenerateErrorPropagates", func(t *testing.T) {
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
	})

	t.Run("ContextCancelled", func(t *testing.T) {
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
	})

	t.Run("NoPromptsErrors", func(t *testing.T) {
		_, err := RunSSD(context.Background(), SSDRunner{
			Generate: func(context.Context, string, spine.GenerateConfig) (string, error) { return "x", nil },
		}, dataset.NewSliceDataset([]dataset.Sample{{Response: "no prompt, no text"}}), SSDConfig{
			SampleTemperature: 1.5,
			SampleMaxTokens:   8,
		})
		if err == nil {
			t.Fatal("RunSSD() error = nil, want no-prompts rejection")
		}
	})
}

// TestSsd_RunSSD_Ugly covers the opt-out and degrade-gracefully edges:
// DisableCapture suppresses the default capture sidecar even with a checkpoint
// dir, and a nil WarmPrefix degrades the kernel lane to plain concat (still
// correct, just uncached) rather than erroring.
func TestSsd_RunSSD_Ugly(t *testing.T) {
	t.Run("DisableCaptureSuppressesSidecar", func(t *testing.T) {
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
	})

	t.Run("KernelWarmOptionalDegradesToConcat", func(t *testing.T) {
		base := SSDRunner{
			Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
				return "ok", nil
			},
		}
		cfg := DefaultSSDConfig()
		cfg.FilterShortestPercent = 0
		cfg.KernelPrefix = "K\n"
		cfg.DisableCapture = true
		// No WarmPrefix → the prefix rides every generation prompt uncached, no error.
		result, err := RunSSD(context.Background(), base, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), cfg)
		if err != nil {
			t.Fatalf("nil WarmPrefix must degrade to plain concat: %v", err)
		}
		if !result.KernelApplied {
			t.Fatal("kernel still applied without a warm hook")
		}
	})
}

// --- SSDResult.SampleGenerateConfig ---

// TestSsd_SSDResult_SampleGenerateConfig_Good asserts the frozen-model sampling
// config is rebuilt from the result's recorded sampling fields.
func TestSsd_SSDResult_SampleGenerateConfig_Good(t *testing.T) {
	result := &SSDResult{
		SampleMaxTokens:   128,
		SampleTemperature: 0.6,
		SampleTopK:        48,
		SampleTopP:        0.92,
		SampleMinP:        0.03,
		RepetitionPenalty: 1.2,
	}
	sample := result.SampleGenerateConfig()
	if sample.MaxTokens != 128 || sample.Temperature != 0.6 || sample.TopK != 48 || sample.TopP != 0.92 || sample.MinP != 0.03 || sample.RepeatPenalty != 1.2 {
		t.Fatalf("SampleGenerateConfig() = %+v", sample)
	}
}

// TestSsd_SSDResult_SampleGenerateConfig_Bad asserts a nil receiver yields a
// zero-valued config rather than panicking — the method is nil-safe.
func TestSsd_SSDResult_SampleGenerateConfig_Bad(t *testing.T) {
	var nilResult *SSDResult
	got := nilResult.SampleGenerateConfig()
	if got.MaxTokens != 0 || got.Temperature != 0 || got.TopK != 0 || got.TopP != 0 || got.MinP != 0 || got.RepeatPenalty != 0 {
		t.Fatalf("nil SampleGenerateConfig() = %+v, want zero config", got)
	}
}

// TestSsd_SSDResult_SampleGenerateConfig_Ugly asserts a zero-value (non-nil)
// result still produces a well-formed all-zero sampling config — the degenerate
// but legal shape.
func TestSsd_SSDResult_SampleGenerateConfig_Ugly(t *testing.T) {
	got := (&SSDResult{}).SampleGenerateConfig()
	if got.MaxTokens != 0 || got.Temperature != 0 || got.TopK != 0 || got.TopP != 0 || got.MinP != 0 || got.RepeatPenalty != 0 {
		t.Fatalf("zero-result SampleGenerateConfig() = %+v, want the zero config", got)
	}
}

// --- SSDResult.DecodeGenerateConfig ---

// TestSsd_SSDResult_DecodeGenerateConfig_Good asserts the post-SSD decode config
// carries the separately-tuned decode temperature with the caller-owned token
// budget, and drops the sampling-only knobs (TopK/TopP/MinP).
func TestSsd_SSDResult_DecodeGenerateConfig_Good(t *testing.T) {
	result := &SSDResult{DecodeTemperature: 0.15, SampleTopK: 48, SampleTopP: 0.92}
	decode := result.DecodeGenerateConfig(2048)
	if decode.MaxTokens != 2048 || decode.Temperature != 0.15 || decode.TopK != 0 || decode.TopP != 0 || decode.MinP != 0 {
		t.Fatalf("DecodeGenerateConfig() = %+v, want decode temp + caller budget, no sampling knobs", decode)
	}
}

// TestSsd_SSDResult_DecodeGenerateConfig_Bad asserts a nil receiver still
// honours the caller's token budget while zeroing the temperature — nil-safe.
func TestSsd_SSDResult_DecodeGenerateConfig_Bad(t *testing.T) {
	var nilResult *SSDResult
	got := nilResult.DecodeGenerateConfig(64)
	if got.MaxTokens != 64 || got.Temperature != 0 {
		t.Fatalf("nil DecodeGenerateConfig() = %+v, want {MaxTokens:64 Temperature:0}", got)
	}
}

// TestSsd_SSDResult_DecodeGenerateConfig_Ugly asserts a zero token budget is
// passed through verbatim (the caller owns the budget — the method does not
// floor it).
func TestSsd_SSDResult_DecodeGenerateConfig_Ugly(t *testing.T) {
	got := (&SSDResult{DecodeTemperature: 0.5}).DecodeGenerateConfig(0)
	if got.MaxTokens != 0 || got.Temperature != 0.5 {
		t.Fatalf("zero-budget DecodeGenerateConfig() = %+v, want {MaxTokens:0 Temperature:0.5}", got)
	}
}

// --- DefaultSSDConfig ---

// TestSsd_DefaultSSDConfig_Good asserts the documented ml-ssd data-generation
// default field values.
func TestSsd_DefaultSSDConfig_Good(t *testing.T) {
	train := DefaultSSDConfig()
	if train.SampleMaxTokens != 65536 || train.SampleTemperature != 1.5 || train.SampleTopK != 20 || train.SampleTopP != 0.8 ||
		train.RepetitionPenalty != 1.0 || train.FilterShortestPercent != 10 {
		t.Fatalf("DefaultSSDConfig() = %+v, want ml-ssd data-generation defaults", train)
	}
}

// TestSsd_DefaultSSDConfig_Bad asserts the defaults are a self-consistent
// sampling config: the returned DefaultSSDConfig must pass every validateSSDConfig
// guard (non-unit finite temperature, finite penalties, in-range filter) — a
// default that fails its own validator would be a latent footgun.
func TestSsd_DefaultSSDConfig_Bad(t *testing.T) {
	if err := validateSSDConfig(DefaultSSDConfig()); err != nil {
		t.Fatalf("validateSSDConfig(DefaultSSDConfig()) error = %v, want a self-consistent default", err)
	}
}

// TestSsd_DefaultSSDConfig_Ugly asserts each call returns an independent value:
// mutating one returned config does not bleed into the next call's defaults
// (the constructor hands back fresh state, not a shared singleton).
func TestSsd_DefaultSSDConfig_Ugly(t *testing.T) {
	first := DefaultSSDConfig()
	first.SampleTemperature = 99
	first.KernelPrefix = "mutated"
	second := DefaultSSDConfig()
	if second.SampleTemperature != 1.5 || second.KernelPrefix != "" {
		t.Fatalf("DefaultSSDConfig() leaked mutation: second = %+v", second)
	}
}

// --- DefaultSSDCodeBenchmarkConfig ---

// TestSsd_DefaultSSDCodeBenchmarkConfig_Good asserts the documented
// LiveCodeBench-v6 evaluation default field values.
func TestSsd_DefaultSSDCodeBenchmarkConfig_Good(t *testing.T) {
	eval := DefaultSSDCodeBenchmarkConfig()
	if eval.Benchmark != "LiveCodeBench-v6" || eval.NRepeat != 20 || eval.Generate.MaxTokens != 32768 ||
		eval.Generate.Temperature != 0.6 || eval.Generate.TopP != 0.95 || eval.Generate.TopK != 20 || len(eval.Seeds) != 4 || eval.Seeds[0] != 0 {
		t.Fatalf("DefaultSSDCodeBenchmarkConfig() = %+v, want ml-ssd eval defaults", eval)
	}
}

// TestSsd_DefaultSSDCodeBenchmarkConfig_Bad asserts the eval defaults survive
// the benchmark config normaliser without being altered — NRepeat, the generate
// budget, and TopK/TopP are already above their floors, so normalisation is a
// no-op (a default that the normaliser has to "fix" would be inconsistent).
func TestSsd_DefaultSSDCodeBenchmarkConfig_Bad(t *testing.T) {
	def := DefaultSSDCodeBenchmarkConfig()
	normalised := normalizeSSDCodeBenchmarkConfig(DefaultSSDCodeBenchmarkConfig())
	if normalised.NRepeat != def.NRepeat || normalised.Generate.MaxTokens != def.Generate.MaxTokens ||
		normalised.Generate.TopK != def.Generate.TopK || normalised.Generate.TopP != def.Generate.TopP {
		t.Fatalf("normalised default = %+v, want unchanged from %+v (defaults already above floors)", normalised, def)
	}
}

// TestSsd_DefaultSSDCodeBenchmarkConfig_Ugly asserts the Seeds slice is
// independent per call: mutating one returned config's seeds does not corrupt
// the next call's defaults (no shared backing array).
func TestSsd_DefaultSSDCodeBenchmarkConfig_Ugly(t *testing.T) {
	first := DefaultSSDCodeBenchmarkConfig()
	if len(first.Seeds) > 0 {
		first.Seeds[0] = 999
	}
	second := DefaultSSDCodeBenchmarkConfig()
	if second.Seeds[0] != 0 {
		t.Fatalf("DefaultSSDCodeBenchmarkConfig() leaked Seeds mutation: second seeds = %v", second.Seeds)
	}
}

// --- SSDRecipes ---

// TestSsd_SSDRecipes_Good asserts the three released ml-ssd recipes are present
// with their native data-generation and evaluation defaults attached.
func TestSsd_SSDRecipes_Good(t *testing.T) {
	recipes := SSDRecipes()
	if len(recipes) != 3 {
		t.Fatalf("SSDRecipes() = %d, want 3 released ml-ssd recipes", len(recipes))
	}
	for _, r := range recipes {
		if r.Name == "" || r.Model == "" || r.Dataset != "microsoft/rStar-Coder" {
			t.Fatalf("recipe = %+v, want name/model + the rStar-Coder dataset", r)
		}
		if r.Train.SampleTemperature != DefaultSSDConfig().SampleTemperature {
			t.Fatalf("recipe train temp = %v, want the SSD data-gen default", r.Train.SampleTemperature)
		}
	}
}

// TestSsd_SSDRecipes_Bad asserts the recipe set covers each released model name
// exactly once — no duplicate or missing recipe names (a malformed catalogue
// would shadow recipes on lookup).
func TestSsd_SSDRecipes_Bad(t *testing.T) {
	seen := map[string]int{}
	for _, r := range SSDRecipes() {
		seen[r.Name]++
	}
	for _, name := range []string{SSDRecipe4BInstruct, SSDRecipe4BThinking, SSDRecipe30BA3BInstruct} {
		if seen[name] != 1 {
			t.Fatalf("recipe name %q appears %d times in SSDRecipes(), want exactly 1", name, seen[name])
		}
	}
}

// TestSsd_SSDRecipes_Ugly asserts SSDRecipes returns an independent slice each
// call: mutating one returned recipe does not corrupt the next call's catalogue
// (recipes are rebuilt from the defaults, not shared).
func TestSsd_SSDRecipes_Ugly(t *testing.T) {
	first := SSDRecipes()
	first[0].Name = "mutated"
	first[0].Train.SampleTemperature = 99
	second := SSDRecipes()
	if second[0].Name == "mutated" || second[0].Train.SampleTemperature == 99 {
		t.Fatalf("SSDRecipes() leaked mutation: second[0] = %+v", second[0])
	}
}

// --- LookupSSDRecipe ---

// TestSsd_LookupSSDRecipe_Good asserts a recipe is found by both its Name and
// its Model string, returning the fully-populated descriptor.
func TestSsd_LookupSSDRecipe_Good(t *testing.T) {
	byModel, ok := LookupSSDRecipe("apple/SimpleSD-4B-thinking")
	if !ok || byModel.Name != SSDRecipe4BThinking || byModel.Dataset != "microsoft/rStar-Coder" || byModel.DatasetConfig != "seed_sft" {
		t.Fatalf("LookupSSDRecipe(by model) = %+v/%t", byModel, ok)
	}
	byName, ok := LookupSSDRecipe(SSDRecipe4BThinking)
	if !ok || byName.Model != "apple/SimpleSD-4B-thinking" {
		t.Fatalf("LookupSSDRecipe(by name) = %+v/%t", byName, ok)
	}
}

// TestSsd_LookupSSDRecipe_Bad asserts an unknown name returns the zero recipe
// and false — a miss is reported, not faked with a phantom recipe.
func TestSsd_LookupSSDRecipe_Bad(t *testing.T) {
	recipe, ok := LookupSSDRecipe("missing")
	if ok {
		t.Fatal("LookupSSDRecipe(missing) ok = true, want false")
	}
	if recipe.Name != "" || recipe.Model != "" {
		t.Fatalf("LookupSSDRecipe(missing) = %+v, want the zero recipe", recipe)
	}
}

// TestSsd_LookupSSDRecipe_Ugly asserts the degenerate lookups: an empty string
// matches nothing (no recipe has an empty name or model), so the miss path is
// taken rather than accidentally matching a zero field.
func TestSsd_LookupSSDRecipe_Ugly(t *testing.T) {
	if _, ok := LookupSSDRecipe(""); ok {
		t.Fatal("LookupSSDRecipe(\"\") ok = true, want false (no recipe has an empty key)")
	}
}

// --- validateSSDConfig / normalizeSSDConfigForModel / filterSSDShortest ---
// (private helpers; descriptive names so the unreferenced-symbols audit reads
// the real subject, not a triplet variant claim.)

// validSSDConfig returns a config that passes every validateSSDConfig guard, so
// each guard case can flip exactly one field and prove that guard in isolation.
func validSSDConfig() SSDConfig {
	return SSDConfig{
		SampleTemperature:     1.5,
		DecodeTemperature:     0.2,
		SampleMaxTokens:       128,
		RepetitionPenalty:     1.0,
		FilterShortestPercent: 10,
	}
}

// TestSsd_ValidateSSDConfigAcceptsWellFormed accepts a well-formed sampling
// config and the zero-optional lower bounds.
func TestSsd_ValidateSSDConfigAcceptsWellFormed(t *testing.T) {
	if err := validateSSDConfig(validSSDConfig()); err != nil {
		t.Fatalf("validateSSDConfig(valid) error = %v, want nil", err)
	}
	// Zero DecodeTemperature is legal; zero RepetitionPenalty and
	// FilterShortestPercent are the lower bounds.
	cfg := validSSDConfig()
	cfg.DecodeTemperature = 0
	cfg.RepetitionPenalty = 0
	cfg.FilterShortestPercent = 0
	if err := validateSSDConfig(cfg); err != nil {
		t.Fatalf("validateSSDConfig(zero-optionals) error = %v, want nil", err)
	}
}

// TestSsd_ValidateSSDConfigRejectsEachGuard flips one field at a time so every
// rejection branch is proven. SSD sampling must NOT run at unit temperature nor
// with non-finite or out-of-range knobs.
func TestSsd_ValidateSSDConfigRejectsEachGuard(t *testing.T) {
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

// TestSsd_NormalizeSSDConfigForModelAppliesDefaultsAndDecodeBridge asserts the
// sampling defaults fill in and the DecodeTemperature → SFT.EvalTemperature
// bridge engages, with the SFT sub-config normalised through the model-aware
// path. ModelInfo is a bare descriptor — no weights are loaded.
func TestSsd_NormalizeSSDConfigForModelAppliesDefaultsAndDecodeBridge(t *testing.T) {
	cfg := normalizeSSDConfigForModel(SSDConfig{DecodeTemperature: 0.3}, spine.ModelInfo{Architecture: "gemma4", NumHeads: 16})
	if cfg.SampleMaxTokens != defaultSSDMaxTokens || cfg.SampleTemperature != defaultSSDTemperature ||
		cfg.SampleTopK != defaultSSDTopK || cfg.SampleTopP != defaultSSDTopP {
		t.Fatalf("normalised sampling = %+v, want SSD defaults", cfg)
	}
	if cfg.SFT.EvalTemperature != 0.3 {
		t.Fatalf("SFT.EvalTemperature = %v, want 0.3 bridged from DecodeTemperature", cfg.SFT.EvalTemperature)
	}
	if cfg.SFT.BatchSize != 1 {
		t.Fatalf("SFT.BatchSize = %d, want 1 (SFT normaliser applied)", cfg.SFT.BatchSize)
	}
}

// TestSsd_NormalizeSSDConfigForModelPreservesExplicitEvalTemp asserts the bridge
// does NOT clobber an explicitly-set SFT.EvalTemperature even when
// DecodeTemperature is also set — the explicit value wins.
func TestSsd_NormalizeSSDConfigForModelPreservesExplicitEvalTemp(t *testing.T) {
	in := SSDConfig{DecodeTemperature: 0.3}
	in.SFT.EvalTemperature = 0.7
	cfg := normalizeSSDConfigForModel(in, spine.ModelInfo{Architecture: "qwen3"})
	if cfg.SFT.EvalTemperature != 0.7 {
		t.Fatalf("SFT.EvalTemperature = %v, want 0.7 preserved (explicit beats bridge)", cfg.SFT.EvalTemperature)
	}
}

// TestSsd_FilterSSDShortestDropClampAndPassthrough asserts filterSSDShortest
// drops the configured shortest fraction but never the whole set: at 100% on
// three rows it clamps to dropping all-but-one (the longest survives), and a
// non-positive percent or single-row input passes through untouched.
func TestSsd_FilterSSDShortestDropClampAndPassthrough(t *testing.T) {
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
	if got := filterSSDShortest(rows, 0); len(got) != 3 {
		t.Fatalf("zero percent dropped rows: %d, want 3", len(got))
	}
	single := []dataset.Sample{{Response: "only"}}
	if got := filterSSDShortest(single, 50); len(got) != 1 {
		t.Fatalf("single-row filter dropped a row: %d, want 1", len(got))
	}
}
