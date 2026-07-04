// SPDX-Licence-Identifier: EUPL-1.2

// Tests for ssd.go — the Model-bound SSD entry surface. The pipeline
// itself lives in dappco.re/go/mlx/train; these tests assert the root
// package's thin wrappers forward and shape data as the CLI consumes it,
// without loading a model (the *Model methods are covered by the live
// suite — see ssd_smoke / sft_smoke).

package mlx

import (
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/train"
)

func TestSSD_DefaultSSDConfig_Good(t *testing.T) {
	cfg := DefaultSSDConfig()
	if cfg.SampleMaxTokens != 65536 {
		t.Fatalf("SampleMaxTokens = %d, want 65536", cfg.SampleMaxTokens)
	}
	if cfg.SampleTemperature != 1.5 {
		t.Fatalf("SampleTemperature = %v, want 1.5", cfg.SampleTemperature)
	}
	if cfg.SampleTopK != 20 || cfg.SampleTopP != 0.8 {
		t.Fatalf("topK/topP = %d/%v, want 20/0.8", cfg.SampleTopK, cfg.SampleTopP)
	}
	if cfg.FilterShortestPercent != 10 {
		t.Fatalf("FilterShortestPercent = %v, want 10", cfg.FilterShortestPercent)
	}
}

func TestSSD_DefaultSSDCodeBenchmarkConfig_Good(t *testing.T) {
	cfg := DefaultSSDCodeBenchmarkConfig()
	if cfg.Benchmark != "LiveCodeBench-v6" {
		t.Fatalf("Benchmark = %q, want LiveCodeBench-v6", cfg.Benchmark)
	}
	if cfg.NRepeat != 20 {
		t.Fatalf("NRepeat = %d, want 20", cfg.NRepeat)
	}
	if cfg.Generate.MaxTokens != 32768 || cfg.Generate.Temperature != 0.6 {
		t.Fatalf("generate = %+v, want maxTokens 32768 temp 0.6", cfg.Generate)
	}
	if len(cfg.Seeds) != 4 || cfg.Seeds[0] != 0 {
		t.Fatalf("Seeds = %v, want [0 1234 1234 1234]", cfg.Seeds)
	}
}

func TestSSD_SSDRecipes_Good(t *testing.T) {
	recipes := SSDRecipes()
	if len(recipes) != 3 {
		t.Fatalf("recipe count = %d, want 3", len(recipes))
	}
	// Every released recipe carries the data-gen + eval defaults.
	wantTrain := DefaultSSDConfig()
	for _, recipe := range recipes {
		if recipe.Name == "" || recipe.Model == "" {
			t.Fatalf("recipe missing name/model: %+v", recipe)
		}
		if recipe.Train.SampleMaxTokens != wantTrain.SampleMaxTokens {
			t.Fatalf("recipe %q train = %+v, want default SampleMaxTokens", recipe.Name, recipe.Train)
		}
	}
}

func TestSSD_LookupSSDRecipe_Good(t *testing.T) {
	// Lookup resolves by short name.
	recipe, ok := LookupSSDRecipe(train.SSDRecipe4BInstruct)
	if !ok {
		t.Fatalf("LookupSSDRecipe(%q) ok = false", train.SSDRecipe4BInstruct)
	}
	if recipe.Name != train.SSDRecipe4BInstruct {
		t.Fatalf("recipe.Name = %q, want %q", recipe.Name, train.SSDRecipe4BInstruct)
	}
	// And by full model path.
	byModel, ok := LookupSSDRecipe("apple/SimpleSD-4B-instruct")
	if !ok || byModel.Name != train.SSDRecipe4BInstruct {
		t.Fatalf("LookupSSDRecipe by model = %+v ok=%v", byModel, ok)
	}
}

func TestSSD_LookupSSDRecipe_Bad(t *testing.T) {
	_, ok := LookupSSDRecipe("no-such-recipe")
	if ok {
		t.Fatal("LookupSSDRecipe(unknown) ok = true, want false")
	}
}

func TestSSD_LoadSSDCodeBenchmarkJSONL_Good(t *testing.T) {
	raw := `{"id":"p1","prompt":"add two numbers","tests":["assert add(1,2)==3"]}
{"task_id":"p2","question_content":"reverse a string","public_test_cases":["assert rev(\"ab\")==\"ba\""]}`
	samples, err := LoadSSDCodeBenchmarkJSONL(raw)
	if err != nil {
		t.Fatalf("LoadSSDCodeBenchmarkJSONL error = %v", err)
	}
	if len(samples) != 2 {
		t.Fatalf("sample count = %d, want 2", len(samples))
	}
	if samples[0].ID != "p1" || samples[0].Prompt != "add two numbers" {
		t.Fatalf("sample[0] = %+v", samples[0])
	}
	if samples[1].ID != "p2" || samples[1].Prompt != "reverse a string" {
		t.Fatalf("sample[1] = %+v", samples[1])
	}
}

func TestSSD_LoadSSDCodeBenchmarkJSONL_Bad(t *testing.T) {
	// Malformed JSON on a non-blank line is a hard parse error.
	_, err := LoadSSDCodeBenchmarkJSONL(`{"prompt": not-json}`)
	if err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONL(malformed) error = nil, want parse error")
	}
}

func TestSSD_LoadSSDCodeBenchmarkJSONL_Ugly(t *testing.T) {
	// Blank lines and records with no usable prompt are skipped; an
	// all-empty input produces a no-samples error rather than a panic.
	_, err := LoadSSDCodeBenchmarkJSONL("\n   \n{\"id\":\"only-id\"}\n")
	if err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONL(no prompts) error = nil, want no-samples error")
	}
}

func TestSSD_LoadSSDLiveCodeBenchV6JSONL_Good(t *testing.T) {
	// Two records; only the one inside the v6 contest window survives the filter.
	raw := `{"id":"in","prompt":"in window","contest_date":"2025-03-15","tests":["t"]}
{"id":"out","prompt":"out of window","contest_date":"2024-01-01","tests":["t"]}`
	samples, err := LoadSSDLiveCodeBenchV6JSONL(raw)
	if err != nil {
		t.Fatalf("LoadSSDLiveCodeBenchV6JSONL error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "in" {
		t.Fatalf("v6 samples = %+v, want only the in-window record", samples)
	}
}

func TestSSD_LoadSSDLiveCodeBenchV6JSONL_Bad(t *testing.T) {
	// A valid record outside the contest window leaves the v6 set empty → error.
	_, err := LoadSSDLiveCodeBenchV6JSONL(`{"id":"old","prompt":"ancient","contest_date":"2020-01-01","tests":["t"]}`)
	if err == nil {
		t.Fatal("LoadSSDLiveCodeBenchV6JSONL(no v6 rows) error = nil, want no-samples error")
	}
}

func TestSSD_FilterSSDLiveCodeBenchV6Samples_Good(t *testing.T) {
	in := []SSDCodeBenchmarkSample{
		{ID: "keep", Prompt: "p", Meta: map[string]string{"contest_date": "2025-04-01"}},
		{ID: "early", Prompt: "p", Meta: map[string]string{"contest_date": "2025-01-31"}},
		{ID: "late", Prompt: "p", Meta: map[string]string{"contest_date": "2025-06-01"}},
		{ID: "no-date", Prompt: "p"},
	}
	out := FilterSSDLiveCodeBenchV6Samples(in)
	if len(out) != 1 || out[0].ID != "keep" {
		t.Fatalf("filtered = %+v, want only [keep]", out)
	}
}

func TestSSD_FilterSSDLiveCodeBenchV6Samples_Ugly(t *testing.T) {
	// Empty input returns an empty (non-nil) slice, never a panic.
	out := FilterSSDLiveCodeBenchV6Samples(nil)
	if len(out) != 0 {
		t.Fatalf("filtered nil = %+v, want empty", out)
	}
}

func TestSSD_LoadSSDCodeBenchmarkJSONLFile_Good(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bench.jsonl")
	if res := core.WriteFile(path, []byte(`{"id":"f1","prompt":"file task","tests":["t"]}`), 0o644); !res.OK {
		t.Fatalf("write fixture: %v", res.Value)
	}
	samples, err := LoadSSDCodeBenchmarkJSONLFile(path)
	if err != nil {
		t.Fatalf("LoadSSDCodeBenchmarkJSONLFile error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "f1" {
		t.Fatalf("samples = %+v, want [f1]", samples)
	}
}

func TestSSD_LoadSSDCodeBenchmarkJSONLFile_Bad(t *testing.T) {
	_, err := LoadSSDCodeBenchmarkJSONLFile(filepath.Join(t.TempDir(), "missing.jsonl"))
	if err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONLFile(missing) error = nil, want read error")
	}
}

func TestSSD_LoadSSDLiveCodeBenchV6JSONLFile_Good(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v6.jsonl")
	raw := `{"id":"v6","prompt":"window task","contest_date":"2025-05-31","tests":["t"]}`
	if res := core.WriteFile(path, []byte(raw), 0o644); !res.OK {
		t.Fatalf("write fixture: %v", res.Value)
	}
	samples, err := LoadSSDLiveCodeBenchV6JSONLFile(path)
	if err != nil {
		t.Fatalf("LoadSSDLiveCodeBenchV6JSONLFile error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "v6" {
		t.Fatalf("samples = %+v, want [v6]", samples)
	}
}

func TestSSD_LoadSSDLiveCodeBenchV6JSONLFile_Bad(t *testing.T) {
	_, err := LoadSSDLiveCodeBenchV6JSONLFile(filepath.Join(t.TempDir(), "missing.jsonl"))
	if err == nil {
		t.Fatal("LoadSSDLiveCodeBenchV6JSONLFile(missing) error = nil, want read error")
	}
}

func TestSSD_SSDPostProcessCode_Good(t *testing.T) {
	// The last fenced python block is extracted, language line dropped.
	response := "Here is my answer:\n```python\nprint('hi')\n```\nDone."
	code, ok := SSDPostProcessCode(response)
	if !ok {
		t.Fatal("SSDPostProcessCode ok = false, want true")
	}
	if code != "print('hi')\n" {
		t.Fatalf("code = %q, want %q", code, "print('hi')\n")
	}
}

func TestSSD_SSDPostProcessCode_Bad(t *testing.T) {
	// No fenced block → no code, no panic.
	code, ok := SSDPostProcessCode("just prose, no fences here")
	if ok {
		t.Fatalf("SSDPostProcessCode ok = true (code=%q), want false", code)
	}
}

func TestSSD_SSDPostProcessCode_Ugly(t *testing.T) {
	// Multiple fences: the LAST one wins.
	response := "```python\nfirst()\n```\nthen\n```python\nlast()\n```"
	code, ok := SSDPostProcessCode(response)
	if !ok || code != "last()\n" {
		t.Fatalf("code = %q ok=%v, want last()\\n / true", code, ok)
	}
}

func TestSSD_FormatSSDLiveCodeBenchPrompt_Good(t *testing.T) {
	// Default (stdin) shape: stdin/stdout instruction + the problem text.
	sample := SSDCodeBenchmarkSample{Prompt: "Sum the input numbers."}
	prompt := FormatSSDLiveCodeBenchPrompt(sample)
	if prompt == "" {
		t.Fatal("FormatSSDLiveCodeBenchPrompt = empty, want non-empty")
	}
	if !core.Contains(prompt, "Sum the input numbers.") {
		t.Fatalf("prompt missing problem text: %q", prompt)
	}
	if !core.Contains(prompt, "stdin") {
		t.Fatalf("default prompt should mention stdin: %q", prompt)
	}
}

func TestSSD_FormatSSDLiveCodeBenchPrompt_Bad(t *testing.T) {
	// Blank problem → empty prompt (the caller falls back to the raw prompt).
	if got := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{Prompt: "   "}); got != "" {
		t.Fatalf("FormatSSDLiveCodeBenchPrompt(blank) = %q, want empty", got)
	}
}

func TestSSD_FormatSSDLiveCodeBenchPrompt_Ugly(t *testing.T) {
	// Function-call (non-stdin) shape with a starter entry point: the
	// rendered prompt switches to the "Starter code" instruction form.
	sample := SSDCodeBenchmarkSample{
		Prompt: "Implement the function.",
		Meta: map[string]string{
			"is_stdin":    "false",
			"entry_point": "def solve(nums):",
		},
	}
	prompt := FormatSSDLiveCodeBenchPrompt(sample)
	if !core.Contains(prompt, "Starter code:") || !core.Contains(prompt, "def solve(nums):") {
		t.Fatalf("function-call prompt missing starter code: %q", prompt)
	}
}
