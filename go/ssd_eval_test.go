// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"strings"
	"testing"

	core "dappco.re/go"
)

func TestRunSimpleSelfDistillationCodeBenchmark_RepeatsAndWritesReport_Good(t *testing.T) {
	outputPath := core.PathJoin(t.TempDir(), "reports", "lcb.json")
	var prompts []string
	var configs []GenerateConfig
	var executed []string

	report, err := RunSimpleSelfDistillationCodeBenchmark(context.Background(), SimpleSelfDistillationCodeBenchmarkRunner{
		Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (string, error) {
			prompts = append(prompts, prompt)
			configs = append(configs, cfg)
			if strings.Contains(prompt, "add") {
				return "```python\ndef add(a, b): return a + b\n```", nil
			}
			return "```python\ndef sub(a, b): return a - b\n```", nil
		},
		RunTests: func(_ context.Context, sample SimpleSelfDistillationCodeBenchmarkSample, candidate SimpleSelfDistillationCodeCandidate) (SimpleSelfDistillationCodeExecution, error) {
			executed = append(executed, sample.ID+"/"+candidate.Solution)
			return SimpleSelfDistillationCodeExecution{
				Passed:      strings.Contains(candidate.Solution, "+"),
				TotalTests:  len(sample.Tests),
				PassedTests: boolToCodeBenchmarkPassedTests(strings.Contains(candidate.Solution, "+"), len(sample.Tests)),
				DurationMS:  12,
			}, nil
		},
	}, []SimpleSelfDistillationCodeBenchmarkSample{
		{ID: "add", Prompt: "write add", Tests: []string{"assert add(1, 2) == 3"}},
		{ID: "sub", Prompt: "write sub", Tests: []string{"assert sub(3, 1) == 2"}},
	}, SimpleSelfDistillationCodeBenchmarkConfig{
		Benchmark:  "LiveCodeBench-v6",
		NRepeat:    2,
		Seeds:      []uint64{7, 1234},
		OutputPath: outputPath,
		Generate: GenerateConfig{
			MaxTokens:     128,
			Temperature:   0.8,
			TopP:          0.95,
			TopK:          64,
			RepeatPenalty: 1.1,
		},
	})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillationCodeBenchmark() error = %v", err)
	}
	if len(prompts) != 4 || len(executed) != 4 {
		t.Fatalf("generated=%d executed=%d, want n_repeat per sample", len(prompts), len(executed))
	}
	if configs[0].MaxTokens != 128 || configs[0].Temperature != 0.8 || configs[0].TopP != 0.95 || configs[0].TopK != 64 || configs[0].RepeatPenalty != 1.1 {
		t.Fatalf("generate config = %+v, want caller sampling config", configs[0])
	}
	if len(configs) != 4 || !configs[0].SeedSet || configs[0].Seed != 7 || configs[1].Seed != 8 || configs[2].Seed != 7 || configs[3].Seed != 8 {
		t.Fatalf("generate seeds = %+v, want seed base plus repeat per sample", configs)
	}
	if report.Benchmark != "LiveCodeBench-v6" || report.Config.NRepeat != 2 || report.Config.OutputPath != outputPath {
		t.Fatalf("report config = %+v benchmark=%q", report.Config, report.Benchmark)
	}
	if report.Metrics.Samples != 2 || report.Metrics.Candidates != 4 || report.Metrics.Passed != 2 || report.Metrics.PassRate != 0.5 {
		t.Fatalf("metrics = %+v, want 2/4 candidates passing", report.Metrics)
	}
	if report.Metrics.PassAtK["pass@1"] != 0.5 {
		t.Fatalf("pass@k = %+v, want pass@1=0.5", report.Metrics.PassAtK)
	}
	if len(report.Results) != 2 || len(report.Results[0].Candidates) != 2 {
		t.Fatalf("results = %+v, want per-sample candidate results", report.Results)
	}
	if !report.Results[0].Candidates[0].Candidate.HasCode || !strings.Contains(report.Results[0].Candidates[0].Candidate.RawSolution, "```python") {
		t.Fatalf("candidate = %+v, want raw fenced output and extracted code marker", report.Results[0].Candidates[0].Candidate)
	}
	data := core.ReadFile(outputPath)
	if !data.OK {
		t.Fatalf("ReadFile(%s) error = %v", outputPath, data.Value)
	}
	if !strings.Contains(string(data.Value.([]byte)), `"benchmark": "LiveCodeBench-v6"`) {
		t.Fatalf("report file = %s, want benchmark JSON", string(data.Value.([]byte)))
	}
}

func TestRunSimpleSelfDistillationCodeBenchmark_DefaultsAndValidation_Bad(t *testing.T) {
	_, err := RunSimpleSelfDistillationCodeBenchmark(context.Background(), SimpleSelfDistillationCodeBenchmarkRunner{}, nil, SimpleSelfDistillationCodeBenchmarkConfig{})
	if err == nil {
		t.Fatal("RunSimpleSelfDistillationCodeBenchmark() error = nil, want missing Generate")
	}
	_, err = RunSimpleSelfDistillationCodeBenchmark(context.Background(), SimpleSelfDistillationCodeBenchmarkRunner{
		Generate: func(context.Context, string, GenerateConfig) (string, error) { return "", nil },
	}, nil, SimpleSelfDistillationCodeBenchmarkConfig{})
	if err == nil {
		t.Fatal("RunSimpleSelfDistillationCodeBenchmark() error = nil, want missing RunTests")
	}

	report, err := RunSimpleSelfDistillationCodeBenchmark(context.Background(), SimpleSelfDistillationCodeBenchmarkRunner{
		Generate: func(context.Context, string, GenerateConfig) (string, error) { return "solution", nil },
		RunTests: func(context.Context, SimpleSelfDistillationCodeBenchmarkSample, SimpleSelfDistillationCodeCandidate) (SimpleSelfDistillationCodeExecution, error) {
			return SimpleSelfDistillationCodeExecution{Passed: true, TotalTests: 1, PassedTests: 1}, nil
		},
	}, []SimpleSelfDistillationCodeBenchmarkSample{{Prompt: "p"}}, SimpleSelfDistillationCodeBenchmarkConfig{})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillationCodeBenchmark(defaults) error = %v", err)
	}
	if report.Config.NRepeat != 1 || report.Config.Generate.MaxTokens != defaultSimpleSelfDistillationMaxTokens {
		t.Fatalf("default config = %+v", report.Config)
	}
}

func TestRunSimpleSelfDistillationCodeBenchmark_PassAtK_Good(t *testing.T) {
	calls := map[string]int{}
	report, err := RunSimpleSelfDistillationCodeBenchmark(context.Background(), SimpleSelfDistillationCodeBenchmarkRunner{
		Generate: func(_ context.Context, prompt string, _ GenerateConfig) (string, error) {
			call := calls[prompt]
			calls[prompt] = call + 1
			return core.Sprintf("```python\n%s/%d\n```", prompt, call), nil
		},
		RunTests: func(_ context.Context, _ SimpleSelfDistillationCodeBenchmarkSample, candidate SimpleSelfDistillationCodeCandidate) (SimpleSelfDistillationCodeExecution, error) {
			solution := core.Trim(candidate.Solution)
			return SimpleSelfDistillationCodeExecution{
				Passed:      strings.HasSuffix(solution, "/0") || strings.HasSuffix(solution, "/1"),
				TotalTests:  1,
				PassedTests: boolToCodeBenchmarkPassedTests(strings.HasSuffix(solution, "/0") || strings.HasSuffix(solution, "/1"), 1),
			}, nil
		},
	}, []SimpleSelfDistillationCodeBenchmarkSample{
		{ID: "a", Prompt: "a", Tests: []string{"test"}, Meta: map[string]string{"difficulty": "easy"}},
		{ID: "b", Prompt: "b", Tests: []string{"test"}, Meta: map[string]string{"difficulty": "hard"}},
	}, SimpleSelfDistillationCodeBenchmarkConfig{NRepeat: 10})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillationCodeBenchmark() error = %v", err)
	}
	if math.Abs(report.Metrics.PassAtK["pass@1"]-0.2) > 0.000001 {
		t.Fatalf("pass@1 = %f, want 0.2", report.Metrics.PassAtK["pass@1"])
	}
	if math.Abs(report.Metrics.PassAtK["pass@5"]-0.777777) > 0.000001 {
		t.Fatalf("pass@5 = %f, want estimated 0.777777", report.Metrics.PassAtK["pass@5"])
	}
	if _, ok := report.Metrics.PassAtK["pass@10"]; ok {
		t.Fatalf("pass@k = %+v, did not want pass@10 for n_repeat=10", report.Metrics.PassAtK)
	}
	if math.Abs(report.Metrics.Difficulty["pass@5_easy"]-0.777777) > 0.000001 || math.Abs(report.Metrics.Difficulty["pass@5_hard"]-0.777777) > 0.000001 {
		t.Fatalf("difficulty metrics = %+v, want pass@5 per difficulty", report.Metrics.Difficulty)
	}
}

func TestLoadSimpleSelfDistillationCodeBenchmarkJSONL_Good(t *testing.T) {
	raw := `{"question_id":"q1","question_content":"Write add.","starter_code":"def add(a,b):\n    pass","entry_point":"def add(a,b):\n    pass","is_stdin":false,"contest_date":"2025-03-01","public_test_cases":["assert add(1, 2) == 3"],"private_test_cases":["assert add(-1, 1) == 0"],"difficulty":"easy","platform":"leetcode"}`
	raw += "\n"
	raw += `{"id":"q2","prompt":"Write sub.","test":"assert sub(3, 1) == 2"}`
	samples, err := LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw)
	if err != nil {
		t.Fatalf("LoadSimpleSelfDistillationCodeBenchmarkJSONL() error = %v", err)
	}
	if len(samples) != 2 {
		t.Fatalf("samples = %d, want 2", len(samples))
	}
	if samples[0].ID != "q1" || !strings.Contains(samples[0].Prompt, "Write add.") || !strings.Contains(samples[0].Prompt, "starter code") {
		t.Fatalf("sample[0] = %+v, want id and starter-code prompt", samples[0])
	}
	if len(samples[0].Tests) != 2 || samples[0].Tests[0] != "assert add(1, 2) == 3" || samples[0].Meta["difficulty"] != "easy" || samples[0].Meta["platform"] != "leetcode" ||
		samples[0].Meta["entry_point"] == "" || samples[0].Meta["is_stdin"] != "false" || samples[0].Meta["contest_date"] != "2025-03-01" {
		t.Fatalf("sample[0] tests/meta = %+v/%+v", samples[0].Tests, samples[0].Meta)
	}
	if samples[1].ID != "q2" || samples[1].Tests[0] != "assert sub(3, 1) == 2" {
		t.Fatalf("sample[1] = %+v", samples[1])
	}
}

func TestLoadSimpleSelfDistillationCodeBenchmarkJSONLFile_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "lcb.jsonl")
	write := core.WriteFile(path, []byte(`{"id":"q","prompt":"Write identity.","tests":["assert f(1) == 1"]}`+"\n"), 0o644)
	if !write.OK {
		t.Fatalf("WriteFile() error = %v", write.Value)
	}
	samples, err := LoadSimpleSelfDistillationCodeBenchmarkJSONLFile(path)
	if err != nil {
		t.Fatalf("LoadSimpleSelfDistillationCodeBenchmarkJSONLFile() error = %v", err)
	}
	if len(samples) != 1 || samples[0].Tests[0] != "assert f(1) == 1" {
		t.Fatalf("samples = %+v", samples)
	}
}

func TestLoadSimpleSelfDistillationLiveCodeBenchV6JSONL_Good(t *testing.T) {
	raw := `{"id":"jan","prompt":"old","contest_date":"2025-01-31"}`
	raw += "\n"
	raw += `{"id":"feb","prompt":"first v6","contest_date":"2025-02-01","difficulty":"easy"}`
	raw += "\n"
	raw += `{"id":"may","prompt":"last v6","contest_date":"2025-05-31","difficulty":"hard"}`
	raw += "\n"
	raw += `{"id":"jun","prompt":"new","contest_date":"2025-06-01"}`

	samples, err := LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(raw)
	if err != nil {
		t.Fatalf("LoadSimpleSelfDistillationLiveCodeBenchV6JSONL() error = %v", err)
	}
	if len(samples) != 2 || samples[0].ID != "feb" || samples[1].ID != "may" {
		t.Fatalf("samples = %+v, want Feb-May 2025 subset", samples)
	}
	if samples[0].Meta["difficulty"] != "easy" || samples[1].Meta["difficulty"] != "hard" {
		t.Fatalf("sample metadata = %+v/%+v", samples[0].Meta, samples[1].Meta)
	}

	_, err = LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(`{"id":"old","prompt":"old","contest_date":"2025-01-01"}`)
	if err == nil {
		t.Fatal("LoadSimpleSelfDistillationLiveCodeBenchV6JSONL() error = nil, want empty v6 subset")
	}
}

func TestLoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "lcb-v6.jsonl")
	write := core.WriteFile(path, []byte(`{"id":"q","prompt":"Write identity.","contest_date":"2025-03-15","tests":["assert f(1) == 1"]}`+"\n"), 0o644)
	if !write.OK {
		t.Fatalf("WriteFile() error = %v", write.Value)
	}
	samples, err := LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile(path)
	if err != nil {
		t.Fatalf("LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile() error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "q" || samples[0].Meta["contest_date"] != "2025-03-15" {
		t.Fatalf("samples = %+v", samples)
	}
}

func TestFormatSimpleSelfDistillationLiveCodeBenchPrompt_Good(t *testing.T) {
	stdinPrompt := FormatSimpleSelfDistillationLiveCodeBenchPrompt(SimpleSelfDistillationCodeBenchmarkSample{
		Prompt: "Add two numbers.",
		Meta:   map[string]string{"is_stdin": "true"},
	})
	if !strings.Contains(stdinPrompt, "stdin") || !strings.Contains(stdinPrompt, "Add two numbers.") {
		t.Fatalf("stdin prompt = %q", stdinPrompt)
	}
	functionPrompt := FormatSimpleSelfDistillationLiveCodeBenchPrompt(SimpleSelfDistillationCodeBenchmarkSample{
		Prompt: "Implement add.",
		Meta:   map[string]string{"is_stdin": "false", "entry_point": "def add(a, b):\n    pass"},
	})
	if !strings.Contains(functionPrompt, "Starter code") || !strings.Contains(functionPrompt, "def add") {
		t.Fatalf("function prompt = %q", functionPrompt)
	}
}

func TestSimpleSelfDistillationPostProcessCode_Good(t *testing.T) {
	response := "analysis\n```go\nnot this\n```\nfinal\n```python\n<code>def add(a, b):\n    return a + b</code>\n```\n"
	code, ok := SimpleSelfDistillationPostProcessCode(response)
	if !ok {
		t.Fatal("SimpleSelfDistillationPostProcessCode() ok = false")
	}
	if core.Trim(code) != "def add(a, b):\n    return a + b" {
		t.Fatalf("code = %q", code)
	}
	if code, ok := SimpleSelfDistillationPostProcessCode("no fenced code"); ok || code != "" {
		t.Fatalf("missing fence = %q/%t, want empty false", code, ok)
	}
}

func boolToCodeBenchmarkPassedTests(pass bool, total int) int {
	if pass {
		return total
	}
	return 0
}
